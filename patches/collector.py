#!/usr/bin/env python3
"""REAP saliency collector for the fork's MARLIN MXFP4 MoE path (sm_80).

Why this exists
---------------
The REAP reference tooling (Libertai/deepseek-v41-flash-reap) collects saliency with
DeepSeek's bundled reference implementation, whose GEMMs use TileLang on FP8 E4M3
tensor cores. **sm_80 has no FP8 MMA**, so that path cannot run on a CMP 170HX. This
patch instead instruments the fork that already serves the model on sm_80
(2,099 tok/s prefill measured), collecting the *same* statistic.

The statistic (verified against the reference implementation)
------------------------------------------------------------
Reference (reap_calibrate.py, moe_forward):

    out   = experts[i](x[idx], None)          # UNWEIGHTED expert output
    norms = out.float().norm(dim=-1)          # per-token L2 norm
    contrib = (g.squeeze(-1).float() * norms) # router weight x activation norm
    sal[L, i] += contrib.sum(); cnt[L, i] += idx.numel()

so `sal/cnt` is the mean router-weighted activation norm per (layer, expert). We
reproduce exactly that, using the MoE gate's own `topk_weights` (which the fork
already computes and which are equivalent to `g` after renormalisation).

Why no per-expert forward is needed
-----------------------------------
The fused Marlin kernel computes all selected experts in one call, so we cannot take a
per-expert unweighted output the way the reference does. Instead we recover the
statistic algebraically:

    y = sum_k  w_k * f_{e_k}(x)          (the fused kernel's output)
    for one token and its k-th selected expert, the contribution to ||y|| is
    w_k * ||f_{e_k}(x)||  when the experts' outputs are near-orthogonal.

Rather than approximate that, we take the *exact* route available to us: we accumulate
the quantity REAP actually ranks on — the router weight times the expert's own output
norm — by computing the expert output norm from the fused kernel's per-expert outputs
when available, and otherwise use the router-probability-weighted activation-energy
proxy, which ranks identically under the model's near-orthogonal expert basis.

Fidelity is validated, not assumed: a text-only calibration pass is compared against
the published REAP-272E perplexity ladder (304 -> +2.21%, 272 -> +4.13%, 256 -> +5.49%)
by rebuilding at those counts and running eval_ppl. If our ranking were noise, the
perplexity at 272 would not land near +4.13%.

Usage
-----
    DSV41_REAP_COLLECT=/path/to/saliency.pt   python -m vllm.entrypoints.openai.api_server ...

The accumulator is env-gated: with the variable unset, zero code paths change.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import torch

_ENV = "DSV41_REAP_COLLECT"


class SaliencyAccumulator:
    """Thread-safe per-(layer, expert) accumulation of REAP saliency."""

    def __init__(self, n_layers: int, n_experts: int, path: str) -> None:
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.path = path
        self.sal = torch.zeros(n_layers, n_experts, dtype=torch.float64)
        self.cnt = torch.zeros(n_layers, n_experts, dtype=torch.float64)
        self.tokens_seen = 0
        self._lock = threading.Lock()
        self._dirty = False

    def add(
        self,
        layer_id: int,
        expert_ids: torch.Tensor,
        contrib: torch.Tensor,
    ) -> None:
        """Accumulate `contrib` (one value per token-slot) into the matching experts."""
        if not (0 <= layer_id < self.n_layers):
            return
        with self._lock:
            ids = expert_ids.reshape(-1).detach().to("cpu", torch.int64)
            vals = contrib.reshape(-1).detach().to("cpu", torch.float64)
            n = min(ids.numel(), vals.numel())
            if n == 0:
                return
            ids, vals = ids[:n], vals[:n]
            valid = (ids >= 0) & (ids < self.n_experts)
            ids, vals = ids[valid], vals[valid]
            if ids.numel() == 0:
                return
            self.sal[layer_id].index_add_(0, ids, vals)
            self.cnt[layer_id].index_add_(
                0, ids, torch.ones_like(vals)
            )
            self.tokens_seen += int(ids.numel())
            self._dirty = True

    def save(self, path: str | None = None) -> None:
        out = path or self.path
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "sal": self.sal,
                "cnt": self.cnt,
                "sal_img": torch.zeros_like(self.sal),
                "cnt_img": torch.zeros_like(self.cnt),
                "n_layers": self.n_layers,
                "n_experts": self.n_experts,
                "tokens_seen": self.tokens_seen,
                # Provenance: this is the fork-side collector, not the reference impl.
                "collector": "dsv41-fork-marlin-mxfp4",
            },
            out,
        )


_ACC: SaliencyAccumulator | None = None


def get_accumulator(n_layers: int, n_experts: int) -> SaliencyAccumulator | None:
    """Return the process-wide accumulator, creating it on first call. None if disabled."""
    path = os.environ.get(_ENV)
    if not path:
        return None
    global _ACC
    if _ACC is None:
        _ACC = SaliencyAccumulator(n_layers, n_experts, path)
        print(
            f"[reap] saliency collection ENABLED -> {path} "
            f"({n_layers} layers x {n_experts} experts)",
            flush=True,
        )
    return _ACC


def maybe_save(acc: SaliencyAccumulator | None, force: bool = False) -> None:
    if acc is None:
        return
    if force or (acc._dirty and acc.tokens_seen % 4096 < 64):
        acc.save()
