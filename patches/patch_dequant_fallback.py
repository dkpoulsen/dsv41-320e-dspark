#!/usr/bin/env python3
"""Make the REAP reference implementation runnable on sm_80 by dequantizing on the fly.

THE PROBLEM
-----------
`<ckpt>/inference/model.py::linear()` dispatches on weight dtype:

    fp4  -> act_quant + fp4_gemm   (TileLang, needs FP4 MMA)
    fp8  -> act_quant + fp8_gemm   (TileLang, needs FP8 MMA)
    else -> F.linear               (plain matmul -- works everywhere, incl. sm_80)

sm_80 has neither FP8 nor FP4 tensor-core MMA, so the first two branches die at kernel
launch (`CUDA_ERROR_ASSERT` / `txm.error.InternalError`). That is why saliency collection
was blocked on this hardware.

THE FIX
-------
Add a `DEQUANT_FALLBACK` path to `linear()`: when the weight is fp4 or fp8 AND the
fallback is enabled, dequantize the weight to bf16 and call `F.linear`. Both formats
carry their scales on the weight object itself (`weight.scale`, set in `Linear.__init__`),
so the dequantization needs no extra plumbing:

    fp4 : weight [N, K/2] packed E2M1  + scale [N, K/32] E8M0   -> [N, K] bf16
    fp8 : weight [N, K]  E4M3          + scale [N/32, K/32] E8M0 -> [N, K] bf16

Cost: the dequantized weight is 8x (fp4) / 2x (fp8) larger in VRAM. Calibration is a
one-off pass where throughput is irrelevant, so this is acceptable. The maths is exact
for fp8 (E4M3 is exactly representable in bf16) and near-exact for fp4 (E2M1 values are
0, .5, 1, 1.5, 2, 3, 4, 6 -- all exactly representable in bf16; only the scale product
rounds).

USAGE
-----
    python patch_dequant_fallback.py <inference-dir>
"""
from __future__ import annotations

import os
import sys


PATCH_MARKER = "# --- DSV41_SM80_DEQUANT_FALLBACK ---"

DEQUANT_BLOCK = '''

# --- DSV41_SM80_DEQUANT_FALLBACK ---
# See patch_dequant_fallback.py. Enabled by DSV41_DEQUANT_FALLBACK=1.
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_fp4_to_bf16(weight, scale, block_size):
    """[N, K/2] packed E2M1 + [N, K/32] E8M0 -> [N, K] bf16 (MX block format, pack along K)."""
    raw = weight.view(torch.uint8).to(torch.int16)
    n, k_half = raw.shape
    k = k_half * 2
    lo = (raw & 0x0F).to(torch.long)
    hi = ((raw >> 4) & 0x0F).to(torch.long)
    vals = torch.empty(n, k, dtype=torch.float32, device=weight.device)
    vals[:, 0::2] = _FP4_TABLE.to(weight.device)[lo.reshape(n, -1)]
    vals[:, 1::2] = _FP4_TABLE.to(weight.device)[hi.reshape(n, -1)]
    s = scale.view(torch.uint8).to(torch.float32)
    s = torch.pow(2.0, s - 127.0)                       # E8M0: byte is the exponent
    s = s.repeat_interleave(block_size, dim=1)
    if s.shape[1] < k:
        s = torch.nn.functional.pad(s, (0, k - s.shape[1]), value=1.0)
    else:
        s = s[:, :k]
    return (vals * s).to(torch.bfloat16)


def _dequant_fp8_to_bf16(weight, scale, block_size):
    """[N, K] E4M3 + [ceil(N/b), ceil(K/b)] E8M0 -> [N, K] bf16 (32x32 blockwise)."""
    n, k = weight.shape
    ob = max(1, n // scale.shape[0])
    ib = max(1, k // scale.shape[1])
    w = weight.float().unflatten(0, (-1, ob)).unflatten(-1, (-1, ib))
    s = scale.view(torch.uint8).to(torch.float32).unsqueeze(1).unsqueeze(-1)
    s = torch.pow(2.0, s - 127.0)                       # E8M0
    out = (w * s).flatten(2, 3).flatten(0, 1)
    if out.shape[0] != n or out.shape[1] != k:
        out = out[:n, :k]
    return out.to(torch.bfloat16)


def _dequant_fallback_enabled() -> bool:
    return os.environ.get("DSV41_DEQUANT_FALLBACK", "0") == "1"


def _dequant_weight(weight) -> torch.Tensor:
    """Dequantize a quantized weight to bf16 using its attached scale."""
    scale = getattr(weight, "scale", None)
    if scale is None:
        raise RuntimeError("dequant fallback: weight has no attached .scale")
    if weight.dtype == torch.float4_e2m1fn_x2:
        return _dequant_fp4_to_bf16(weight, scale, fp4_block_size)
    return _dequant_fp8_to_bf16(weight, scale, fp8_block_size)
# --- END DSV41_SM80_DEQUANT_FALLBACK ---

'''


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    p = os.path.join(d, "model.py")
    src = open(p).read()
    if PATCH_MARKER in src:
        print(f"already patched: {p}")
        return 0

    # 1. insert the helpers just before `def linear(`
    anchor = "\ndef linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:"
    if anchor not in src:
        print("ERROR: could not find `def linear(` anchor", file=sys.stderr)
        return 1
    src = src.replace(anchor, DEQUANT_BLOCK + anchor[1:], 1)

    # 2. add the fallback branch at the top of linear()'s body
    body_anchor = (
        '    """Pick a GEMM from the weight dtype. Quantized weights need a quantized activation, and both\n'
        "    fp4 and fp8 weights take an fp8 one -- for fp4 the kernel handles the mixed precision.\"\"\"\n"
        "    assert bias is None\n"
    )
    if body_anchor not in src:
        print("ERROR: could not find linear() body anchor", file=sys.stderr)
        return 1
    fallback = (
        body_anchor
        + "\n"
        + "    # sm_80 has no FP8/FP4 tensor-core MMA, so the TileLang GEMM branches below\n"
        + "    # cannot launch there. Dequantize to bf16 and use a plain matmul instead.\n"
        + "    if _dequant_fallback_enabled() and weight.dtype in (\n"
        + "        torch.float4_e2m1fn_x2,\n"
        + "        torch.float8_e4m3fn,\n"
        + "    ):\n"
        + "        return F.linear(x, _dequant_weight(weight), bias)\n"
    )
    src = src.replace(body_anchor, fallback, 1)

    open(p, "w").write(src)
    print(f"patched {p}: dequant fallback installed (set DSV41_DEQUANT_FALLBACK=1 to enable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
