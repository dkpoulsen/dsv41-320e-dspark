#!/usr/bin/env python3
"""Measure held-out perplexity through the SERVING stack (not the reference impl).

Why through the server: it measures the artifact that actually ships — the pruned
checkpoint, the Marlin MXFP4 path, fp8 KV, the pp5 pipeline. The reference-impl ladder
the REAP card publishes is measured a different way; this is our number.

Method: send a held-out text as a completion prompt with `prompt_logprobs`, sum the
log-probability of each token *given its full prefix* (standard causal NLL), and take
exp(-mean NLL). Only positions 1..n-1 count (position 0 has no prefix).

Usage:
    perplexity.py --url http://localhost:9004/v1 --label 320E --n-chunks 24 --chunk-tokens 1024
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import requests


def harvest(paths: list[str], max_bytes: int) -> str:
    exts = {".py", ".md", ".txt", ".rst", ".c", ".cu", ".h", ".cpp", ".rs", ".go", ".js", ".ts"}
    chunks, total = [], 0
    for root in paths:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in {".git", "__pycache__", "node_modules", ".venv"}]
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) > 1_000_000:
                        continue
                    t = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if len(t) > 400:
                    chunks.append(t)
                    total += len(t)
                    if total >= max_bytes:
                        break
            if total >= max_bytes:
                break
        if total >= max_bytes:
            break
    return "\n\n".join(chunks)


def nll(client: requests.Session, url: str, model: str, token_ids: list[int], timeout: int):
    """Return (sum_nll, n_scored) for token_ids[1:] given their prefixes."""
    body = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,   # 0 => just the chosen token's logprob
    }
    r = client.post(f"{url}/completions", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    pl = j["choices"][0].get("prompt_logprobs")
    if not pl:
        raise RuntimeError("server returned no prompt_logprobs")
    total = 0.0
    n = 0
    # pl is a list aligned with prompt positions; entry 0 is None (no prefix)
    for i in range(1, len(token_ids)):
        entry = pl[i] if i < len(pl) else None
        if not entry:
            continue
        # with prompt_logprobs=0 the entry is {token_id: logprob_dict}
        want = token_ids[i]
        got = None
        for k, v in entry.items():
            if int(k) == int(want):
                got = v.get("logprob") if isinstance(v, dict) else float(v)
                break
        if got is None:
            # fall back: the dict is {tok: info} only for the sampled token
            for k, v in entry.items():
                got = v.get("logprob") if isinstance(v, dict) else float(v)
                break
        if got is not None:
            total += float(got)
            n += 1
    return total, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9004/v1")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--label", default="model")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--chunk-tokens", type=int, default=1024)
    ap.add_argument("--n-chunks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ.get("TOK_DIR", "/home/dkp/models/DeepSeek-V4.1-Flash"))
    text = harvest(args.roots, 8_000_000)
    ids_all = tok(text, add_special_tokens=False)["input_ids"]
    print(f"[ppl] harvested {len(ids_all):,} tokens", flush=True)

    need = args.chunk_tokens * args.n_chunks
    if len(ids_all) < need:
        args.n_chunks = max(1, len(ids_all) // args.chunk_tokens)
        print(f"[ppl] reduced to {args.n_chunks} chunks", flush=True)

    rnd = random.Random(args.seed)
    starts = sorted(rnd.sample(range(0, len(ids_all) - args.chunk_tokens), args.n_chunks))

    s = requests.Session()
    tot_nll = 0.0
    tot_n = 0
    per_chunk = []
    t0 = time.time()
    for i, st in enumerate(starts):
        chunk = ids_all[st : st + args.chunk_tokens]
        nll_sum, n = nll(s, args.url, args.model, chunk, args.timeout)
        tot_nll += nll_sum
        tot_n += n
        per_chunk.append({"start": st, "nll": nll_sum, "n": n})
        ppl_so_far = math.exp(-tot_nll / max(tot_n, 1))
        print(
            f"  chunk {i+1}/{len(starts)}: nll {nll_sum:.2f} over {n} tok "
            f"| running ppl {ppl_so_far:.4f}",
            flush=True,
        )
    ppl = math.exp(-tot_nll / max(tot_n, 1))
    print(f"\n[{args.label}] PERPLEXITY = {ppl:.4f}  (nll/tok {tot_nll/max(tot_n,1):.4f}, {tot_n:,} tokens, {time.time()-t0:.0f}s)")
    print(json.dumps({"label": args.label, "ppl": ppl, "tokens": tot_n,
                      "nll_per_tok": tot_nll / max(tot_n, 1),
                      "per_chunk": per_chunk}))
    with open(f"/home/dkp/ppl-{args.label}.json", "w") as fh:
        json.dump({"label": args.label, "ppl": ppl, "tokens": tot_n,
                   "nll_per_tok": tot_nll / max(tot_n, 1),
                   "per_chunk": per_chunk}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
