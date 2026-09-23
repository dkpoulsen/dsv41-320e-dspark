#!/usr/bin/env python3
"""Build a REAP saliency calibration corpus from LOCAL text — no network, no `datasets`.

Why local text is the right choice here
---------------------------------------
The REAP authors stress that saliency should be measured over the distribution you
actually serve, and note that this model's evaluation profile is dominated by
agentic/code workloads. This node serves a coding agent (long technical reasoning, code,
tool output). So a corpus of Python source + technical prose is a *better* match than
generic web text — and it avoids both a network dependency and the `datasets` package
whose install upgrades NCCL inside the serving image.

Output format matches what reap_calibrate.py expects:
    int32 array of [n_seq, seq_len] token ids, saved with .npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def collect_text(roots: list[str], max_bytes: int) -> str:
    """Walk the given roots and concatenate text/code files up to max_bytes."""
    exts = {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh",
        ".c", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".rs", ".go", ".js", ".ts",
        ".proto", ".cfg", ".ini", ".rst", ".tex",
    }
    chunks: list[str] = []
    total = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # keep the walk on-topic and bounded
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "node_modules", "__pycache__", ".venv", "build",
                             ".cache", "dist", ".mypy_cache", ".pytest_cache"}
            ]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > 2_000_000:
                        continue
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        t = f.read()
                except OSError:
                    continue
                if len(t) < 200:
                    continue
                chunks.append(t)
                total += len(t)
                if total >= max_bytes:
                    break
            if total >= max_bytes:
                break
        if total >= max_bytes:
            break
    return "\n\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (for the tokenizer)")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--roots", nargs="+", required=True, help="dirs to harvest text from")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--max-bytes", type=int, default=40_000_000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    want_tokens = args.n_seq * args.seq_len
    print(f"[calib] target {want_tokens:,} tokens = {args.n_seq} x {args.seq_len}", flush=True)

    text = collect_text(args.roots, args.max_bytes)
    print(f"[calib] harvested {len(text):,} chars from {len(args.roots)} roots", flush=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"[calib] tokenized -> {len(ids):,} tokens", flush=True)

    if len(ids) < want_tokens:
        print(
            f"[calib] WARNING: only {len(ids):,} tokens available for {want_tokens:,} wanted; "
            f"reducing to {len(ids) // args.seq_len} sequences",
            flush=True,
        )
        args.n_seq = max(1, len(ids) // args.seq_len)
        want_tokens = args.n_seq * args.seq_len

    arr = np.array(ids[:want_tokens], dtype=np.int32).reshape(args.n_seq, args.seq_len)
    np.save(args.out, arr)
    print(f"[calib] wrote {args.out}: shape {arr.shape}, dtype {arr.dtype}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
