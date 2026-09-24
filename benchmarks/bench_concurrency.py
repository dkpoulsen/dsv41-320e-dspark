#!/usr/bin/env python3
"""Measure decode throughput and tau at several concurrency levels.

The dynamic-spec-token schedule exists because the optimal draft budget depends on
batch size: at batch 1 a long draft is nearly free, at high concurrency each extra
draft token costs a full extra forward across the whole batch. This measures that
curve so the schedule can be set from data instead of guessed.
"""
import argparse, json, os, statistics, sys, threading, time
import requests

BASE = os.environ.get("DSV41_URL", "http://localhost:9006")
PROMPT = ("Write a detailed technical explanation of how mixture-of-experts routing works, "
          "covering gating, capacity factors, load balancing and expert parallelism. "
          "Be thorough and do not stop early.")
COUNTERS = {"accepted": "vllm:spec_decode_num_accepted_tokens_total",
            "drafts": "vllm:spec_decode_num_drafts_total",
            "draft": "vllm:spec_decode_num_draft_tokens_total"}


def snap():
    out = {}
    try:
        for line in requests.get(f"{BASE}/metrics", timeout=60).text.splitlines():
            if line.startswith("#"):
                continue
            for k, name in COUNTERS.items():
                if line.startswith(name) and (len(line) == len(name) or line[len(name)] in "{ "):
                    out[k] = float(line.rsplit(None, 1)[1])
    except Exception:
        pass
    return out


def worker(results, idx, max_tokens):
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.0,
        }, timeout=3600)
        d = r.json()
        wall = time.time() - t0
        results[idx] = (d.get("usage", {}).get("completion_tokens", 0), wall)
    except Exception as e:
        results[idx] = (0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="conc")
    a = ap.parse_args()

    out = {"label": a.label, "levels": {}}
    print(f"  {'conc':>5}{'agg tok/s':>11}{'per-req':>9}{'tau':>7}{'accept':>8}")
    print("  " + "-"*42)
    for c in a.levels:
        s0 = snap()
        t0 = time.time()
        res = [None] * c
        ths = [threading.Thread(target=worker, args=(res, i, a.max_tokens)) for i in range(c)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        s1 = snap()
        total = sum(r[0] for r in res if r)
        agg = total / wall if wall else 0
        acc = s1.get("accepted", 0) - s0.get("accepted", 0)
        dr = s1.get("drafts", 0) - s0.get("drafts", 0)
        drt = s1.get("draft", 0) - s0.get("draft", 0)
        tau = (acc + dr) / dr if dr else 0
        accr = acc / drt if drt else 0
        out["levels"][c] = {"agg_tok_s": round(agg, 2), "tau": round(tau, 3),
                            "acceptance": round(accr, 4), "total_tokens": total,
                            "wall_s": round(wall, 1)}
        print(f"  {c:>5}{agg:>11.1f}{agg/c:>9.1f}{tau:>7.2f}{accr:>8.3f}")
    with open(f"/home/dkp/bench-{a.label}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  -> /home/dkp/bench-{a.label}.json")


if __name__ == "__main__":
    sys.exit(main())
