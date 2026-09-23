#!/usr/bin/env python3
"""Long-context needle with a generous token budget, separating reasoning from content.

The failure mode this catches: a thinking model that spends the whole max_tokens budget
in its `reasoning` field and returns empty `content` with finish_reason=length. That
looks exactly like a retrieval MISS in a naive harness and is not one.
"""
import argparse, os, random, sys, time
import requests

URL = os.environ.get("DSV41_URL", "http://localhost:9006") + "/v1/chat/completions"
NEEDLE = "ORCHID-MAGNET-42"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=int, default=32000)
    ap.add_argument("--depth", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    body = " ".join(
        f"Note {i}: the survey recorded {rng.randint(100,999)} sites." for i in range(a.sentences)
    )
    cut = int(len(body) * a.depth)
    prompt = (body[:cut] + f" The secret vault passphrase is {NEEDLE}." + body[cut:]
              + " What is the secret vault passphrase? Answer with the passphrase only.")

    t0 = time.time()
    r = requests.post(URL, json={
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.max_tokens, "temperature": 0,
    }, timeout=3000)
    d = r.json()
    ch = d["choices"][0]
    m = ch["message"]
    # Some builds put the streamed text in content, others in reasoning, others both.
    content = m.get("content") or ""
    reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
    wall = time.time() - t0
    pt = d.get("usage", {}).get("prompt_tokens", 0)
    hit_c, hit_r = NEEDLE in content, NEEDLE in reasoning
    print(f"  prompt {pt:,} tok | wall {wall:.0f}s | prefill {pt/wall:,.0f} tok/s")
    print(f"  finish={ch.get('finish_reason')}  content_len={len(content)}  reasoning_len={len(reasoning)}")
    print(f"  needle in content={hit_c}  in reasoning={hit_r}  -> {'HIT' if (hit_c or hit_r) else 'MISS'}")
    print(f"  answer: {content.strip()[:90]!r}")
    if not (hit_c or hit_r) and ch.get("finish_reason") == "length":
        print("  NOTE: budget exhausted (finish=length) — raise --max-tokens before calling this a MISS")
    return 0 if (hit_c or hit_r) else 1


if __name__ == "__main__":
    sys.exit(main())
