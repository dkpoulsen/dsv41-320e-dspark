#!/usr/bin/env python3
"""Harness-independent decode measurement straight from the engine's own counters.

Tokens/sec = (accepted_spec_tokens + draft_steps) / wall_time, because every draft
step emits one bonus token beyond the accepted drafts. This cannot be fooled by
SSE framing, the reasoning/content split, or early EOS, so it is the ground truth
against which the streaming harness is checked.

Also writes results JSON so numbers stop living only in terminal scrollback.
"""
import argparse, json, os, statistics, sys, time
import requests

PROMPTS = [
    "Explain how speculative decoding works, covering draft models, verification, and how acceptance rate affects effective throughput. Write at length.",
    "Describe the engineering tradeoffs between pipeline parallelism and tensor parallelism for serving large MoE models on PCIe-only multi-GPU systems.",
    "Walk through how a Mixture-of-Experts layer routes tokens to experts, and why expert parallelism interacts badly with pipeline parallelism on slow interconnects.",
    "Explain the design of sparse attention for million-token contexts, including indexer-based top-k selection and KV cache compression.",
]

COUNTERS = {
    "draft": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "steps": "vllm:spec_decode_num_drafts_total",
}


def snapshot(base):
    """Read spec-decode counters. Metric lines look like `name{labels} value`."""
    out = {}
    try:
        text = requests.get(f"{base}/metrics", timeout=60).text
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, name in COUNTERS.items():
            if line.startswith(name) and (
                len(line) == len(name) or line[len(name)] in "{ "
            ):
                try:
                    out[key] = float(line.rsplit(None, 1)[1])
                except ValueError:
                    pass
    return out


def stream_request(base, model, prompt, max_tokens):
    """Streaming request; returns (ttft, end, completion_tokens, first_tok, last_tok)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft = last = None
    n = 0
    usage = None
    with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                break
            j = json.loads(payload)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning"):
                    now = time.time()
                    if ttft is None:
                        ttft = now
                    last = now
                    n += 1
    return ttft, time.time(), (usage or {}).get("completion_tokens", n), t0


def out_harness_median(rows):
    """Fallback for non-spec-decode servers (counters absent): use the harness rate."""
    vals = [r["harness_tok_s"] for r in rows if r["harness_tok_s"]]
    return round(statistics.median(vals), 2) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("DSV41_URL", "http://localhost:9004"))
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    base = args.url.rstrip("/")

    # warm up so load-time compilation and cold faulting are excluded
    stream_request(base, args.model, PROMPTS[0], 64)

    print("per-request (fresh engine counters, so the pipeline rate is exact):")
    rows = []
    for rep in range(args.repeats):
        for i, p in enumerate(PROMPTS):
            a = snapshot(base)
            ttft, end, comp, t0 = stream_request(base, args.model, p, args.max_tokens)
            b = snapshot(base)
            acc = b.get("accepted", 0) - a.get("accepted", 0)
            steps = b.get("steps", 0) - a.get("steps", 0)
            drafted = b.get("draft", 0) - a.get("draft", 0)
            emitted = acc + steps
            wall = end - t0
            row = {
                "rep": rep, "prompt": i, "wall_s": round(wall, 3),
                "ttft_s": round(ttft - t0, 3) if ttft else None,
                "completion_tokens": comp,
                "accepted": int(acc), "steps": int(steps),
                "tau": round(emitted / steps, 3) if steps else None,
                "acceptance": round(acc / drafted if drafted else 0, 4),
                "server_tok_s": round(emitted / wall, 2) if wall else None,
                "harness_tok_s": round(comp / (end - ttft), 2) if ttft and end > ttft else None,
            }
            rows.append(row)
            print(f"  rep{rep} p{i}: server={row['server_tok_s']} tok/s  "
                  f"harness={row['harness_tok_s']}  tau={row['tau']}  "
                  f"acc={row['acceptance']}")

    srv = [r["server_tok_s"] for r in rows if r["server_tok_s"]]
    har = [r["harness_tok_s"] for r in rows if r["harness_tok_s"]]
    out = {
        "label": args.label,
        "url": base,
        "max_tokens": args.max_tokens,
        "server_tok_s_median": (round(statistics.median(srv), 2) if srv
                                else out_harness_median(rows)),
        "harness_tok_s_median": round(statistics.median(har), 2) if har else None,
        "acceptance_median": (round(statistics.median([r["acceptance"] for r in rows]), 4)
                              if any(r["accepted"] for r in rows) else None),
        "tau_median": (round(statistics.median([r["tau"] for r in rows if r["tau"]]), 3)
                       if any(r["tau"] for r in rows) else None),
        "runs": rows,
    }
    print()
    print(f"[{args.label}] server-side median : {out['server_tok_s_median']} tok/s")
    print(f"[{args.label}] harness median     : {out['harness_tok_s_median']} tok/s")
    print(f"[{args.label}] acceptance / tau   : {out['acceptance_median']} / {out['tau_median']}")
    with open(f"/home/dkp/bench-{args.label}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"  -> /home/dkp/bench-{args.label}.json")


if __name__ == "__main__":
    sys.exit(main())
