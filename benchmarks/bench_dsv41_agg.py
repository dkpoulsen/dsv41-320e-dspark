#!/usr/bin/env python3
"""Aggregate throughput sweep: concurrency 1,2,4,8,12 on dsv41-pp5 :9004.

Counts ALL streamed delta fields (content + reasoning_content), so the decode
window is measured over the true generation span rather than post-reasoning prose.
"""
import json, time, requests, statistics
from concurrent.futures import ThreadPoolExecutor as TPE

URL = "http://localhost:9004/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash"
GEN = 512

PROMPTS = [
    "Write a detailed technical explanation of how rotary position embeddings work in transformer attention, covering the math and the practical implementation details.",
    "Explain the tradeoffs between pipeline parallelism and tensor parallelism for serving large MoE language models on PCIe-only multi-GPU systems.",
    "Describe how speculative decoding can accelerate autoregressive generation, including draft model design, verification, and acceptance-rate effects on effective throughput.",
    "Give a thorough overview of quantization methods for LLM inference: INT8, FP8, MXFP4, AWQ, and how each affects memory bandwidth versus accuracy.",
    "Explain the design of sparse attention mechanisms for million-token contexts, including indexer-based top-k selection and KV cache compression.",
    "Discuss the engineering challenges of running a 500GB frontier model on consumer-adjacent 64GB accelerators without NVLink, from weight residency to host-memory offload.",
    "Walk through how a Mixture-of-Experts layer routes tokens to experts, and why expert parallelism interacts badly with pipeline parallelism on slow interconnects.",
    "Explain unified virtual addressing (UVA) based weight offloading: how pinned host memory is exposed to the GPU, bandwidth implications, and when it is worth it.",
    "Describe how CUDA graph capture works for decode steps, including the constraints it places on dynamic control flow and collective communication.",
    "Give a rigorous explanation of FP8 KV cache quantization for MLA attention, including scaling factor handling and accuracy considerations.",
    "Explain the difference between prefill and decode phases in LLM serving, and why they have opposite bottlenecks (compute vs memory bandwidth).",
    "Describe how a pipeline-parallel inference engine schedules microbatches, and what bubble sizes result from uneven stage workloads.",
]


def one(i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "max_tokens": GEN, "temperature": 0.6, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    t_first = None      # first token of ANY kind (content or reasoning)
    n = 0
    with requests.post(URL, json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            d = line[6:]
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                n = j["usage"]["completion_tokens"]
            for ch in j.get("choices", []):
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    if t_first is None:
                        t_first = time.time()
    return {"t0": t0, "t_first": t_first, "t_end": time.time(), "n": n}


for conc in [1, 2, 4, 6, 8]:
    one(0)  # warm
    t0 = time.time()
    with TPE(conc) as ex:
        rs = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    total = sum(r["n"] for r in rs)
    firsts = [r["t_first"] for r in rs if r["t_first"]]
    ttfts = [r["t_first"] - r["t0"] for r in rs if r["t_first"]]
    ends = [r["t_end"] for r in rs]
    # steady-state window: from the LAST stream's first token to the FIRST stream's end
    span = min(ends) - max(firsts) if len(firsts) > 1 else (ends[0] - firsts[0])
    if span <= 0: span = float("nan")
    dec_tokens = sum(max(r["n"] - 1, 0) for r in rs)
    print(f"conc={conc:2d} | e2e {total/wall:7.1f} tok/s | steady-state {dec_tokens/max(span,1e-9):7.1f} tok/s | "
          f"per-stream {dec_tokens/max(span,1e-9)/conc:6.1f} | TTFT {statistics.median(ttfts):5.2f}s | "
          f"wall {wall:6.1f}s | gen {total}", flush=True)
