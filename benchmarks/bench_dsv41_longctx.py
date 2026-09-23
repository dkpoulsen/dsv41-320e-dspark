#!/usr/bin/env python3
"""Long-context bench for dsv41-pp5: prefill speed + needle recall at 100K/250K/500K/900K."""
import json, time, requests, random, sys
from concurrent.futures import ThreadPoolExecutor as TPE

URL = "http://localhost:9004/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash"

def build_prompt(target_tokens):
    # filler ~1.3 tokens/word; use varied sentences to defeat trivial dedup
    random.seed(42)
    topics = ["harbor logistics", "glacier cores", "radio astronomy", "textile mills",
              "mycorrhizal networks", "tidal power", "cave paintings", "satellite orbits"]
    words = []
    while True:
        t = random.choice(topics)
        words.append(f"Notes on {t}: the {random.randint(1800,2024)} survey recorded "
                     f"{random.randint(3,900)} sites with variance {random.random():.4f}. ")
        if len(words) > target_tokens / 25:  # ~25 tokens per sentence chunk
            break
    filler = "".join(words)
    while len(filler.split()) < target_tokens:  # pad to target
        filler += f"Segment {random.random():.6f} logged at latitude {random.uniform(-60,70):.3f}. "
    return filler

def run(prompt, q, max_tokens=120):
    body = {"model": MODEL, "messages": [{"role":"user","content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=7200)
    t1 = time.time()
    r.raise_for_status()
    j = r.json()
    u = j["usage"]
    text = j["choices"][0]["message"]["content"] or ""
    return t1 - t0, u["prompt_tokens"], text

NEEDLE = "The secret passphrase for the lighthouse archive is ZEBRA-QUARTZ-77."
Q = " What is the secret passphrase for the lighthouse archive? Answer with the passphrase only."

for target in [100_000, 250_000, 500_000, 900_000]:
    filler = build_prompt(target)
    # plant needle ~70% into the text
    cut = int(len(filler) * 0.7)
    prompt = filler[:cut] + NEEDLE + filler[cut:] + Q
    dt, ptoks, text = run(prompt, Q)
    ttft_est = dt  # max_tokens small; dominated by prefill
    ok = "ZEBRA-QUARTZ-77" in text
    print(f"target~{target//1000}K | prompt {ptoks:,} tok | wall {dt:.1f}s | "
          f"prefill {ptoks/dt:,.0f} tok/s | needle: {'HIT' if ok else 'MISS'} | out: {text.strip()[:80]!r}", flush=True)
