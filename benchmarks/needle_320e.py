import os, requests, json, random, time
random.seed(7)
filler = " ".join(f"Note {i}: the survey recorded {random.randint(100,999)} sites." for i in range(9000))
NEEDLE = " The secret vault passphrase is ORCHID-MAGNET-42."
Q = " What is the secret vault passphrase? Answer with the passphrase only."
cut = int(len(filler) * 0.7)
prompt = filler[:cut] + NEEDLE + filler[cut:] + Q
t0 = time.time()
r = requests.post(
    os.environ.get("DSV41_URL","http://localhost:9004")+"/v1/chat/completions",
    json={"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 200, "temperature": 0},
    timeout=3600,
)
dt = time.time() - t0
j = r.json()
u = j["usage"]
txt = j["choices"][0]["message"]["content"]
hit = "HIT" if "ORCHID-MAGNET-42" in txt else "MISS"
print(f"prompt {u['prompt_tokens']:,} tokens | wall {dt:.1f}s | prefill {u['prompt_tokens']/dt:,.0f} tok/s")
print(f"needle: {hit} -> {txt.strip()[:90]!r}")
