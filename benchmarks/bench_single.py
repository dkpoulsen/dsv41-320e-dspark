import json, requests, time

def run(prompt, max_tokens):
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time(); tf = None; usage = None; r_chunks = 0; c_chunks = 0
    with requests.post("http://localhost:9004/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            d = line[6:]
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                dl = ch.get("delta") or {}
                if dl.get("reasoning"):
                    r_chunks += 1
                    if tf is None: tf = time.time()
                if dl.get("content"):
                    c_chunks += 1
                    if tf is None: tf = time.time()
    dt = time.time() - tf
    n = usage["completion_tokens"]
    print("max_tokens=%-5d completion=%-5d reasoning_chunks=%-5d content_chunks=%-4d | TTFT=%5.2fs decode=%5.1f tok/s wall=%5.1fs"
          % (max_tokens, n, r_chunks, c_chunks, tf - t0, n / dt, time.time() - t0))
    return n / dt

for mt, p in [(60, "What is the capital of France? One word."),
              (600, "Write a detailed technical explanation of rotary position embeddings, covering math and implementation."),
              (600, "Count from 1 to 300, one number per line.")]:
    run(p, mt)
