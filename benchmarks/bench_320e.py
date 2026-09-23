#!/usr/bin/env python3
"""Measure 320E+DSpark throughput and acceptance, counting `reasoning` + `content`."""
import json, time, requests, statistics, sys

URL = "http://localhost:9004/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash"

PROMPTS = [
 "Explain how speculative decoding works, covering draft models, verification, and how acceptance rate affects effective throughput. Write at length.",
 "Describe the engineering tradeoffs between pipeline parallelism and tensor parallelism for serving large MoE models on PCIe-only multi-GPU systems.",
 "Walk through how a Mixture-of-Experts layer routes tokens to experts, and why expert parallelism interacts badly with pipeline parallelism on slow interconnects.",
 "Explain the design of sparse attention for million-token contexts, including indexer-based top-k selection and KV cache compression.",
]

def run(prompt, max_tokens=512):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],
          "max_tokens":max_tokens,"temperature":0.0,"stream":True,
          "stream_options":{"include_usage":True}}
    t0=time.time(); tf=None; usage=None; n=0
    with requests.post(URL,json=body,stream=True,timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "): continue
            d=line[6:]
            if d==b"[DONE]": break
            j=json.loads(d)
            if j.get("usage"): usage=j["usage"]
            for ch in j.get("choices",[]):
                dl=ch.get("delta") or {}
                if dl.get("content") or dl.get("reasoning"):
                    if tf is None: tf=time.time()
                    n+=1
    t1=time.time()
    return t0,tf,t1,usage

# warm
run(PROMPTS[0], 64)

def metrics():
    out={}
    try:
        for line in requests.get("http://localhost:9004/metrics",timeout=30).text.splitlines():
            if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                out['acc']=float(line.rsplit(None,1)[1])
            elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                out['draft']=float(line.rsplit(None,1)[1])
            elif line.startswith("vllm:spec_decode_num_drafts_total"):
                out['drafts']=float(line.rsplit(None,1)[1])
    except Exception:
        pass
    return out

m0=metrics(); t0=time.time()
ttft=[]; rates=[]
for p in PROMPTS:
    a,tf,b,u=run(p)
    if tf:
        ttft.append(tf-a); rates.append(u["completion_tokens"]/max(b-tf,1e-9))
m1=metrics(); wall=time.time()-t0

print(f"decode (steady, per-request): {statistics.median(rates):.1f} tok/s   (n={len(rates)})")
print(f"TTFT (median): {statistics.median(ttft):.2f}s")
print()
dd=m1.get('draft',0)-m0.get('draft',0)
da=m1.get('acc',0)-m0.get('acc',0)
dr=m1.get('drafts',0)-m0.get('drafts',0)
if dd>0:
    print(f"draft tokens   : {dd:,.0f}")
    print(f"accepted tokens: {da:,.0f}")
    print(f"acceptance rate: {da/dd:.3f}")
if dr>0:
    print(f"drafts (steps) : {dr:,.0f}")
    print(f"=> accepted length tau = {(da+dr)/dr:.2f} tokens per step (incl. the bonus token)")
