# REAP-320E + DSpark on 5× CMP 170HX (sm_80)

Running **DeepSeek-V4.1-Flash** (510 GB, native FP4) with **DSpark speculative decoding enabled** on
five NVIDIA CMP 170HX cards (GA100, sm_80, 64 GiB, PCIe-only, no P2P) — no sixth card, no expert
offload.

The draft head needs 7.39 GiB on the last pipeline stage, and that stage had ~1.2 GiB free. We
made room by **pruning the routed experts from 384 to 320 per layer with REAP** and calibrating
the saliency ranking on this hardware, which turned out to be possible after all.

## Results (all measured on the target node)

| metric | 384E baseline (no DSpark) | **320E + DSpark(5)** |
|---|---|---|
| single-stream decode | 42.4 tok/s | **57.8 tok/s median — 1.36×** (1.64× best prompt) |
| acceptance rate | — | 29.4% (τ = 2.47 tok/step) |
| prefill @107K ctx | 2,099 tok/s | 3,099 tok/s |
| needle-in-haystack @107K | HIT | **HIT** |
| free VRAM, ranks 0–3 | 1.4–3.0 GiB | **12.0–13.1 GiB** |
| rank 4 load (draft host) | 57.92 GiB | 58.02 GiB *(incl. 3-stage draft)* |
| KV pool | 1,814,599 tok | **5,471,116 tok — 3× more** (see [KV-CAPACITY.md](KV-CAPACITY.md)) |

**Correction.** An earlier version of this table claimed 72.7 tok/s / 1.71×. Fixed — that
figure was the *maximum* of a noisy sample set (the engine's own logs from that session:
n=19, median 59.3, max 71.5), not the typical rate. Re-measured both sides on the same
node with two independent methods, the number is **57.8 tok/s median, 1.36×**. Throughput
is prompt-dependent (55.5–69.4 tok/s depending on how well the drafter accepts), so treat
it as a range. Full accounting in [KV-CAPACITY.md](KV-CAPACITY.md#correction-the-727-toks-in-the-readme-is-wrong).

DSpark did not cost KV capacity — the rebalanced partition below *tripled* it.

**Quality cost: +1.93% held-out perplexity** (4.3329 → 4.4164, paired on 49,104 held-out tokens;
bootstrap 95% CI [+0.66%, +3.60%]). Task suite: 12/12 on both models.

## Hardware / software

- 5× CMP 170HX (GA100, sm_80, 64 GiB each, PCIe Gen2 x4, **no P2P**), 251 GB host RAM
- Model: `deepseek-ai/DeepSeek-V4.1-Flash` (MXFP4 routed experts, FP8 dense, 189 GiB Engram tables)
- Engine: [`wtdcode/vllm-backport`](https://github.com/wtdcode/vllm-backport) (Python-only fork of
  vLLM 0.13.1; adds shadow-source PP partitioning, staged Marlin repack, exact-size pinned Engram)
- Serving: PP5 `8,8,8,8,8`, full residency, `--kv-cache-dtype fp8_ds_mla`, 1M context

## How it works

### 1. Why the draft didn't fit

The draft head is three MTP stages (7.39 GiB) that must be VRAM-resident on the **last** pipeline
stage, which also carries the model head. With the unpruned checkpoint that stage loaded at
57.92 GiB against a 62.08 GiB budget at `gpu_memory_utilization 0.97` — and the draft needs ~8.2 GiB
once its own KV windows and CUDA graph are counted.

### 2. Pruning 384 → 320 experts

REAP (router-weighted expert activation pruning) ranks experts per layer by
`mean(router_weight × ‖expert_output‖)` and keeps the top K. Published pruned checkpoints exist for
272 and 256 experts, but **not 320** — and 320 matters because it is the largest count that is
already a valid entry in the fork's fused-router kernel table:

```
csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu
  case 256: ... case 320: ... case 384: ...     # 272 and 288 are ABSENT
```

So 320E serves with **no C++ rebuild**, whereas 272E would need a new kernel instantiation.

### 3. The calibration blocker, and how it was solved

REAP's reference tooling needs a forward pass exposing per-expert router statistics. DeepSeek's
bundled reference implementation (`inference/`) uses TileLang GEMMs on **FP8 E4M3 tensor cores** —
and sm_80 has **no FP8/FP4 MMA**. Hence the published conclusion that calibration is impossible on
this hardware.

**That conclusion was needlessly pessimistic.** The reference's `linear()` dispatches on weight
dtype, and its final branch is a plain `F.linear` that works on any GPU. So
`patches/patch_dequant_fallback.py` adds an env-gated branch that dequantizes MXFP4 (E2M1 nibbles
packed along K + E8M0 scales) and FP8 (32×32 blockwise) to bf16 on the fly:

```python
if _dequant_fallback_enabled() and weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
    return F.linear(x, _dequant_weight(weight), bias)
```

Scales ride on `weight.scale`, so no extra plumbing is needed. The reference implementation then
runs on sm_80 — its selftest produces `"The capital of France is" → ' Paris'` at logit 19.85.

Calibration is a one-off pass where throughput is irrelevant, so dequantizing on every call is an
acceptable cost.

### 4. Pipeline

```bash
# 1. patch the reference implementation
cp -r $CKPT/inference inference-patched
python patches/patch_dequant_fallback.py inference-patched
python patches/patch_kernels.py ...          # from Libertai/deepseek-v41-flash-reap
python patches/patch_engram.py  inference-patched

# 2. build a calibration corpus (local text; no network, no `datasets`)
DSV41_DEQUANT_FALLBACK=1 python patches/build_calib_local.py \
    --ckpt $CKPT --out calib_text.npy --roots <dirs> --seq-len 2048 --n-seq 128

# 3. collect saliency (~100 min for 262K tokens on 5 cards)
DSV41_DEQUANT_FALLBACK=1 python reap_calibrate.py \
    --inference-dir inference-patched --ckpt $CKPT \
    --calib calib_text.npy --batch 16 --out saliency_text.pt

# 4. prune to 320 experts
python prune_experts.py --src $CKPT --out dsv41-320E \
    --saliency saliency_text.pt --keep 320 --mode mean --allow-text-only

# 5. serve with DSpark
./scripts/serve-dsv41-320e.sh
```

`--allow-text-only` is legitimate here: this fork hard-skips the vision tower
(`vision.`/`aligner.`/`image_` weights), so text-only saliency cannot damage a vision path the
deployment never loads.

### 5. Measured calibration stats

- 262,144 calibration tokens (128 × 2048) from local technical/code text — the REAP paper's
  calibration size, and a closer domain match for an agent workload than generic web text
- 62.9M routed-token-expert pairs over 8 passes
- Saliency spread is real: 10×–475× between best and worst expert per layer; **126 of 15,360
  experts were never selected at all**

## Quality

Measured through the live serving stack (`prompt_logprobs`), so it reflects the shipping artifact
including Marlin MXFP4, fp8 KV and the PP5 pipeline.

| | 384E | 320E |
|---|---|---|
| held-out perplexity (49,104 tok) | 4.3329 | 4.4164 |
| Δ | — | **+1.93%** (95% CI +0.66% … +3.60%) |
| task suite (12 deterministic prompts) | 12/12 | 12/12 |

This sits right on the published REAP ladder (304E +2.21%, 288E +3.16%, 272E +4.13%, 256E +5.49%),
which is also decent evidence the calibration ranking is sound.

**Caveats, stated plainly:**
- 49k tokens gives a CI of roughly ±1.5pp; the true drop could be nearer +1%
- **Perplexity is not a benchmark** — the REAP authors say so themselves, and they published no
  downstream evals. Our 12-prompt suite scores 12/12 on both models, so it cannot resolve a 2%
  difference. If you need certainty for a specific workload, run a task eval on that workload.
- Calibration was text-only (see above for why that is sound here)

## KV capacity — and a free 3×

Once the draft fits, the KV pool is limited by whichever rank has the least free memory,
because vLLM allocates the same block count on every rank. With the default balanced
partition that rank is PP4 (it carries the head *and* the draft), so 11–14 GiB sitting idle
on the other four cards bought nothing.

Two independent levers fix it — see **[KV-CAPACITY.md](KV-CAPACITY.md)** for the full
measurement:

```bash
VLLM_PP_LAYER_PARTITION=8,9,9,8,6 GPU_UTIL=0.98 ./scripts/serve-dsv41-320e-rebalanced.sh
```

That takes the pool from 1,729,736 tokens (1.65× a 1M request) to **5,471,116 (5.22×)** at
the same speed and 12/12 correctness. Rebalancing alone gives 2.86×, utilization alone
1.92×, both together 3.47×.

## Repo layout

```
patches/     the dequant fallback + calibration corpus builder + saliency collector
scripts/     launch scripts for 320E+DSpark and for the 384E baseline
benchmarks/  perplexity, capability, decode/aggregate/long-context benchmarks
results/     raw measured JSON (perplexity, capability)
logs/        calibration, prune, and server logs from the reported run
diagnostics/ minimal probes establishing the sm_80 FP8/FP4 MMA absence
docs-research-report.md   the full prior analysis of the fit problem
```

## Gotchas worth knowing

- **Prefer the engine's own counters over a streaming harness.** `(accepted_spec_tokens +
draft_steps) / wall_time` cannot be fooled by SSE framing, the `reasoning`/`content` field
split, or an early EOS. Three separate bugs in this project were harness artifacts that the
counters would have caught immediately. `benchmarks/bench_spec.py` reads them.
- **Never quote a single throughput number without its spread.** Acceptance varies by prompt
  (τ=2.42–3.05 across four prompts here), which moves decode rate by 25%. The figure that
  shipped first in this repo was the max of a noisy sample set and had to be corrected down
  from 1.71× to 1.36×.
- **`prompt_logprobs` is required for perplexity** through a vLLM server; the openai-compatible
  `/completions` endpoint with `prompt_logprobs=0` returns the chosen token's logprob per position.
- **PPL is `exp(-nll_per_token)`**, so a paired ratio is `exp(-mean_delta)` — not `exp(mean_delta)`.
  Getting this backwards flips the sign of the quality delta.
- **Don't compare perplexity across different chunk samples.** A 24-chunk run scored 4.59 where the
  48-chunk run scored 4.33 on the same model; sampling dominates at small n. Pair on identical
  spans and bootstrap the confidence interval.
- **The full bf16-dequantized checkpoint does not fit on disk** (268.9 GiB of MXFP4 experts becomes
  ~1,012 GiB). Dequantize on the fly instead — that is what the fallback patch does.
- **`--gpu-memory-utilization` above 0.98 is fragile here.** The engine will load and serve, but
  the binding rank is left with ~400 MiB and a transient peak can kill the boot. 0.97–0.98 is the
  usable band.
- **`persistent_topk` has no capability gate on SM8x** in vLLM and can silently corrupt at prompt
  lengths 2049–4096 (see the linked PR thread). Worth verifying before trusting batch>1 output.
- `case 320:` exists in the fused-router kernel table; `272`/`288` do not.

## References

- REAP recipe and pruned checkpoints: [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap),
  [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E)
- Engine fork: [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)
- sm_80 DSpark field reports and the `persistent_topk` finding:
  [vllm-project/vllm#50576](https://github.com/vllm-project/vllm/issues/50576)
- DSpark: *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* (arXiv 2607.05147)

## License

Code here is provided as-is for reference; the model weights retain their original licenses.
