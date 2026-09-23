# Measurement notes

Everything in the README table was measured on the target node (5× CMP 170HX). This file records
*how*, so the numbers can be checked or reproduced.

## Decode throughput

Single-stream, greedy, 512-token generations over four varied technical prompts, counting **both**
`content` and `reasoning` deltas.

That detail matters: this model streams its chain of thought in a separate `reasoning` field
(~595 of 600 chunks on a technical prompt), and an earlier harness that timed first-token on
`content` alone reported **74 tok/s** for a configuration that actually ran at **42 tok/s** — it
divided all completion tokens by the short content tail. Server-side telemetry
(`vllm:avg_generation_throughput`) confirmed 40.3 tok/s, which is why 42.4 is used here.

The corrected harnesses are `benchmarks/bench_320e.py` and `benchmarks/bench_single.py`.

## Acceptance rate and τ

Read from the server's own Prometheus counters rather than inferred:

```
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_drafts_total
```

Over the reported run: 4,025 draft tokens, 1,242 accepted → **30.9%**; 805 draft steps →
**τ = 2.54** accepted tokens per step including the bonus token. The per-position breakdown showed
the usual decay (171 → 104 → 65 → 34 across positions 0–3), which is the signature of a working
drafter rather than a degenerate one.

## Perplexity

`benchmarks/perplexity.py` sends held-out text to `/v1/completions` with `prompt_logprobs=0` and
accumulates the logprob of each token given its full prefix, scoring positions 1..n-1.
`PPL = exp(-Σ logprob / n)`.

**Design choices that matter:**

1. **Measured through the serving stack, not the reference implementation.** This captures the
   artifact that actually ships — pruning + Marlin MXFP4 + fp8 KV + PP5.
2. **Held-out text is disjoint from the calibration corpus.** Calibration harvested
   `dsv41-flash-pp5/vllm`, `reap-tools/scripts`, `dsv41-exl3/results`; evaluation harvested
   `dsv41-exl3/research/notes`, `.claude/skills`, and the pi documentation tree. No overlap.
3. **Paired on identical token ranges** (same seed → same 48 chunk offsets), because between-chunk
   NLL varies by more than an order of magnitude (−204 to −2890).
4. **Determinism verified.** Re-running 320E on the same chunks reproduced all 48 per-chunk NLLs
   bit-identically (`results/ppl-320E-48.json` vs `results/ppl-320E-rerun.json`).

**Result:** 4.3329 (384E) vs 4.4164 (320E) = **+1.93%**. Bootstrap over 48 chunks (2,000 resamples)
gives a 95% CI of [+0.66%, +3.60%].

### Sign convention trap

`PPL = exp(-NLL)`, so the paired ratio is `exp(-mean(delta))`. A first pass computed `exp(+mean(delta))`
and reported **−1.89%** (i.e. the pruned model as *better*). The sign was wrong; the reported figure
is the corrected one.

### Sample-size trap

A 24-chunk run reported 4.59 and a 48-chunk run reported 4.33 **for the same model** — the
difference was entirely which chunks each draw sampled. Never compare perplexity across different
chunk samples; always pair.

## Capability suite

`benchmarks/capability_test.py` — 12 deterministic prompts (arithmetic, a syllogism, exact string
reversal, letter counting, binary→decimal, a code one-liner, a capital) scored by exact-answer
matchers, greedy decoding so runs are reproducible.

Both models score **12/12**, so the suite cannot resolve a 2% perplexity difference. It is included
as a regression check, not as evidence of parity at the margin.

## Long context

`benchmarks/bench_dsv41_longctx.py` plants a passphrase at 70% depth in filler text and asks for it.
At 107,060 tokens (prompt 34.7s) the needle was retrieved correctly with prefill at 3,086 tok/s.

## Memory

Per-GPU usage from `nvidia-smi` after load and warmup, plus the engine's own accounting:

```
gpu_worker.py:660   Available KV cache memory: <value>
kv_cache_utils.py   GPU KV cache size: 1,729,736 tokens …
model_runner.py     Model loading took <value>
```

Rank 4 is the draft host. Its load went from 57.92 GiB (384E, no draft) to 58.02 GiB (320E + 3-stage
draft): pruning saved ~9 GiB and the draft consumed ~8.2 GiB of it.

## What is NOT measured here

- **Real agent-workload acceptance.** The 30.9% figure comes from four technical-explanation prompts.
  Reasoning-heavy traffic is the case that matters and it is worth re-measuring with
  `benchmarks/bench_320e.py` against your own prompts.
- **Downstream task quality at scale.** Perplexity and a 12-prompt suite are all we ran.
- **Long-run stability.** The reported run was ~1 hour; no soak test.
- **The `persistent_topk` SM8x corruption** was not exercised. It reportedly affects prompt lengths
  2049–4096; verify before trusting batch>1 output.
