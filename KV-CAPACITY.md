# KV capacity: rebalancing the pipeline, and a throughput correction

Two results here. The second one corrects a number in the README.

## The lever: only one rank matters

vLLM sizes the KV pool from the rank with the **least** free memory, then allocates the
same block count everywhere (`kv_cache_utils.py`: *"Change the num_blocks of each rank to
the smallest among all ranks"*). Attention across a pipeline needs a uniform pool.

With `VLLM_PP_LAYER_PARTITION=8,8,8,8,8` the binding rank is **PP4**, because it carries the
output head *and* the 3-stage DSpark draft:

| rank | layers | weights | free for KV |
|---|---|---|---|
| PP0 | 8 (+embedding) | 49.22 GiB | 12.86 GiB |
| PP1 | 8 | 47.77 GiB | 14.31 GiB |
| PP2 | 8 | 47.74 GiB | 14.34 GiB |
| PP3 | 8 | 47.72 GiB | 14.36 GiB |
| **PP4** | 8 (+head +draft) | **58.02 GiB** | **4.06 GiB** |

~11–14 GiB of headroom on four ranks was unusable while PP4 had 4 GiB. Two independent
levers attack that one rank.

## Lever A — rebalance the partition

`VLLM_PP_LAYER_PARTITION` accepts any split summing to 40, and `8+8+8+8+8` is exactly
balanced, so **any** deviation forces some rank to 9 layers. Moving layers off PP4 means
moving them onto a rank that then holds nine:

```
VLLM_PP_LAYER_PARTITION=8,9,9,8,6
```

PP4 drops to 6 layers (+head+draft = 46.09 GiB) while PP1/PP2 absorb 9.
Predicted by a cost model fitted to the measured per-rank loads (per-layer 5.968 GiB,
head+draft 10.28 GiB, embedding 1.48 GiB) which reproduces all five measured loads to
within 0.03 GiB.

## Lever B — raise `--gpu-memory-utilization`

PP4's weights are fixed, but the *fraction of the card the engine may claim* is not.

## Measured results

| config | KV pool | × of 1M | decode | correct | min free |
|---|---|---|---|---|---|
| `[8,8,8,8,8]` util 0.97 | 1,729,736 | 1.65 | 57.4 tok/s | 12/12 | 822 MiB |
| `[8,8,8,8,8]` util 0.99 | 3,318,465 | 3.16 | 57.8 tok/s | 12/12 | 414 MiB |
| `[8,9,9,8,6]` util 0.97 | 4,941,504 | 4.71 | 57.5 tok/s | 12/12 | 1.7 GiB |
| **`[8,9,9,8,6]` util 0.98** | **5,471,116** | **5.22** | **57.6 tok/s** | **12/12** | **1.06 GiB** |
| `[8,9,9,8,6]` util 0.99 | 6,000,728 | 5.72 | 57.8 tok/s | 12/12 | 396 MiB |

- rebalance alone: **2.86×**
- utilization alone: **1.92×**
- both: **3.47×** (5.72× of a 1M request), at the same speed and 12/12 correctness

`util 0.99` leaves ~400 MiB free, which is fragile — prefer the `0.98` row. Long-context
retrieval still HITs at 107K tokens on the recommended config (prefill 3,099 tok/s).

## Correction: the 72.7 tok/s in the README is wrong

It should be **57–69 tok/s by prompt, median 57.8**.

The 72.7 was the *maximum* of a noisy sample set, not the typical rate. The engine's own
`Avg generation throughput` lines from that same session:

```
n=19   min=0.1   p25=53.2   median=59.3   p75=62.4   max=71.5
```

Measured today with two independent methods — the streaming harness and the engine's own
spec-decode counters (`(accepted + drafts) / wall`) — the config gives **57.3–57.8 tok/s**
across repeated runs, and 69.4 on the single best-accepting prompt. The figure is stable
to ±0.1 tok/s between repeats.

So the honest comparison, both sides re-measured on the same node with the same harness:

| | 384E baseline | 320E + DSpark |
|---|---|---|
| tok/s | 42.4 (n=12, spread 0.03) | 57.8 (median; 55.5–69.4 by prompt) |
| speedup | — | **1.36× median, 1.64× best prompt** |

Previously published: *"72.7 tok/s, 1.71×"*. Corrected: **"1.36× median"**.

An earlier draft of this file also claimed rebalancing cost 19% of throughput. That was
also an artifact of the bad 72.7 baseline: measured against a properly re-measured
baseline, the rebalanced config is **not slower** (57.45 vs 57.35 tok/s server-side,
within run-to-run noise). No throughput tradeoff exists between the configs at all.

### Why this matters

Throughput on this node is prompt-dependent — acceptance varies from τ=2.42 to τ=3.05
across four prompts, which moves decode rate by 25%. Any single number quoted without its
spread will be wrong for some workload. Report the median **and** the range, and prefer
the engine's own counters over a streaming harness, which can be fooled by SSE framing,
the reasoning/content field split, or an early EOS.

## Reproduce

```bash
# rebalanced + higher utilization
VLLM_PP_LAYER_PARTITION=8,9,9,8,6 GPU_UTIL=0.98 ./scripts/serve-dsv41-320e-rebalanced.sh

# measure with the engine's own counters (not fooled by SSE framing)
python3 benchmarks/bench_spec.py --url http://localhost:9006 --label my-run --repeats 3
```
