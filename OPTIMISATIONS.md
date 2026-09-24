# Optimisation survey — what helps, what does not, and what breaks

Five candidate optimisations tested after the KV work. **One is worth acting on, three are
dead ends, and one is an active trap.** Measured, not assumed.

Baseline for every row: `[8,9,9,8,6]` util 0.97, `num_speculative_tokens=5`,
`max_num_batched_tokens=2048`, 4,941,504-token KV pool, 57.5 tok/s.

## Summary

| lever | result | verdict |
|---|---|---|
| Keep the prompt prefix byte-stable (prefix caching) | 20.5s → **0.50s** TTFT on a 40K prompt | **act on this — biggest win by far** |
| `num_speculative_tokens=10` | 57.9 → **24.7 tok/s**, τ 2.50 → 2.25 | dead end, 2.3× regression |
| `max_num_batched_tokens=8192` | engine wedged, PP desync | breaks — do not use |
| Batch-size draft schedule | τ flat across batch 1–8 | no evidence it helps |
| Adaptive verification | hard-blocked by PP | unavailable |

## 1. Prefix caching is the real win (act on this)

Already enabled; the payoff is enormous and easy to destroy:

| call | prompt | wall | rate |
|---|---|---|---|
| cold | 40,042 tok | 20.48 s | 1,955 tok/s |
| warm | 40,042 tok | **0.67 s** | 59,973 tok/s |
| warm | 40,042 tok | **0.50 s** | 79,400 tok/s |

**30–40× TTFT reduction.** Decode is pinned at ~58 tok/s no matter what you tune, but an agent
loop resends a growing conversation every turn. Without cache reuse every turn re-prefills the
whole history — 20 s for 40K tokens, *per turn*. With reuse it is half a second.

So the highest-leverage optimisation here is not a server flag, it is prompt discipline: **keep
the start of the prompt byte-identical across turns.** Anything that perturbs it silently
destroys the win:

- a timestamp or session id at the top of the system message
- reshuffled or reordered tool definitions
- non-deterministic JSON key ordering in a serialised context block
- re-rendering a prompt template whose whitespace varies

Measured counter-example: 502 queries with a 0% hit rate, because each prompt differed at
"Question {i}" in the *user* turn — but note that varying the user turn is fine. Only the
**prefix** must be stable, so put variable content last.

## 2. `num_speculative_tokens=10` is 2.3× slower (dead end)

The tempting inference: per-position acceptance showed ~66% survival even at the last draft
position, so the 5-token budget looked truncating rather than saturating. Worth testing. It is
not:

| config | tok/s | τ | acceptance |
|---|---|---|---|
| `num_speculative_tokens=5` | 57.9 | **2.50** | 0.301 |
| `num_speculative_tokens=10` | **24.7** | 2.25 | 0.125 |

Reproducible across three measurement runs on the same boot (24.7 / 24.9 / 25.3), 12/12
correct, same KV pool — so this is a clean throughput regression, not a broken boot.

Two reasons:

- **The budget is quantised.** `num_speculative_tokens` must be a multiple of `n_predict` (5),
  so the only options are 5, 10, 15. There is no gentle step from 5 to 6.
- **n=10 reuses the MTP module a second time on its own output** — a regime it was not trained
  for. Per-position survival collapses to ~45% in the second block and hits 0% at position 9.
  τ therefore *falls*, and the extra draft passes are charged in full on a PCIe-only PP5 node.

The lesson is about my own inference, not the flag: **τ cannot decrease when you draft more
unless the drafts got worse** — which is the signal that the extra tokens were out-of-distribution,
not merely unhelpful.

### A contaminated measurement to avoid repeating

The "66% survival at position 4" that motivated the experiment came from a cumulative counter
that included my long-context needle tests — 9,000 repetitions of
`"Note N: the survey recorded X sites."`, near-perfectly predictable text that inflates
acceptance. Cumulative Prometheus counters span everything you have run. Use fresh deltas over
a representative prompt set.

## 3. `max_num_batched_tokens=8192` breaks the pipeline (do not use)

Loads fine and serves nothing. On the first real request the engine wedges permanently:

```
(PP2) RuntimeError: torch_call_dispatcher("aten::new_empty" ...) API call failed
(PP3) RuntimeError: PP intermediate tensor 'hidden_states' has 2048 rows but this
      step expects 8192; the upstream rank likely failed mid-step and the pipeline
      is out of sync.
(EngineCore) No available shared memory broadcast block found in 60 seconds ...
```

PP2 — the 9-layer rank — fails first, on a CUDA allocation. 8192 batched tokens needs 4× the
activation rows, and that rank has no reserve left. It then never recovers: the shm-broadcast
warnings repeat indefinitely while `/v1/models` still answers 200, so a health check will not
tell you it is dead.

Left at 2048. Anyone raising this must raise it together with freeing memory on the tightest
rank.

## 4. No batch-size draft schedule (no evidence)

`num_speculative_tokens_per_batch_size` takes `[(lo, hi, n), ...]` ranges and is available here
(it is only disabled under data parallelism, and we are DP1). The theory is that at high
concurrency each extra draft token costs a full extra forward across the batch, so you should
draft less.

Measured τ by batch size:

| concurrency | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| τ | 2.43 | 2.35 | 2.43 | 2.47 | 2.43 |

**Flat.** If drafting less at high batch helped, τ would rise with batch — it does not. And
aggregate throughput scales 53 → 139 tok/s from concurrency 1 → 8, so draft cost is not
dominating either. Adding a schedule would be config surface with no measured gain.

Caveat: `max_num_seqs=8`, so "high concurrency" here means batch 8. A larger server could
differ.

(The raw concurrency sweep also produced one physically impossible point — concurrency 2 at
44.1 tok/s, below concurrency 1 at 53.2. That is an artifact of measuring wall time across
prefill + decode for simultaneously-started threads; prefill dominates at small batch. Treat
the τ column as the reliable output and the aggregate column as indicative.)

## 5. Adaptive verification is unavailable

`enable_adaptive_verification` is a real DSpark-only feature — sizing the draft-verification
budget from per-request confidence, which is exactly the right idea for this setup. It is
hard-blocked:

```python
if self.parallel_config.pipeline_parallel_size > 1:
    raise ValueError("Adaptive verification is not currently compatible "
                     "with pipeline parallelism")
```

We run PP5. It also requires cudagraphs (we have those) and is incompatible with LoRA. The
upstream TODO says cost curves and confidences only exist on the last PP rank today. Not
available; would need a fork change to broadcast them.
