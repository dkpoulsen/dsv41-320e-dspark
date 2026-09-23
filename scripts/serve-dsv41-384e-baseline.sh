#!/usr/bin/env bash
# PP5 full-residency DeepSeek-V4.1-Flash on 5x CMP 170HX (sm80), docker adaptation
# of 344303947/dsv41-flash-pp5-170hx dsv41-pp5/start_vllm_ds4.1.sh
set -euo pipefail
IMAGE=dsv41-pp5-sm80
NAME=dsv41-pp5-384e
PORT=${PORT:-9004}
MAXLEN=${MAXLEN:-1048576}
GPU_UTIL=${GPU_UTIL:-0.97}
MAX_BATCHED=${MAX_BATCHED:-2048}
MAX_SEQS=${MAX_SEQS:-8}
MODEL_DIR=${MODEL_DIR:-/home/dkp/models/DeepSeek-V4.1-Flash}
LOG_DIR=${LOG_DIR:-/home/dkp/logs}
mkdir -p "$LOG_DIR"
LOG=/home/dkp/logs/vllm_v41_pp5_$(date +%Y%m%d_%H%M%S).log

# 20GiB engram x ~9.5 shards/rank flows through page cache; expandable segments
# to avoid fragmentation OOM; sparse indexer logits chunk cap per fork README.
docker rm -f $NAME >/dev/null 2>&1 || true
set -x
docker run -d --name $NAME --gpus all --network host --ipc host \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v /home/dkp/models:/models:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGRAM_EXACT_PIN=1 \
  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \
  -e VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=14400 \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint vllm $IMAGE serve /models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port $PORT \
  --served-model-name DeepSeek-V4.1-Flash \
  --pipeline-parallel-size 5 --tensor-parallel-size 1 \
  --max-model-len $MAXLEN \
  --max-num-batched-tokens $MAX_BATCHED \
  --max-num-seqs $MAX_SEQS \
  --gpu-memory-utilization $GPU_UTIL \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code \
  --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  > $LOG 2>&1
set +x
echo "container started, log: $LOG  (tail -f $LOG)"
