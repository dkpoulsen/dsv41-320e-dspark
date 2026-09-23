#!/usr/bin/env bash
set -euo pipefail
CKPT=${CKPT:-/models/DeepSeek-V4.1-Flash}
IMG=${IMG:-dsv41-pp5-sm80}
NAME=${NAME:-reap-cal}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --name "$NAME" --gpus all --ipc host --network host \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v /home/dkp/models:/models:ro \
  -v /home/dkp/reap:/reap \
  -v /home/dkp/reap-tools:/tools:ro \
  -v /home/dkp/dsv41-flash-pp5/vllm:/src/vllm:ro \
  -v /home/dkp/dsv41-exl3/results:/src/results:ro \
  -e DSV41_ENGRAM_SRC="$CKPT" \
  -e DSV41_DEQUANT_FALLBACK="${DEQUANT:-0}" \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=8.0 \
  -w /tools/scripts \
  --entrypoint python3 "$IMG" "$@"
