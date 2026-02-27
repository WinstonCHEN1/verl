#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/wait_for_8gpu_and_smoke.sh \
#     --venv .venv \
#     --install \
#     --gpu-count 8 \
#     --mem-threshold-mb 500 \
#     --util-threshold 5 \
#     --poll-seconds 30 \
#     --timeout-seconds 0 \
#     --run-cmd "python -c 'import torch; print(torch.cuda.device_count())'"

VENV_DIR=".venv"
DO_INSTALL=0
GPU_COUNT=8
MEM_THRESHOLD_MB=500
UTIL_THRESHOLD=5
POLL_SECONDS=30
TIMEOUT_SECONDS=0
RUN_CMD="python -c \"import torch; print('torch', torch.__version__); print('cuda_count', torch.cuda.device_count())\""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  sed -n '1,40p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --install)
      DO_INSTALL=1
      shift
      ;;
    --gpu-count)
      GPU_COUNT="$2"
      shift 2
      ;;
    --mem-threshold-mb)
      MEM_THRESHOLD_MB="$2"
      shift 2
      ;;
    --util-threshold)
      UTIL_THRESHOLD="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --run-cmd)
      RUN_CMD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script requires NVIDIA GPUs." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ "$DO_INSTALL" -eq 1 ]]; then
  log "Installing dependencies into $VENV_DIR"
  python -m pip install --upgrade pip setuptools wheel

  if [[ -f requirements.txt ]]; then
    python -m pip install -r requirements.txt
  fi

  if [[ -f requirements-test.txt ]]; then
    python -m pip install -r requirements-test.txt
  fi

  python -m pip install -e .
fi

check_gpus_free() {
  local q
  q="index,memory.used,utilization.gpu"

  mapfile -t LINES < <(nvidia-smi --query-gpu="$q" --format=csv,noheader,nounits)

  local total=${#LINES[@]}
  if (( total < GPU_COUNT )); then
    log "Detected GPU count=$total < required=$GPU_COUNT"
    return 1
  fi

  local free=0
  local details=()
  local i
  for ((i=0; i<GPU_COUNT; i++)); do
    IFS=',' read -r idx mem util <<<"${LINES[$i]}"
    idx=$(echo "$idx" | xargs)
    mem=$(echo "$mem" | xargs)
    util=$(echo "$util" | xargs)

    details+=("GPU${idx}:mem=${mem}MB,util=${util}%")

    if (( mem <= MEM_THRESHOLD_MB && util <= UTIL_THRESHOLD )); then
      free=$((free+1))
    fi
  done

  log "GPU status: ${details[*]}"

  if (( free == GPU_COUNT )); then
    return 0
  fi
  return 1
}

start_ts=$(date +%s)
log "Waiting for $GPU_COUNT free GPUs (mem<=${MEM_THRESHOLD_MB}MB, util<=${UTIL_THRESHOLD}%)"

while true; do
  if check_gpus_free; then
    log "All required GPUs are free. Starting smoke command..."
    set -x
    bash -lc "$RUN_CMD"
    set +x
    log "Smoke command finished."
    exit 0
  fi

  if (( TIMEOUT_SECONDS > 0 )); then
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    if (( elapsed >= TIMEOUT_SECONDS )); then
      log "Timeout reached (${TIMEOUT_SECONDS}s). Exit with failure."
      exit 2
    fi
  fi

  sleep "$POLL_SECONDS"
done
