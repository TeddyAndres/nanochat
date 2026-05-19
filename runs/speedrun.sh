#!/bin/bash
set -euo pipefail

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.
#
# Defaults preserve the original dense speedrun behavior.
# Override with env vars when you want a faster sparse+manifest launch path on pre-staged nodes.
#
# 1) Example dense launch:
# bash runs/speedrun.sh
# 2) Example sparse launch with pre-staged data + manifest:
# SPEEDRUN_SPARSE_MODE=1 \
# SPEEDRUN_SPARSE_MANIFEST=manifests/65kvocab_2kseq_16batch_2accum_8ddp_10kstep.json \
# SPEEDRUN_SKIP_DOWNLOADS=1 \
# bash runs/speedrun.sh
# 3) Example launch in a screen session:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

die() {
    echo "speedrun.sh: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local message="$2"
    [[ -f "$path" ]] || die "$message ($path)"
}

require_dir() {
    local path="$1"
    local message="$2"
    [[ -d "$path" ]] || die "$message ($path)"
}

tokenizer_ready() {
    local tokenizer_dir="$1"
    [[ -f "$tokenizer_dir/tokenizer.model" || -f "$tokenizer_dir/tokenizer.json" || -f "$tokenizer_dir/tokenizer.pkl" ]]
}

prepare_dataset_placeholders_from_token_cache() {
    local data_dir="$1"
    local token_cache_dir="$2"
    python - "$token_cache_dir" "$data_dir" <<'PY'
import json
import sys
from pathlib import Path

token_cache_dir = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
parquet_files = set()

for split in ("train", "val"):
    metadata_path = token_cache_dir / split / "metadata.json"
    if not metadata_path.is_file():
        continue
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    parquet_files.update(payload.get("parquet_files", []))

if not parquet_files:
    raise SystemExit(f"Token cache metadata under {token_cache_dir} does not list any parquet files")

data_dir.mkdir(parents=True, exist_ok=True)
for filename in sorted(parquet_files):
    (data_dir / filename).touch(exist_ok=True)

print(f"Prepared {len(parquet_files)} placeholder parquet filenames in {data_dir}")
PY
}

export OMP_NUM_THREADS=1

# Preserve user-provided storage roots and keep large runtime caches off /tmp.
: "${NANOCHAT_BASE_DIR:=$HOME/.cache/nanochat}"
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"

: "${TORCHINDUCTOR_CACHE_DIR:=$NANOCHAT_BASE_DIR/torchinductor-cache}"
: "${TRITON_CACHE_DIR:=$NANOCHAT_BASE_DIR/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

: "${SPEEDRUN_VENV_PATH:=.venv}"
: "${SPEEDRUN_SKIP_VENV_SETUP:=0}"
: "${SPEEDRUN_SKIP_UV_SYNC:=0}"
: "${SPEEDRUN_SKIP_DOWNLOADS:=0}"
: "${SPEEDRUN_SKIP_DATASET_DOWNLOAD:=0}"
: "${SPEEDRUN_SKIP_IDENTITY_DOWNLOAD:=0}"
: "${SPEEDRUN_SKIP_TOKENIZER_TRAIN:=1}"
: "${SPEEDRUN_SKIP_BASE_EVAL:=0}"
: "${SPEEDRUN_SKIP_SFT:=0}"
: "${SPEEDRUN_SKIP_CHAT_EVAL:=0}"
: "${SPEEDRUN_NPROC_PER_NODE:=8}"
: "${SPEEDRUN_MODEL_DEPTH:=24}"
: "${SPEEDRUN_TARGET_PARAM_DATA_RATIO:=8}"
: "${SPEEDRUN_ENABLE_FP8:=1}"
: "${SPEEDRUN_DEVICE_BATCH_SIZE:=}"
: "${SPEEDRUN_EVAL_DEVICE_BATCH_SIZE:=}"
: "${SPEEDRUN_SFT_DEVICE_BATCH_SIZE:=}"
: "${SPEEDRUN_TOTAL_BATCH_SIZE:=}"
: "${SPEEDRUN_SPARSE_MODE:=0}"
: "${SPEEDRUN_SPARSE_MANIFEST:=}"
: "${SPEEDRUN_TOKEN_CACHE_DIR:=}"
: "${SPEEDRUN_TRAIN_EVAL_EVERY:=-1}"
: "${SPEEDRUN_TRAIN_CORE_METRIC_EVERY:=-1}"
: "${SPEEDRUN_TRAIN_SAMPLE_EVERY:=-1}"
: "${SPEEDRUN_BASE_TRAIN_EXTRA_ARGS:=}"
: "${SPEEDRUN_BASE_EVAL_EXTRA_ARGS:=}"
: "${SPEEDRUN_CHAT_SFT_EXTRA_ARGS:=}"
: "${SPEEDRUN_CHAT_EVAL_EXTRA_ARGS:=}"

if [[ "$SPEEDRUN_SKIP_DOWNLOADS" == "1" ]]; then
    SPEEDRUN_SKIP_DATASET_DOWNLOAD=1
    SPEEDRUN_SKIP_IDENTITY_DOWNLOAD=1
fi

DATA_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
IDENTITY_CONVERSATIONS_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"

# -----------------------------------------------------------------------------
# Python venv setup with uv

if [[ "$SPEEDRUN_SKIP_VENV_SETUP" != "1" ]]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [[ -d "$SPEEDRUN_VENV_PATH" ]] || uv venv "$SPEEDRUN_VENV_PATH"
    if [[ "$SPEEDRUN_SKIP_UV_SYNC" != "1" ]]; then
        uv sync --extra gpu
    fi
    source "$SPEEDRUN_VENV_PATH/bin/activate"
fi

# -----------------------------------------------------------------------------
# wandb setup

if [[ -z "${WANDB_RUN:-}" ]]; then
    WANDB_RUN=dummy
fi

if [[ "$SPEEDRUN_SPARSE_MODE" == "1" ]]; then
    [[ -n "$SPEEDRUN_SPARSE_MANIFEST" ]] || die "SPEEDRUN_SPARSE_MODE=1 requires SPEEDRUN_SPARSE_MANIFEST"
    require_file "$SPEEDRUN_SPARSE_MANIFEST" "Sparse manifest not found"

    mapfile -t manifest_meta < <(python - "$SPEEDRUN_SPARSE_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

for key in ("device_batch_size", "total_batch_size", "ddp_world_size", "grad_accum_steps", "u_max", "grad_accum_u_max"):
    print(payload.get(key, ""))
PY
    )

    manifest_device_batch_size="${manifest_meta[0]}"
    manifest_total_batch_size="${manifest_meta[1]}"
    manifest_world_size="${manifest_meta[2]}"
    manifest_grad_accum_steps="${manifest_meta[3]}"
    manifest_u_max="${manifest_meta[4]}"
    manifest_grad_accum_u_max="${manifest_meta[5]}"

    [[ "$manifest_world_size" == "$SPEEDRUN_NPROC_PER_NODE" ]] || die "Sparse manifest world size $manifest_world_size does not match SPEEDRUN_NPROC_PER_NODE=$SPEEDRUN_NPROC_PER_NODE"
    [[ -n "$manifest_device_batch_size" ]] || die "Sparse manifest is missing device_batch_size"
    [[ -n "$manifest_total_batch_size" ]] || die "Sparse manifest is missing total_batch_size"

    if [[ -z "$SPEEDRUN_DEVICE_BATCH_SIZE" ]]; then
        SPEEDRUN_DEVICE_BATCH_SIZE="$manifest_device_batch_size"
    fi
    if [[ -z "$SPEEDRUN_TOTAL_BATCH_SIZE" ]]; then
        SPEEDRUN_TOTAL_BATCH_SIZE="$manifest_total_batch_size"
    fi

    echo "Sparse speedrun mode enabled"
    echo "  manifest: $SPEEDRUN_SPARSE_MANIFEST"
    echo "  world size: $manifest_world_size"
    echo "  device batch size: $SPEEDRUN_DEVICE_BATCH_SIZE"
    echo "  total batch size: $SPEEDRUN_TOTAL_BATCH_SIZE"
    echo "  grad accum steps: $manifest_grad_accum_steps"
    echo "  u_max: $manifest_u_max"
    echo "  grad_accum_u_max: $manifest_grad_accum_u_max"
fi

if [[ -z "$SPEEDRUN_DEVICE_BATCH_SIZE" ]]; then
    SPEEDRUN_DEVICE_BATCH_SIZE=16
fi
if [[ -z "$SPEEDRUN_EVAL_DEVICE_BATCH_SIZE" ]]; then
    SPEEDRUN_EVAL_DEVICE_BATCH_SIZE="$SPEEDRUN_DEVICE_BATCH_SIZE"
fi
if [[ -z "$SPEEDRUN_SFT_DEVICE_BATCH_SIZE" ]]; then
    SPEEDRUN_SFT_DEVICE_BATCH_SIZE="$SPEEDRUN_DEVICE_BATCH_SIZE"
fi

echo "Using NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "Using TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "Using TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

DATASET_DOWNLOAD_PID=""
if [[ "$SPEEDRUN_SKIP_DATASET_DOWNLOAD" != "1" ]]; then
    # Download the first ~2B characters of pretraining dataset.
    python -m nanochat.dataset -n 8
    # Immediately also kick off downloading more shards in the background while tokenizer trains.
    python -m nanochat.dataset -n 170 &
    DATASET_DOWNLOAD_PID=$!
else
    if [[ ! -d "$DATA_DIR" && "$SPEEDRUN_SPARSE_MODE" == "1" && -n "$SPEEDRUN_TOKEN_CACHE_DIR" ]]; then
        require_dir "$SPEEDRUN_TOKEN_CACHE_DIR" "Sparse token cache directory is missing"
        prepare_dataset_placeholders_from_token_cache "$DATA_DIR" "$SPEEDRUN_TOKEN_CACHE_DIR"
    fi
    require_dir "$DATA_DIR" "Dataset download skipped but dataset directory is missing"
fi

if [[ "$SPEEDRUN_SKIP_TOKENIZER_TRAIN" != "1" ]]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    tokenizer_ready "$TOKENIZER_DIR" || die "Tokenizer training skipped but tokenizer artifacts are missing from $TOKENIZER_DIR"
    echo "Skipping tokenizer training; using pre-staged tokenizer from $TOKENIZER_DIR"
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

if [[ -n "$DATASET_DOWNLOAD_PID" ]]; then
    echo "Waiting for dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
fi

base_train_args=(
    "--depth=$SPEEDRUN_MODEL_DEPTH"
    "--target-param-data-ratio=$SPEEDRUN_TARGET_PARAM_DATA_RATIO"
    "--device-batch-size=$SPEEDRUN_DEVICE_BATCH_SIZE"
    "--eval-every=$SPEEDRUN_TRAIN_EVAL_EVERY"
    "--core-metric-every=$SPEEDRUN_TRAIN_CORE_METRIC_EVERY"
    "--sample-every=$SPEEDRUN_TRAIN_SAMPLE_EVERY"
    "--run=$WANDB_RUN"
)

if [[ "$SPEEDRUN_ENABLE_FP8" == "1" ]]; then
    base_train_args+=("--fp8")
fi
if [[ -n "$SPEEDRUN_TOTAL_BATCH_SIZE" ]]; then
    base_train_args+=("--total-batch-size=$SPEEDRUN_TOTAL_BATCH_SIZE")
fi
if [[ -n "$SPEEDRUN_TOKEN_CACHE_DIR" ]]; then
    base_train_args+=("--token-cache-dir=$SPEEDRUN_TOKEN_CACHE_DIR")
fi
if [[ "$SPEEDRUN_SPARSE_MODE" == "1" ]]; then
    base_train_args+=("--sparse-mode" "--sparse-manifest=$SPEEDRUN_SPARSE_MANIFEST")
fi

torchrun --standalone --nproc_per_node="$SPEEDRUN_NPROC_PER_NODE" -m scripts.base_train -- "${base_train_args[@]}" ${SPEEDRUN_BASE_TRAIN_EXTRA_ARGS}

if [[ "$SPEEDRUN_SKIP_BASE_EVAL" != "1" ]]; then
    base_eval_args=(
        "--eval=core,bpb,sample"
        "--device-batch-size=$SPEEDRUN_EVAL_DEVICE_BATCH_SIZE"
    )
    if [[ -n "$SPEEDRUN_TOKEN_CACHE_DIR" ]]; then
        base_eval_args+=("--token-cache-dir=$SPEEDRUN_TOKEN_CACHE_DIR")
    fi
    torchrun --standalone --nproc_per_node="$SPEEDRUN_NPROC_PER_NODE" -m scripts.base_eval -- "${base_eval_args[@]}" ${SPEEDRUN_BASE_EVAL_EXTRA_ARGS}
fi

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

if [[ "$SPEEDRUN_SKIP_SFT" != "1" ]]; then
    if [[ "$SPEEDRUN_SKIP_IDENTITY_DOWNLOAD" != "1" ]]; then
        curl -L -o "$IDENTITY_CONVERSATIONS_PATH" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    else
        require_file "$IDENTITY_CONVERSATIONS_PATH" "Identity conversations download skipped but file is missing"
    fi

    torchrun --standalone --nproc_per_node="$SPEEDRUN_NPROC_PER_NODE" -m scripts.chat_sft -- --device-batch-size="$SPEEDRUN_SFT_DEVICE_BATCH_SIZE" --run="$WANDB_RUN" ${SPEEDRUN_CHAT_SFT_EXTRA_ARGS}

    if [[ "$SPEEDRUN_SKIP_CHAT_EVAL" != "1" ]]; then
        torchrun --standalone --nproc_per_node="$SPEEDRUN_NPROC_PER_NODE" -m scripts.chat_eval -- -i sft ${SPEEDRUN_CHAT_EVAL_EXTRA_ARGS}
    fi
fi

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
