#!/bin/bash
# Multi-GPU offline patch feature extraction. Wraps extract_img_features.py,
# one process per split_name/slides_part{i}.txt file per dataset.
#
# Usage:
#   VLM_BACKEND=conch DATASETS="gtex panda" ./extract_img_features.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISCERN_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_ROOT="${MODELS_ROOT:-$DISCERN_ROOT/models}"
DATA_ROOT="${DATA_ROOT:-$DISCERN_ROOT/data}"

VLM_BACKEND="${VLM_BACKEND:-conch}"
if [[ "$VLM_BACKEND" == "conch" ]]; then
  VLM_CKPT="${VLM_CKPT:-$MODELS_ROOT/CONCH/pytorch_model.bin}"
  FEATURE_SUBDIR="${FEATURE_SUBDIR:-img_features_conch}"
else
  VLM_CKPT="${VLM_CKPT:-$MODELS_ROOT/plip}"
  FEATURE_SUBDIR="${FEATURE_SUBDIR:-img_features}"
fi

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-12}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# GPUs to spread split parts across; defaults to all visible GPUs, or
# override with GPUS="0 1" to pin explicit device indices.
if [[ -n "${GPUS:-}" ]]; then
  read -ra GPUS <<< "$GPUS"
else
  DETECTED_GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
  if [[ "$DETECTED_GPU_COUNT" -lt 1 ]]; then
    echo "No GPUs detected (nvidia-smi -L returned none) and GPUS was not set. Aborting." >&2
    exit 1
  fi
  GPUS=($(seq 0 $((DETECTED_GPU_COUNT - 1))))
fi
NUM_GPUS=${#GPUS[@]}

if [[ -n "${DATASETS:-}" ]]; then
  read -ra DATASETS <<< "$DATASETS"
else
  DATASETS=(gtex panda tcga tcga_expert_vqa tcga_slidebench)
fi

VLM_FLAGS=(--vlm_backend "$VLM_BACKEND" --vlm_ckpt "$VLM_CKPT")

echo "VLM backend: $VLM_BACKEND ($VLM_CKPT)"
echo "Output subdir: $FEATURE_SUBDIR"
echo "Datasets: ${DATASETS[*]}"
echo "GPUs: ${GPUS[*]}"

part_idx=0
for DATASET in "${DATASETS[@]}"; do
    DATASET_ROOT="$DATA_ROOT/$DATASET"
    SPLIT_DIR="$DATASET_ROOT/split_name"

    if [[ ! -d "$SPLIT_DIR" ]]; then
        echo "No split_name/ dir for $DATASET at $SPLIT_DIR, skipping."
        continue
    fi

    echo "=============================================="
    echo "Processing dataset: $DATASET"
    echo "=============================================="

    for TASK_FILE in "$SPLIT_DIR"/slides_part*.txt; do
        [[ -e "$TASK_FILE" ]] || continue
        PART_NAME="$(basename "$TASK_FILE" .txt)"
        LOG_FILE="$LOG_DIR/${DATASET}_${VLM_BACKEND}_${PART_NAME}.log"
        GPU_ID=${GPUS[$(( part_idx % NUM_GPUS ))]}
        part_idx=$((part_idx + 1))

        echo "Assigning [$DATASET] $PART_NAME to GPU $GPU_ID. Logging to $LOG_FILE"

        CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=1 nohup python "$SCRIPT_DIR/extract_img_features.py" \
            "${VLM_FLAGS[@]}" \
            --task_file "$TASK_FILE" \
            --img_root "$DATASET_ROOT/patches_output" \
            --output_dir "$DATASET_ROOT/$FEATURE_SUBDIR" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            > "$LOG_FILE" 2>&1 &
    done
done

wait
echo "All extraction jobs finished. Logs in $LOG_DIR"
