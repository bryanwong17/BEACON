SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEACON_ROOT="$SCRIPT_DIR"
MODELS_ROOT="${MODELS_ROOT:-$BEACON_ROOT/models}"
DATA_ROOT="${DATA_ROOT:-$BEACON_ROOT/data}"

SEED="${SEED:-128}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-0}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

QWEN_CKPT="${QWEN_CKPT:-$MODELS_ROOT/Qwen3-4B}"

VLM_BACKEND="${VLM_BACKEND:-plip}"
if [[ "$VLM_BACKEND" == "conch" ]]; then
  VLM_CKPT="${VLM_CKPT:-$MODELS_ROOT/CONCH/pytorch_model.bin}"
else
  VLM_CKPT="${VLM_CKPT:-$MODELS_ROOT/plip}"
fi
PATHO_R1_CKPT="${PATHO_R1_CKPT:-$MODELS_ROOT/Patho-R1-7B}"
RERANKER_CKPT="${RERANKER_CKPT:-$MODELS_ROOT/Qwen3-Reranker-4B}"
PATHGEN_LLAVA_CKPT="${PATHGEN_LLAVA_CKPT:-$MODELS_ROOT/PathGen-LLaVA}"
PATHGEN_LLAVA_INPUT_SIZE="${PATHGEN_LLAVA_INPUT_SIZE:-1024}"
QUESTIONS_FILE="$BEACON_ROOT/MultiPathQA.csv"


PERCEPTOR="${PERCEPTOR:-patho_r1}"
EVIDENCE_CONTROLLER="${EVIDENCE_CONTROLLER:-qwen}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-}"
OPENROUTER_PERCEPTOR_MODEL="${OPENROUTER_PERCEPTOR_MODEL:-openai/gpt-5}"
OPENROUTER_CONTROLLER_MODEL="${OPENROUTER_CONTROLLER_MODEL:-openai/gpt-5}"
OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-high}"
OPENROUTER_PERCEPTOR_IMAGE_SIZE="${OPENROUTER_PERCEPTOR_IMAGE_SIZE:-500}"

VLM_FILTER_TOP_K="${VLM_FILTER_TOP_K:-25}"
PATCHES_PER_ROUND="${PATCHES_PER_ROUND:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
PRESCREEN_BATCH_SIZE="${PRESCREEN_BATCH_SIZE:-4}"
MAGNIFICATION="${MAGNIFICATION:-5}"
PATHO_R1_INPUT_SIZE="${PATHO_R1_INPUT_SIZE:-1024}"

RERANKER_LABEL="${RERANKER_CKPT##*/}"
RERANKER_LABEL="${RERANKER_LABEL#Qwen3-Reranker-}"

if [[ "$PERCEPTOR" == "patho_r1" ]]; then
  PERCEPTOR_LABEL="${PATHO_R1_CKPT##*/}"
elif [[ "$PERCEPTOR" == "pathgen_llava" ]]; then
  PERCEPTOR_LABEL="${PATHGEN_LLAVA_CKPT##*/}"
else
  PERCEPTOR_LABEL="${OPENROUTER_PERCEPTOR_MODEL//\//_}"
fi
if [[ "$EVIDENCE_CONTROLLER" == "qwen" ]]; then
  CONTROLLER_LABEL="${QWEN_CKPT##*/}"
else
  CONTROLLER_LABEL="${OPENROUTER_CONTROLLER_MODEL//\//_}"
fi

RESULTS_DIR="results/BEACON_${VLM_BACKEND^^}_M${VLM_FILTER_TOP_K}_B${PATCHES_PER_ROUND}_reranker_qwen3-${RERANKER_LABEL}_perceptor_${PERCEPTOR_LABEL}_controller_${CONTROLLER_LABEL}"

if [[ "$VLM_BACKEND" == "conch" ]]; then
  FEATURE_SUBDIR="img_features_conch"
else
  FEATURE_SUBDIR="img_features"
fi

STRICT_FLAG=()
if [[ "$STRICT_DETERMINISM" == "1" ]]; then
  STRICT_FLAG=(--strict_determinism)
fi

BACKEND_FLAGS=(--perceptor "$PERCEPTOR" --evidence_controller "$EVIDENCE_CONTROLLER")
if [[ "$PERCEPTOR" == "patho_r1" ]]; then
  BACKEND_FLAGS+=(--patho_r1_ckpt "$PATHO_R1_CKPT")
elif [[ "$PERCEPTOR" == "pathgen_llava" ]]; then
  BACKEND_FLAGS+=(--pathgen_llava_ckpt "$PATHGEN_LLAVA_CKPT" --pathgen_llava_input_size "$PATHGEN_LLAVA_INPUT_SIZE")
fi
if [[ "$EVIDENCE_CONTROLLER" == "qwen" ]]; then
  BACKEND_FLAGS+=(--qwen_ckpt "$QWEN_CKPT")
fi

BACKEND_FLAGS+=(--vlm_backend "$VLM_BACKEND" --vlm_ckpt "$VLM_CKPT")
if [[ "$PERCEPTOR" == "gpt5" || "$EVIDENCE_CONTROLLER" == "gpt5" ]]; then
  BACKEND_FLAGS+=(
    --openrouter_perceptor_model "$OPENROUTER_PERCEPTOR_MODEL"
    --openrouter_controller_model "$OPENROUTER_CONTROLLER_MODEL"
    --openrouter_reasoning_effort "$OPENROUTER_REASONING_EFFORT"
  )
  if [[ -n "$OPENROUTER_BASE_URL" ]]; then
    BACKEND_FLAGS+=(--openrouter_base_url "$OPENROUTER_BASE_URL")
  fi
  if [[ -n "$OPENROUTER_API_KEY" ]]; then
    BACKEND_FLAGS+=(--openrouter_api_key "$OPENROUTER_API_KEY")
  fi
fi
if [[ "$PERCEPTOR" == "gpt5" ]]; then
  BACKEND_FLAGS+=(--openrouter_perceptor_image_size "$OPENROUTER_PERCEPTOR_IMAGE_SIZE")
fi

if [[ -n "${DATASETS:-}" ]]; then
  read -ra DATASETS <<< "$DATASETS"
else
  DATASETS=(
    tcga_slidebench
    tcga_expert_vqa
    tcga
    gtex
    panda
  )
fi

for DATASET in "${DATASETS[@]}"
do
    echo "========================================"
    echo "Running BEACON on dataset: $DATASET"
    echo "Seed: $SEED  |  M=$VLM_FILTER_TOP_K  B=$PATCHES_PER_ROUND  T=$MAX_ROUNDS  reranker=$RERANKER_LABEL"
    echo "Acquisition: top-B by EIG"
    echo "Results: $RESULTS_DIR/multipathqa_${DATASET}"
    echo "========================================"

    python "$BEACON_ROOT/beacon.py" \
        "${BACKEND_FLAGS[@]}" \
        --reranker_ckpt "$RERANKER_CKPT" \
        --descriptions_file "$DATA_ROOT/${DATASET}/desc/patches_descriptions.json" \
        --questions_file "$QUESTIONS_FILE" \
        --feature_dir "$DATA_ROOT/${DATASET}/${FEATURE_SUBDIR}" \
        --patch_root "$DATA_ROOT/${DATASET}/patches_output" \
        --save_dir "$BEACON_ROOT/${RESULTS_DIR}/multipathqa_${DATASET}" \
        --dataset_name "multipathqa_${DATASET}" \
        --plip_top_k "$VLM_FILTER_TOP_K" \
        --patches_per_round "$PATCHES_PER_ROUND" \
        --max_rounds "$MAX_ROUNDS" \
        --prescreen_batch_size "$PRESCREEN_BATCH_SIZE" \
        --magnification "$MAGNIFICATION" \
        --patho_r1_input_size "$PATHO_R1_INPUT_SIZE" \
        --seed "$SEED" \
        "${STRICT_FLAG[@]}"
    PYTHON_EXIT_CODE=$?
    if [[ "$PYTHON_EXIT_CODE" -ne 0 ]]; then
        echo "ERROR: beacon.py failed for dataset '$DATASET' with exit code $PYTHON_EXIT_CODE" >&2
        OVERALL_EXIT_CODE="$PYTHON_EXIT_CODE"
    fi
done

exit "${OVERALL_EXIT_CODE:-0}"
