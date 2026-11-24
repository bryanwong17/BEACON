#!/bin/bash
# LLM-based evaluator for a BEACON results directory: infers the intended answer choice
# from each free-form prediction, then scores it. See eval/llm_evaluator.py for the logic.
#
# Usage:
#   BASE_DIR=results/<run_name> DATASETS="tcga_slidebench" bash eval/llm_evaluator.sh
# Prefer running via run_eval.sh, which also sets up the conda env and model paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEACON_ROOT="${BEACON_ROOT:-$(dirname "$SCRIPT_DIR")}"
BASE_DIR="${BASE_DIR:?Set BASE_DIR to the results directory to score, e.g. results/BEACON_PLIP_M25_B5_...}"
QWEN_CKPT="${QWEN_CKPT:-${BEACON_ROOT}/models/Qwen3-4B}"

SEED="${SEED:-128}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
VERBOSE="${VERBOSE:-0}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

EXTRA_FLAGS=()
[[ "$STRICT_DETERMINISM" == "1" ]] && EXTRA_FLAGS+=(--strict_determinism)
[[ "$VERBOSE" == "1" ]] && EXTRA_FLAGS+=(--verbose)

if [[ -n "${DATASETS:-}" ]]; then
  read -ra DATASETS <<< "$DATASETS"
else
  DATASETS=(tcga_slidebench tcga_expert_vqa tcga gtex panda)
fi

echo "========================================"
echo "LLM-based evaluation"
echo "Base dir: ${BASE_DIR}"
echo "Qwen ckpt: ${QWEN_CKPT}"
echo "Datasets: ${DATASETS[*]}"
echo "========================================"

OVERALL_EXIT_CODE=0
for dataset in "${DATASETS[@]}"; do
  RESULTS_DIR="${BASE_DIR}/multipathqa_${dataset}"
  if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "-- ${dataset}: SKIPPED (no results dir at ${RESULTS_DIR})"
    continue
  fi

  python "${BEACON_ROOT}/eval/llm_evaluator.py" \
    --results_dir "${RESULTS_DIR}" \
    --qwen_ckpt "${QWEN_CKPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}"
  PYTHON_EXIT_CODE=$?
  if [[ "$PYTHON_EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: evaluation failed for dataset '${dataset}' with exit code ${PYTHON_EXIT_CODE}" >&2
    OVERALL_EXIT_CODE="$PYTHON_EXIT_CODE"
  fi
done

echo "========================================"
echo "Summary (${BASE_DIR})"
# Reads the eval_summary.json files llm_evaluator.py wrote for each dataset above.
python - "$BASE_DIR" "${DATASETS[@]}" <<'PYEOF'
import json, os, sys

base_dir, datasets = sys.argv[1], sys.argv[2:]
for dataset in datasets:
    summary_path = os.path.join(base_dir, f"multipathqa_{dataset}", "eval_summary.json")
    if not os.path.exists(summary_path):
        print(f"  {dataset:<20} (no summary)")
        continue
    s = json.load(open(summary_path))
    print(f"  {dataset:<20} {s['metric_name'].upper()}={s['metric_value']:.2f}  n={s['total_mcq']}")
PYEOF
echo "========================================"

exit "$OVERALL_EXIT_CODE"
