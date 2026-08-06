set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate beacon

export MODELS_ROOT="${MODELS_ROOT:?Set MODELS_ROOT to your models root, e.g. /path/to/models}"
export QWEN_CKPT="${QWEN_CKPT:-$MODELS_ROOT/Qwen3-4B}"

export BASE_DIR="${BASE_DIR:?Set BASE_DIR to the results directory to score, e.g. results/BEACON_PLIP_M25_B5_...}"
export DATASETS="${DATASETS:-tcga_slidebench}"  # tcga_slidebench, tcga_expert_vqa, tcga, gtex, panda

bash "$SCRIPT_DIR/eval/llm_evaluator.sh"
