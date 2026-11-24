set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate discern

export DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to your data root, e.g. /path/to/data}"
export MODELS_ROOT="${MODELS_ROOT:?Set MODELS_ROOT to your models root, e.g. /path/to/models}"

export QWEN_CKPT="${QWEN_CKPT:-$MODELS_ROOT/Qwen3-4B}"
export PATHO_R1_CKPT="${PATHO_R1_CKPT:-$MODELS_ROOT/Patho-R1-7B}"
export RERANKER_CKPT="${RERANKER_CKPT:-$MODELS_ROOT/Qwen3-Reranker-4B}"
export PATHGEN_LLAVA_CKPT="${PATHGEN_LLAVA_CKPT:-$MODELS_ROOT/PathGen-LLaVA}"

export VLM_BACKEND="${VLM_BACKEND:-plip}"  # plip, conch
export DATASETS="${DATASETS:-tcga_slidebench}"  # tcga_slidebench, tcga_expert_vqa, tcga, gtex, panda
export PERCEPTOR="${PERCEPTOR:-patho_r1}"  # patho_r1, pathgen_llava, gpt5
export EVIDENCE_CONTROLLER="${EVIDENCE_CONTROLLER:-qwen}"  # qwen, gpt5

export VLM_FILTER_TOP_K="${VLM_FILTER_TOP_K:-25}"       # M: coarse candidate pool size
export PATCHES_PER_ROUND="${PATCHES_PER_ROUND:-5}"      # B: patches acquired per round
export MAX_ROUNDS="${MAX_ROUNDS:-5}"                    # T: max acquisition rounds per case

# Only needed when PERCEPTOR or EVIDENCE_CONTROLLER is "gpt5":
# export OPENROUTER_API_KEY=sk-or-v1-...
# export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

bash "$SCRIPT_DIR/discern.sh"
