set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate beacon

export DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to your data root, e.g. /path/to/data}"
export MODELS_ROOT="${MODELS_ROOT:?Set MODELS_ROOT to your models root, e.g. /path/to/models}"

export VLM_BACKEND="${VLM_BACKEND:-plip}"  # plip, conch
export DATASETS="${DATASETS:-tcga_slidebench}"  # tcga_slidebench, tcga_expert_vqa, tcga, gtex, panda

bash "$SCRIPT_DIR/data_utils/extract_img_features.sh"
