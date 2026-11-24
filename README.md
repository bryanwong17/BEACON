<h1 align="center">Resemblance Alone Is Not Evidence: Uncertainty-Guided Patch Acquisition for Agentic Whole-Slide Image Reasoning</h1>

<p align="center">
  <a href="https://github.com/bryanwong17/DISCERN">
    <img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2608.05757">
    <img src="https://img.shields.io/badge/arXiv-2608.05757-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://arxiv.org/pdf/2608.05757">
    <img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper PDF">
  </a>
</p>

## 📣 Updates

* **2026.11.24** — Released **DISCERN** (arXiv v2), formerly **BEACON** (arXiv v1).
* **2026.08.06** — Released the initial code.

## 🚀 Overview

Answering a diagnostic question about a gigapixel whole-slide image (WSI) requires an agent to decide which few of its thousands of patches to read. Existing training-free agents read the patches that best match the question, assuming that **a patch resembling the question is also evidence for answering it**. In pathology this often fails: competing diagnoses share morphology, so many patches match the question yet are equally consistent with several answer options.

We propose **DISCERN**, a **training-free, plug-and-play** agent that keeps similarity as a fast **System 1** to shortlist candidate patches and adds a deliberate, uncertainty-guided **System 2** that reads the patches expected to **tell the options apart**, ranked by their **expected uncertainty reduction (EUR)**.

<p align="center">
  <img src="figures/DISCERN_Figure1.png" width="800">
</p>

### Highlights

- **Similarity finds the region but cannot choose within it.** Vision-language similarity retrieves a shortlist that usually contains useful patches, but ranks them at chance level within it, even when an LLM rewrites the query.
- **DISCERN** adds a System 2 that estimates an answer distribution from the captions of the shortlisted patches and acquires the patches with the highest EUR.
- On five MultiPathQA benchmarks, DISCERN achieves the **highest average accuracy** among training-free agents with both a pathology MLLM and GPT-5, **cuts GPT-5 API cost by 47%** compared with PathAgent, and improves over existing agents across pathology, medical, and general-purpose MLLMs.

<p align="center">
  <img src="figures/DISCERN_Figure2.png" width="900">
</p>

### How DISCERN Works

Each round has four stages:

1. **Coarse Candidate Retrieval (System 1)**: a pathology VLM shortlists the top-*M* patches most similar to the current query.
2. **Answer Distribution Construction (System 2)**: a frozen reranker scores how strongly each candidate's offline caption supports each answer option; averaging over the pool gives the answer distribution.
3. **Uncertainty-Guided Patch Acquisition (System 2)**: the top-*B* candidates by EUR are read by the MLLM.
4. **Evidence Control**: an LLM controller decides whether to answer, zoom in for a missing detail, or continue with a refined query.

## 🛠 1. Installation

```bash
git clone https://github.com/bryanwong17/DISCERN.git
cd DISCERN

conda env create -f environment.yml
conda activate discern

pip install --no-deps git+https://github.com/Mahmoodlab/CONCH.git
pip install --no-deps git+https://github.com/aldraus/quilt-llava.git
```

## 2. Data Preparation

Download MultiPathQA slides: see [GIANT: Downloading MultiPathQA Slides](https://github.com/KianWeihrauch/GIANT#-downloading-multipathqa-slides).

Next, arrange the data into a data root with one folder per dataset (`tcga_slidebench`, `tcga_expert_vqa`, `tcga`, `gtex`, `panda`), laid out like this:

```text
DATA_ROOT/
└── tcga_slidebench/
    ├── desc/patches_descriptions.json
    ├── img_features/
    │   ├── slide_a.npy
    │   └── ...
    └── patches_output/
        ├── slide_a/
        │   ├── 0_23904.png
        │   └── ...
        └── ...
```

- **`patches_output/`** (5x image patches): see [PathAgent: Section 2.1 (Patch Generation)](https://github.com/G14nTDo4/PathAgent#21-patch-generation).
- **`desc/patches_descriptions.json`** (offline 5x patch descriptions): see [PathAgent: Section 2.2 (Description Generation)](https://github.com/G14nTDo4/PathAgent#22-description-generation).
- **`img_features/`** (precomputed 5x VLM embeddings; saved as `img_features_conch/` when `VLM_BACKEND=conch`).

Separately, download model checkpoints and place them under a `MODELS_ROOT/` of your choice:

  | Model | Hugging Face |
  |---|---|
  | PLIP | [vinid/plip](https://huggingface.co/vinid/plip) |
  | CONCH | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
  | Patho-R1-7B | [WenchuanZhang/Patho-R1-7B](https://huggingface.co/WenchuanZhang/Patho-R1-7B) |
  | PathGen-LLaVA | [jamessyx/PathGen-LLaVA](https://huggingface.co/jamessyx/PathGen-LLaVA) |
  | Qwen3-Reranker-4B | [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |
  | Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |

To extract precomputed 5x VLM embeddings:

```bash
  export DATA_ROOT=/path/to/your/data/root
  export MODELS_ROOT=/path/to/your/models/root

  export VLM_BACKEND=plip           # plip, conch
  export DATASETS=tcga_slidebench   # tcga_slidebench, tcga_expert_vqa, tcga, gtex, panda

  bash run_extract_img_features.sh
```

## 3. Inference

```bash
export DATA_ROOT=/path/to/your/data/root      # data root from Section 2
export MODELS_ROOT=/path/to/your/models/root  # folder containing all model checkpoints

export VLM_BACKEND=plip           # plip, conch
export DATASETS=tcga_slidebench   # tcga_slidebench, tcga_expert_vqa, tcga, gtex, panda
export PERCEPTOR=patho_r1         # patho_r1, pathgen_llava, gpt5
export EVIDENCE_CONTROLLER=qwen   # qwen, gpt5

export RERANKER_CKPT=$MODELS_ROOT/Qwen3-Reranker-4B

export PATHO_R1_CKPT=$MODELS_ROOT/Patho-R1-7B           # if PERCEPTOR=patho_r1
# export PATHGEN_LLAVA_CKPT=$MODELS_ROOT/PathGen-LLaVA  # if PERCEPTOR=pathgen_llava

export QWEN_CKPT=$MODELS_ROOT/Qwen3-4B                  # if EVIDENCE_CONTROLLER=qwen

# if PERCEPTOR or EVIDENCE_CONTROLLER is gpt5, also export:
# export OPENROUTER_API_KEY=sk-or-v1-...
# export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

bash run_discern.sh
```

Results are written to `results/DISCERN_<VLM>_M<M>_B<B>_reranker_qwen3-<size>_perceptor_<perceptor>_controller_<controller>/multipathqa_<dataset>/`.

## 4. Evaluation

To avoid format-dependent mismatches and ensure consistent evaluation, we use an LLM-based evaluator (Qwen3-4B) to infer the intended answer choice.

```bash
export MODELS_ROOT=/path/to/your/models/root  # same root as Section 3

export DATASETS=tcga_slidebench  # tcga_slidebench, tcga_expert_vqa, tcga, gtex, panda
export BASE_DIR=results/DISCERN_PLIP_M25_B5_reranker_qwen3-4B_perceptor_Patho-R1-7B_controller_Qwen3-4B  # results directory from Section 3

export QWEN_CKPT=$MODELS_ROOT/Qwen3-4B

bash run_eval.sh
```

Results are written to `$BASE_DIR/multipathqa_<dataset>/eval_summary.json`.

ACC is reported for `tcga_slidebench` and `tcga_expert_vqa`.
BACC is reported for `tcga`, `gtex`, and `panda`.

## Citation

If you find our work useful in your research, please consider starring 🌟 this repo and citing our [paper](https://arxiv.org/abs/2608.05757) at:

```
@article{wong2026discern,
  title={Resemblance Alone Is Not Evidence: Uncertainty-Guided Patch Acquisition for Agentic Whole-Slide Image Reasoning},
  author={Wong, Bryan and Xu, Xun and Fu, Huazhu and Chen, Nancy F. and Yi, Mun Yong},
  journal={arXiv preprint arXiv:2608.05757},
  year={2026}
}
```

## Acknowledgements

This project is based on [PathAgent](https://github.com/G14nTDo4/PathAgent), [GIANT](https://github.com/KianWeihrauch/GIANT/tree/main?tab=readme-ov-file), [PLIP](https://github.com/PathologyFoundation/plip), [CONCH](https://github.com/mahmoodlab/CONCH), [Patho-R1](https://github.com/wenchuan-zhang/patho-r1), [PathGen-LLaVA](https://github.com/PathFoundation/PathGen-1.6M), and [Qwen](https://github.com/QwenLM). We sincerely thank the authors of these excellent works.

## Contact

If you have any questions, feedback, or issues regarding this project, please reach out via email at [bryan.wong@kaist.ac.kr](mailto:bryan.wong@kaist.ac.kr) or open an issue on GitHub.
