import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DISCERN: uncertainty-guided patch acquisition for agentic WSI reasoning")

    parser.add_argument("--vlm_backend", type=str, default="plip", choices=["plip", "conch"],
                        help="VLM backend for coarse candidate retrieval: 'plip' or 'conch'.")
    parser.add_argument("--vlm_ckpt", type=str, required=True,
                        help="Path or HF id for the VLM checkpoint.")

    parser.add_argument("--perceptor", type=str, default="patho_r1",
                        choices=["patho_r1", "pathgen_llava", "gpt5"],
                        help="Vision-language backend that produces question-aware patch descriptions.")
    parser.add_argument("--evidence_controller", type=str, default="qwen", choices=["qwen", "gpt5"],
                        help="Text-reasoning backend for sufficiency/zoom judgement and answer synthesis.")
    parser.add_argument("--openrouter_api_key", type=str, default=None,
                        help="OpenRouter API key (falls back to the OPENROUTER_API_KEY env var).")
    parser.add_argument("--openrouter_base_url", type=str, default=None,
                        help="OpenRouter-compatible API base URL (falls back to the "
                             "OPENROUTER_BASE_URL env var; required when --perceptor gpt5 "
                             "or --evidence_controller gpt5).")
    parser.add_argument("--openrouter_perceptor_model", type=str, default="openai/gpt-5",
                        help="OpenRouter model id used when --perceptor gpt5.")
    parser.add_argument("--openrouter_controller_model", type=str, default="openai/gpt-5",
                        help="OpenRouter model id used when --evidence_controller gpt5.")
    parser.add_argument("--openrouter_reasoning_effort", type=str, default="high",
                        help="Reasoning effort for gpt5 backends; pass '' for the provider default.")
    parser.add_argument("--openrouter_perceptor_image_size", type=int, default=500,
                        help="Resize patches to this square size before the gpt5 perceptor; <=0 disables.")

    parser.add_argument("--qwen_ckpt", type=str, default=None,
                        help="Path to Qwen checkpoint. Required when --evidence_controller qwen.")
    parser.add_argument("--patho_r1_ckpt", type=str, default=None,
                        help="Path to Patho-R1 checkpoint. Required when --perceptor patho_r1.")
    parser.add_argument("--pathgen_llava_ckpt", type=str, default=None,
                        help="Path to the full PathGen-LLaVA checkpoint. Required when --perceptor pathgen_llava.")
    parser.add_argument("--reranker_ckpt", type=str, required=True,
                        help="Path or HF id for the Qwen3-Reranker checkpoint (f_R).")

    parser.add_argument("--descriptions_file", type=str, required=True,
                        help="Path to patch descriptions JSON file.")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="Path to questions/VQA dataset file.")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing image features.")
    parser.add_argument("--patch_root", type=str, required=True,
                        help="Root directory for image patches.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save results.")

    parser.add_argument("--dataset_name", type=str, default="wsi_vqa",
                        help="Name of the dataset (e.g., wsi_vqa, slidechat).")
    parser.add_argument("--plip_top_k", type=int, default=25,
                        help="Coarse candidate pool size M.")
    parser.add_argument("--patches_per_round", type=int, default=5,
                        help="Batch size B per acquisition round (top-B EUR selection).")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Maximum acquisition rounds per case (T).")
    parser.add_argument("--prescreen_batch_size", type=int, default=4,
                        help="Batch size for reranker scoring of answer statements.")
    parser.add_argument("--magnification", type=int, default=5,
                        help="Initial magnification level passed to the perceptor.")
    parser.add_argument("--patho_r1_input_size", type=int, default=1024,
                        help="Resize perceptor input images to this square size; <=0 disables.")
    parser.add_argument("--pathgen_llava_input_size", type=int, default=1024,
                        help="Resize patches before the pathgen_llava perceptor; <=0 disables.")
    parser.add_argument("--seed", type=int, default=128,
                        help="Global seed for reproducible reruns.")
    parser.add_argument("--strict_determinism", action="store_true",
                        help="Raise on nondeterministic torch ops instead of warning.")

    parsed_args = parser.parse_args()

    if parsed_args.perceptor == "patho_r1" and not parsed_args.patho_r1_ckpt:
        parser.error("--patho_r1_ckpt is required when --perceptor patho_r1")
    if parsed_args.perceptor == "pathgen_llava" and not parsed_args.pathgen_llava_ckpt:
        parser.error("--pathgen_llava_ckpt is required when --perceptor pathgen_llava")
    if parsed_args.evidence_controller == "qwen" and not parsed_args.qwen_ckpt:
        parser.error("--qwen_ckpt is required when --evidence_controller qwen")
    if parsed_args.perceptor == "gpt5" or parsed_args.evidence_controller == "gpt5":
        if not (parsed_args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")):
            parser.error(
                "--openrouter_api_key (or the OPENROUTER_API_KEY env var) is required "
                "when --perceptor gpt5 or --evidence_controller gpt5"
            )
        if not (parsed_args.openrouter_base_url or os.environ.get("OPENROUTER_BASE_URL")):
            parser.error(
                "--openrouter_base_url (or the OPENROUTER_BASE_URL env var) is required "
                "when --perceptor gpt5 or --evidence_controller gpt5"
            )
    return parsed_args


def resolve_perceptor_image_size(args: argparse.Namespace) -> int:
    if args.perceptor == "gpt5":
        return args.openrouter_perceptor_image_size
    if args.perceptor == "pathgen_llava":
        return args.pathgen_llava_input_size
    return args.patho_r1_input_size


