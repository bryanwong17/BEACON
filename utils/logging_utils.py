import argparse
import os


def _ckpt_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def print_run_banner(args: argparse.Namespace) -> None:
    print("=" * 40)
    print("BEACON: Bayesian Evidence Acquisition for WSI Reasoning")
    print(f"VLM backend:         {args.vlm_backend} ({args.vlm_ckpt})")
    print(f"Evidence controller: " + (
        f"{_ckpt_name(args.qwen_ckpt)} ({args.qwen_ckpt})" if args.evidence_controller == "qwen"
        else f"{args.openrouter_controller_model} (via OpenRouter)"
    ))

    if args.perceptor == "patho_r1":
        perceptor_line = f"{_ckpt_name(args.patho_r1_ckpt)} ({args.patho_r1_ckpt})"
    elif args.perceptor == "pathgen_llava":
        perceptor_line = f"{_ckpt_name(args.pathgen_llava_ckpt)} ({args.pathgen_llava_ckpt})"
    else:
        perceptor_line = f"{args.openrouter_perceptor_model} (via OpenRouter)"
    print(f"Perceptor:           {perceptor_line}")

    if args.perceptor == "gpt5" or args.evidence_controller == "gpt5":
        print(f"OpenRouter reasoning effort: {args.openrouter_reasoning_effort or '(provider default)'}")

    print(f"Reranker:            {_ckpt_name(args.reranker_ckpt)} ({args.reranker_ckpt})")
    print(f"Dataset:             {args.dataset_name}")
    print(f"Results to:          {args.save_dir}")
    print(
        f"Acquisition config: M={args.plip_top_k}, B={args.patches_per_round}, "
        f"T={args.max_rounds}, rule=top-B by EIG"
    )
    print("=" * 40)
