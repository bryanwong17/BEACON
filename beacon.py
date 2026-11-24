import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import sys
import warnings
from copy import deepcopy
from typing import Dict, Optional

import json
import numpy as np
import torch

from cli import parse_args, resolve_perceptor_image_size

warnings.filterwarnings("ignore")

from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

args = parse_args()


from vlm.factory import build_vlm_encoder
from data_utils.utils import (
    build_descriptions_with_meta,
    extract_coords_from_name,
    get_patch_fullpath,
    get_specific_case_descriptions,
    load_all_vqa_pairs,
    load_image,
    make_unique_id,
    maybe_resize_for_perceptor,
    split_patch_for_zoom,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.evidence_controller import build_evidence_controller
from models.perceptor import build_perceptor
from utils.reproducibility import set_seed, configure_generation_defaults
from utils.logging_utils import print_run_banner
from pipeline.belief import aggregate_prior_from_R, normalize_R_rows
from pipeline.retrieval import coarse_candidate_retrieval
from models.reranker import prepare_runtime, score_candidates
from pipeline.acquisition import select_top_b_by_eig


def main():
    set_seed(args.seed, strict=args.strict_determinism)

    os.makedirs(args.save_dir, exist_ok=True)

    perceptor_image_size = resolve_perceptor_image_size(args)
    print_run_banner(args)

    print(f"Loading VQA pairs from {args.questions_file}...")
    pairs = load_all_vqa_pairs(
        args.questions_file,
        dataset_name=args.dataset_name,
        image_dir=args.patch_root
    )
    print(f"Successfully loaded {len(pairs)} VQA pairs.")

    zoom_level_val = args.magnification

    openrouter_client = None
    if args.perceptor == "gpt5" or args.evidence_controller == "gpt5":
        from models.openrouter_client import OpenRouterClient
        openrouter_client = OpenRouterClient(
            api_key=args.openrouter_api_key,
            base_url=args.openrouter_base_url,
        )
        print("OpenRouter client initialized.")

    openrouter_reasoning_effort = args.openrouter_reasoning_effort or None

    vlm_encoder = build_vlm_encoder(args.vlm_backend, args.vlm_ckpt)
    vlm_encoder.model.eval()
    print(f"VLM encoder loaded successfully: {args.vlm_backend}")

    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_ckpt, padding_side="left")
    reranker_model = AutoModelForCausalLM.from_pretrained(
        args.reranker_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    reranker_model.eval()
    configure_generation_defaults(reranker_model)
    reranker_runtime = prepare_runtime(reranker_tokenizer)
    print("Reranker model loaded successfully.")

    perceptor = build_perceptor(
        args.perceptor,
        patho_r1_ckpt=args.patho_r1_ckpt,
        pathgen_llava_ckpt=args.pathgen_llava_ckpt,
        openrouter_client=openrouter_client,
        openrouter_model=args.openrouter_perceptor_model,
        openrouter_reasoning_effort=openrouter_reasoning_effort,
    )
    print(f"Perceptor backend ready: {args.perceptor}")

    controller = build_evidence_controller(
            args.evidence_controller,
            qwen_ckpt=args.qwen_ckpt,
            openrouter_client=openrouter_client,
            openrouter_model=args.openrouter_controller_model,
            openrouter_reasoning_effort=openrouter_reasoning_effort,
        )
    print(f"Evidence controller backend ready: {args.evidence_controller}")

    run_config = {
        "perceptor_model": perceptor.model_name,
        "evidence_controller_model": controller.model_name,
        "openrouter_reasoning_effort": openrouter_reasoning_effort,
        "perceptor_image_size": perceptor_image_size,
        "reranker_ckpt": args.reranker_ckpt,
        "vlm_backend": args.vlm_backend,
        "vlm_ckpt": args.vlm_ckpt,
    }
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    finished_cases = set()
    if os.path.exists(args.save_dir):
        for f in os.listdir(args.save_dir):
            if f.endswith(".json"):
                finished_cases.add(f.replace(".json", ""))

    with open(args.descriptions_file, "r", encoding="utf-8") as f:
        all_descriptions = json.load(f)
    all_keys = list(all_descriptions.keys())
    print(f"Pre-loaded descriptions file, containing {len(all_keys)} keys.")

    def run_single_case(idx: int, pair: dict) -> Optional[dict]:
        nonlocal zoom_level_val
        target_long_id = pair["long_id"]

        matching_keys = [k for k in all_keys if k.startswith(target_long_id)]
        if len(matching_keys) == 1:
            target_long_id = matching_keys[0]
        elif len(matching_keys) > 1:
            print(f"Warning: Found multiple keys starting with '{target_long_id}', using the first one: {matching_keys[0]}")
            target_long_id = matching_keys[0]
        else:
            print(f"Target ID '{target_long_id}' not found in description file, keeping as is.")

        question_text = pair["question"]
        question_choices = pair["choices"]
        correct_answer = pair["answer"]

        unique_id = make_unique_id(target_long_id, question_text)

        save_path = os.path.join(args.save_dir, f"{unique_id}.json")

        if unique_id in finished_cases:
            print(f"Skipping Case {idx+1}/{len(pairs)}: {unique_id} (Already completed)")
            return None

        print(f"\nStarting Case {idx+1}/{len(pairs)}: {unique_id}")

        print("="*80)
        print(f"Question: {question_text}")
        print(f"Choices: {question_choices}")
        print(f"Ground Truth: {correct_answer}")

        descriptions_dict = get_specific_case_descriptions(args.descriptions_file, target_long_id)
        descriptions_dict = deepcopy(descriptions_dict)
        print(f"Extracted {len(descriptions_dict)} patch descriptions.")

        feature_cache = {}
        slide_npy_path = os.path.join(args.feature_dir, f"{target_long_id}.npy")

        if os.path.exists(slide_npy_path):
            try:
                slide_data = np.load(slide_npy_path, allow_pickle=True).item()
                embeddings = slide_data['embeddings']
                filenames = slide_data['filenames']
                for name, emb in zip(filenames, embeddings):
                    feature_cache[name] = emb
                print(f"Successfully cached {len(feature_cache)} / {len(descriptions_dict)} patch features")
            except Exception as e:
                print(f"Error loading slide features from {slide_npy_path}: {e}")
        else:
            print(f"Warning: Feature file not found for slide {target_long_id} at {slide_npy_path}")

        all_patch_names = list(descriptions_dict.keys())
        missing_count = sum(1 for p in all_patch_names if p not in feature_cache)
        if missing_count > 0:
            print(f"Note: {missing_count} patches in descriptions do not have corresponding features.")

        retrieval_query = question_text
        accumulated_patch_names = []
        queried_patch_names = []
        remaining_patch_names = list(all_patch_names)
        patches_to_evaluate_names = []
        belief = {}

        case_R_cache: Dict[str, Dict[str, float]] = {}

        attempt = 0
        max_attempts = max(1, args.max_rounds)
        final_answer = None
        all_results = []

        while attempt < max_attempts:
            attempt += 1

            if not remaining_patch_names:
                print("No remaining candidate patches available, stopping.")
                break

            # Stage 1: Coarse Candidate Retrieval
            print(f"\n[ACQ] Retrieval query for round {attempt}: {retrieval_query}")
            candidate_patch_names = coarse_candidate_retrieval(
                vlm_encoder,
                retrieval_query,
                remaining_patch_names,
                feature_cache,
                top_m=max(1, args.plip_top_k),
            )
            candidate_patch_names = [
                name for name in candidate_patch_names
                if name in descriptions_dict
            ]

            if not candidate_patch_names:
                print("No candidate patches available after coarse retrieval, stopping.")
                break

            num_to_select = min(args.patches_per_round, len(candidate_patch_names))

            # Stage 2: Belief Construction
            uncached_names = [n for n in candidate_patch_names if n not in case_R_cache]
            if uncached_names:
                new_R = score_candidates(
                    reranker_model,
                    reranker_tokenizer,
                    reranker_runtime,
                    uncached_names,
                    descriptions_dict,
                    question_text,
                    question_choices,
                    batch_size=args.prescreen_batch_size,
                )
                case_R_cache.update(normalize_R_rows(new_R, question_choices))
            R = {name: case_R_cache[name] for name in candidate_patch_names}

            belief = aggregate_prior_from_R(R, question_choices)

            # Stage 3: Bayesian evidence acquisition
            patches_to_evaluate_names = select_top_b_by_eig(
                candidate_patch_names,
                R,
                belief,
                question_choices,
                batch_size=num_to_select,
            )
            if not patches_to_evaluate_names:
                print("Acquisition returned no patches, stopping.")
                break

            selected_set = set(patches_to_evaluate_names)
            queried_patch_names.extend(patches_to_evaluate_names)
            accumulated_patch_names.extend(patches_to_evaluate_names)
            remaining_patch_names = [
                p for p in remaining_patch_names
                if p not in selected_set
            ]

            print(
                f"\nRound {attempt}/{max_attempts}: selected {len(patches_to_evaluate_names)} "
                f"patches (top-B by EIG): {patches_to_evaluate_names}"
            )

            num_top_patches_to_describe = min(args.patches_per_round, len(patches_to_evaluate_names))

            def _load_patch(i):
                patch_name = patches_to_evaluate_names[i]
                patch_path = get_patch_fullpath(args.patch_root, target_long_id, patch_name)
                patch_img = load_image(patch_path)
                perceptor_img = maybe_resize_for_perceptor(
                    patch_img,
                    size=perceptor_image_size,
                )
                x, y = extract_coords_from_name(patch_name)
                return patch_name, x, y, perceptor_img

            def _describe_patch(i):
                patch_name, x, y, perceptor_img = _load_patch(i)
                question_specific_desc = perceptor.describe(
                    perceptor_img,
                    question=question_text,
                    coords=(x, y),
                    magnification=zoom_level_val,
                    choices=question_choices,
                )
                return patch_name, x, y, question_specific_desc.strip()

            if args.perceptor == "gpt5" and num_top_patches_to_describe > 1:
                loaded = [_load_patch(i) for i in range(num_top_patches_to_describe)]
                images_batch = [img for _, _, _, img in loaded]
                coords_batch = [(x, y) for _, x, y, _ in loaded]
                descs = perceptor.describe_batch(
                    images_batch,
                    question=question_text,
                    coords_list=coords_batch,
                    magnification=zoom_level_val,
                    choices=question_choices,
                )
                patch_results = [
                    (loaded[i][0], loaded[i][1], loaded[i][2], descs[i].strip())
                    for i in range(num_top_patches_to_describe)
                ]
            else:
                patch_results = [_describe_patch(i) for i in range(num_top_patches_to_describe)]

            for patch_name, x, y, desc in patch_results:
                descriptions_dict[patch_name] = desc
                print(f"[PERCEPTOR] {patch_name} @ ({x},{y}): {len(desc)} chars")

            # Stage 4: Evidence Control
            current_items_to_evaluate = [(name, descriptions_dict[name]) for name in patches_to_evaluate_names]
            desc_for_evaluation = build_descriptions_with_meta(current_items_to_evaluate, mag_level=zoom_level_val, include_header=True, include_coords=True)

            evaluation_result = controller.evaluate_with_llm_chain(
                desc_for_evaluation,
                question_text,
                question_choices,
            )
            can_now = str(evaluation_result.get("sufficient", "")).strip().lower() == "yes"
            can_zoom = str(evaluation_result.get("zoom_recommendation", "")).strip().lower() == "yes"

            if can_now:
                print("\nEvidence controller judgment: Sufficient -- answering now.")
                final_desc_text = controller.summarize_patches_in_chunks(
                    descriptions_dict, accumulated_patch_names,
                    question_text=question_text,
                    chunk_size=10, threshold=50,
                    magnification=zoom_level_val,
                )
                final_answer = controller.slide_llm_answer(
                    final_desc_text,
                    question_text,
                    question_choices,
                    magnification=zoom_level_val,
                    case_name=unique_id,
                )

                all_results.append({
                    "attempt": attempt,
                    "mode": "sufficient",
                    "retrieval_query": retrieval_query,
                    "evaluated_patches_this_round": patches_to_evaluate_names,
                    "total_accumulated_patches": len(accumulated_patch_names),
                    "evaluation_result": evaluation_result,
                    "answer": final_answer["answer"],
                    "explanation": final_answer["explanation"]
                })
                break

            elif can_zoom:
                final_desc_text = controller.summarize_patches_in_chunks(
                    descriptions_dict, accumulated_patch_names,
                    question_text=question_text,
                    chunk_size=10, threshold=50,
                    magnification=zoom_level_val,
                )
                zoom_reason = evaluation_result.get("zoom_reason", "").strip()
                print(f"\nEvidence controller judgment: Zoom. Reason: {zoom_reason}")
                zoom_level_raw = evaluation_result.get("zoom_level", zoom_level_val)
                try:
                    zoom_level_val = int(zoom_level_raw)
                except (ValueError, TypeError):
                    pass

                print(f"  -> Zoom level: {zoom_level_val}x")

                patch_feats = []
                patch_names_valid = []
                for p in patches_to_evaluate_names:
                    if p in feature_cache:
                        patch_feats.append(feature_cache[p])
                        patch_names_valid.append(p)
                if not patch_feats:
                    print("No valid patch features, skipping zoom process.")
                else:
                    patch_feats = np.stack(patch_feats, axis=0)
                    text_emb = vlm_encoder.encode_text([question_text], batch_size=1)
                    text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
                    sims = np.dot(patch_feats, text_emb.T).squeeze()

                    top2_idx = np.argsort(sims)[-2:][::-1]
                    top2_names = [patch_names_valid[i] for i in top2_idx]
                    print(f"  -> Selected Top2 patches: {top2_names}")

                    candidate_images = []
                    candidate_names = []
                    for patch_name in top2_names:
                        patch_path = get_patch_fullpath(args.patch_root, target_long_id, patch_name)
                        sub_patches = split_patch_for_zoom(patch_path, zoom_level_val)
                        for sub_img, (x, y) in sub_patches:
                            candidate_images.append(sub_img)
                            candidate_names.append(f"{x}_{y}")

                    if not candidate_images:
                        print("No sub-patches generated, skipping zoom process.")
                    else:
                        image_embs = vlm_encoder.encode_images(candidate_images, batch_size=4)
                        image_embs = image_embs / np.linalg.norm(image_embs, axis=-1, keepdims=True)
                        sims_sub = np.dot(image_embs, text_emb.T).squeeze()
                        best_idx = int(np.argmax(sims_sub))
                        best_img = candidate_images[best_idx]
                        best_name = candidate_names[best_idx]
                        print(f"Selected zoomed sub-patch: {best_name}")

                        x, y = best_name.split("_")
                        best_img_for_perceptor = maybe_resize_for_perceptor(
                            best_img,
                            size=perceptor_image_size,
                        )

                        zoomed_patch_desc = perceptor.describe(
                            best_img_for_perceptor,
                            question=question_text,
                            coords=(x, y),
                            magnification=zoom_level_val,
                            choices=question_choices,
                        )

                        extra_note = (
                            f"\n\n[NOTE] A key sub-patch was identified after zooming.\n"
                            f"Patch=({x},{y}), Magnification={zoom_level_val}x\n"
                            f"Detail: {zoomed_patch_desc}"
                        )
                        final_desc_text = final_desc_text + extra_note

                        final_answer = controller.slide_llm_answer(
                            final_desc_text,
                            question_text, question_choices,
                            magnification=zoom_level_val,
                            case_name=unique_id,
                        )

                        all_results.append({
                            "attempt": attempt,
                            "mode": "zoom_then_answer",
                            "retrieval_query": retrieval_query,
                            "evaluated_patches_this_round": patches_to_evaluate_names,
                            "total_accumulated_patches": len(accumulated_patch_names),
                            "evaluation_result": evaluation_result,
                            "selected_zoom_patch": best_name,
                            "zoom_patch_desc": zoomed_patch_desc,
                            "answer": final_answer["answer"],
                            "explanation": final_answer["explanation"]
                        })
                        break

            else:
                missing_info = evaluation_result.get("missing_info", "pathology details").strip()
                print(f"Evidence controller identifies missing info: '{missing_info}' -> continuing acquisition.")
                all_results.append({
                    "attempt": attempt,
                    "mode": "continue",
                    "retrieval_query": retrieval_query,
                    "next_retrieval_query": missing_info,
                    "evaluated_patches_this_round": patches_to_evaluate_names,
                    "total_accumulated_patches": len(accumulated_patch_names),
                    "remaining_pool": len(remaining_patch_names),
                    "evaluation_result": evaluation_result,
                })
                retrieval_query = missing_info or question_text

        if final_answer is None:
            print("Answer not found, entering fallback...")
            final_desc_text = controller.summarize_patches_in_chunks(
                descriptions_dict, accumulated_patch_names,
                question_text=question_text,
                chunk_size=10, threshold=50,
                magnification=zoom_level_val,
            )
            final_answer = controller.slide_llm_answer(
                final_desc_text,
                question_text,
                question_choices,
                magnification=zoom_level_val,
                case_name=unique_id,
            )

            all_results.append({
                "attempt": "final_fallback",
                "mode": "fallback_final_attempt",
                "total_accumulated_patches": len(accumulated_patch_names),
                "answer": final_answer["answer"],
                "explanation": final_answer["explanation"]
            })
        print('single_case_result:', final_answer["answer"])
        print('ground_truth:', correct_answer)

        case_result = {
            "long_id": target_long_id,
            "question": question_text,
            "choices": question_choices,
            "ground_truth": correct_answer,
            "pred_answer": final_answer["answer"],
            "explanation": final_answer["explanation"],
            "acquisition_config": {
                "plip_top_k_M": args.plip_top_k,
                "patches_per_round_B": args.patches_per_round,
                "max_rounds_T": args.max_rounds,
                "acquisition_rule": "top_b_by_eig",
                "perceptor_model": perceptor.model_name,
                "evidence_controller_model": controller.model_name,
            },
            "total_patches_selected": len(accumulated_patch_names),
            "process": all_results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)

        return case_result

    num_completed = 0
    for idx, pair in enumerate(pairs):
        case_result = run_single_case(idx, pair)
        if case_result is None:
            continue

        num_completed += 1

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Completed {num_completed}/{len(pairs)} cases this run.")


if __name__ == "__main__":
    main()
