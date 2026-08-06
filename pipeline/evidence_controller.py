"""Stage 4: Evidence Control"""

import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Optional

from utils.json_utils import parse_json_loose


class BaseEvidenceController:

    def _generate(self, messages: List[dict], max_new_tokens: int, stage: str) -> str:
        raise NotImplementedError

    def _call_json(self, messages, max_new_tokens, stage, retries=1):
        for attempt in range(retries + 1):
            text = self._generate(messages, max_new_tokens, stage)
            parsed = parse_json_loose(text)
            if parsed is not None:
                return text, parsed
            if attempt < retries:
                messages = deepcopy(messages)
                messages.append({
                    "role": "system",
                    "content": "Reminder: Respond with only a single valid JSON object containing the requested keys."
                })
                continue
            return text, None
        return None, None

    def evaluate_with_llm_chain(
        self,
        description: str,
        question: str,
        choices: Optional[List[str]] = None,
        max_new_tokens_a: int = 512,
        max_new_tokens_b: int = 256,
        max_new_tokens_c: int = 256,
        retries: int = 1,
    ) -> dict:
        system_a = (
            "You are an expert AI pathology assistant.\n"
            "Task: Based on the patch descriptions, try to answer the question step-by-step.\n"
            "Output ONLY a JSON object:\n"
            "{\n"
            '  "answer": "the final predicted answer (string)",\n'
            '  "thinking_steps": "your detailed reasoning, step-by-step (string)"\n'
            "}"
        )

        choices_text = f"\nChoices: {choices}" if choices else ""
        user_a = f"--- Patch Descriptions ---\n{description}\n--- End ---\nQuestion: {question}{choices_text}\nNow output the JSON with 'answer' and 'thinking_steps'."

        messages_a = [
            {"role": "system", "content": system_a},
            {"role": "user", "content": user_a},
        ]
        full_a, parsed_a = self._call_json(messages_a, max_new_tokens_a, "controller_step_a", retries=retries)

        if parsed_a is None:
            parsed_a = {
                "answer": "Uncertain",
                "thinking_steps": full_a or ""
            }

        parsed_a.setdefault("answer", "Uncertain")
        parsed_a.setdefault("thinking_steps", "")

        system_b = (
            "You are an expert AI pathology assistant.\n"
            "Task: Judge whether the current patch descriptions are sufficient to confidently support the answer.\n"
            "Output ONLY a JSON object like:\n"
            '{"sufficient": "Yes" or "No" }'
        )
        user_b = (
            f"Descriptions:\n{description}\n\n"
            f"Question: {question}{choices_text}\n\n"
            f"Previous answer and reasoning:\n{json.dumps(parsed_a, ensure_ascii=False)}\n\n"
            "Return JSON only."
        )
        messages_b = [{"role": "system", "content": system_b}, {"role": "user", "content": user_b}]
        full_b, parsed_b = self._call_json(messages_b, max_new_tokens_b, "controller_step_b", retries=retries)

        if parsed_b is None:
            parsed_b = {"sufficient": "Uncertain"}

        suff = str(parsed_b.get("sufficient", "")).strip().lower()

        if suff == "yes":
            return {
                "answer": parsed_a.get("answer"),
                "thinking_steps": parsed_a.get("thinking_steps"),
                "sufficient": parsed_b.get("sufficient", ""),
            }

        system_c = (
            "You are an expert AI pathology assistant.\n"
            "Task: If current data is insufficient, specify what visual evidence is missing, "
            "and whether zooming in could help obtain that evidence.\n"
            "Output ONLY a JSON object like:\n"
            "{\n"
            '  "missing_info": "short noun phrase",\n'
            '  "zoom_recommendation": "Yes" or "No",\n'
            '  "recommended_zoom_level": "None" or an integer like 10 or 20 or 40,\n'
            '  "zoom_reason": "brief reason why zooming helps"\n'
            "}"
        )

        user_c = (
            f"Descriptions:\n{description}\n\n"
            f"Question: {question}{choices_text}\n\n"
            f"Previous answer: {json.dumps(parsed_a, ensure_ascii=False)}\n"
            f"Sufficiency judgement: {json.dumps(parsed_b, ensure_ascii=False)}\n\n"
            "Now provide the JSON for missing info and zoom recommendation."
        )
        messages_c = [{"role": "system", "content": system_c}, {"role": "user", "content": user_c}]
        full_c, parsed_c = self._call_json(messages_c, max_new_tokens_c, "controller_step_c", retries=retries)

        if parsed_c is None:
            parsed_c = {
                "missing_info": "Uncertain",
                "zoom_recommendation": "Uncertain",
                "recommended_zoom_level": "Uncertain",
                "zoom_reason": full_c or ""
            }

        return {
            "answer": parsed_a.get("answer"),
            "thinking_steps": parsed_a.get("thinking_steps"),
            "sufficient": parsed_b.get("sufficient"),
            "missing_info": parsed_c.get("missing_info"),
            "zoom_recommendation": parsed_c.get("zoom_recommendation"),
            "zoom_level": parsed_c.get("recommended_zoom_level", 5),
            "zoom_reason": parsed_c.get("zoom_reason"),
        }

    def slide_llm_answer(
        self,
        descriptions_text: str,
        question: str,
        choices: Optional[List[str]] = None,
        magnification: Optional[int] = None,
        case_name: Optional[str] = None,
    ) -> Dict[str, str]:
        system_prompt = (
            "You are an expert slide-level pathology assistant. "
            "You will be given a question and detailed patch-level descriptions of a pathology slide. "
            "Your task is to infer the specific **slide-level diagnostic result or quantitative value** "
            "based on the provided evidence — not to define or explain the medical term itself. "
            "The answer should directly reflect the information observable in the slide, such as biomarker expression level, "
            "presence or absence of features, or a numeric measurement.\n\n"
            "Rules:\n"
            "1. When the question asks about a biomarker (e.g., HER2, progesterone receptor, Ki-67), "
            "output the **observed status or score** (e.g., 'positive', 'negative', '2+', 'high expression'), not the definition.\n"
            "2. When the question asks about survival time or other quantitative results, "
            "output only the numerical value for the answer key in the response JSON.\n"
            "3. Always keep the 'answer' short — short phrase, or one number.\n"
            "4. Always include a brief reasoning in 'explanation', summarizing how the evidence supports the answer.\n"
            "5. **If choices are provided, your 'answer' must be exactly one of the given options. "
            "Never generate an answer outside the provided choices.**\n\n"
            "Respond strictly in JSON format with keys: 'answer' and 'explanation'."
        )

        if magnification is not None:
            system_prompt = f"[Slide-level Context | Magnification={magnification}x]\n" + system_prompt

        choices_text = f"\nChoices: {choices}" if choices else ""

        user_prompt = (
            f"Question: {question}{choices_text}\n\n"
            "Now, based on the following patch-level descriptions of the slide, "
            "determine the **slide-level result** that directly answers the question.\n\n"
            f"--- Patch Descriptions ---\n{descriptions_text}\n--- End of Descriptions ---\n\n"
            "Answer in JSON format:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_answer = self._generate(messages, 1024, "answer_synthesis")

        parsed = parse_json_loose(raw_answer)
        if parsed is None:
            json_candidates = re.findall(r"\{.*?\}", raw_answer or "", flags=re.DOTALL)
            for candidate in reversed(json_candidates):
                try:
                    parsed = json.loads(candidate)
                    break
                except Exception:
                    continue

        if parsed is None:
            print(f"JSON parsing failed ({case_name if case_name else 'unknown_case'}), raw output:\n{raw_answer}\n")
            answer = {
                "answer": raw_answer.split()[0] if raw_answer else "Unknown",
                "explanation": "Model did not return valid JSON."
            }
        else:
            answer_text = str(parsed.get("answer", "")).strip()
            if not answer_text:
                answer_text = "Unknown"

            explanation_text = str(parsed.get("explanation", "")).strip()
            if not explanation_text:
                explanation_text = "No explanation provided."

            answer = {
                "answer": answer_text,
                "explanation": explanation_text
            }

        return answer

    def summarize_patches_in_chunks(
        self,
        descriptions_dict: Dict[str, str],
        patch_names: List[str],
        question_text: Optional[str] = None,
        chunk_size: int = 5,
        threshold: int = 50,
        magnification: Optional[int] = None,
    ) -> str:
        from data_utils.utils import extract_coords_from_name, build_descriptions_with_meta

        if len(patch_names) <= threshold:
            items = [(name, descriptions_dict[name]) for name in patch_names]
            return build_descriptions_with_meta(
                items, mag_level=magnification, include_header=True, include_coords=True
            )

        print(f"Patch count {len(patch_names)} exceeds {threshold}, performing chunked summarization...")
        summaries = []
        for i in range(0, len(patch_names), chunk_size):
            chunk_names = patch_names[i:i + chunk_size]
            items = [(name, descriptions_dict[name]) for name in chunk_names]

            coords_list = []
            for name, _ in items:
                x, y = extract_coords_from_name(name)
                coords_list.append(f"({x},{y})" if x is not None and y is not None else "(unknown)")
            coords_str = ", ".join(coords_list)

            chunk_text = build_descriptions_with_meta(
                items, mag_level=magnification, include_header=False, include_coords=True
            )

            system_prompt = (
                "You are an expert pathology assistant. "
                "You will be given multiple patch-level descriptions of histopathology images. "
                "Your task is to summarize the key pathological features across these patches. "
                "The summary must be concise yet informative, highlighting significant morphological patterns. "
                "Additionally, focus on information that could help answer the following question "
                "about the slide, emphasizing details relevant to the diagnostic or interpretive context."
            )

            if magnification is not None:
                system_prompt = f"[Patch-level Summarization | Magnification={magnification}x]\n" + system_prompt

            user_prompt = (
                f"--- Patch Descriptions (with coordinates) ---\n{chunk_text}\n"
                "--- End of Descriptions ---\n\n"
            )

            if question_text:
                user_prompt += f"Related Question: {question_text}\n\n"

            user_prompt += (
                "Summarize the main pathological findings across these patches. "
                "Your summary should:\n"
                "- Capture key morphological features and cellular details.\n"
                "- If a question is provided, emphasize features that are relevant to answering it.\n"
                "- Do not provide a final answer or diagnosis.\n"
                "Output only the summary text, no JSON or extra formatting."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            summary_text = self._generate(messages, 512, "evidence_summarization")

            summaries.append(
                f"[Chunk {i // chunk_size + 1} | Patches={coords_str}]\n{summary_text}"
            )

            print(f"Completed summary for chunk {i // chunk_size + 1} (patch count: {len(chunk_names)})")

        combined_summary = (
            f"[Current Magnification: {magnification}x]\n\n" +
            "\n\n".join(summaries)
        )
        return combined_summary


class QwenEvidenceController(BaseEvidenceController):

    def __init__(self, ckpt_path, device_map="auto", torch_dtype="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch_dtype, device_map=device_map
        )
        self.model.eval()

        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.num_beams = 1
            generation_config.num_beam_groups = 1

        self.model_name = f"qwen:{os.path.basename(os.path.normpath(ckpt_path))}"
        self._torch = torch

    def _generate(self, messages, max_new_tokens, stage):
        torch = self._torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        with torch.no_grad():
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        out_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        del model_inputs, generated_ids, output_ids
        torch.cuda.empty_cache()
        return out_text


class OpenRouterEvidenceController(BaseEvidenceController):
    """Evidence controller backed by any OpenRouter-hosted chat model (e.g. GPT-5)."""

    def __init__(self, client, model, reasoning_effort: Optional[str] = "high"):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.model_name = f"openrouter:{model}"

    def _generate(self, messages, max_new_tokens, stage):
        # max_new_tokens is unused; the OpenRouter client applies its own cap.
        return self.client.chat(
            messages, model=self.model, reasoning_effort=self.reasoning_effort,
        )


def build_evidence_controller(
    backend: str,
    qwen_ckpt: Optional[str] = None,
    openrouter_client=None,
    openrouter_model: Optional[str] = None,
    openrouter_reasoning_effort: Optional[str] = "high",
) -> BaseEvidenceController:
    if backend == "qwen":
        if not qwen_ckpt:
            raise ValueError("qwen_ckpt is required for the 'qwen' evidence controller backend.")
        return QwenEvidenceController(qwen_ckpt)
    if backend == "gpt5":
        if openrouter_client is None:
            raise ValueError("An OpenRouterClient is required for the 'gpt5' evidence controller backend.")
        return OpenRouterEvidenceController(openrouter_client, openrouter_model, reasoning_effort=openrouter_reasoning_effort)
    raise ValueError(f"Unknown evidence controller backend: {backend!r} (expected 'qwen' or 'gpt5')")
