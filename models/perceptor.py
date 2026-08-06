import os
import re
from typing import List, Optional, Tuple


def _parse_batch_descriptions(text: str, n: int) -> Optional[List[str]]:
    headers = list(re.finditer(r"===\s*PATCH\s*(\d+)\s*===", text, flags=re.IGNORECASE))
    if len(headers) != n:
        return None
    indices = [int(m.group(1)) for m in headers]
    if indices != list(range(1, n + 1)):
        return None

    descriptions = []
    for i, header in enumerate(headers):
        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        descriptions.append(text[start:end].strip())
    return descriptions


class BasePerceptor:

    def _build_prompt(
        self, question, coords, magnification, choices, missing_info
    ) -> Tuple[str, str]:
        meta_lines = []
        if coords is not None:
            meta_lines.append(f"Patch coordinates: ({coords[0]},{coords[1]})")
        if magnification is not None:
            meta_lines.append(f"Magnification: {magnification}x")
        meta_text = ""
        if meta_lines:
            meta_text = "[IMAGE META] " + " | ".join(meta_lines) + "\n\n"

        if question is None:
            prompt_body = "Please describe the pathology features in this image."
        else:
            prompt_body = (
                f"Question: {question}\n\n"
                "Answer the question and list the pathological features visible in the image that support your answer."
            )

        if choices is not None:
            prompt_body += f"\nChoices: {choices}"
        if missing_info is not None:
            prompt_body += f"\nMissing information: {missing_info}"

        full_text = meta_text + prompt_body

        system_prompt = (
            "A conversation between a curious user and an AI medical assistant specialized in pathology image analysis. "
            "The assistant can interpret pathology images, describe observed features, and provide possible explanations based on medical knowledge, "
            "but will never give a definitive diagnosis or prescribe treatment. "
            "The assistant must always maintain a polite, clear, and professional tone. "
            "All answers should be supported by established, reliable medical sources. "
            "The assistant should carefully consider visual details in pathology images, such as cell morphology, staining patterns, and tissue architecture. "
            "If choices are given, the answer must be given from the choices."
            "If Missing information is given, you need to focus on this part of the image."
        )
        return system_prompt, full_text

    def describe(
        self,
        image,
        question: Optional[str] = None,
        coords: Optional[Tuple[int, int]] = None,
        magnification: Optional[int] = None,
        max_new_tokens: int = 1024,
        choices: Optional[List[str]] = None,
        missing_info: Optional[str] = None,
    ) -> str:
        system_prompt, user_text = self._build_prompt(question, coords, magnification, choices, missing_info)
        return self._generate(image, system_prompt, user_text, max_new_tokens)

    def _generate(self, image, system_prompt, user_text, max_new_tokens):
        raise NotImplementedError

    def describe_batch(
        self,
        images: List,
        question: Optional[str] = None,
        coords_list: Optional[List[Tuple[int, int]]] = None,
        magnification: Optional[int] = None,
        max_new_tokens: int = 1024,
        choices: Optional[List[str]] = None,
        missing_info: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError(f"{type(self).__name__} does not support describe_batch()")


class PathoR1Perceptor(BasePerceptor):

    def __init__(self, ckpt_path, device_map="auto", torch_dtype="auto"):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, torch_dtype=torch_dtype, device_map=device_map
        )
        self.model.eval()

        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.num_beams = 1
            generation_config.num_beam_groups = 1

        self.processor = AutoProcessor.from_pretrained(ckpt_path)
        self.model_name = f"patho_r1:{os.path.basename(os.path.normpath(ckpt_path))}"
        self._torch = torch

    def _generate(self, image, system_prompt, user_text, max_new_tokens):
        from qwen_vl_utils import process_vision_info
        torch = self._torch

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text


def _import_llava_model_module():
    import sys
    import types
    if "llava.model" in sys.modules:
        return

    if "llava.model.language_model.llava_mpt" not in sys.modules:
        _mpt_stub = types.ModuleType("llava.model.language_model.llava_mpt")
        _mpt_stub.LlavaMPTForCausalLM = None
        _mpt_stub.LlavaMPTConfig = None
        sys.modules["llava.model.language_model.llava_mpt"] = _mpt_stub

    import transformers
    _orig_register = transformers.AutoConfig.register
    transformers.AutoConfig.register = lambda model_type, config, exist_ok=False: _orig_register(
        model_type, config, exist_ok=True
    )
    try:
        import importlib
        importlib.import_module("llava.model")
    finally:
        transformers.AutoConfig.register = _orig_register


class PathGenLlavaPerceptor(BasePerceptor):

    def __init__(self, ckpt_path, device_map="auto", load_8bit=False, load_4bit=False, device="cuda"):
        _import_llava_model_module()
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(ckpt_path)
        self.tokenizer, self.model, self.image_processor, _context_len = load_pretrained_model(
            ckpt_path, None, model_name, load_8bit=load_8bit, load_4bit=load_4bit,
            device_map=device_map, device=device,
        )
        self.model.eval()
        self.model_name = f"pathgen_llava:{os.path.basename(os.path.normpath(ckpt_path))}"

    def _generate(self, image, system_prompt, user_text, max_new_tokens):
        import torch
        from llava.constants import (
            DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import Conversation, SeparatorStyle
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_text
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + user_text

        conv = Conversation(
            system=system_prompt,
            roles=("USER", "ASSISTANT"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        stopping_criteria = KeywordsStoppingCriteria([conv.sep2], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        raw_input_len = int(input_ids.shape[1])
        output_text = self.tokenizer.decode(output_ids[0, raw_input_len:], skip_special_tokens=True).strip()
        if output_text.endswith(conv.sep2):
            output_text = output_text[: -len(conv.sep2)].strip()

        return output_text


class OpenRouterPerceptor(BasePerceptor):
    """Perceptor backed by any OpenRouter-hosted vision model (e.g. GPT-5)."""

    def __init__(self, client, model, reasoning_effort: Optional[str] = "high"):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.model_name = f"openrouter:{model}"

    def _generate(self, image, system_prompt, user_text, max_new_tokens):
        # max_new_tokens is unused; the OpenRouter client applies its own cap.
        from models.openrouter_client import image_to_data_url

        data_url = image_to_data_url(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        text = self.client.chat(
            messages, model=self.model, reasoning_effort=self.reasoning_effort,
        )
        return text

    def describe_batch(
        self,
        images: List,
        question: Optional[str] = None,
        coords_list: Optional[List[Tuple[int, int]]] = None,
        magnification: Optional[int] = None,
        max_new_tokens: int = 1024,
        choices: Optional[List[str]] = None,
        missing_info: Optional[str] = None,
    ) -> List[str]:
        from models.openrouter_client import image_to_data_url

        n = len(images)
        system_prompt, shared_body = self._build_prompt(question, None, magnification, choices, missing_info)

        strict_note = ""
        for attempt in range(2):
            batch_instructions = (
                f"\n\nYou will be shown {n} separate pathology image patches from the same slide, "
                f"labeled Patch 1 through Patch {n} in the order presented below. Analyze each patch "
                "independently on its own visual merits -- do not assume patches are related or repeat "
                "content across sections.\n"
                f"Respond with exactly {n} sections, one per patch, using precisely this format "
                "(including the '===' markers, one line by themselves):\n"
                "=== PATCH 1 ===\n<description and answer for patch 1>\n"
                "=== PATCH 2 ===\n<description and answer for patch 2>\n"
                f"... continuing through '=== PATCH {n} ==='. "
                "Do not omit any patch, merge patches together, or add commentary outside these sections."
                + strict_note
            )
            user_text = shared_body + batch_instructions

            content = [{"type": "text", "text": user_text}]
            for i, image in enumerate(images):
                coord_label = ""
                if coords_list is not None and coords_list[i] is not None:
                    x, y = coords_list[i]
                    coord_label = f" (coordinates: {x},{y})"
                content.append({"type": "text", "text": f"--- Patch {i + 1}{coord_label} ---"})
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            text = self.client.chat(
                messages, model=self.model, reasoning_effort=self.reasoning_effort,
            )

            descriptions = _parse_batch_descriptions(text, n)
            if descriptions is not None:
                return descriptions

            strict_note = (
                f"\n\nSTRICT: your previous response did not contain exactly {n} correctly numbered "
                "'=== PATCH k ===' sections. Follow the format exactly this time."
            )

        raise RuntimeError(
            f"OpenRouterPerceptor.describe_batch: could not parse {n} patch sections from the model "
            f"response after 2 attempts. Last raw response:\n{text}"
        )


def build_perceptor(
    backend: str,
    patho_r1_ckpt: Optional[str] = None,
    pathgen_llava_ckpt: Optional[str] = None,
    openrouter_client=None,
    openrouter_model: Optional[str] = None,
    openrouter_reasoning_effort: Optional[str] = "high",
) -> BasePerceptor:
    if backend == "patho_r1":
        if not patho_r1_ckpt:
            raise ValueError("patho_r1_ckpt is required for the 'patho_r1' perceptor backend.")
        return PathoR1Perceptor(patho_r1_ckpt)
    if backend == "pathgen_llava":
        if not pathgen_llava_ckpt:
            raise ValueError("pathgen_llava_ckpt is required for the 'pathgen_llava' perceptor backend.")
        return PathGenLlavaPerceptor(pathgen_llava_ckpt)
    if backend == "gpt5":
        if openrouter_client is None:
            raise ValueError("An OpenRouterClient is required for the 'gpt5' perceptor backend.")
        return OpenRouterPerceptor(openrouter_client, openrouter_model, reasoning_effort=openrouter_reasoning_effort)
    raise ValueError(
        f"Unknown perceptor backend: {backend!r} (expected 'patho_r1', "
        "'pathgen_llava', or 'gpt5')"
    )
