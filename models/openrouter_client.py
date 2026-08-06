import base64
import io
import os
import time
from typing import List, Optional

_env_max_output_tokens = os.environ.get("OPENROUTER_MAX_OUTPUT_TOKENS", "32768")
DEFAULT_MAX_OUTPUT_TOKENS = int(_env_max_output_tokens) or None


def image_to_data_url(image, fmt: str = "JPEG", quality: int = 75) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _to_responses_input(messages: List[dict]) -> List[dict]:
    input_items = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            input_items.append({"role": role, "type": "message", "content": content})
            continue

        parts = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "input_text", "text": part["text"]})
            elif part_type == "image_url":
                image_url = part["image_url"]
                url = image_url["url"] if isinstance(image_url, dict) else image_url
                parts.append({"type": "input_image", "image_url": url})
            else:
                parts.append(part)
        input_items.append({"role": role, "type": "message", "content": parts})
    return input_items


def _extract_output_text(response) -> str:
    text_parts = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content_item in getattr(item, "content", None) or []:
            text = getattr(content_item, "text", None)
            if text:
                text_parts.append(text)
    return "".join(text_parts).strip()


class OpenRouterClient:
    """Wraps the `openai` SDK pointed at OpenRouter for any hosted reasoning model (e.g. GPT-5)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenRouter API key provided. Pass --openrouter_api_key or set "
                "the OPENROUTER_API_KEY environment variable."
            )

        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL")
        if not base_url:
            raise ValueError(
                "No OpenRouter-compatible API base URL provided. Pass --openrouter_base_url or set "
                "the OPENROUTER_BASE_URL environment variable."
            )

        from openai import OpenAI

        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )

    def chat(
        self,
        messages: List[dict],
        model: str,
        retries: int = 2,
        timeout: int = 600,
        reasoning_effort: Optional[str] = "high",
        max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> str:
        last_err = None
        for attempt in range(retries + 1):
            try:
                request_params = {
                    "input": _to_responses_input(messages),
                    "model": model,
                    "timeout": timeout,
                }
                if reasoning_effort:
                    request_params["reasoning"] = {"effort": reasoning_effort}
                if max_output_tokens:
                    request_params["max_output_tokens"] = max_output_tokens

                response = self.client.responses.create(**request_params)

                text = _extract_output_text(response)
                status = getattr(response, "status", None)
                if not text:
                    incomplete_reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                    print(
                        f"[OpenRouter] WARNING: empty completion from model={model} "
                        f"(status={status!r}, incomplete_reason={incomplete_reason!r})."
                    )

                return text
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
        raise RuntimeError(
            f"OpenRouter Responses API call failed (model={model}) after "
            f"{retries + 1} attempts: {last_err}"
        )
