"""Thin wrapper over a (possibly remote, ngrok-tunnelled) Ollama server with structured JSON output."""

import base64
from pathlib import Path
from typing import TypeVar

from ollama import Client
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class VisionLLM:
    def __init__(self, host: str, basic_auth: str = ""):
        # ngrok's free tier serves an HTML interstitial unless this header is present.
        headers = {"ngrok-skip-browser-warning": "true"}
        if basic_auth:
            headers["Authorization"] = "Basic " + base64.b64encode(basic_auth.encode()).decode()
        self.client = Client(host=host, headers=headers, timeout=600)

    def structured(
        self,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        images: list[Path] | None = None,
        temperature: float = 0.4,
    ) -> T:
        """Ask `model` for a reply that validates against `schema`; images are attached to the user turn."""
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        response = self.client.chat(
            model=model,
            messages=[{"role": "system", "content": system}, user_msg],
            format=schema.model_json_schema(),
            options={"temperature": temperature},
            # Thinking models (Qwen3.5, Qwen3-VL-Thinking) leave `content` empty while their reasoning
            # fills `thinking`, which breaks structured output. Ollama silently ignores `think` on
            # non-thinking models, so it is safe to always pass False.
            think=False,
        )
        content = response.message.content or ""
        if not content.strip():
            raise RuntimeError(
                f"{model} returned empty content (thinking={bool(getattr(response.message, 'thinking', None))}). "
                "If the model requires thinking, pin a non-thinking tag or switch EVAL_MODEL."
            )
        return schema.model_validate_json(content)

    def check(self, *models: str) -> None:
        """Fail fast if the server is unreachable or a model has not been pulled."""
        available = {m.model for m in self.client.list().models}
        missing = [m for m in models if m not in available and f"{m}:latest" not in available]
        if missing:
            raise RuntimeError(
                f"Ollama at the configured host is missing {missing}. Run `ollama pull <model>` on the MacBook. "
                f"Available: {sorted(available)}"
            )
