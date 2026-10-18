"""Settings loaded from the environment (and .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _opt_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def _opt_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    ollama_basic_auth: str
    prompt_model: str
    eval_model: str

    image_backend: str
    studio_url: str
    studio_api_key: str
    studio_steps: int | None
    studio_guidance: float | None
    diffusers_model: str

    max_iterations: int
    pass_score: float
    max_side: int
    output_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        default_model = os.getenv("OLLAMA_MODEL", "gemma4:12b")
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_basic_auth=os.getenv("OLLAMA_BASIC_AUTH", ""),
            prompt_model=os.getenv("PROMPT_MODEL", default_model),
            eval_model=os.getenv("EVAL_MODEL", default_model),
            image_backend=os.getenv("IMAGE_BACKEND", "studio").lower(),
            studio_url=os.getenv("STUDIO_URL", "http://127.0.0.1:8888").rstrip("/"),
            studio_api_key=os.getenv("STUDIO_API_KEY", ""),
            studio_steps=_opt_int("STUDIO_STEPS"),
            studio_guidance=_opt_float("STUDIO_GUIDANCE"),
            diffusers_model=os.getenv("DIFFUSERS_MODEL", "black-forest-labs/FLUX.2-klein-4B"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
            pass_score=float(os.getenv("PASS_SCORE", "8")),
            max_side=int(os.getenv("MAX_SIDE", "1024")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
        )
