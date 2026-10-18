"""Image generation through a running Unsloth Studio (any model loaded in its Image tab, e.g. Qwen-Image-2.1).

Studio handles the low-VRAM work (GGUF, int8 text encoder, layer streaming, VAE tiling), so this backend
is just HTTP. Uses /api/inference/images/generate rather than the OpenAI /v1/images/generations shape
because only the former accepts reference images.
"""

import base64
import mimetypes
from pathlib import Path

import httpx

from ..config import Settings
from .base import GenerationRequest

# (repo-id substring, max reference images incl. the first). Mirrors Studio's per-family limits.
_REFERENCE_LIMITS = (("qwen-image-2.1", 10), ("qwenimage21", 10), ("flux.2", 4))


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class StudioBackend:
    name = "studio"

    def __init__(self, settings: Settings):
        if not settings.studio_api_key:
            raise RuntimeError("STUDIO_API_KEY is empty. Create one in Unsloth Studio: avatar -> Settings -> API.")
        self.settings = settings
        # A Qwen-Image-2.1 render on 8 GB streams layers from RAM and takes minutes.
        self.http = httpx.Client(
            base_url=settings.studio_url,
            headers={"Authorization": f"Bearer {settings.studio_api_key}"},
            timeout=httpx.Timeout(900, connect=10),
        )
        status = self._status()
        self.model = status.get("display_repo_id") or status.get("repo_id") or "unknown"
        lowered = self.model.lower()
        self.max_references = next((n for key, n in _REFERENCE_LIMITS if key in lowered), 0)

    def _status(self) -> dict:
        try:
            resp = self.http.get("/api/inference/images/status")
        except httpx.ConnectError as exc:
            raise RuntimeError(f"Unsloth Studio is not reachable at {self.settings.studio_url}. Start it first.") from exc
        resp.raise_for_status()
        status = resp.json()
        if not status.get("loaded"):
            raise RuntimeError("No image model is loaded in Unsloth Studio. Load one in its Image tab first.")
        return status

    def generate(self, request: GenerationRequest, out_path: Path) -> Path:
        body: dict = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
        }
        if request.negative_prompt:
            body["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            body["seed"] = request.seed
        # Leaving steps/guidance unset would fall back to the schema default (9 steps), not the model preset.
        steps, guidance = self._preset()
        body["steps"] = self.settings.studio_steps or steps
        body["guidance"] = self.settings.studio_guidance if self.settings.studio_guidance is not None else guidance

        refs = request.references[: self.max_references]
        if refs:
            # Studio takes the first image as init_image and the rest as reference_images.
            body["workflow"] = "reference"
            body["init_image"] = _data_url(refs[0])
            body["reference_images"] = [_data_url(p) for p in refs[1:]]

        resp = self.http.post("/api/inference/images/generate", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"Studio generate failed ({resp.status_code}): {resp.text[:500]}")
        record = resp.json()["images"][0]

        png = self.http.get(record["url"])
        png.raise_for_status()
        out_path.write_bytes(png.content)
        return out_path

    def _preset(self) -> tuple[int, float]:
        """Steps/guidance Studio's UI uses for the loaded model (its ComfyUI-derived defaults)."""
        m = self.model.lower()
        if "qwen" in m and "turbo" in m:
            return 8, 1.0
        if "qwen-image-2.1" in m or "qwenimage21" in m:
            return 25, 1.0
        if "flux.2-klein-base" in m:
            return 20, 5.0
        if "flux.2-klein" in m:
            return 4, 1.0
        return 20, 4.0
