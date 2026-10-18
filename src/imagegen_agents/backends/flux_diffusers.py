"""FLUX.2 [klein] 4B in-process via diffusers, 4-bit quantized so it fits an 8 GB GPU.

The pipeline is loaded once and kept for the whole run. The VLM agents run remotely, so nothing else
competes for VRAM.
"""

from pathlib import Path

from ..config import Settings
from .base import GenerationRequest


class FluxKleinBackend:
    name = "diffusers"
    max_references = 4

    def __init__(self, settings: Settings):
        try:
            import torch
            from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig
        except ImportError as exc:
            raise RuntimeError("Diffusers backend needs extras: `uv sync --extra diffusers`.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to torch; reinstall with `uv sync --extra diffusers`.")

        self.torch = torch
        self.model = settings.diffusers_model
        # Distilled klein runs 4 steps without CFG; the -base variants need full CFG sampling.
        self.steps, self.guidance = (50, 4.0) if "base" in self.model else (4, 1.0)

        quant = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder"],
        )
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.model, torch_dtype=torch.bfloat16, quantization_config=quant
        )
        # Keeps only the active component on the GPU; text encoder and transformer take turns.
        self.pipe.enable_model_cpu_offload()

    def generate(self, request: GenerationRequest, out_path: Path) -> Path:
        from PIL import Image

        refs = [Image.open(p).convert("RGB") for p in request.references[: self.max_references]]
        generator = None
        if request.seed is not None:
            generator = self.torch.Generator("cpu").manual_seed(request.seed)

        image = self.pipe(
            image=refs or None,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
        ).images[0]
        image.save(out_path)
        return out_path
