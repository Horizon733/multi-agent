from ..config import Settings
from .base import GenerationRequest, ImageBackend


def make_backend(settings: Settings) -> ImageBackend:
    if settings.image_backend == "studio":
        from .studio import StudioBackend

        return StudioBackend(settings)
    if settings.image_backend == "diffusers":
        from .flux_diffusers import FluxKleinBackend

        return FluxKleinBackend(settings)
    raise ValueError(f"Unknown IMAGE_BACKEND {settings.image_backend!r}; use 'studio' or 'diffusers'.")


__all__ = ["GenerationRequest", "ImageBackend", "make_backend"]
