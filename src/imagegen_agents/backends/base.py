from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class GenerationRequest:
    prompt: str
    width: int
    height: int
    negative_prompt: str = ""
    seed: int | None = None
    references: list[Path] = field(default_factory=list)


class ImageBackend(Protocol):
    name: str
    model: str
    max_references: int

    def generate(self, request: GenerationRequest, out_path: Path) -> Path:
        """Render one image to `out_path` (PNG) and return it."""
        ...
