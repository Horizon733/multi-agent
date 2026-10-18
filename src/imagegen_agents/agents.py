"""The two LLM agents: prompt writer (agent 1) and visual evaluator (agent 3). Agent 2 is the image backend."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .categories import Category
from .llm import VisionLLM

AspectRatio = Literal["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"]


class PromptSpec(BaseModel):
    prompt: str = Field(description="The full prompt for the image model")
    negative_prompt: str = Field(default="", description="Things to avoid; empty if none")
    aspect_ratio: AspectRatio = "1:1"
    rationale: str = Field(description="One or two sentences on the choices made")


class Evaluation(BaseModel):
    text_audit: list[str] = Field(
        description="Every readable text region in the image, verbatim. One entry per region "
        "(headline, body text, logos, watermarks, badges, signs, UI, background text). "
        "Mark [GARBLED] after any that is misspelled, nonsense, or wasn't asked for."
    )
    score: float = Field(ge=0, le=10, description="Overall 0-10 match to the request and references")
    prompt_adherence: float = Field(ge=0, le=10)
    reference_fidelity: float | None = Field(default=None, ge=0, le=10, description="null when no references")
    technical_quality: float = Field(ge=0, le=10, description="Artifacts, anatomy, text rendering, composition")
    issues: list[str] = Field(description="Concrete problems, most important first")
    suggestions: list[str] = Field(description="Concrete changes to the prompt that would fix the issues")


PROMPT_WRITER_SYSTEM = """You are a prompt engineer for a text-to-image diffusion model ({model}).
Turn the user's request into one detailed, unambiguous image prompt.
- Describe subject, composition, framing, lighting, color palette, style/medium and mood in natural sentences.
- Put any text that must appear in the image in double quotes, exactly as it should be spelled.
- If reference images are attached, they are passed to the image model in the same order. Refer to them as
  "image 1", "image 2", ... and say precisely what to take from each (identity, outfit, style, layout...).
  Also describe the important visual details in words, since the model may follow text more than pixels.
- If previous attempts and critique are given, keep what worked and fix every listed issue.
- Choose the aspect ratio that best fits the subject.
Reply only with the requested JSON."""

EVALUATOR_SYSTEM = """You are a strict art director reviewing a generated image for production use.
Compare it with the user's request{refs}. The FIRST attached image is the generated one{ref_order}.

Run these checks before you score. Each one that fails is an issue.

1. TEXT AUDIT — scan the ENTIRE image, including corners, backgrounds, logos, watermarks, badges,
   signs and ribbons. Transcribe every readable text region into `text_audit`, each marked [GARBLED]
   if it is misspelled, nonsense, broken letterforms, or appears in the image but was never asked for
   (e.g. a hallucinated brand/watermark). ANY [GARBLED] entry caps the score at 5 and MUST appear in
   `issues` with a prompt-change suggestion ("add 'no watermark, no extra text' to the negative
   prompt", "remove unlabeled areas by filling with ...", etc.).
2. SUBJECT & COUNT — right number of people/objects, correct identity if a reference was given.
3. COMPOSITION — framing, layout, aspect, elements placed where the prompt asked.
4. ANATOMY & ARTIFACTS — hands, eyes, extra limbs, warped objects, glitched textures.
5. STYLE & LIGHTING — matches the requested style, medium and mood.

Scoring guide (be strict; the agent loop depends on honest scores):
- 10: shippable as-is, no fixes needed. Reserve for genuinely clean images.
- 7-9: minor issues, usable with small edits.
- 4-6: visible problems that need another pass.
- 0-3: broken (garbled text, wrong subject, severe artifacts).

Suggestions must be concrete PROMPT changes, not vague advice. Reply only with the requested JSON."""


class PromptWriter:
    def __init__(self, llm: VisionLLM, model: str, image_model: str, category: Category):
        self.llm, self.model, self.image_model, self.category = llm, model, image_model, category

    def run(self, request: str, references: list[Path], history: list[dict]) -> PromptSpec:
        parts = [f"User request:\n{request}"]
        if self.category.writer_guidance:
            parts.insert(0, f"Category: {self.category.label}\n{self.category.writer_guidance}")
        if self.category.default_negative:
            parts.append(f"Baseline negatives to include (add more if the critique calls for it): "
                         f"{self.category.default_negative}")
        if self.category.default_aspect != "1:1":
            parts.append(f"Preferred aspect ratio for this category: {self.category.default_aspect} "
                         f"(override if the subject clearly wants another).")
        if references:
            parts.append(f"{len(references)} reference image(s) are attached, in order.")
        for i, attempt in enumerate(history, 1):
            ev = attempt["evaluation"]
            garbled = [t for t in ev.get("text_audit", []) if "[GARBLED]" in t]
            parts.append(
                f"Attempt {i} prompt:\n{attempt['prompt']}\n"
                f"Attempt {i} negative prompt: {attempt['spec'].get('negative_prompt') or '(empty)'}\n"
                f"Score {ev['score']}/10. Issues: {'; '.join(ev['issues']) or 'none'}. "
                f"Suggestions: {'; '.join(ev['suggestions']) or 'none'}."
                + (f" Garbled/unwanted text found: {garbled}. Add precise negatives to remove it."
                   if garbled else "")
            )
        return self.llm.structured(
            self.model,
            PROMPT_WRITER_SYSTEM.format(model=self.image_model),
            "\n\n".join(parts),
            PromptSpec,
            images=references,
            temperature=0.7,
        )


class Evaluator:
    def __init__(self, llm: VisionLLM, model: str, category: Category):
        self.llm, self.model, self.category = llm, model, category

    def run(self, request: str, image: Path, references: list[Path], prompt: str) -> Evaluation:
        system = EVALUATOR_SYSTEM.format(
            refs=" and the reference images" if references else "",
            ref_order="; the rest are the user's references, in order" if references else "",
        )
        if self.category.evaluator_guidance:
            system += f"\n\nCategory-specific focus ({self.category.label}): {self.category.evaluator_guidance}"
        if self.category.text_free:
            system += ("\n\nNOTE: this category is TEXT-FREE. Any readable text in the image is a defect, "
                       "not just garbled text. Transcribe it in text_audit and cap the score at 5.")
        user = f"User request:\n{request}\n\nPrompt that was used:\n{prompt}"
        return self.llm.structured(self.model, system, user, Evaluation, images=[image, *references], temperature=0.1)
