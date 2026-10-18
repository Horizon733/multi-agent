"""Illustration category profiles. Each profile biases the prompt writer and the
evaluator toward one visual domain, plus sensible defaults for aspect ratio and
negative prompts that the user does not want to type every time.

Keep the text concrete — these are prepended verbatim to the system prompt and
drive what the models actually attend to."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    description: str
    writer_guidance: str
    evaluator_guidance: str
    default_negative: str
    default_aspect: str
    # True when text in the image is undesirable (user adds labels in post).
    text_free: bool = False


CATEGORIES: dict[str, Category] = {
    "paper": Category(
        key="paper",
        label="Research paper illustration",
        description="Clean, publication-ready figures and diagrams for academic papers.",
        writer_guidance=(
            "Produce a figure that would fit a Nature/IEEE paper. Clean, uncluttered, "
            "high contrast, white or very pale background. Prefer vector-style flat shading, "
            "line art or isometric. Describe the exact concept, components and their spatial "
            "relationship. If labels are needed, say so, but keep them to short common words. "
            "Avoid photorealism, bokeh, dramatic lighting and decorative elements."
        ),
        evaluator_guidance="Judge like a reviewer: clarity, correctness of the depicted concept, no visual noise.",
        default_negative="photorealistic, 3d render, blurry, decorative, cluttered background, watermark, signature",
        default_aspect="4:3",
        text_free=True,
    ),
    "poster": Category(
        key="poster",
        label="Festival / event poster",
        description="Vibrant, culturally rich posters for festivals, events and celebrations.",
        writer_guidance=(
            "Design an eye-catching festival or event poster. Rich color palette appropriate "
            "to the occasion (Diwali: warm golds/reds/oranges with diyas; Holi: splashes of "
            "pigment; Christmas: cool reds/greens; Eid: deep blues/golds, crescent motifs; "
            "generic: bold complementary palette). Strong central subject, decorative motifs "
            "around the edges, poster-style composition. If a short headline is essential, "
            "put ONE line of large text in double quotes — never more. Leave empty space at "
            "the top or bottom where the user will overlay their own date/venue text."
        ),
        evaluator_guidance="Judge visual impact, cultural accuracy, composition and whether the empty text area is clean.",
        default_negative="garbled text, extra text, watermark, low contrast, muddy colors",
        default_aspect="3:4",
    ),
    "textbook": Category(
        key="textbook",
        label="Textbook illustration",
        description="Clear educational illustrations for school and college textbooks.",
        writer_guidance=(
            "Create an educational illustration suitable for a textbook. Clear, friendly, "
            "slightly stylized (not photorealistic). Light, neutral background. Every element "
            "readable at thumbnail size. If a process has steps, show them left-to-right or "
            "top-to-bottom with arrows. Characters should look age-appropriate and inclusive "
            "when humans are depicted. Avoid any in-image text — labels will be added later "
            "in InDesign; just leave clean empty space next to the thing to be labeled."
        ),
        evaluator_guidance="Judge pedagogical clarity: can a student understand what is being shown without a caption?",
        default_negative="in-image text, labels, letters, numbers, watermark, dark background, moody lighting",
        default_aspect="4:3",
        text_free=True,
    ),
    "medical": Category(
        key="medical",
        label="Medical / anatomical diagram",
        description="Anatomically accurate medical and biological diagrams.",
        writer_guidance=(
            "Produce an anatomically accurate medical diagram in the style of Netter, Gray's "
            "Anatomy or a modern medical textbook. Cross-section or cutaway view when it helps. "
            "Realistic proportions and relative positions. Muted anatomical palette (pinks, reds, "
            "muted blues, cream background). Clean linework. Leave empty callout space around "
            "key structures for later labeling — do NOT render any text in the image."
        ),
        evaluator_guidance=(
            "Judge anatomical accuracy strictly. Call out wrong structures, missing organs, "
            "incorrect proportions or anatomical impossibilities. A pretty picture that is "
            "medically wrong is unusable."
        ),
        default_negative="in-image text, labels, incorrect anatomy, extra organs, cartoonish, horror elements",
        default_aspect="1:1",
        text_free=True,
    ),
    "science": Category(
        key="science",
        label="Science diagram",
        description="Chemistry, physics, biology concept diagrams (molecules, forces, cells).",
        writer_guidance=(
            "Produce a scientific concept diagram. Molecules in ball-and-stick or space-filling "
            "notation. Physics with clean vector arrows for forces/vectors. Cells with labeled-"
            "ready organelles. White or very pale grid background. Precise, textbook-grade "
            "accuracy — if the atom count, charge, or formula is wrong, the image is wrong. "
            "No in-image text; leave clean space next to each element so labels can be added."
        ),
        evaluator_guidance=(
            "Judge scientific accuracy: correct atoms/bonds/formulas, correct arrows, "
            "correct cellular structures. Flag any physical/chemical impossibility."
        ),
        default_negative="in-image text, labels, formulas as text, incorrect chemistry, cluttered background",
        default_aspect="4:3",
        text_free=True,
    ),
    "tech": Category(
        key="tech",
        label="Tech illustration",
        description="Modern illustrations for tech blogs, documentation and architecture diagrams.",
        writer_guidance=(
            "Create a modern tech illustration in flat, isometric or line-art style. Clean "
            "geometry, limited palette (2-4 colors plus neutrals), lots of whitespace. "
            "Devices, servers, cloud, data flows depicted abstractly. The style of Stripe, "
            "Vercel or Linear marketing art. Leave labels out — the user composites them later."
        ),
        evaluator_guidance="Judge composition, modern aesthetic and whether the concept is instantly readable.",
        default_negative="photorealistic, old-fashioned, cluttered, garbled text, 3d render with shadows",
        default_aspect="16:9",
        text_free=True,
    ),
    "general": Category(
        key="general",
        label="General image (no category hints)",
        description="Fallback — writer behaves as before, no category bias.",
        writer_guidance="",
        evaluator_guidance="",
        default_negative="",
        default_aspect="1:1",
    ),
}


def get(key: str | None) -> Category:
    return CATEGORIES.get(key or "general", CATEGORIES["general"])
