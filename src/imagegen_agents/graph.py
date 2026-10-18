"""LangGraph wiring: write_prompt -> generate -> evaluate -> (retry with critique | finish)."""

import json
import operator
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import Evaluator, PromptSpec, PromptWriter
from .backends import GenerationRequest, ImageBackend
from .config import Settings

_RATIOS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "16:9": (16, 9), "9:16": (9, 16), "3:2": (3, 2), "2:3": (2, 3)}


def dimensions(aspect_ratio: str, max_side: int, multiple: int = 32) -> tuple[int, int]:
    """Width/height for a ratio with the long side at `max_side`, both rounded to `multiple`.
    Qwen-Image-2.1 requires 32; FLUX / most others accept 16 but 32 is a safe superset."""
    w, h = _RATIOS.get(aspect_ratio, (1, 1))
    scale = max_side / max(w, h)
    return int(w * scale) // multiple * multiple, int(h * scale) // multiple * multiple


class State(TypedDict, total=False):
    request: str
    references: list[str]
    run_dir: str
    spec: dict
    image: str
    evaluation: dict
    history: Annotated[list[dict], operator.add]  # one entry per finished attempt


def build_graph(settings: Settings, writer: PromptWriter, backend: ImageBackend, evaluator: Evaluator, log=print):
    def write_prompt(state: State) -> State:
        n = len(state.get("history", [])) + 1
        log(f"\n[1/3 prompt writer] attempt {n}/{settings.max_iterations}")
        spec = writer.run(state["request"], [Path(p) for p in state["references"]], state.get("history", []))
        log(f"  prompt: {spec.prompt}\n  aspect: {spec.aspect_ratio}  ({spec.rationale})")
        return {"spec": spec.model_dump()}

    def generate(state: State) -> State:
        spec = PromptSpec(**state["spec"])
        n = len(state.get("history", [])) + 1
        width, height = dimensions(spec.aspect_ratio, settings.max_side)
        refs = [Path(p) for p in state["references"]]
        if len(refs) > backend.max_references:
            log(f"  ! backend accepts {backend.max_references} reference(s); extra ones are only seen by the agents")
        out = Path(state["run_dir"]) / f"attempt_{n}.png"
        log(f"[2/3 generator:{backend.name}] {width}x{height}, {min(len(refs), backend.max_references)} ref(s)...")
        t0 = time.perf_counter()
        backend.generate(
            GenerationRequest(spec.prompt, width, height, spec.negative_prompt, references=refs), out
        )
        log(f"  saved {out} in {time.perf_counter() - t0:.0f}s")
        return {"image": str(out)}

    def evaluate(state: State) -> State:
        log("[3/3 evaluator]")
        ev = evaluator.run(
            state["request"], Path(state["image"]), [Path(p) for p in state["references"]], state["spec"]["prompt"]
        )
        log(f"  score {ev.score}/10; issues: {'; '.join(ev.issues) or 'none'}")
        attempt = {"prompt": state["spec"]["prompt"], "spec": state["spec"], "image": state["image"],
                   "evaluation": ev.model_dump()}
        return {"evaluation": ev.model_dump(), "history": [attempt]}

    def route(state: State) -> str:
        if state["evaluation"]["score"] >= settings.pass_score:
            return "done"
        if len(state["history"]) >= settings.max_iterations:
            return "done"
        return "retry"

    g = StateGraph(State)
    g.add_node("write_prompt", write_prompt)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_edge(START, "write_prompt")
    g.add_edge("write_prompt", "generate")
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges("evaluate", route, {"retry": "write_prompt", "done": END})
    return g.compile()


def summarize(state: State) -> dict:
    """Pick the best-scoring attempt and write a JSON report next to the images."""
    best = max(state["history"], key=lambda a: a["evaluation"]["score"])
    report = {"request": state["request"], "references": state["references"], "best": best,
              "attempts": state["history"]}
    Path(state["run_dir"], "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
