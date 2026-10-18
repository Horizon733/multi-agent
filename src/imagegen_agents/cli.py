"""imagegen "a red fox reading a newspaper" --ref fox.jpg --ref style.png"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .agents import Evaluator, PromptWriter
from .backends import make_backend
from .categories import CATEGORIES, get as get_category
from .config import Settings
from .graph import build_graph, summarize
from .llm import VisionLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent image generator")
    parser.add_argument("request", help="What you want to see")
    parser.add_argument("--ref", action="append", default=[], type=Path, help="Reference image (repeatable)")
    parser.add_argument("--iterations", type=int, help="Override MAX_ITERATIONS")
    parser.add_argument(
        "--category", "-c", choices=list(CATEGORIES.keys()), default="general",
        help="Specialize the agents for one illustration domain. Default: general.",
    )
    args = parser.parse_args()
    category = get_category(args.category)

    settings = Settings.load()
    if args.iterations:
        settings = Settings(**{**settings.__dict__, "max_iterations": args.iterations})
    for ref in args.ref:
        if not ref.is_file():
            sys.exit(f"Reference not found: {ref}")

    llm = VisionLLM(settings.ollama_host, settings.ollama_basic_auth)
    llm.check(settings.prompt_model, settings.eval_model)
    backend = make_backend(settings)
    print(f"LLM: {settings.prompt_model} / {settings.eval_model} @ {settings.ollama_host}")
    print(f"Image backend: {backend.name} ({backend.model}), up to {backend.max_references} reference(s)")

    run_dir = settings.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"Category: {category.label}")
    graph = build_graph(
        settings,
        PromptWriter(llm, settings.prompt_model, backend.model, category),
        backend,
        Evaluator(llm, settings.eval_model, category),
    )
    # Each loop visits 3 nodes; leave headroom over LangGraph's default recursion limit.
    final = graph.invoke(
        {"request": args.request, "references": [str(p.resolve()) for p in args.ref], "run_dir": str(run_dir)},
        {"recursion_limit": settings.max_iterations * 3 + 5},
    )
    report = summarize(final)
    best = Path(report["best"]["image"])
    shutil.copy(best, run_dir / "best.png")
    print(f"\nBest: {best.name} scored {report['best']['evaluation']['score']}/10 -> {run_dir / 'best.png'}")


if __name__ == "__main__":
    main()
