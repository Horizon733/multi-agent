"""FastAPI frontend for the illustration pipeline. Runs the same LangGraph loop as the CLI
but streams per-stage events to the browser over Server-Sent Events, so the user watches each
attempt as it finishes."""

import asyncio
import json
import secrets
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agents import Evaluator, PromptWriter
from ..backends import make_backend
from ..categories import CATEGORIES, get as get_category
from ..config import Settings
from ..graph import build_graph, summarize
from ..llm import VisionLLM

# One queue per run: the worker thread pushes dicts, the SSE endpoint drains them. The queue
# outlives the HTTP request so a late-connecting client still catches up to the final event.
_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()
_SENTINEL = {"__done__": True}


def _emit(run_id: str, event: dict[str, Any]) -> None:
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        return
    run["events"].append(event)
    run["loop"].call_soon_threadsafe(run["queue"].put_nowait, event)


def _worker(run_id: str, request: str, references: list[Path], category_key: str,
            max_iterations: int | None) -> None:
    """Run the whole graph on a background thread, emitting events as it goes."""
    with _runs_lock:
        run = _runs[run_id]
    run_dir: Path = run["dir"]
    settings = Settings.load()
    if max_iterations:
        settings = Settings(**{**settings.__dict__, "max_iterations": max_iterations})
    try:
        category = get_category(category_key)
        llm = VisionLLM(settings.ollama_host, settings.ollama_basic_auth)
        llm.check(settings.prompt_model, settings.eval_model)
        backend = make_backend(settings)
        _emit(run_id, {"type": "started", "backend": backend.name, "image_model": backend.model,
                       "category": category.label, "prompt_model": settings.prompt_model,
                       "eval_model": settings.eval_model, "max_iterations": settings.max_iterations})

        def log(msg: str) -> None:
            _emit(run_id, {"type": "log", "message": msg.rstrip()})

        graph = build_graph(
            settings,
            PromptWriter(llm, settings.prompt_model, backend.model, category),
            backend,
            Evaluator(llm, settings.eval_model, category),
            log=log,
        )
        # Stream node outputs so the browser gets per-attempt updates, not just the final state.
        # NOTE: dict.update() does not respect LangGraph reducers, so `history` (operator.add)
        # has to be accumulated by hand — otherwise each attempt clobbers the previous one.
        state: dict = {"request": request, "references": [str(p) for p in references],
                       "run_dir": str(run_dir), "history": []}
        for chunk in graph.stream(state, {"recursion_limit": settings.max_iterations * 3 + 5}):
            for node, delta in chunk.items():
                for key, value in delta.items():
                    if key == "history":
                        state["history"] = state.get("history", []) + list(value)
                    else:
                        state[key] = value
                if node == "generate" and "image" in delta:
                    rel = Path(delta["image"]).name
                    _emit(run_id, {"type": "attempt_image",
                                   "image_url": f"/outputs/{run_id}/{rel}",
                                   "attempt": len(state["history"]) + 1})
                if node == "evaluate" and "evaluation" in delta:
                    _emit(run_id, {"type": "attempt_scored",
                                   "attempt": len(state["history"]),
                                   "evaluation": delta["evaluation"],
                                   "spec": state["spec"]})
        report = summarize(state)
        best_rel = Path(report["best"]["image"]).name
        _emit(run_id, {"type": "done", "best_image_url": f"/outputs/{run_id}/{best_rel}",
                       "best_score": report["best"]["evaluation"]["score"],
                       "report_url": f"/outputs/{run_id}/report.json",
                       "attempts": [{"image_url": f"/outputs/{run_id}/{Path(a['image']).name}",
                                     "score": a["evaluation"]["score"],
                                     "prompt": a["prompt"],
                                     "issues": a["evaluation"].get("issues", []),
                                     "text_audit": a["evaluation"].get("text_audit", [])}
                                    for a in state["history"]]})
    except Exception as exc:
        _emit(run_id, {"type": "error", "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        _emit(run_id, _SENTINEL)


def create_app() -> FastAPI:
    settings = Settings.load()
    output_dir = settings.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Illustration Agents")
    static_dir = Path(__file__).parent / "static"
    app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def root():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/categories")
    def list_categories():
        return [{"key": c.key, "label": c.label, "description": c.description,
                 "default_aspect": c.default_aspect, "text_free": c.text_free}
                for c in CATEGORIES.values()]

    @app.get("/api/runs")
    def list_runs():
        """Browse past runs by scanning the outputs folder — no database."""
        runs = []
        for d in sorted(output_dir.iterdir(), reverse=True):
            report = d / "report.json"
            if not report.is_file():
                continue
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            best = data.get("best", {})
            best_img = Path(best.get("image", "")).name
            runs.append({"run_id": d.name, "request": data.get("request", ""),
                         "best_score": best.get("evaluation", {}).get("score"),
                         "best_image_url": f"/outputs/{d.name}/{best_img}" if best_img else None,
                         "attempt_count": len(data.get("attempts", []))})
        return runs[:50]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        report = output_dir / run_id / "report.json"
        if not report.is_file():
            raise HTTPException(404, "run not found")
        return JSONResponse(json.loads(report.read_text(encoding="utf-8")))

    @app.post("/api/generate")
    async def generate(request: str = Form(...), category: str = Form("general"),
                       iterations: int | None = Form(None), ref: list[UploadFile] = []):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        run_dir = output_dir / run_id
        run_dir.mkdir()
        refs_dir = run_dir / "refs"
        references: list[Path] = []
        if ref:
            refs_dir.mkdir()
            for i, upload in enumerate(ref):
                if upload.filename:  # skip empty slots from the form
                    dest = refs_dir / f"{i:02d}_{Path(upload.filename).name}"
                    dest.write_bytes(await upload.read())
                    references.append(dest)
        loop = asyncio.get_running_loop()
        with _runs_lock:
            _runs[run_id] = {"queue": asyncio.Queue(), "events": [], "dir": run_dir, "loop": loop}
        threading.Thread(target=_worker, args=(run_id, request, references, category, iterations),
                         daemon=True).start()
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str):
        with _runs_lock:
            run = _runs.get(run_id)
        if run is None:
            raise HTTPException(404, "run not active (completed runs: use /api/runs/{id})")

        async def events():
            # Replay anything that already happened so a late client doesn't miss the start.
            for past in list(run["events"]):
                if past is _SENTINEL:
                    return
                yield f"data: {json.dumps(past)}\n\n"
            while True:
                item = await run["queue"].get()
                if item is _SENTINEL:
                    return
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("imagegen_agents.web.server:app", host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
