# Multi-agent image generator

Three agents in a LangGraph loop:

```
user request + reference images
        │
        ▼
 ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ 1 Prompt     │──▶│ 2 Image generator│──▶│ 3 Evaluator  │──▶ score ≥ PASS_SCORE or
 │   writer     │   │  Studio/diffusers│   │   (VLM)      │    MAX_ITERATIONS → best.png
 │   (VLM)      │◀──┴──────────────────┴───│ issues +     │
 └──────────────┘        retry w/ critique │ suggestions  │
                                           └──────────────┘
```

- **Agents 1 and 3**: one vision LLM (default `gemma4:12b`) on the MacBook's Ollama, reached over ngrok.
  The prompt writer *sees* the reference images; the evaluator compares the result against them.
- **Agent 2**: switchable with `IMAGE_BACKEND`:
  - `studio`: whatever is loaded in Unsloth Studio's Image tab (Qwen-Image-2.1, its Turbo, FLUX.2 klein...).
    Studio does the low-VRAM handling. Qwen-Image-2.1 takes up to 10 reference images.
  - `diffusers`: FLUX.2 [klein] 4B in-process, 4-bit, CPU offload. Up to 4 reference images.

Because the LLM runs on the Mac, the whole 8 GB GPU is free for image generation.

## Setup

### MacBook (Ollama)
```bash
ollama pull gemma4:12b
ngrok http 11434 --host-header="localhost:11434" --basic-auth "me:a-long-password"
```
`--host-header` is required or Ollama rejects tunnelled requests. Put the https URL and the
`user:pass` into `.env`. Without basic auth, anyone with the URL can use your Ollama.

### This PC
```bash
uv sync                      # Studio backend only
uv sync --extra diffusers    # also the in-process FLUX.2 klein backend (~3 GB torch download)
copy .env.example .env       # then fill it in
```
For `IMAGE_BACKEND=studio`: start Unsloth Studio, load the image model in the Image tab, and create an
API key under avatar → Settings → API.

## Run
```bash
uv run imagegen "a red fox reading a newspaper in a Paris café, golden hour, film photo"
uv run imagegen "put this person in a cyberpunk street at night" --ref me.jpg --ref style.png
```
Each run writes `outputs/<timestamp>/attempt_N.png`, `best.png` and `report.json` (prompts, scores, critiques).

## Notes for an 8 GB RTX 4060
- Keep `MAX_SIDE=1024`. A 1920×1088 Qwen-Image-2.1 render in Studio ran out of memory at VAE decode.
- Qwen-Image-2.1 (full) streams layers from RAM: ~2 min per image. The Turbo GGUF (8 steps) or FLUX.2 klein
  (4 steps) are much faster for iterating. Use Qwen for final quality.
- Try a different evaluator with `EVAL_MODEL=qwen3-vl:8b`; it is often stricter on details.

## Illustration categories

Specialize the agents for one of six domains (plus a generic fallback). The category biases
the prompt writer's style, the evaluator's checklist, and the default aspect ratio and
negative prompts.

| key | best for |
|---|---|
| `paper` | Research-paper figures (Nature/IEEE style) |
| `poster` | Festival / event posters (Diwali, Holi, Christmas, Eid …) |
| `textbook` | School-textbook illustrations (clean, no in-image text) |
| `medical` | Anatomical diagrams (Netter/Gray's style) |
| `science` | Molecules, cells, physics vectors |
| `tech` | Stripe/Vercel/Linear-style flat/isometric tech art |

CLI:
```bash
uv run imagegen "Mitosis in an animal cell, 4 stages left to right" --category textbook
uv run imagegen "Diwali poster with a diya centerpiece and rangoli border" -c poster
```

Categories that are text-free (`paper`, `textbook`, `medical`, `science`, `tech`) tell the
evaluator to flag ANY readable text as a defect and cap the score at 5, so the retry loop
actively strips hallucinated labels and watermarks.

## Web UI

Browser frontend with live progress, side-by-side attempts and a gallery of past runs.
Serves the same pipeline the CLI uses.

```bash
uv run imagegen-web            # http://127.0.0.1:8787
```

The pane shows each attempt as it finishes (image + score + the evaluator's text audit + the
critique), and the final best image on top. Past runs are browsable from the Gallery tab —
nothing is stored in a database, it just reads `outputs/*/report.json`.

Prerequisites same as the CLI: Studio with an image model loaded, Mac Ollama reachable via
`.env`, image backend configured.
