// Illustration Studio frontend — talks to FastAPI, drives the tabs, renders live SSE updates.

const $ = (id) => document.getElementById(id);
const panes = { new: $("pane-new"), run: $("pane-run"), gallery: $("pane-gallery") };
const tabs = document.querySelectorAll(".tab");
const statusPill = $("status-pill");
const statusLabel = $("status-label");

function activate(tabKey) {
  for (const el of tabs) el.classList.toggle("active", el.dataset.tab === tabKey);
  const show = (key) => Object.entries(panes).forEach(([k, el]) => el.classList.toggle("hidden", k !== key));
  if (tabKey === "new") show("new");
  if (tabKey === "gallery") { show("gallery"); loadGallery(); }
}
tabs.forEach((t) => t.addEventListener("click", () => activate(t.dataset.tab)));

function setStatus(label, state = "idle") {
  statusLabel.textContent = label;
  statusPill.classList.remove("working", "error");
  if (state === "working") statusPill.classList.add("working");
  if (state === "error") statusPill.classList.add("error");
}

function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("toast-root").appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function esc(s) {
  return (s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ── Categories ─────────────────────────────────────────
async function loadCategories() {
  try {
    const cats = await fetch("/api/categories").then((r) => r.json());
    const sel = $("category");
    for (const c of cats) {
      const o = document.createElement("option");
      o.value = c.key; o.textContent = c.label; o.dataset.description = c.description;
      sel.appendChild(o);
    }
    sel.value = "paper";
    const refresh = () => { $("cat-desc").textContent = sel.selectedOptions[0]?.dataset.description || ""; };
    sel.onchange = refresh; refresh();
  } catch (e) { toast("Could not load categories: " + e, "error"); }
}

// ── File drop UX ───────────────────────────────────────
const fileWrap = $("file-wrap"), fileInput = $("ref"), fileLabel = $("file-label");
fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);
  fileWrap.classList.toggle("has-files", files.length > 0);
  fileLabel.textContent = files.length === 0
    ? "Drop images here or click to browse"
    : files.length === 1 ? files[0].name : `${files.length} files selected`;
});

// ── Gallery ────────────────────────────────────────────
async function loadGallery() {
  const g = $("gallery");
  g.innerHTML = `<div class="empty">loading…</div>`;
  try {
    const runs = await fetch("/api/runs").then((r) => r.json());
    g.innerHTML = runs.length === 0
      ? `<div class="empty">No runs yet. Generate your first illustration →</div>` : "";
    for (const r of runs) {
      const a = document.createElement("a");
      a.href = "#";
      a.innerHTML = `
        ${r.best_image_url ? `<img src="${r.best_image_url}" alt="" />` : `<div class="empty">no image</div>`}
        <div class="meta">
          <div class="req">${esc(r.request)}</div>
          <div class="sub">${r.best_score != null ? r.best_score + "/10" : "—"} · ${r.attempt_count} attempt(s)</div>
        </div>`;
      a.onclick = (e) => { e.preventDefault(); openPastRun(r.run_id); };
      g.appendChild(a);
    }
  } catch (e) { g.innerHTML = `<div class="empty">Failed to load: ${esc(String(e))}</div>`; }
}

async function openPastRun(runId) {
  try {
    const data = await fetch(`/api/runs/${runId}`).then((r) => r.json());
    activate("new"); // not strictly, we'll show the run pane manually
    for (const el of tabs) el.classList.remove("active");
    panes.new.classList.add("hidden"); panes.gallery.classList.add("hidden");
    panes.run.classList.remove("hidden");
    $("run-id").textContent = runId;
    $("status-log").innerHTML = `<div class="line hint">Past run · ${esc(runId)}</div>`;
    $("attempts").innerHTML = "";
    data.attempts.forEach((a, i) => {
      const name = a.image.split(/[\\/]/).pop();
      renderAttempt(i + 1, `/outputs/${runId}/${name}`, a.evaluation, a.spec);
    });
    const bestName = (data.best?.image || "").split(/[\\/]/).pop();
    showBest(`/outputs/${runId}/${bestName}`, data.best?.evaluation?.score, `/outputs/${runId}/report.json`);
    setStatus("Idle");
  } catch (e) { toast("Could not open run: " + e, "error"); }
}

// ── Render helpers ─────────────────────────────────────
function renderAttempt(n, imgUrl, ev, spec) {
  const existing = $("attempts").querySelector(`[data-n="${n}"]`);
  const el = existing || document.createElement("div");
  el.className = "attempt"; el.dataset.n = n;

  const auditHtml = (ev?.text_audit || [])
    .map((t) => `<li class="${t.includes("[GARBLED]") ? "garbled" : ""}">${esc(t)}</li>`).join("");
  const issuesHtml = (ev?.issues || []).map((t) => `<li>${esc(t)}</li>`).join("");
  const sugHtml = (ev?.suggestions || []).map((t) => `<li>${esc(t)}</li>`).join("");

  el.innerHTML = `
    <div class="attempt-head">
      <span>Attempt ${n}</span>
      <span class="score">${ev ? ev.score + "/10" : "generating…"}</span>
    </div>
    ${imgUrl
      ? `<img class="attempt-img" src="${imgUrl}" alt="attempt ${n}" />`
      : `<div class="attempt-img loading">rendering…</div>`}
    ${ev ? `
      <details>
        <summary>prompt, critique and text audit</summary>
        <div class="block"><span class="k">Prompt</span><div>${esc(spec?.prompt || "")}</div></div>
        ${spec?.negative_prompt ? `<div class="block"><span class="k">Negative</span><div>${esc(spec.negative_prompt)}</div></div>` : ""}
        ${issuesHtml ? `<div class="block"><span class="k">Issues</span><ul>${issuesHtml}</ul></div>` : ""}
        ${sugHtml ? `<div class="block"><span class="k">Suggestions</span><ul>${sugHtml}</ul></div>` : ""}
        ${auditHtml ? `<div class="block"><span class="k">Text in image</span><ul>${auditHtml}</ul></div>` : ""}
      </details>` : ""}`;
  if (!existing) $("attempts").appendChild(el);
}

function showBest(url, score, reportUrl) {
  $("best").classList.remove("hidden");
  $("best-img").src = url;
  $("best-score").textContent = score != null ? `${score}/10` : "—";
  $("report-link").href = reportUrl;
}

// ── Submit a new run ───────────────────────────────────
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("go"); btn.disabled = true;
  setStatus("Starting", "working");
  try {
    const fd = new FormData($("form"));
    if (!fd.get("iterations")) fd.delete("iterations");
    const { run_id } = await fetch("/api/generate", { method: "POST", body: fd }).then((r) => r.json());

    panes.new.classList.add("hidden"); panes.gallery.classList.add("hidden");
    panes.run.classList.remove("hidden");
    for (const el of tabs) el.classList.remove("active");
    $("run-id").textContent = run_id;
    $("attempts").innerHTML = ""; $("best").classList.add("hidden");
    $("status-log").innerHTML = `<div class="line hint">Starting run <b>${esc(run_id)}</b>…</div>`;

    const es = new EventSource(`/api/runs/${run_id}/stream`);
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "started") {
        $("info-prompt").textContent = msg.prompt_model;
        $("info-eval").textContent = msg.eval_model;
        $("info-image").textContent = msg.image_model;
        $("status-log").insertAdjacentHTML("beforeend",
          `<div class="line">Backend <b>${esc(msg.backend)}</b> · model <b>${esc(msg.image_model)}</b> · category <b>${esc(msg.category)}</b> · up to ${msg.max_iterations} iterations</div>`);
        setStatus(`Running ${msg.image_model}`, "working");
      } else if (msg.type === "log") {
        $("status-log").insertAdjacentHTML("beforeend", `<div class="line hint">· ${esc(msg.message)}</div>`);
        $("status-log").scrollTop = $("status-log").scrollHeight;
      } else if (msg.type === "attempt_image") {
        renderAttempt(msg.attempt, msg.image_url, null, null);
      } else if (msg.type === "attempt_scored") {
        const existing = $("attempts").querySelector(`[data-n="${msg.attempt}"] img`);
        renderAttempt(msg.attempt, existing?.src, msg.evaluation, msg.spec);
      } else if (msg.type === "done") {
        showBest(msg.best_image_url, msg.best_score, msg.report_url);
        setStatus(`Done · ${msg.best_score}/10`);
        es.close(); btn.disabled = false;
      } else if (msg.type === "error") {
        $("status-log").insertAdjacentHTML("beforeend", `<div class="line err">Error: ${esc(msg.message)}</div>`);
        toast(msg.message, "error");
        setStatus("Error", "error");
        es.close(); btn.disabled = false;
      }
    };
    es.onerror = () => { es.close(); setStatus("Disconnected", "error"); btn.disabled = false; };
  } catch (err) {
    toast("Failed to start: " + err, "error");
    setStatus("Error", "error");
    btn.disabled = false;
  }
});

loadCategories();
