"use strict";
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const CITATION = /\[((?:d\d{2,}-\d{4,})(?:\s*,\s*d\d{2,}-\d{4,})*)\]/g;
const OPTS = $("#opts") ? JSON.parse($("#opts").textContent) : { providers: [] };

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = typeof body === "object" ? body.detail : body;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || res.statusText);
  }
  return body;
}

const postJson = (url, data) =>
  api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function showError(el, err) {
  el.textContent = err.message || String(err);
  el.hidden = false;
}

function withCitations(text) {
  return esc(text).replace(CITATION, (_m, ids) =>
    ids.split(/\s*,\s*/).map((id) => `<button type="button" class="chip" data-chunk="${id}">${id}</button>`).join(""));
}

const paragraphs = (text) => text.split(/\n\s*\n/).map((p) => `<p>${withCitations(p)}</p>`).join("");
const chips = (ids, cls = "chip quiet") =>
  ids.map((id) => `<button type="button" class="${cls}" data-chunk="${esc(id)}">${esc(id)}</button>`).join("");

// Small per-browser conveniences (last model picked); the page works without them.
const remember = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

// --- chunk side panel ------------------------------------------------------
const corpusId = () => ($("#compare") || $("#corpus"))?.dataset.corpus;

document.addEventListener("click", async (ev) => {
  const chip = ev.target.closest(".chip[data-chunk]");
  if (chip) {
    ev.stopPropagation();
    const panel = $("#chunk-panel");
    $("#chunk-title").textContent = chip.dataset.chunk;
    $("#chunk-body").textContent = "Loading…";
    panel.hidden = false;
    try {
      $("#chunk-body").textContent = (await api(`/api/corpora/${corpusId()}/chunks/${chip.dataset.chunk}`)).text;
    } catch (err) {
      $("#chunk-body").textContent = err.message;
    }
  }
  if (ev.target.id === "chunk-close") $("#chunk-panel").hidden = true;
}, true);
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") $("#chunk-panel").hidden = true; });

// --- model picker: one dropdown, grouped by provider ----------------------------
const targetValue = (provider, model) => `${provider}::${model}`;
const parseTarget = (v) => { const [provider, ...rest] = v.split("::"); return { provider, model: rest.join("::") || null }; };

/** Fill a <select> with every model. `taken` values are shown but disabled. */
function fillModelPicker(select, { taken = new Set(), preferred = null } = {}) {
  select.innerHTML = OPTS.providers.map((p) => `<optgroup label="${esc(p.label)}">${p.models.map((m) => {
    const v = targetValue(p.name, m.id);
    return `<option value="${esc(v)}" ${taken.has(v) ? "disabled" : ""}>${esc(m.label)}${taken.has(v) ? " (already here)" : ""}</option>`;
  }).join("")}</optgroup>`).join("");
  const free = [...select.options].filter((o) => !o.disabled);
  const want = free.find((o) => o.value === preferred)
    || free.find((o) => o.value.startsWith(`${OPTS.default_provider}::`))
    || free[0];
  if (want) select.value = want.value;
  return free.length;
}

// --- step 1: upload ----------------------------------------------------------------
$("#upload-form")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const button = $("button[type=submit]", form);
  button.disabled = true;
  button.textContent = "Uploading…";
  try {
    const corpus = await api("/api/corpora", { method: "POST", body: new FormData(form) });
    location.href = `/corpora/${corpus.corpus_id}/sources`;
  } catch (err) {
    showError($("#upload-error"), err);
    button.disabled = false;
    button.textContent = "Upload";
  }
});

// Sources page: while indexing, wait, then move on to step 2 by itself.
const corpusEl = $("#corpus");
if (corpusEl?.dataset.status === "ingesting") {
  const poll = async () => {
    const c = await api(`/api/corpora/${corpusEl.dataset.corpus}`).catch(() => null);
    if (c && c.status === "ready") location.href = `/corpora/${c.corpus_id}`;
    else if (c && c.status !== "ingesting") location.reload();
    else setTimeout(poll, 2000);
  };
  setTimeout(poll, 2000);
}

// --- step 2: ask (story | question tabs, one or more models) ---------------------------
const askForm = $("#ask-form");
if (askForm && OPTS.providers.length) {
  const rows = $(".model-rows", askForm);
  const addRow = $(".add-row", askForm);
  const submit = $("button[type=submit]", askForm);
  const pickers = () => [...rows.querySelectorAll("select")];

  // A model picked in one row is greyed out in the others.
  const refreshTaken = () => {
    const values = pickers().map((p) => p.value);
    for (const p of pickers()) {
      for (const o of p.options) o.disabled = o.value !== p.value && values.includes(o.value);
    }
  };
  const sync = () => {
    const n = pickers().length;
    for (const b of rows.querySelectorAll(".remove")) b.hidden = n === 1;
    addRow.hidden = n >= Math.min(OPTS.max_compare_targets, OPTS.providers.reduce((k, p) => k + p.models.length, 0));
    $(".models-label", askForm).textContent = n === 1 ? "Model" : "Models";
    const verb = askForm.dataset.kind === "ask" ? "Ask" : "Write the story";
    submit.textContent = n === 1 ? verb : `${verb} with ${n} models`;
    refreshTaken();
  };
  const newRow = (preferred) => {
    const row = document.createElement("div");
    row.className = "model-row";
    row.innerHTML = `<select aria-label="Model"></select>
      <button type="button" class="secondary remove" aria-label="Remove this model" title="Remove">×</button>`;
    const taken = new Set(pickers().map((p) => p.value));
    rows.append(row);
    fillModelPicker($("select", row), { taken, preferred });
    for (const o of $("select", row).options) o.textContent = o.textContent.replace(" (already here)", "");
    sync();
  };
  newRow(remember.get("model"));
  addRow.addEventListener("click", () => { newRow(null); pickers().at(-1).focus(); });
  rows.addEventListener("click", (ev) => {
    if (ev.target.closest(".remove")) { ev.target.closest(".model-row").remove(); sync(); }
  });
  rows.addEventListener("change", refreshTaken);

  const setTab = (kind) => {
    askForm.dataset.kind = kind;
    for (const t of askForm.querySelectorAll(".tab")) t.setAttribute("aria-selected", String(t.dataset.tab === kind));
    for (const p of askForm.querySelectorAll(".tab-panel")) p.hidden = p.dataset.panel !== kind;
    askForm.question.required = kind === "ask";
    sync();
    const url = new URL(location.href);
    if (kind === "ask") url.searchParams.set("tab", "ask"); else url.searchParams.delete("tab");
    history.replaceState(null, "", url);
    if (kind === "ask") askForm.question.focus();
  };
  askForm.addEventListener("click", (ev) => { const t = ev.target.closest(".tab"); if (t) setTab(t.dataset.tab); });
  if (new URLSearchParams(location.search).get("tab") === "ask") setTab("ask");

  askForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const values = pickers().map((p) => p.value);
    const label = submit.textContent;
    submit.disabled = true;
    submit.textContent = "Starting…";
    try {
      remember.set("model", values[0]);
      const body = { kind: askForm.dataset.kind, targets: values.map(parseTarget) };
      if (body.kind === "ask") body.question = askForm.question.value;
      const res = await postJson(`/api/corpora/${corpusEl.dataset.corpus}/compare`, body);
      location.href = `/corpora/${corpusEl.dataset.corpus}/results/${res.result_id}`;
    } catch (err) {
      showError($("#action-error"), err);
      submit.disabled = false;
      submit.textContent = label;
    }
  });
}

// --- step 3: results (one column per model; hide, details, add a model) ----------------
const isActive = (r) => r.status === "queued" || r.status === "running";

function elapsed(r) {
  const end = r.finished_at ? new Date(r.finished_at) : new Date();
  const s = Math.max(0, Math.round((end - new Date(r.created_at)) / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** One plain line: what the model is doing, or how it went. */
function statusLine(r) {
  if (r.status === "queued") return `<span class="dot running"></span> Waiting to start`;
  if (r.status === "running") {
    const what = r.kind === "story" ? `${r.progress} of ${r.total} questions` : "Working";
    return `<span class="dot running"></span> ${what} · ${elapsed(r)}`;
  }
  if (r.status === "failed") return `<span class="dot failed"></span> Failed after ${elapsed(r)}`;
  const parts = [`Done in ${elapsed(r)}`];
  if (r.kind === "story" && r.answers.length) {
    parts.push(`${r.answers.filter((a) => a.answer).length} of ${r.answers.length} answered`);
  }
  const v = r.verification;
  if (v) {
    parts.push(v.flagged_count
      ? `<span class="error">${plural(v.flagged_count, "flagged sentence")}</span>`
      : "no flagged sentences");
  }
  return `<span class="dot"></span> ${parts.join(" · ")}`;
}

function columnBody(r) {
  if (isActive(r)) {
    return r.kind === "story" ? `<progress max="${r.total || 1}" value="${r.progress}"></progress>` : `<span class="spinner"></span>`;
  }
  if (r.status === "failed") return `<p class="error">${esc(r.error)}</p>`;
  return `<div class="story">${paragraphs(r.story || "Nothing relevant was found in the sources.")}</div>`;
}

function columnHtml(r, { selected, alone }) {
  const tools = alone ? "" : `
      <button type="button" class="ghost" data-act="details" aria-pressed="${selected}">${selected ? "Hide details" : "Details"}</button>
      <button type="button" class="ghost icon" data-act="hide" title="Minimize" aria-label="Minimize ${esc(r.model)}">–</button>`;
  return `<header class="col-head">
      <div class="col-title"><strong>${esc(r.model)}</strong><span class="muted small">${esc(r.provider)}</span></div>
      <div class="col-tools">${tools}</div>
    </header>
    <p class="status-line small">${statusLine(r)}</p>
    <div class="col-body">${columnBody(r)}</div>`;
}

function detailHtml(r, alone) {
  const v = r.verification;
  let html = `<div class="card">
    <div class="detail-head">
      <h2>${alone ? "Details" : `Details: ${esc(r.model)}`}</h2>
      <span class="small muted"><a href="/api/runs/${esc(r.run_id)}/log">log</a> · <a href="/api/runs/${esc(r.run_id)}">JSON</a>
        ${alone ? "" : `<button type="button" class="ghost icon" data-act="close-detail" aria-label="Close details">×</button>`}</span>
    </div>`;
  if (v && v.checks.length) {
    html += `<h3>Sentence check <span class="muted small">${Math.round(v.traced_ratio * 100)}% backed by a cited passage</span></h3>
      <p class="muted small">A sentence is flagged if it cites nothing, cites a passage that wasn't retrieved, has names or numbers
      the passages don't, or wasn't supported when checked. Flagged sentences probably came from the model's memory, not your sources.</p>
      <div class="scroll"><table><thead><tr><th>#</th><th>Sentence</th><th>Result</th><th>Support</th><th>Note</th></tr></thead><tbody>
      ${v.checks.map((c, i) => `<tr class="${c.flagged ? "flagged" : ""}"><td>${i + 1}</td><td>${withCitations(c.sentence)}</td>
        <td><span class="badge ${esc(c.verdict)}">${esc(c.verdict.replace("_", " "))}</span></td>
        <td>${Math.round(c.support * 100)}%</td><td class="muted">${esc(c.note)}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else if (isActive(r)) {
    html += `<p class="muted">The sentence check shows up when this model finishes.</p>`;
  }
  if (r.answers.length) {
    html += `<h3>${r.kind === "story" ? "Answers" : "Answer"}</h3>
      <div class="scroll"><table><thead><tr><th>#</th><th>Question</th><th>Answer</th><th>Passages found</th></tr></thead><tbody>
      ${r.answers.map((a) => `<tr class="${a.answer ? "" : "absent"}"><td>${a.query_id || "–"}</td><td>${esc(a.question)}</td>
        <td>${withCitations(a.display)}</td><td>${chips(a.retrieved_ids)}</td></tr>`).join("")}
      </tbody></table></div>`;
  }
  return html + `</div>`;
}

function matrixHtml(runs) {
  const withAnswers = runs.filter((r) => r.answers.length);
  if (runs.length < 2 || !withAnswers.length) return "";
  const questions = withAnswers[0].answers.map((a) => [a.query_id, a.question]);
  const head = `<thead><tr><th>#</th><th>Question</th>${runs.map((r) => `<th>${esc(r.model)}</th>`).join("")}</tr></thead>`;
  const rows = questions.map(([id, q]) => `<tr><td>${id}</td><td>${esc(q)}</td>${runs.map((r) => {
    const a = r.answers.find((x) => x.query_id === id);
    if (!a) return `<td class="muted">…</td>`;
    if (!a.answer) return `<td class="muted" title="not in the sources">–</td>`;
    return `<td title="${esc(a.answer)}">✓ ${chips(a.cited_ids)}</td>`;
  }).join("")}</tr>`).join("");
  return head + `<tbody>${rows}</tbody>`;
}

const cmpEl = $("#compare");
if (cmpEl) {
  const resultId = location.pathname.split("/").pop();
  const base = `/api/corpora/${cmpEl.dataset.corpus}/comparisons/${resultId}`;
  let cmp = JSON.parse($("#cmp-data").textContent);
  let selected = new URLSearchParams(location.search).get("run");
  let polling = false;
  const grid = $("#cmp-grid");
  const drawn = new Map(); // run_id -> last rendered key, so polls don't redraw what didn't change
  const hiddenKey = `hidden:${resultId}`;
  let hidden = new Set(JSON.parse(remember.get(hiddenKey) || "[]"));
  const saveHidden = () => remember.set(hiddenKey, JSON.stringify([...hidden]));

  // --- add a model (toolbar) ---
  const addOpen = $("#add-open");
  const addForm = $("#add-form");
  const addPicker = $("select", addForm);
  const taken = () => new Set(cmp.runs.map((r) => targetValue(r.provider, r.model)));
  const closeAdd = () => { addForm.hidden = true; addOpen.hidden = false; };

  addOpen.addEventListener("click", () => {
    if (!fillModelPicker(addPicker, { taken: taken(), preferred: remember.get("model") })) {
      showError($("#cmp-error"), new Error("All the models are already here."));
      return;
    }
    addOpen.hidden = true;
    addForm.hidden = false;
    addPicker.focus();
  });
  $("#add-cancel").addEventListener("click", closeAdd);
  addForm.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeAdd(); });
  addForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("button[type=submit]", addForm);
    btn.disabled = true;
    btn.textContent = "Starting…";
    try {
      remember.set("model", addPicker.value);
      const run = await postJson(`${base}/models`, parseTarget(addPicker.value));
      cmp.runs.push(run);
      $("#cmp-error").hidden = true;
      closeAdd();
      render();
      grid.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
      startPolling();
    } catch (err) {
      showError($("#cmp-error"), err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run";
    }
  });

  // --- details, minimize, restore ---
  const select = (runId) => {
    selected = selected === runId ? null : runId;
    const url = new URL(location.href);
    if (selected) url.searchParams.set("run", selected); else url.searchParams.delete("run");
    history.replaceState(null, "", url);
    render();
    if (selected) $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const setHidden = (runId, hide) => {
    if (hide) hidden.add(runId); else hidden.delete(runId);
    if (hide && selected === runId) select(runId);
    saveHidden();
    render();
  };

  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const runId = btn.closest("[data-run]")?.dataset.run;
    if (btn.dataset.act === "details") select(runId);
    else if (btn.dataset.act === "hide") setHidden(runId, true);
    else if (btn.dataset.act === "show") setHidden(runId, false);
    else if (btn.dataset.act === "close-detail") select(selected);
  });

  function render() {
    const runs = cmp.runs;
    hidden = new Set([...hidden].filter((id) => runs.some((r) => r.run_id === id)));
    if (hidden.size >= runs.length) hidden.clear(); // never hide everything
    const shown = runs.filter((r) => !hidden.has(r.run_id));
    const alone = runs.length === 1;
    if (alone) selected = runs[0].run_id; // one model: its details are simply part of the page
    if (selected && !shown.some((r) => r.run_id === selected)) selected = null;

    grid.classList.toggle("single", shown.length === 1);
    grid.style.setProperty("--cols", String(shown.length));
    addOpen.hidden = !addForm.hidden || runs.length >= OPTS.max_compare_targets || !OPTS.providers.length;
    $("#cmp-sub").textContent = alone ? "1 model" : `${runs.length} models`;

    const chipsEl = $("#hidden-models");
    chipsEl.hidden = !hidden.size;
    chipsEl.innerHTML = hidden.size ? `Minimized: ${runs.filter((r) => hidden.has(r.run_id)).map((r) =>
      `<span data-run="${esc(r.run_id)}"><button type="button" class="pill" data-act="show" title="Show again">
        <span class="dot ${isActive(r) ? "running" : r.status === "failed" ? "failed" : ""}"></span>${esc(r.model)} +</button></span>`).join("")}` : "";

    for (const col of grid.querySelectorAll(".compare-col")) {
      if (!shown.some((r) => r.run_id === col.dataset.run)) { col.remove(); drawn.delete(col.dataset.run); }
    }
    let prev = null;
    for (const r of shown) {
      let col = grid.querySelector(`.compare-col[data-run="${r.run_id}"]`);
      if (!col) {
        col = document.createElement("article");
        col.className = "card compare-col";
        col.dataset.run = r.run_id;
      }
      if (col.previousElementSibling !== prev || col.parentNode !== grid) {
        prev ? prev.after(col) : grid.prepend(col);
      }
      prev = col;
      const isSel = r.run_id === selected && !alone;
      col.classList.toggle("selected", isSel);
      const key = JSON.stringify([r, isSel, alone, isActive(r) ? Math.floor(Date.now() / 1000) : 0]);
      if (drawn.get(r.run_id) !== key) { col.innerHTML = columnHtml(r, { selected: isSel, alone }); drawn.set(r.run_id, key); }
    }

    const detail = $("#detail");
    const run = shown.find((r) => r.run_id === selected);
    detail.hidden = !run;
    const detailKey = run ? JSON.stringify([run, alone]) : "";
    if (run && detail.dataset.key !== detailKey) {
      detail.innerHTML = detailHtml(run, alone);
      detail.dataset.key = detailKey;
      detail.dataset.run = run.run_id;
    }

    const matrix = $("#cmp-matrix");
    if (matrix) {
      const html = matrixHtml(shown);
      $("#cmp-matrix-card").hidden = !html;
      if (matrix.dataset.html !== html) { matrix.innerHTML = html; matrix.dataset.html = html; }
    }
    return runs.some(isActive);
  }

  async function poll() {
    try {
      cmp = await api(base);
      $("#cmp-error").hidden = true;
      if (render()) setTimeout(poll, 2000); else polling = false;
    } catch (err) {
      showError($("#cmp-error"), err);
      setTimeout(poll, 5000);
    }
  }
  function startPolling() {
    if (polling) return;
    polling = true;
    setTimeout(poll, 1500);
  }

  if (render()) startPolling();
}
