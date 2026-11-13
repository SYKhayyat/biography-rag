"use strict";
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const CITATION = /\[((?:d\d{2,}-\d{4,})(?:\s*,\s*d\d{2,}-\d{4,})*)\]/g;

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = typeof body === "object" ? body.detail : body;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || res.statusText);
  }
  return body;
}

function showError(el, err) {
  el.textContent = err.message || String(err);
  el.hidden = false;
}

function withCitations(text) {
  return esc(text).replace(CITATION, (_m, ids) =>
    ids.split(/\s*,\s*/).map((id) => `<button type="button" class="chip" data-chunk="${id}">${id}</button>`).join(""));
}

// --- chunk side panel ------------------------------------------------------
function corpusId() {
  return ($("#run") || $("#compare") || $("#corpus"))?.dataset.corpus;
}

document.addEventListener("click", async (ev) => {
  const chip = ev.target.closest(".chip");
  if (chip) {
    const panel = $("#chunk-panel");
    $("#chunk-title").textContent = chip.dataset.chunk;
    $("#chunk-body").textContent = "Loading…";
    panel.hidden = false;
    try {
      const c = await api(`/api/corpora/${corpusId()}/chunks/${chip.dataset.chunk}`);
      $("#chunk-body").textContent = c.text;
    } catch (err) {
      $("#chunk-body").textContent = err.message;
    }
  }
  if (ev.target.id === "chunk-close") $("#chunk-panel").hidden = true;
});

// --- index: upload -----------------------------------------------------------
$("#upload-form")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const button = $("button", form);
  button.disabled = true;
  button.textContent = "Uploading…";
  try {
    const corpus = await api("/api/corpora", { method: "POST", body: new FormData(form) });
    location.href = `/corpora/${corpus.corpus_id}`;
  } catch (err) {
    showError($("#upload-error"), err);
    button.disabled = false;
    button.textContent = "Upload & index";
  }
});

// --- corpus page ---------------------------------------------------------------
const corpusEl = $("#corpus");
if (corpusEl?.dataset.status === "ingesting") {
  const poll = async () => {
    const c = await api(`/api/corpora/${corpusEl.dataset.corpus}`).catch(() => null);
    if (c && c.status !== "ingesting") location.reload();
    else setTimeout(poll, 2000);
  };
  setTimeout(poll, 2000);
}

async function postJson(url, data) {
  return api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}

// Model rows: each form holds one or more (provider, model) pairs. One pair is an ordinary
// run; several start a side-by-side comparison.
const OPTS = $("#opts") ? JSON.parse($("#opts").textContent) : null;

function modelOptions(providerName, selected) {
  const p = OPTS.providers.find((x) => x.name === providerName);
  return (p?.models || []).map((m) =>
    `<option value="${esc(m.id)}" ${m.id === selected ? "selected" : ""}>${esc(m.label)}</option>`).join("");
}

function addTarget(form, provider = OPTS.default_provider, model = null) {
  const list = $(".targets", form);
  const row = document.createElement("div");
  row.className = "row target";
  row.innerHTML = `
    <select name="provider" aria-label="Provider">${OPTS.providers.map((p) =>
      `<option value="${esc(p.name)}" ${p.name === provider ? "selected" : ""}>${esc(p.label)}</option>`).join("")}</select>
    <select name="model" aria-label="Model">${modelOptions(provider, model)}</select>
    <button type="button" class="remove-target secondary" aria-label="Remove this model">×</button>`;
  list.append(row);
  syncTargets(form);
}

function syncTargets(form) {
  const rows = [...form.querySelectorAll(".target")];
  rows.forEach((r) => { $(".remove-target", r).hidden = rows.length === 1; });
  $(".add-target", form).disabled = rows.length >= OPTS.max_compare_targets;
  const submit = $("button[type=submit]", form);
  const base = form.dataset.kind === "ask" ? "Ask" : "Generate story";
  submit.textContent = rows.length > 1 ? `${base} with ${rows.length} models` : base;
}

function targetsOf(form) {
  return [...form.querySelectorAll(".target")].map((r) => ({
    provider: $("[name=provider]", r).value,
    model: $("[name=model]", r).value || null,
  }));
}

for (const form of document.querySelectorAll(".target-form")) {
  if (!OPTS?.providers.length) break;
  addTarget(form);
  form.addEventListener("change", (ev) => {
    if (ev.target.name === "provider") {
      $("[name=model]", ev.target.closest(".target")).innerHTML = modelOptions(ev.target.value);
    }
  });
  form.addEventListener("click", (ev) => {
    if (ev.target.closest(".add-target")) {
      const last = [...form.querySelectorAll(".target")].pop();
      addTarget(form, $("[name=provider]", last).value);
      // default the new row to the next model of the same provider, so "+" reads as "also try"
      const sel = $("[name=model]", [...form.querySelectorAll(".target")].pop());
      const prev = $("[name=model]", last).selectedIndex;
      if (sel.options.length > 1) sel.selectedIndex = (prev + 1) % sel.options.length;
    }
    if (ev.target.closest(".remove-target")) {
      ev.target.closest(".target").remove();
      syncTargets(form);
    }
  });
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const submit = $("button[type=submit]", form);
    const label = submit.textContent;
    submit.disabled = true;
    submit.textContent = "Starting…";
    try {
      const body = { kind: form.dataset.kind, targets: targetsOf(form) };
      if (form.question) body.question = form.question.value;
      const res = await postJson(`/api/corpora/${corpusEl.dataset.corpus}/compare`, body);
      location.href = res.comparison_id
        ? `/corpora/${corpusEl.dataset.corpus}/compare/${res.comparison_id}`
        : `/runs/${res.run_ids[0]}`;
    } catch (err) {
      showError($("#action-error"), err);
      submit.disabled = false;
      submit.textContent = label;
    }
  });
}

// --- run page --------------------------------------------------------------------
function renderRun(run) {
  const progress = $("#progress");
  const active = run.status === "queued" || run.status === "running";
  progress.hidden = !active;
  if (active) {
    $("progress", progress).max = run.total || 1;
    $("progress", progress).value = run.progress;
    $("#progress-text").textContent = `${run.status}: ${run.progress}/${run.total} questions answered`;
  }
  if (run.error) showError($("#run-error"), new Error(run.error));
  if (run.kind === "ask") $("#run-title").textContent = run.question || run.answers[0]?.question || run.person;

  if (run.status === "done") {
    $("#story-card").hidden = false;
    $("#story-heading").textContent = run.kind === "ask" ? "Answer" : `The story of ${run.person}`;
    const story = run.story || "Nothing relevant was found in the sources.";
    $("#story").innerHTML = story.split(/\n\s*\n/).map((p) => `<p>${withCitations(p)}</p>`).join("");
    const v = run.verification;
    if (v) {
      const found = run.answers.filter((a) => a.answer).length;
      $("#summary").innerHTML =
        `<span class="stat"><b>${Math.round(v.traced_ratio * 100)}%</b> of sentences traced to chunks</span>` +
        `<span class="stat"><b>${v.flagged_count}</b> flagged of ${v.sentence_count}</span>` +
        `<span class="stat"><b>${found}</b>/${run.answers.length} questions answered from sources</span>`;
      $("#audit-card").hidden = v.checks.length === 0;
      $("#audit").innerHTML = v.checks.map((c, i) => `
        <tr class="${c.flagged ? "flagged" : ""}">
          <td>${i + 1}</td><td>${withCitations(c.sentence)}</td>
          <td><span class="badge ${esc(c.verdict)}">${esc(c.verdict)}</span></td>
          <td>${Math.round(c.support * 100)}%</td><td class="muted">${esc(c.note)}</td>
        </tr>`).join("");
    }
  }
  if (run.answers.length) {
    $("#answers-card").hidden = false;
    $("#answers").innerHTML = run.answers.map((a) => `
      <tr class="${a.answer ? "" : "absent"}">
        <td>${a.query_id || "–"}</td><td>${esc(a.question)}</td><td>${withCitations(a.display)}</td>
        <td>${a.retrieved_ids.map((id) => `<button type="button" class="chip quiet" data-chunk="${id}">${id}</button>`).join("")}</td>
      </tr>`).join("");
  }
  return active;
}

const runEl = $("#run");
if (runEl) {
  const poll = async () => {
    try {
      if (renderRun(await api(`/api/runs/${runEl.dataset.run}`))) setTimeout(poll, 1500);
    } catch (err) {
      showError($("#run-error"), err);
      setTimeout(poll, 5000);
    }
  };
  poll();
}

// --- compare page -------------------------------------------------------------------
const isActive = (r) => r.status === "queued" || r.status === "running";

function elapsed(r) {
  const end = r.finished_at ? new Date(r.finished_at) : new Date();
  const s = Math.max(0, Math.round((end - new Date(r.created_at)) / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function compareColumn(r) {
  const v = r.verification;
  const answered = r.answers.filter((a) => a.answer).length;
  const stats = [`<span class="stat"><b>${elapsed(r)}</b>${isActive(r) ? " so far" : ""}</span>`];
  if (r.kind === "story" && r.answers.length) stats.push(`<span class="stat"><b>${answered}</b>/${r.answers.length} answered</span>`);
  if (v) {
    stats.push(`<span class="stat"><b>${Math.round(v.traced_ratio * 100)}%</b> traced</span>`);
    stats.push(`<span class="stat ${v.flagged_count ? "error" : ""}"><b>${v.flagged_count}</b> flagged</span>`);
  }
  let body = "";
  if (isActive(r)) {
    body = `<p><progress max="${r.total || 1}" value="${r.progress}"></progress>
      <span class="muted">${r.progress}/${r.total} questions</span></p>`;
  } else if (r.status === "failed") {
    body = `<p class="error">${esc(r.error)}</p>`;
  } else {
    const story = r.story || "Nothing relevant was found in the sources.";
    body = `<div class="story">${story.split(/\n\s*\n/).map((p) => `<p>${withCitations(p)}</p>`).join("")}</div>`;
  }
  return `<article class="card compare-col">
    <header><strong>${esc(r.model)}</strong> <span class="badge ${esc(r.status)}">${esc(r.status)}</span>
      <div class="muted small">${esc(r.provider)}</div></header>
    <div class="summary">${stats.join("")}</div>
    ${body}
    <p class="small"><a href="/runs/${esc(r.run_id)}">full run, audit and log →</a></p>
  </article>`;
}

function compareMatrix(runs) {
  const withAnswers = runs.filter((r) => r.answers.length);
  if (!withAnswers.length) return "";
  const questions = withAnswers[0].answers.map((a) => [a.query_id, a.question]);
  const head = `<thead><tr><th>#</th><th>Question</th>${runs.map((r) => `<th>${esc(r.model)}</th>`).join("")}</tr></thead>`;
  const rows = questions.map(([id, q]) => `<tr><td>${id}</td><td>${esc(q)}</td>${runs.map((r) => {
    const a = r.answers.find((x) => x.query_id === id);
    if (!a) return `<td class="muted">…</td>`;
    if (!a.answer) return `<td class="muted" title="not in the sources">–</td>`;
    return `<td title="${esc(a.answer)}">✓ ${a.cited_ids.map((c) => `<button type="button" class="chip quiet" data-chunk="${esc(c)}">${esc(c)}</button>`).join("")}</td>`;
  }).join("")}</tr>`).join("");
  return head + `<tbody>${rows}</tbody>`;
}

function renderCompare(cmp) {
  $("#cmp-grid").innerHTML = cmp.runs.map(compareColumn).join("");
  const matrix = $("#cmp-matrix");
  if (matrix) {
    const html = compareMatrix(cmp.runs);
    $("#cmp-matrix-card").hidden = !html;
    matrix.innerHTML = html;
  }
  return cmp.runs.some(isActive);
}

const cmpEl = $("#compare");
if (cmpEl) {
  const url = `/api/corpora/${cmpEl.dataset.corpus}/comparisons/${cmpEl.dataset.comparison}`;
  const poll = async () => {
    try {
      if (renderCompare(await api(url))) setTimeout(poll, 2000);
    } catch (err) {
      showError($("#cmp-error"), err);
      setTimeout(poll, 5000);
    }
  };
  if (renderCompare(JSON.parse($("#cmp-data").textContent))) setTimeout(poll, 2000);
}
