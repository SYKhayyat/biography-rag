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
  return ($("#run") || $("#corpus"))?.dataset.corpus;
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

$("#run-form")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const button = $("button", ev.target);
  button.disabled = true;
  try {
    const run = await postJson(`/api/corpora/${corpusEl.dataset.corpus}/runs`, { provider: ev.target.provider.value });
    location.href = `/runs/${run.run_id}`;
  } catch (err) {
    showError($("#action-error"), err);
    button.disabled = false;
  }
});

$("#ask-form")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const button = $("button", ev.target);
  button.disabled = true;
  button.textContent = "Asking…";
  try {
    const run = await postJson(`/api/corpora/${corpusEl.dataset.corpus}/ask`, {
      question: ev.target.question.value,
      provider: ev.target.provider.value,
    });
    location.href = `/runs/${run.run_id}`;
  } catch (err) {
    showError($("#action-error"), err);
    button.disabled = false;
    button.textContent = "Ask";
  }
});

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
  if (run.kind === "ask") $("#run-title").textContent = run.answers[0]?.question || run.person;

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
