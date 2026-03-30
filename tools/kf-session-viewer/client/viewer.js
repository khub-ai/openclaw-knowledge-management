import { renderArcGrid, summarizeArcComponents } from "./renderers/arc-grid.js";

const state = {
  session: null,
  stepIndex: 0,
  zoom: 8,
  autoplayId: null,
  sourceLabel: null,
};

const els = {
  fileInput: document.getElementById("fileInput"),
  sampleSelect: document.getElementById("sampleSelect"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  playBtn: document.getElementById("playBtn"),
  diffToggle: document.getElementById("diffToggle"),
  zoomRange: document.getElementById("zoomRange"),
  statusText: document.getElementById("statusText"),
  sessionTitle: document.getElementById("sessionTitle"),
  sessionSubtitle: document.getElementById("sessionSubtitle"),
  rendererValue: document.getElementById("rendererValue"),
  stepValue: document.getElementById("stepValue"),
  actionValue: document.getElementById("actionValue"),
  stateValue: document.getElementById("stateValue"),
  levelsValue: document.getElementById("levelsValue"),
  componentsValue: document.getElementById("componentsValue"),
  sourceValue: document.getElementById("sourceValue"),
  notesValue: document.getElementById("notesValue"),
  stateSnapshotValue: document.getElementById("stateSnapshotValue"),
  entityCountsValue: document.getElementById("entityCountsValue"),
  rulesList: document.getElementById("rulesList"),
  toolsList: document.getElementById("toolsList"),
  statesList: document.getElementById("statesList"),
  goalsList: document.getElementById("goalsList"),
  changeTypesValue: document.getElementById("changeTypesValue"),
  metadataValue: document.getElementById("metadataValue"),
  canvas: document.getElementById("artifactCanvas"),
};

const ctx = els.canvas.getContext("2d");
const renderers = { "arc-grid": renderArcGrid };

async function init() {
  bindEvents();
  await loadSampleList();
}

function bindEvents() {
  els.fileInput.addEventListener("change", onFileSelected);
  els.loadSampleBtn.addEventListener("click", onLoadSample);
  els.prevBtn.addEventListener("click", () => showStep(state.stepIndex - 1));
  els.nextBtn.addEventListener("click", () => showStep(state.stepIndex + 1));
  els.playBtn.addEventListener("click", toggleAutoplay);
  els.zoomRange.addEventListener("input", (event) => {
    state.zoom = Number(event.target.value);
    renderCurrentStep();
  });
  els.diffToggle.addEventListener("change", renderCurrentStep);
  window.addEventListener("keydown", (event) => {
    if (!state.session) return;
    if (event.code === "Space" || event.code === "ArrowRight") {
      event.preventDefault();
      showStep(state.stepIndex + 1);
    } else if (event.code === "ArrowLeft") {
      event.preventDefault();
      showStep(state.stepIndex - 1);
    } else if (event.key.toLowerCase() === "p") {
      toggleAutoplay();
    } else if (event.key.toLowerCase() === "d") {
      els.diffToggle.checked = !els.diffToggle.checked;
      renderCurrentStep();
    }
  });
}

async function loadSampleList() {
  try {
    const response = await fetch("/api/samples");
    const samples = await response.json();
    els.sampleSelect.innerHTML = samples.length
      ? samples.map((name) => `<option value="${name}">${name}</option>`).join("")
      : '<option value="">No samples found</option>';
  } catch {
    els.sampleSelect.innerHTML = '<option value="">Sample API unavailable</option>';
  }
}

async function onLoadSample() {
  const sample = els.sampleSelect.value;
  if (!sample) return;
  const response = await fetch(`/samples/${sample}`);
  const session = await response.json();
  loadSession(session, sample);
}

async function onFileSelected(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  const session = JSON.parse(text);
  loadSession(session, file.name);
}

function loadSession(session, sourceLabel) {
  state.session = session;
  state.stepIndex = 0;
  state.sourceLabel = sourceLabel;
  stopAutoplay();
  els.sessionTitle.textContent = session.title || session.session_id || "KF Session";
  els.sessionSubtitle.textContent = `${session.session_type || "session"} · ${session.session_id || "-"}`;
  els.rendererValue.textContent = session.renderer || "-";
  showStep(0);
}

function currentStep() {
  return state.session?.steps?.[state.stepIndex] || null;
}

function showStep(index) {
  if (!state.session) return;
  const maxIndex = state.session.steps.length - 1;
  state.stepIndex = Math.max(0, Math.min(index, maxIndex));
  renderCurrentStep();
}

function renderCurrentStep() {
  const step = currentStep();
  if (!step) return;
  const prevStep = state.stepIndex > 0 ? state.session.steps[state.stepIndex - 1] : null;
  const renderer = renderers[state.session.renderer];
  if (!renderer) {
    els.statusText.textContent = `Unknown renderer: ${state.session.renderer}`;
    return;
  }

  renderer({ ctx, canvas: els.canvas, step, zoom: state.zoom, showDiff: els.diffToggle.checked });

  const entities = mergeEntityBundle(state.session, step);
  const prevEntities = prevStep ? mergeEntityBundle(state.session, prevStep) : emptyEntityBundle();

  setField(els.statusText, `Frame ${state.stepIndex + 1}/${state.session.steps.length}`, null, "Current playback position within the loaded session.");
  setField(els.stepValue, `${step.index}`, prevStep ? `${prevStep.index}` : null, "Current session step index.");
  setField(els.actionValue, step.action?.label || step.action?.type || "-", prevStep ? (prevStep.action?.label || prevStep.action?.type || "-") : null, "Action taken at this step.");
  setField(els.stateValue, step.state?.status || "-", prevStep ? (prevStep.state?.status || "-") : null, "High-level runtime state reported by the session.");
  setField(els.levelsValue, `${step.state?.levels_completed ?? "-"} / ${step.state?.win_levels ?? "-"}`, prevStep ? `${prevStep.state?.levels_completed ?? "-"} / ${prevStep.state?.win_levels ?? "-"}` : null, "Current level progress summary.");
  setField(els.componentsValue, summarizeArcComponents(step.transition?.diff?.components || []), prevStep ? summarizeArcComponents(prevStep.transition?.diff?.components || []) : null, "Connected diff components detected between the previous and current step.");
  setField(els.sourceValue, state.sourceLabel || "-", null, "Source sample or uploaded session file currently being viewed.");
  setField(els.notesValue, step.commentary?.note || "-", prevStep ? (prevStep.commentary?.note || "-") : null, "Step commentary or player note.");
  setField(els.stateSnapshotValue, JSON.stringify(step.state || {}, null, 2), prevStep ? JSON.stringify(prevStep.state || {}, null, 2) : null, "Detailed state snapshot for this step.");
  setField(els.changeTypesValue, JSON.stringify(step.transition?.diff?.change_types || {}, null, 2), prevStep ? JSON.stringify(prevStep.transition?.diff?.change_types || {}, null, 2) : null, "Grouped transition counts by before/after value.");
  setField(els.metadataValue, JSON.stringify(step.metadata || {}, null, 2), prevStep ? JSON.stringify(prevStep.metadata || {}, null, 2) : null, "Additional metadata emitted for this step.");
  renderEntityLists(entities, prevEntities);
}

function mergeEntityBundle(session, step) {
  const categories = ["rules", "tools", "states", "goals"];
  const sessionEntities = normalizeEntitySource(session.entities);
  const stepEntities = normalizeEntitySource(step.entities);
  const merged = {};
  for (const category of categories) {
    const base = [...sessionEntities[category], ...stepEntities[category]];
    if (category === "goals") {
      base.push(...normalizeEntityItems(session.goals));
      base.push(...normalizeEntityItems(step.goals));
    }
    merged[category] = dedupeEntities(base);
  }
  return merged;
}

function normalizeEntitySource(source) {
  return {
    rules: normalizeEntityItems(source?.rules),
    tools: normalizeEntityItems(source?.tools),
    states: normalizeEntityItems(source?.states),
    goals: normalizeEntityItems(source?.goals),
  };
}

function normalizeEntityItems(items) {
  if (!Array.isArray(items)) return [];
  return items.map((item, index) => {
    if (typeof item === "string") {
      return {
        id: `item-${index}-${item}`,
        label: item,
        description: "",
        status: "",
        progress: null,
        tags: [],
      };
    }
    return {
      id: item.id || `item-${index}-${item.label || item.name || item.title || "entity"}`,
      label: item.label || item.name || item.title || item.id || "Unnamed entity",
      description: item.description || "",
      status: item.status || "",
      progress: item.progress || null,
      tags: item.tags || [],
    };
  });
}

function dedupeEntities(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const key = item.id || item.label;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

function emptyEntityBundle() {
  return { rules: [], tools: [], states: [], goals: [] };
}

function renderEntityLists(bundle, prevBundle) {
  renderEntityList(els.rulesList, bundle.rules, prevBundle.rules, "No rules in this session step", "Rules currently available in the session or activated at this step.");
  renderEntityList(els.toolsList, bundle.tools, prevBundle.tools, "No tools in this session step", "Tools currently available in the session or activated at this step.");
  renderEntityList(els.statesList, bundle.states, prevBundle.states, "No interpreted states in this session step", "Interpreted Knowledge Fabric states relevant to this step.");
  renderEntityList(els.goalsList, bundle.goals, prevBundle.goals, "No goals in this session step", "Goals and goal progress visible at this step.");
  setField(
    els.entityCountsValue,
    `rules ${bundle.rules.length} · tools ${bundle.tools.length} · states ${bundle.states.length} · goals ${bundle.goals.length}`,
    prevBundle ? `rules ${prevBundle.rules.length} · tools ${prevBundle.tools.length} · states ${prevBundle.states.length} · goals ${prevBundle.goals.length}` : null,
    "Counts of currently visible Knowledge Fabric entities by category."
  );
}

function renderEntityList(target, items, prevItems, emptyText, detail) {
  if (!items.length) {
    target.className = "entity-list empty";
    target.textContent = emptyText;
    target.title = detail;
    return;
  }
  target.className = "entity-list";
  target.title = detail;
  const prevMap = new Map(prevItems.map((item) => [item.id || item.label, JSON.stringify(item)]));
  target.innerHTML = items.map((item) => renderEntityCard(item, prevMap.get(item.id || item.label))).join("");
}

function renderEntityCard(item, previousSignature) {
  const changed = previousSignature !== JSON.stringify(item);
  const statusClass = item.status ? `entity-status ${String(item.status).toLowerCase()}` : "entity-status";
  const statusHtml = item.status ? `<span class="${statusClass}">${escapeHtml(item.status)}</span>` : "";
  const descriptionHtml = item.description ? `<div class="entity-description">${escapeHtml(item.description)}</div>` : "";
  const progressHtml = item.progress && Number.isFinite(item.progress.total) && item.progress.total > 0
    ? `<div class="progress"><div class="progress-bar"><div class="progress-fill" style="width:${Math.max(0, Math.min(100, (item.progress.value / item.progress.total) * 100))}%"></div></div><div class="progress-label">${item.progress.value}/${item.progress.total}</div></div>`
    : "";
  const tagsHtml = item.tags?.length ? `<div class="tag-row">${item.tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>` : "";
  const tooltip = [
    item.label,
    item.description || "",
    item.status ? `status: ${item.status}` : "",
    item.progress && Number.isFinite(item.progress.total) ? `progress: ${item.progress.value}/${item.progress.total}` : "",
    item.tags?.length ? `tags: ${item.tags.join(", ")}` : "",
  ].filter(Boolean).join("\n");
  return `<div class="entity-card${changed ? " recent-change" : ""}" title="${escapeHtml(tooltip)}"><div class="entity-head"><div class="entity-label">${escapeHtml(item.label)}</div>${statusHtml}</div>${descriptionHtml}${progressHtml}${tagsHtml}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setField(element, text, previousText, detail) {
  element.textContent = text;
  element.title = detail ? `${detail}\n\nCurrent value:\n${text}` : text;
  triggerRecent(element, previousText !== null && previousText !== text);
}

function triggerRecent(element, changed) {
  element.classList.remove("recent-change");
  if (!changed) return;
  void element.offsetWidth;
  element.classList.add("recent-change");
}

function toggleAutoplay() {
  if (!state.session) return;
  if (state.autoplayId) {
    stopAutoplay();
    return;
  }
  els.playBtn.textContent = "Pause";
  state.autoplayId = window.setInterval(() => {
    if (state.stepIndex >= state.session.steps.length - 1) {
      stopAutoplay();
      return;
    }
    showStep(state.stepIndex + 1);
  }, 700);
}

function stopAutoplay() {
  if (state.autoplayId) {
    window.clearInterval(state.autoplayId);
    state.autoplayId = null;
  }
  els.playBtn.textContent = "Play";
}

init();
