// All calls use relative paths: in dev Vite proxies /api, in production FastAPI serves both on one port.
const POLL_MS = 3000;
const MAX_TOTAL_MS = 15 * 60 * 1000;
const MAX_QUEUED_MS = 2 * 60 * 1000;

async function asJson(r) {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

export function startCompare(fileA, fileB) {
  const fd = new FormData();
  fd.append("a", fileA);
  fd.append("b", fileB);
  return fetch("/api/compare", { method: "POST", body: fd }).then(asJson);
}

export const getJob = (id) => fetch(`/api/jobs/${id}`).then(asJson);
export const getReport = (id) => fetch(`/api/jobs/${id}/report`).then(asJson);
export const evidenceUrl = (id, rel) => `/api/jobs/${id}/${rel}`;
export const downloadUrls = (id) => ({
  html: `/api/jobs/${id}/report.html`,
  single: `/api/jobs/${id}/report_embedded.html`,
  text: `/api/jobs/${id}/report.txt`,
  json: `/api/jobs/${id}/report`,
});
export const deleteJob = (id) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).catch(() => {});

// Polls until done / error / timeout. Returns a cancel function.
export function pollJob(id, onUpdate, onDone, onError) {
  const t0 = Date.now();
  let queuedSince = null;
  let stopped = false;
  let timer = null;
  const tick = async () => {
    if (stopped) return;
    try {
      const j = await getJob(id);
      if (stopped) return;
      onUpdate(j);
      if (j.status === "done") return onDone(j);
      if (j.status === "error") return onError(new Error(j.message || "The comparison failed."));
      if (j.status === "queued") {
        queuedSince = queuedSince ?? Date.now();
        if (Date.now() - queuedSince > MAX_QUEUED_MS) return onError(new Error("The job stayed queued for 2 minutes. The worker may have stopped; please try again."));
      } else queuedSince = null;
      if (Date.now() - t0 > MAX_TOTAL_MS) return onError(new Error("Timed out after 15 minutes."));
      timer = setTimeout(tick, POLL_MS);
    } catch (e) {
      if (!stopped) onError(e);
    }
  };
  tick();
  return () => { stopped = true; if (timer) clearTimeout(timer); };
}
