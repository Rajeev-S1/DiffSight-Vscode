import React, { useEffect, useMemo, useRef, useState } from "react";
import { startCompare, pollJob, getReport, evidenceUrl, downloadUrls, deleteJob } from "./api.js";

const ACCEPT = ".pdf,.png,.jpg,.jpeg,.docx,.doc";
const OK_EXT = ["pdf", "png", "jpg", "jpeg", "docx", "doc"];
const fmtSize = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
const extOf = (name) => (name.split(".").pop() || "").toLowerCase();

/* ---------------- Icons: static and decorative ---------------- */
const ICONS = {
  scale: '<path d="M12 3v18M7 21h10M5 7h14"/><path d="M5 7l-3 7a3.5 3.5 0 0 0 6 0zM19 7l-3 7a3.5 3.5 0 0 0 6 0z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  cog: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
  help: '<path d="M4 14v-2a8 8 0 0 1 16 0v2"/><rect x="3" y="14" width="4" height="6" rx="1.5"/><rect x="17" y="14" width="4" height="6" rx="1.5"/><path d="M19 20c0 1.5-2 2-5 2"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/>',
  bell: '<path d="M18 16v-5a6 6 0 0 0-12 0v5l-2 2h16z"/><path d="M10 21a2 2 0 0 0 4 0"/>',
  chev: '<path d="M6 9l6 6 6-6"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
};
const Ico = ({ n }) => (
  <svg className="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" dangerouslySetInnerHTML={{ __html: ICONS[n] }} />
);

/* ---------------- Page frame: sidebar + top bar ---------------- */
const NAV = [["clock", "History"], ["cog", "Settings"], ["help", "Help & Support"], ["info", "About"]];

function Shell({ dark, setDark, onNew, children }) {
  const placeholder = (e) => e.preventDefault();      // History, Settings, Help, About, search, notifications and profile: static for now
  return (
    <div className="app">
      <aside className="sidenav">
        <div className="brandblock"><span className="logo">D</span><div><b>DiffSight</b><small>Every detail. Every difference.</small></div></div>
        <nav aria-label="Main">
          <button className="nav on" aria-current="page" onClick={onNew}><Ico n="scale" /><span>Compare Documents</span></button>
          {NAV.map(([i, l]) => <button key={l} className="nav" title="Coming soon" onClick={placeholder}><Ico n={i} /><span>{l}</span></button>)}
        </nav>
        <svg className="waves" viewBox="0 0 285 300" preserveAspectRatio="none" aria-hidden="true">
          <defs><linearGradient id="wg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#22e0c5" stopOpacity=".55" /><stop offset="1" stopColor="#1fb6f2" stopOpacity=".05" /></linearGradient></defs>
          <path d="M0 120C60 90 110 170 175 200S255 230 285 205V300H0Z" fill="url(#wg)" />
          <path d="M0 175C55 150 120 215 190 240S260 262 285 250V300H0Z" fill="url(#wg)" opacity=".6" />
        </svg>
        <div className="sidefoot"><Ico n="bolt" /><span>Powered by DiffSight AI</span></div>
      </aside>
      <div className="mainwrap">
        <header className="topbar">
          <div className="gsearch"><Ico n="search" /><input readOnly placeholder="Search..." aria-label="Search (coming soon)" title="Coming soon" /></div>
          <div className="tools">
            <button className="pillbtn" onClick={() => setDark(!dark)} aria-label="Toggle theme"><Ico n={dark ? "sun" : "moon"} />{dark ? "Light" : "Dark"}</button>
            <button className="iconbtn" title="Coming soon" aria-label="Notifications (coming soon)" onClick={placeholder}><Ico n="bell" /><i className="dot" /></button>
            <button className="profile" title="Coming soon" aria-label="Profile (coming soon)" onClick={placeholder}><span className="avatar"><Ico n="user" /></span><Ico n="chev" /></button>
          </div>
        </header>
        <main className="page">
          {children}
          <p className="rule"><span>Results are evidence for review, not a legal guarantee; check uncertain items manually.</span></p>
        </main>
      </div>
    </div>
  );
}

/* ---------------- Upload ---------------- */
const Illustration = () => (
  <svg className="illus" viewBox="0 0 480 250" aria-hidden="true">
    <defs>
      <linearGradient id="dg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#2c7fd0" /><stop offset="1" stopColor="#12507c" /></linearGradient>
      <radialGradient id="gl" cx="50%" cy="50%" r="50%"><stop offset="0" stopColor="#22e0c5" stopOpacity=".35" /><stop offset="1" stopColor="#22e0c5" stopOpacity="0" /></radialGradient>
    </defs>
    <ellipse cx="240" cy="205" rx="228" ry="40" fill="url(#gl)" stroke="#22e0c5" strokeOpacity=".55" />
    <g transform="rotate(-6 130 120)">
      <path d="M62 38h96l34 34v112a8 8 0 0 1-8 8H62a8 8 0 0 1-8-8V46a8 8 0 0 1 8-8z" fill="url(#dg)" stroke="#6fe3f7" strokeOpacity=".7" />
      <rect x="76" y="70" width="88" height="9" rx="4" fill="#8fd7ff" /><rect x="76" y="92" width="100" height="9" rx="4" fill="#f5b04c" /><rect x="76" y="114" width="84" height="9" rx="4" fill="#22e0c5" /><rect x="76" y="136" width="96" height="9" rx="4" fill="#8fd7ff" /><rect x="76" y="158" width="60" height="9" rx="4" fill="#8fd7ff" />
    </g>
    <g transform="rotate(7 350 120)">
      <path d="M292 30h96l34 34v112a8 8 0 0 1-8 8H292a8 8 0 0 1-8-8V38a8 8 0 0 1 8-8z" fill="url(#dg)" stroke="#6fe3f7" strokeOpacity=".7" />
      <rect x="306" y="62" width="88" height="9" rx="4" fill="#8fd7ff" /><rect x="306" y="84" width="100" height="9" rx="4" fill="#22e0c5" /><rect x="306" y="106" width="84" height="9" rx="4" fill="#ff8a4c" /><rect x="306" y="128" width="96" height="9" rx="4" fill="#22e0c5" /><rect x="306" y="150" width="60" height="9" rx="4" fill="#8fd7ff" />
    </g>
    <circle cx="240" cy="118" r="36" fill="#0c5f66" stroke="#22e0c5" strokeWidth="2.5" />
    <path d="M222 110h34m-8-8l8 8-8 8M258 128h-34m8-8l-8 8 8 8" fill="none" stroke="#fff" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const FileIcon = () => (
  <svg className="fileico" viewBox="0 0 120 92" aria-hidden="true">
    <ellipse cx="60" cy="70" rx="54" ry="10" fill="none" stroke="#22e0c5" strokeOpacity=".5" />
    <ellipse cx="60" cy="70" rx="36" ry="6" fill="#22e0c5" fillOpacity=".15" />
    <path d="M40 10h26l18 18v40a4 4 0 0 1-4 4H40a4 4 0 0 1-4-4V14a4 4 0 0 1 4-4z" fill="none" stroke="#1fb6f2" strokeWidth="4" strokeLinejoin="round" />
    <path d="M66 10v18h18" fill="none" stroke="#1fb6f2" strokeWidth="4" strokeLinejoin="round" />
    <path d="M60 60V38M50 48l10-10 10 10" fill="none" stroke="#1fb6f2" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

function DropZone({ id, step, label, file, onFile }) {
  const [over, setOver] = useState(false);
  const [err, setErr] = useState("");
  const inputRef = useRef(null);
  const take = (f) => {
    if (!f) return;
    if (!OK_EXT.includes(extOf(f.name))) { setErr("Supported formats: PDF, JPG, PNG and Word (.docx, .doc)."); return; }
    setErr(""); onFile(f);
  };
  return (
    <div className={`slot ${over ? "over" : ""} ${file ? "has" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}>
      <span className="num">{step}</span>
      <FileIcon />
      <label htmlFor={id} className="slot-title">{label}</label>
      <input id={id} ref={inputRef} type="file" accept={ACCEPT} hidden onChange={(e) => take(e.target.files[0])} />
      {file ? <p className="fname"><b>{file.name}</b><span>{fmtSize(file.size)}</span></p> : <p className="drag">Drag a file here</p>}
      <button type="button" className="choose" onClick={() => inputRef.current.click()}><Ico n="folder" />{file ? "Change file" : "Choose file"}</button>
      {err && <p className="err" role="alert">{err}</p>}
    </div>
  );
}

function UploadScreen({ onStart, busy, error }) {
  const [a, setA] = useState(null);
  const [b, setB] = useState(null);
  return (
    <>
      <section className="hero">
        <div className="herotext">
          <span className="aibadge"><Ico n="spark" />AI-Powered Document Intelligence</span>
          <h1>Compare <span className="hl">two</span> documents</h1>
          <p className="lead">Upload the original and the revised file. Every difference is listed row by row, with its location and evidence.</p>
        </div>
        <Illustration />
      </section>
      <div className="pair">
        <DropZone id="fa" step="1" label="Original" file={a} onFile={setA} />
        <DropZone id="fb" step="2" label="Revised" file={b} onFile={setB} />
      </div>
      <p className="infoline" data-testid="formats"><Ico n="info" />Supported formats: PDF, JPG, PNG and Microsoft Word (.docx, .doc). DiffSight chooses the best scan resolution automatically.</p>
      {error && <p className="err center" role="alert">{error}</p>}
      <div className="cta"><button className="compare" disabled={!a || !b || busy} onClick={() => onStart(a, b)}>{busy ? "Uploading…" : <><Ico n="scale" />Compare<Ico n="arrow" /></>}</button></div>
    </>
  );
}

/* ---------------- Progress ---------------- */
function ProgressScreen({ job, error, onRetry }) {
  const pct = Math.round((job?.fraction ?? 0) * 100);
  return (
    <section className="glass narrow">
      <h1>{error ? "Comparison failed" : "Comparing…"}</h1>
      {error ? (
        <>
          <p className="err" role="alert">{error}</p>
          <button className="btn primary big" onClick={onRetry}>Try again</button>
        </>
      ) : (
        <>
          <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}><div className="bar-fill" style={{ width: `${Math.min(100, pct)}%` }} /></div>
          <p className="muted">{job?.message || "Queued…"} {pct ? `(${pct}%)` : ""}</p>
        </>
      )}
    </section>
  );
}

/* ---------------- Evidence viewer ---------------- */
function Viewer({ jobId, ev, onClose }) {
  const [mode, setMode] = useState("side");
  const [pos, setPos] = useState(50);
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const A = ev.a && evidenceUrl(jobId, ev.a), B = ev.b && evidenceUrl(jobId, ev.b), O = ev.overlay && evidenceUrl(jobId, ev.overlay);
  return (
    <div className="modal" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="modal-body" onClick={(e) => e.stopPropagation()}>
        <div className="row between">
          <div className="tabs">
            {[["side", "Side by side"], ["slider", "Overlay slider"], ["overlay", "Highlighted"]].map(([k, l]) => (
              <button key={k} className={`tab ${mode === k ? "on" : ""}`} onClick={() => setMode(k)}>{l}</button>
            ))}
          </div>
          <button className="btn ghost" onClick={onClose}>Close ✕</button>
        </div>
        {mode === "side" && (
          <div className="side">
            {A && <figure><figcaption>A · Original</figcaption><img src={A} alt="Document A evidence" /></figure>}
            {B && <figure><figcaption>B · Modified</figcaption><img src={B} alt="Document B evidence" /></figure>}
          </div>
        )}
        {mode === "slider" && A && B && (
          <div className="slider">
            <div className="stack">
              <img src={B} alt="Document B" />
              <div className="clip" style={{ width: `${pos}%` }}><img src={A} alt="Document A" /></div>
            </div>
            <input type="range" min="0" max="100" value={pos} onChange={(e) => setPos(+e.target.value)} aria-label="Compare A and B" />
            <p className="muted">Left of the slider: A · Right: B</p>
          </div>
        )}
        {mode === "overlay" && (O ? <img className="full" src={O} alt="Highlighted differences" /> : <p className="muted">No overlay for this finding.</p>)}
      </div>
    </div>
  );
}

/* ---------------- Differences: one row per difference ---------------- */
const r0 = (v) => { const f = Math.floor(v), d = v - f; return d === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(v); };   // same rounding as the HTML report
const locText = (d) => (d && d.bbox_mm && d.bbox_px
  ? `x ${r0(d.bbox_mm[0])}-${r0(d.bbox_mm[2])} mm, y ${r0(d.bbox_mm[1])}-${r0(d.bbox_mm[3])} mm from top-left (px ${d.bbox_px.map(Math.trunc).join(", ")})` : "n/a");
const pageText = (f) => (f.page_b == null || f.page_a === f.page_b ? (f.page_a ?? f.page) : `${f.page_a} (A) / ${f.page_b} (B)`);

function DiffTable({ rows, jobId, onZoom }) {
  return (
    <div className="tablewrap difftable">
      <table>
        <thead>
          <tr><th>Type</th><th>Page</th><th>Document A</th><th>Document B</th><th>Difference</th><th>Location (A)</th><th>Location (B)</th><th>Confidence</th><th>Evidence</th></tr>
        </thead>
        <tbody>
          {rows.map((f) => {
            const ev = f.evidence || {};
            const has = ev.overlay || ev.b || ev.a;
            const cat = f.details?.original_category ? `${f.category} · ${f.details.original_category}` : f.category;
            return (
              <tr key={`${f.number}-${f.type}-${f.page_a}`}>
                <td className="c-type"><b>{f.number}. {f.type}</b><span className={`badge cat-${f.category}`}>{cat}</span></td>
                <td className="c-page">{pageText(f)}</td>
                <td className="c-doc">{f.document_a || <em>—</em>}</td>
                <td className="c-doc">{f.document_b || <em>—</em>}</td>
                <td className="c-diff">{f.difference}
                  {f.details?.note && <small className="note">{f.details.note}</small>}
                  {f.details?.quality_note && <small className="note">{f.details.quality_note}</small>}</td>
                <td className="c-loc">{locText(f.location?.a)}</td>
                <td className="c-loc">{locText(f.location?.b)}</td>
                <td className="c-conf"><b className={f.confidence >= 0.9 ? "hi" : f.confidence >= 0.7 ? "mid" : "lo"}>{Math.round((f.confidence || 0) * 100)}%</b>
                  <small>({(f.engines || []).join(", ") || "—"})</small></td>
                <td className="c-ev">{has ? <button className="btn evbtn" onClick={() => onZoom(ev)}>View evidence</button> : <span className="muted">—</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- Automatic scan and resolution block ---------------- */
const dash = (v, unit = "") => (v === null || v === undefined ? "—" : `${v}${unit}`);

function ScanInfo({ sel }) {
  const retryText = !sel.retried ? "No (not needed)" : sel.retry_result === "failed" ? "Yes – could not be completed" : "Yes";
  const retryResult = !sel.retried ? "—" : sel.retry_result === "improved" ? "Improved" : sel.retry_result === "failed" ? "Failed" : "Not improved";
  const rows = [
    ["Source type", sel.source_type],
    ["Sharpness", sel.sharpness === null || sel.sharpness === undefined ? "—" : `${sel.sharpness} (${sel.sharpness_label})`],
    ["Noise", sel.noise === null || sel.noise === undefined ? "—" : `${sel.noise} (${sel.noise_label})`],
    ["Contrast", sel.contrast === null || sel.contrast === undefined ? "—" : `${sel.contrast} (${sel.contrast_label})`],
    ["Tilt", sel.tilt === null || sel.tilt === undefined ? "—" : `${sel.tilt}° (${sel.tilt_label})`],
    ["Source resolution", sel.source_dpi ? `${sel.source_dpi} DPI` : sel.source_dpi_status],
    ["Selected DPI", sel.dpi_applicable === false ? "Not applicable (images keep their own pixel size)" : `${sel.selected_dpi ?? sel.used} DPI`],
    ["Selection reason", sel.reason],
    ["Retry performed", retryText],
    ["Retry result", retryResult],
    ["Selected comparison", sel.selected_comparison || "Original-resolution result"],
  ];
  return (
    <section className="card scaninfo" data-testid="dpi-note">
      <h3>Scan &amp; resolution (chosen automatically)</h3>
      <p className="diff">{sel.message}</p>
      {sel.retried && sel.retry_reason && <p className="muted">{sel.retry_result === "failed" ? "Re-check failed: " : "Re-check: "}{sel.retry_reason}.</p>}
      {(sel.notes || []).map((n, i) => <p key={i} className="note">{n}</p>)}
      <div className="tablewrap">
        <table><tbody>{rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{dash(v)}</td></tr>)}</tbody></table>
      </div>
    </section>
  );
}

/* ---------------- Results ---------------- */
function Results({ jobId, report, onReset }) {
  const [tab, setTab] = useState("confirmed");
  const [cats, setCats] = useState(new Set());
  const [types, setTypes] = useState(new Set());
  const [page, setPage] = useState("all");
  const [q, setQ] = useState("");
  const [zoom, setZoom] = useState(null);
  const confirmed = report.differences || [];
  const uncertain = report.uncertain || [];
  const hr = report.seen_only_at_higher_resolution || [];
  const list = tab === "confirmed" ? confirmed : tab === "uncertain" ? uncertain : hr;
  const dl = downloadUrls(jobId);
  const s = report.summary || {};
  const sel = report.diagnostics?.dpi_selection;

  const allCats = useMemo(() => [...new Set(list.map((f) => f.category))], [list]);
  const allTypes = useMemo(() => [...new Set(list.map((f) => f.type))], [list]);
  const pages = useMemo(() => [...new Set(list.map((f) => f.page))].sort((x, y) => x - y), [list]);
  const shown = list.filter((f) =>
    (!cats.size || cats.has(f.category)) && (!types.size || types.has(f.type)) &&
    (page === "all" || String(f.page) === page) &&
    (!q || `${f.difference} ${f.document_a} ${f.document_b}`.toLowerCase().includes(q.toLowerCase())));
  const toggle = (set, setter, v) => { const n = new Set(set); n.has(v) ? n.delete(v) : n.add(v); setter(n); };
  const pick = (t) => { setTab(t); setCats(new Set()); setTypes(new Set()); setPage("all"); };

  return (
    <>
      <section className="resultbar">
        <div>
          <h1>Differences</h1>
          <p className="pills">
            <span className="pill ok">{s.total_differences ?? confirmed.length} confirmed</span>
            <span className="pill warn">{s.uncertain_findings ?? uncertain.length} uncertain</span>
            {Object.entries(s.by_category || {}).map(([k, v]) => <span key={k} className="pill">{v} {k.toLowerCase()}</span>)}
          </p>
        </div>
        <div className="row wrap">
          <button className="btn primary" onClick={onReset}>+ New comparison</button>
          <a className="btn" href={dl.html} target="_blank" rel="noopener">HTML report</a>
          <a className="btn" href={dl.single} download>Single-file HTML</a>
          <a className="btn" href={dl.text} download>Text</a>
          <a className="btn" href={dl.json} download="diffsight_report.json">JSON</a>
        </div>
      </section>

      {sel && <details className="scan-details"><summary>Scan &amp; resolution (chosen automatically)</summary><ScanInfo sel={sel} /></details>}

      <section className="panel">
        <div className="row between wrap">
          <div className="tabs">
            <button className={`tab ${tab === "confirmed" ? "on" : ""}`} onClick={() => pick("confirmed")}>Confirmed ({confirmed.length})</button>
            <button className={`tab ${tab === "uncertain" ? "on" : ""}`} onClick={() => pick("uncertain")}>Uncertain ({uncertain.length})</button>
            {hr.length > 0 && <button className={`tab ${tab === "hr" ? "on" : ""}`} onClick={() => pick("hr")}>Seen only at higher resolution ({hr.length})</button>}
          </div>
          <div className="filters">
            <input className="search" type="search" placeholder="Search differences…" aria-label="Search differences" value={q} onChange={(e) => setQ(e.target.value)} />
            <label className="sel">Page
              <select value={page} onChange={(e) => setPage(e.target.value)}>
                <option value="all">All</option>{pages.map((p) => <option key={p} value={String(p)}>{p}</option>)}
              </select>
            </label>
          </div>
        </div>
        {tab === "hr" && <p className="note">These appeared only in the higher-resolution re-check, not in the original run. They are listed for manual review and are not counted as independently verified changes.</p>}
        {tab === "uncertain" && <p className="note">Uncertain findings are possible differences that could not be confirmed. Check them manually; they are not counted as confirmed.</p>}
        <div className="chips" aria-label="Filters">
          {allCats.map((c) => <button key={c} className={`chip ${cats.has(c) ? "on" : ""}`} onClick={() => toggle(cats, setCats, c)}>{c}</button>)}
          {allTypes.map((t) => <button key={t} className={`chip alt ${types.has(t) ? "on" : ""}`} onClick={() => toggle(types, setTypes, t)}>{t}</button>)}
        </div>
        {shown.length === 0
          ? <p className="muted empty">{list.length === 0 ? (tab === "confirmed" ? "No confirmed differences were found." : tab === "hr" ? "Nothing appeared only at higher resolution." : "No uncertain findings.") : "No differences match the filters."}</p>
          : <DiffTable rows={shown} jobId={jobId} onZoom={setZoom} />}
      </section>

      {(report.verified_unchanged || []).length > 0 && (
        <details className="scan-details">
          <summary>Verified unchanged content</summary>
          <p className="muted pad">How much text matched exactly on each page pair — proof of what did <em>not</em> change.</p>
          <div className="tablewrap">
            <table>
              <thead><tr><th>Page A</th><th>Page B</th><th>Type</th><th>Lines matched</th><th>Words matched</th><th>Identical</th></tr></thead>
              <tbody>
                {report.verified_unchanged.map((v, i) => (
                  <tr key={i}>
                    <td>{v.page_a ?? "—"}</td><td>{v.page_b ?? "—"}</td><td>{v.kind}</td>
                    <td>{v.compared ? `${v.matched_lines}/${Math.max(v.lines_a, v.lines_b)}` : "—"}</td>
                    <td>{v.compared ? `${v.matched_words}/${Math.max(v.words_a, v.words_b)}` : "—"}</td>
                    <td>{v.compared ? `${v.percent_identical}%` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      <div className="row"><button className="btn primary" onClick={onReset}>New comparison</button></div>
      {zoom && <Viewer jobId={jobId} ev={zoom} onClose={() => setZoom(null)} />}
    </>
  );
}

/* ---------------- App ---------------- */
export default function App() {
  const [stage, setStage] = useState("upload"); // upload | running | results
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dark, setDark] = useState(() => (localStorage.getItem("nexascan-theme") ? localStorage.getItem("nexascan-theme") === "dark" : true));   // dark by default
  const cancel = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try { localStorage.setItem("nexascan-theme", dark ? "dark" : "light"); } catch { /* ignore */ }
  }, [dark]);
  useEffect(() => () => cancel.current && cancel.current(), []);

  const reset = () => { cancel.current && cancel.current(); if (jobId) deleteJob(jobId); setStage("upload"); setJob(null); setReport(null); setError(""); setJobId(null); };

  const start = async (a, b) => {
    setBusy(true); setError("");
    try {
      const r = await startCompare(a, b);
      setJobId(r.job_id); setJob(null); setStage("running");
      cancel.current = pollJob(r.job_id, setJob,
        async () => { try { setReport(await getReport(r.job_id)); setStage("results"); } catch (e) { setError(e.message); } },
        (e) => setError(e.message));
    } catch (e) { setError(e.message); }
    setBusy(false);
  };

  return (
    <Shell dark={dark} setDark={setDark} onNew={reset}>
      {stage === "upload" && <UploadScreen onStart={start} busy={busy} error={error} />}
      {stage === "running" && <ProgressScreen job={job} error={error} onRetry={reset} />}
      {stage === "results" && report && <Results jobId={jobId} report={report} onReset={reset} />}
    </Shell>
  );
}
