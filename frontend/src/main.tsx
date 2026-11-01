import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap-icons/font/bootstrap-icons.css";
import "@fontsource-variable/dm-sans";
import "@fontsource-variable/manrope";
import "./style.css";

type Obj = Record<string, any>;
type Case = Obj & {
  id: number;
  reference: string;
  subject: string;
  customer: string;
  message: string;
  status: string;
  channel: string;
  analysis: Obj;
  sample: boolean;
};
const title = (s: string) =>
  s.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
const date = (s: string) =>
  new Date(s).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
const pct = (n: number | undefined) =>
  n === undefined ? "—" : (n * 100).toFixed(1) + "%";
async function api(path: string, method = "GET", body?: unknown) {
  const res = await fetch("/api" + path, {
    method,
    headers:
      body instanceof FormData ? {} : { "Content-Type": "application/json" },
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  const value = await res.json();
  if (!res.ok)
    throw Error(
      typeof value.detail === "string"
        ? value.detail
        : JSON.stringify(value.detail),
    );
  return value;
}
function Icon({ name }: { name: string }) {
  return <i aria-hidden="true" className={"bi bi-" + name} />;
}
function Status({ value }: { value: string }) {
  return <span className={"status status-" + value}>{title(value)}</span>;
}
function Empty({
  icon,
  children,
}: {
  icon: string;
  children: React.ReactNode;
}) {
  return (
    <div className="empty">
      <Icon name={icon} />
      <p>{children}</p>
    </div>
  );
}
function App() {
  const [page, setPage] = useState("Overview"),
    [cases, setCases] = useState<Case[]>([]),
    [ops, setOps] = useState<Obj>({}),
    [policy, setPolicy] = useState<Obj>({ articles: [] }),
    [models, setModels] = useState<Obj>({ models: [] });
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(""),
    [modal, setModal] = useState(false),
    [selected, setSelected] = useState<Case | null>(null);
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all"),
    [draft, setDraft] = useState(""),
    [answer, setAnswer] = useState<Obj | null>(null),
    [question, setQuestion] = useState("");
  const [comparison, setComparison] = useState<Obj | null>(null),
    [compareText, setCompareText] = useState(
      "I would like to cancel my order before it ships.",
    );
  const [newCase, setNewCase] = useState({
    subject: "",
    customer: "",
    message: "",
    channel: "text",
  });
  const [recording, setRecording] = useState(false),
    [audioName, setAudioName] = useState("");
  const recorder = useRef<MediaRecorder | null>(null),
    chunks = useRef<Blob[]>([]),
    stream = useRef<MediaStream | null>(null),
    timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const msgRef = useRef<HTMLTextAreaElement | null>(null);
  async function refresh() {
    try {
      const [c, o, p, m] = await Promise.all([
        api("/tickets"),
        api("/operations"),
        api("/policies"),
        api("/models"),
      ]);
      setCases(c);
      setOps(o);
      setPolicy(p);
      setModels(m);
    } catch (e) {
      setError(String((e as Error).message));
    }
  }
  useEffect(() => {
    refresh();
    return () => {
      if (timer.current) clearTimeout(timer.current);
      stream.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);
  useEffect(() => {
    setDraft(selected?.approved_reply || selected?.analysis.reply || "");
  }, [selected]);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [page, selected?.id]);
  useEffect(() => {
    function esc(e: KeyboardEvent) {
      if (e.key === "Escape" && !busy && !recording) setModal(false);
    }
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [busy, recording]);
  async function act(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }
  async function openCase(c: Case) {
    await act("Opening case", async () => {
      setSelected(await api("/tickets/" + c.id));
      setPage("Cases");
    });
  }
  function create(voice = false) {
    setNewCase({
      subject: "",
      customer: "",
      message: "",
      channel: voice ? "voice" : "text",
    });
    setAudioName("");
    setModal(true);
  }
  async function uploadAudio(file: Blob, name: string) {
    await act("Transcribing audio", async () => {
      const form = new FormData();
      form.append("file", file, name);
      const r = await api("/transcribe", "POST", form);
      setNewCase((x) => ({ ...x, message: r.text, channel: "voice" }));
      setAudioName(name + " · " + r.duration_seconds + " sec");
      setNotice("Transcript ready. Review the text before creating the case.");
    });
  }
  async function toggleRecording() {
    if (recording) {
      recorder.current?.stop();
      setRecording(false);
      if (timer.current) clearTimeout(timer.current);
      return;
    }
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      recorder.current = new MediaRecorder(stream.current);
      chunks.current = [];
      recorder.current.ondataavailable = (e) => chunks.current.push(e.data);
      recorder.current.onstop = () => {
        const type = recorder.current?.mimeType || "audio/webm";
        const file = new Blob(chunks.current, { type });
        stream.current?.getTracks().forEach((t) => t.stop());
        uploadAudio(file, "voice-note.webm");
      };
      recorder.current.start();
      setRecording(true);
      timer.current = setTimeout(() => {
        recorder.current?.stop();
        setRecording(false);
      }, 175000);
    } catch (e) {
      setError(
        "Microphone access was unavailable. You can upload an audio file instead.",
      );
    }
  }
  const filtered = cases.filter(
    (c) =>
      (filter === "all" || c.status === filter) &&
      `${c.reference} ${c.subject} ${c.customer} ${c.message}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const review = cases.filter((c) => c.status === "review").length,
    total = cases.length,
    voice = cases.filter((c) => c.channel === "voice").length,
    resolved = cases.filter((c) => c.status === "resolved").length;
  const intentCounts: Obj = {};
  cases.forEach((c) => {
    const label = c.analysis.intent?.label;
    if (label) intentCounts[label] = (intentCounts[label] || 0) + 1;
  });
  const topIntents = Object.entries(intentCounts)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 5);
  const sentimentCounts = ["negative", "neutral", "positive"].map((label) => ({
    label,
    count: cases.filter((c) => c.analysis.sentiment?.label === label).length,
  }));
  const nav = [
    ["Overview", "grid-1x2"],
    ["Cases", "inbox"],
    ["Policy library", "journal-text"],
    ["Quality & models", "bar-chart-line"],
    ["Activity", "clock-history"],
  ];
  return (
    <div className="app-shell">
      <aside className="sidebar" inert={modal}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("Overview");
          }}
        >
          <span className="brand-mark">
            <Icon name="compass" />
          </span>
          <span>
            northstar<small>RETAIL SERVICE</small>
          </span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {nav.map(([label, icon]) => (
            <button
              key={label}
              aria-label={label}
              title={label}
              className={"nav-item " + (page === label ? "active" : "")}
              onClick={() => setPage(label)}
            >
              <Icon name={icon} />
              <span>{label}</span>
              {label === "Cases" && review > 0 && (
                <span className="nav-count">{review}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="connection-dot" />{" "}
          {models.environment === "local" ? "Local workspace" : "Lab workspace"}
          <p>Customer enquiries and policy review.</p>
          <div className="team">
            <span className="avatar">NS</span>
            <div>
              Service team<small>Group 15 · CCZG506</small>
            </div>
          </div>
        </div>
      </aside>
      <div className="main-wrap" inert={modal}>
        <header className="topbar">
          <div>
            <span className="muted">Workspace</span>
            <span className="breadcrumb-sep">/</span>
            {page}
          </div>
          <div className="topbar-right">
            <span className="outline-tag">
              <span className="connection-dot" />{" "}
              {models.models.length >= 6 &&
              models.models.every((m: Obj) => m.ready)
                ? "Models ready"
                : "Model setup"}
            </span>
            <button
              className="icon-btn"
              title="Refresh workspace"
              aria-label="Refresh workspace"
              onClick={refresh}
            >
              <Icon name="arrow-clockwise" />
            </button>
            <span className="avatar small-avatar">G15</span>
          </div>
        </header>
        <main>
          {error && (
            <div
              role="alert"
              className="alert alert-danger d-flex justify-content-between"
            >
              <span>
                <Icon name="exclamation-circle" /> {error}
              </span>
              <button
                aria-label="Dismiss error"
                className="btn-close"
                onClick={() => setError("")}
              />
            </div>
          )}
          {notice && (
            <div
              role="status"
              className="alert alert-success d-flex justify-content-between"
            >
              <span>{notice}</span>
              <button
                aria-label="Dismiss notification"
                className="btn-close"
                onClick={() => setNotice("")}
              />
            </div>
          )}
          <div className="page-heading">
            <div>
              <div className="eyebrow">CUSTOMER OPERATIONS</div>
              <h1>{page === "Overview" ? "Service overview" : page}</h1>
              <p>
                {page === "Overview"
                  ? "Every conversation, the right context, and a considered response."
                  : page === "Cases"
                    ? "Understand the issue. Review the evidence. Keep the customer informed."
                    : page === "Policy library"
                      ? "Published guidance with clear sources and room for human judgement."
                      : page === "Quality & models"
                        ? "Measured results, reproducible experiments, and visible limitations."
                        : "A record of case decisions and model operations."}
              </p>
            </div>
            {["Overview", "Cases"].includes(page) && (
              <button className="btn btn-primary" onClick={() => create()}>
                <Icon name="plus-lg" /> New case
              </button>
            )}
          </div>
          {busy && (
            <div className="working" role="status">
              <span className="spinner-border spinner-border-sm" /> {busy}…{" "}
              <span className="muted">
                Local models may take a moment to load.
              </span>
            </div>
          )}
          {page === "Overview" && (
            <>
              <section className="stat-grid" aria-label="Case statistics">
                {[
                  {
                    name: "Total cases",
                    value: total,
                    icon: "inbox",
                    note: "All customer enquiries",
                  },
                  {
                    name: "Awaiting review",
                    value: review,
                    icon: "clipboard-check",
                    note: "Drafts ready for an agent",
                  },
                  {
                    name: "Voice enquiries",
                    value: voice,
                    icon: "soundwave",
                    note: "Audio brought into the workflow",
                  },
                  {
                    name: "Resolved cases",
                    value: resolved,
                    icon: "check2-circle",
                    note: "Reviewed and closed",
                  },
                ].map((s) => (
                  <div className="stat-card" key={s.name}>
                    <div className="stat-top">
                      {s.name}
                      <span>
                        <Icon name={s.icon} />
                      </span>
                    </div>
                    <strong>{s.value}</strong>
                    <small>{s.note}</small>
                  </div>
                ))}
              </section>
              <section className="welcome-strip">
                <div>
                  <div className="eyebrow">FROM ENQUIRY TO RESOLUTION</div>
                  <h2>Turn enquiries into reviewed responses.</h2>
                  <p>
                    Bring in a message or voice note. Review the intent,
                    sentiment,
                    <br className="desktop-br" /> and policy evidence before
                    approving a response.
                  </p>
                  <div className="d-flex flex-wrap gap-2">
                    <button className="btn btn-dark" onClick={() => create()}>
                      <Icon name="chat-left-text" /> Add a message
                    </button>
                    <button
                      className="btn btn-outline-dark"
                      onClick={() => create(true)}
                    >
                      <Icon name="mic" /> Add voice enquiry
                    </button>
                  </div>
                </div>
                <div className="workflow-art" aria-hidden="true">
                  <div>
                    <Icon name="chat-left-text" />
                    <span>Receive</span>
                  </div>
                  <span className="workflow-line" />
                  <div>
                    <Icon name="journal-check" />
                    <span>Understand</span>
                  </div>
                  <span className="workflow-line" />
                  <div>
                    <Icon name="person-check" />
                    <span>Review</span>
                  </div>
                </div>
              </section>
              <div className="dashboard-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Recent enquiries</h2>
                      <p>The latest cases across your workspace</p>
                    </div>
                    <button
                      className="text-btn"
                      onClick={() => {
                        setSelected(null);
                        setPage("Cases");
                      }}
                    >
                      View all <Icon name="arrow-right" />
                    </button>
                  </div>
                  {cases.length ? (
                    <div className="table-responsive">
                      <table className="table case-table">
                        <thead>
                          <tr>
                            <th>CASE</th>
                            <th>CUSTOMER</th>
                            <th>STATUS</th>
                            <th />
                          </tr>
                        </thead>
                        <tbody>
                          {cases.slice(0, 5).map((c) => (
                            <tr key={c.id}>
                              <td>
                                <button
                                  className="case-link"
                                  onClick={() => openCase(c)}
                                >
                                  <small>{c.reference}</small>
                                  {c.subject}
                                </button>
                              </td>
                              <td>
                                {c.customer}
                                <small>
                                  <Icon
                                    name={
                                      c.channel === "voice"
                                        ? "mic"
                                        : "chat-left"
                                    }
                                  />{" "}
                                  {title(c.channel)} enquiry
                                </small>
                              </td>
                              <td>
                                <Status value={c.status} />
                              </td>
                              <td>
                                <button
                                  className="icon-btn"
                                  aria-label={"Open " + c.reference}
                                  onClick={() => openCase(c)}
                                >
                                  <Icon name="arrow-up-right" />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <Empty icon="inbox">
                      Your first customer enquiry will appear here.
                    </Empty>
                  )}
                </section>
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Customer sentiment</h2>
                      <p>Signals from analysed messages</p>
                    </div>
                    <Icon name="emoji-neutral" />
                  </div>
                  <div className="sentiment-bars">
                    {sentimentCounts.map((s) => (
                      <div key={s.label}>
                        <div className="d-flex justify-content-between">
                          <span>{title(s.label)}</span>
                          <strong>{s.count}</strong>
                        </div>
                        <div className="bar-track">
                          <div
                            className={"bar-fill " + s.label}
                            style={{
                              width:
                                Math.max(0, (s.count / (total || 1)) * 100) +
                                "%",
                            }}
                          />
                        </div>
                      </div>
                    ))}
                    <small className="muted">
                      A model estimate to support review; context still matters.
                    </small>
                  </div>
                  <div className="panel-divider" />
                  <div className="panel-heading">
                    <h2>Common intents</h2>
                  </div>
                  <div className="intent-list">
                    {topIntents.length ? (
                      topIntents.map(([name, count]) => (
                        <div key={name}>
                          <span>{title(name)}</span>
                          <span className="count-pill">{String(count)}</span>
                        </div>
                      ))
                    ) : (
                      <p className="muted">
                        Analyse cases to see request categories.
                      </p>
                    )}
                  </div>
                </section>
              </div>
            </>
          )}
          {page === "Cases" && !selected && (
            <section className="panel">
              <div className="case-toolbar">
                <div className="search-field">
                  <Icon name="search" />
                  <input
                    aria-label="Search cases"
                    placeholder="Search cases, customers, or messages"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
                <select
                  className="form-select"
                  aria-label="Filter case status"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  {[
                    "all",
                    "new",
                    "processing",
                    "review",
                    "approved",
                    "resolved",
                  ].map((x) => (
                    <option key={x} value={x}>
                      {x === "all" ? "All statuses" : title(x)}
                    </option>
                  ))}
                </select>
              </div>
              <div className="table-responsive">
                <table className="table case-table">
                  <thead>
                    <tr>
                      <th>CASE</th>
                      <th>CUSTOMER</th>
                      <th>INTENT</th>
                      <th>STATUS</th>
                      <th>RECEIVED</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((c) => (
                      <tr key={c.id}>
                        <td>
                          <button
                            className="case-link"
                            onClick={() => openCase(c)}
                          >
                            <small>
                              {c.reference}
                              {c.sample ? " · Sample case" : ""}
                            </small>
                            {c.subject}
                          </button>
                        </td>
                        <td>
                          {c.customer}
                          <small>{title(c.channel)}</small>
                        </td>
                        <td>
                          {title(
                            c.analysis.intent?.label || "Pending analysis",
                          )}
                        </td>
                        <td>
                          <Status value={c.status} />
                        </td>
                        <td>{date(c.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!filtered.length && (
                <Empty icon="search">No cases match this view.</Empty>
              )}
            </section>
          )}
          {page === "Cases" && selected && (
            <>
              <button
                className="text-btn mb-3"
                onClick={() => setSelected(null)}
              >
                <Icon name="arrow-left" /> Back to cases
              </button>
              <div className="detail-heading">
                <div>
                  <span className="eyebrow">
                    {selected.reference} · {selected.channel.toUpperCase()}{" "}
                    ENQUIRY {selected.sample ? "· SAMPLE CASE" : ""}
                  </span>
                  <h2>{selected.subject}</h2>
                  <p>
                    {selected.customer} · {date(selected.created_at)}
                  </p>
                </div>
                <Status value={selected.status} />
              </div>
              <div className="case-detail-grid">
                <div>
                  <section className="panel p-4 mb-4">
                    <div className="section-title">
                      <Icon name="chat-left-text" />
                      <h3>Customer message</h3>
                    </div>
                    <blockquote>{selected.message}</blockquote>
                    <button
                      className="btn btn-primary"
                      disabled={!!busy || selected.status === "resolved"}
                      onClick={() =>
                        act("Analysing case", async () => {
                          const r = await api(
                            `/tickets/${selected.id}/analyse`,
                            "POST",
                          );
                          setSelected(await api("/tickets/" + r.id));
                          await refresh();
                        })
                      }
                    >
                      <Icon name="layers" />{" "}
                      {selected.analysis.summary
                        ? "Run analysis again"
                        : "Analyse case"}
                    </button>
                  </section>
                  {selected.analysis.summary && (
                    <section className="panel p-4 mb-4">
                      <div className="section-title">
                        <Icon name="card-text" />
                        <h3>Case summary</h3>
                      </div>
                      <p>{selected.analysis.summary}</p>
                      <div className="analysis-chips">
                        <span>
                          <Icon name="tag" />{" "}
                          {title(selected.analysis.intent?.label || "Unknown")}
                        </span>
                        <span>
                          <Icon name="emoji-neutral" />{" "}
                          {title(
                            selected.analysis.sentiment?.label || "Unknown",
                          )}{" "}
                          sentiment
                        </span>
                      </div>
                    </section>
                  )}
                  {selected.analysis.reply && (
                    <section className="panel p-4">
                      <div className="section-title">
                        <Icon name="pencil-square" />
                        <h3>Response for review</h3>
                      </div>
                      <p className="muted">
                        Edit the draft and check it against the policy evidence.
                        Approval records your review.
                      </p>
                      {selected.status === "review" &&
                        selected.analysis.flags?.map((f: string) => (
                          <div className="review-note" key={f}>
                            <Icon name="exclamation-triangle" /> {f}
                          </div>
                        ))}
                      <label
                        htmlFor="response-draft"
                        className="visually-hidden"
                      >
                        Response draft
                      </label>
                      <textarea
                        id="response-draft"
                        className="form-control response-draft"
                        rows={9}
                        value={draft}
                        readOnly={selected.status !== "review"}
                        onChange={(e) => setDraft(e.target.value)}
                      />
                      <div className="d-flex flex-wrap gap-2 mt-3">
                        {selected.status === "review" && (
                          <button
                            className="btn btn-primary"
                            disabled={!!busy || draft.trim().length < 12}
                            onClick={() =>
                              act("Approving response", async () => {
                                await api(
                                  `/tickets/${selected.id}/approve`,
                                  "POST",
                                  { reply: draft },
                                );
                                setSelected(
                                  await api("/tickets/" + selected.id),
                                );
                                setNotice(
                                  "Response approved and saved. No message has been sent.",
                                );
                                await refresh();
                              })
                            }
                          >
                            <Icon name="check2" /> Approve draft
                          </button>
                        )}
                        {selected.status === "approved" && (
                          <button
                            className="btn btn-primary"
                            onClick={() =>
                              act("Resolving case", async () => {
                                await api(
                                  `/tickets/${selected.id}/status?status=resolved`,
                                  "PATCH",
                                );
                                setSelected(
                                  await api("/tickets/" + selected.id),
                                );
                                await refresh();
                              })
                            }
                          >
                            <Icon name="check2-circle" /> Mark resolved
                          </button>
                        )}
                        {["resolved", "approved"].includes(selected.status) && (
                          <button
                            className="btn btn-outline-secondary"
                            onClick={() =>
                              act("Reopening case", async () => {
                                await api(
                                  `/tickets/${selected.id}/status?status=review`,
                                  "PATCH",
                                );
                                setSelected(
                                  await api("/tickets/" + selected.id),
                                );
                                await refresh();
                              })
                            }
                          >
                            Reopen for review
                          </button>
                        )}
                        <button
                          className="btn btn-outline-secondary"
                          onClick={() =>
                            navigator.clipboard
                              .writeText(draft)
                              .then(() => setNotice("Response copied."))
                              .catch(() =>
                                setError(
                                  "Clipboard access was unavailable. Select and copy the text.",
                                ),
                              )
                          }
                        >
                          <Icon name="clipboard" /> Copy response
                        </button>
                      </div>
                      <small className="muted d-block mt-3">
                        This workspace saves drafts and review decisions. It
                        does not send messages or execute account actions.
                      </small>
                    </section>
                  )}
                </div>
                <aside>
                  <section className="panel p-4 mb-4">
                    <div className="section-title">
                      <Icon name="journal-check" />
                      <h3>Policy evidence</h3>
                    </div>
                    {selected.analysis.evidence?.length ? (
                      selected.analysis.evidence.map((e: Obj) => (
                        <article className="evidence" key={e.id}>
                          <span className="eyebrow">
                            {e.id} · VERSION {e.policy_version}
                          </span>
                          <h4>{e.title}</h4>
                          <p>{e.content}</p>
                        </article>
                      ))
                    ) : (
                      <p className="muted">
                        Run analysis to retrieve relevant store policies.
                        Missing evidence requires human review.
                      </p>
                    )}
                  </section>
                  <section className="panel p-4">
                    <div className="section-title">
                      <Icon name="clock-history" />
                      <h3>Case history</h3>
                    </div>
                    <ol className="timeline">
                      {selected.events?.map((e: Obj, i: number) => (
                        <li key={i}>
                          <strong>{title(e.action)}</strong>
                          <small>{date(e.created_at)}</small>
                          <p>{e.detail}</p>
                        </li>
                      ))}
                    </ol>
                  </section>
                </aside>
              </div>
            </>
          )}
          {page === "Policy library" && (
            <div className="policy-grid">
              <section className="panel p-4">
                <div className="section-title">
                  <Icon name="search" />
                  <h2>Ask the policy library</h2>
                </div>
                <p className="muted">
                  Answers use the published Northstar policies. Questions
                  outside the guide are referred for review.
                </p>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    act("Checking policy", async () => {
                      setAnswer(
                        await api("/knowledge/ask", "POST", { question }),
                      );
                      await refresh();
                    });
                  }}
                >
                  <label className="form-label" htmlFor="policy-question">
                    Your question
                  </label>
                  <textarea
                    id="policy-question"
                    className="form-control"
                    rows={3}
                    placeholder="Can an order be cancelled after dispatch?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    minLength={5}
                    required
                  />
                  <button className="btn btn-primary mt-3" disabled={!!busy}>
                    <Icon name="search" /> Find an answer
                  </button>
                </form>
                {answer && (
                  <div className="policy-answer">
                    <div className="eyebrow">
                      {answer.abstained
                        ? "HUMAN REVIEW NEEDED"
                        : "ANSWER WITH EVIDENCE"}
                    </div>
                    <p>{answer.answer}</p>
                    {answer.flags?.map((f: string) => (
                      <div className="review-note" key={f}>
                        {f}
                      </div>
                    ))}
                    {answer.evidence?.map((a: Obj) => (
                      <article className="evidence" key={a.id}>
                        <span className="eyebrow">{a.id}</span>
                        <h4>{a.title}</h4>
                        <p>{a.content}</p>
                      </article>
                    ))}
                  </div>
                )}
                <div className="library-note">
                  <Icon name="info-circle" />
                  <p>
                    Northstar Retail is a fictional store used for this
                    demonstration. Policy version {policy.version}.
                  </p>
                </div>
              </section>
              <section>
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h2 className="h5">Published policies</h2>
                  <span className="count-pill">
                    {policy.articles.length} articles
                  </span>
                </div>
                {policy.articles.map((a: Obj) => (
                  <details className="policy-article" key={a.id}>
                    <summary>
                      <span>
                        <small>
                          {a.id} · {a.category}
                        </small>
                        {a.title}
                      </span>
                      <Icon name="chevron-down" />
                    </summary>
                    <p>{a.content}</p>
                  </details>
                ))}
              </section>
            </div>
          )}
          {page === "Quality & models" && (
            <>
              {ops.intent_experiment ? (
                <>
                  <div className="quality-banner">
                    <Icon name="flask" />
                    <div>
                      <h2>Support intent fine-tuning</h2>
                      <p>
                        {ops.intent_experiment.training.examples.toLocaleString()}{" "}
                        training examples ·{" "}
                        {ops.intent_experiment.labels.length} intents ·{" "}
                        {ops.intent_experiment.training.epochs} epochs ·
                        held-out evaluation
                      </p>
                    </div>
                    <span className="outline-tag">
                      {ops.intent_experiment.experiment_id}
                    </span>
                  </div>
                  <div className="model-score-grid">
                    {ops.intent_experiment.models.map((m: Obj) => (
                      <section className="panel p-4" key={m.name}>
                        <div className="eyebrow">
                          {m.name.includes("Fine")
                            ? "DOMAIN ADAPTED"
                            : m.name.includes("Original")
                              ? "PRETRAINED BASELINE"
                              : "CONVENTIONAL BASELINE"}
                        </div>
                        <h3>{m.name}</h3>
                        <div className="score-number">{pct(m.accuracy)}</div>
                        <p className="muted">
                          Accuracy on {m.examples} held-out examples
                        </p>
                        <div className="metric-row">
                          <span>Macro F1</span>
                          <strong>{m.macro_f1.toFixed(3)}</strong>
                        </div>
                        <div className="metric-row">
                          <span>Invalid labels</span>
                          <strong>{pct(m.invalid_label_rate)}</strong>
                        </div>
                      </section>
                    ))}
                  </div>
                  <section className="panel p-4 mt-4">
                    <h2>Per-intent performance</h2>
                    <p className="muted">
                      Fine-tuned model · class balance and errors remain visible
                    </p>
                    <div className="table-responsive metric-table">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>INTENT</th>
                            <th>PRECISION</th>
                            <th>RECALL</th>
                            <th>F1</th>
                            <th>TEST EXAMPLES</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ops.intent_experiment.labels.map((l: string) => {
                            const r = ops.intent_experiment.models[2].report[l];
                            return (
                              <tr key={l}>
                                <td>{title(l)}</td>
                                <td>{pct(r.precision)}</td>
                                <td>{pct(r.recall)}</td>
                                <td>{r["f1-score"].toFixed(3)}</td>
                                <td>{r.support}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              ) : (
                <section className="panel">
                  <Empty icon="bar-chart">
                    The fine-tuning experiment will appear after training and
                    evaluation complete.
                  </Empty>
                </section>
              )}
              <div className="two-columns mt-4">
                <section className="panel p-4">
                  <h2>Compare an enquiry</h2>
                  <p className="muted">
                    The same request, original and fine-tuned language models.
                  </p>
                  <label className="form-label" htmlFor="compare-text">
                    Customer message
                  </label>
                  <textarea
                    id="compare-text"
                    className="form-control"
                    rows={3}
                    value={compareText}
                    onChange={(e) => setCompareText(e.target.value)}
                  />
                  <button
                    className="btn btn-primary mt-3"
                    disabled={!!busy || compareText.length < 5}
                    onClick={() =>
                      act("Comparing models", async () => {
                        setComparison(
                          await api("/models/compare", "POST", {
                            question: compareText,
                          }),
                        );
                        await refresh();
                      })
                    }
                  >
                    Compare predictions
                  </button>
                  {comparison && (
                    <div className="compare-results">
                      {["original", "fine_tuned"].map((k) => (
                        <div key={k}>
                          <small>{title(k)}</small>
                          <strong>{title(comparison[k].label)}</strong>
                          <span className="muted">
                            Selected label: {comparison[k].raw}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
                <section className="panel p-4">
                  <h2>Model registry</h2>
                  <p className="muted">
                    Versioned open models served through local APIs.
                  </p>
                  {models.models.map((m: Obj) => (
                    <div className="registry-row" key={m.name}>
                      <span
                        className={"connection-dot " + (!m.ready ? "off" : "")}
                      />
                      <div>
                        <strong>{title(m.name)}</strong>
                        <small>{m.model_id}</small>
                        {m.revision && (
                          <small title={m.revision}>
                            Revision {m.revision.slice(0, 12)}
                          </small>
                        )}
                      </div>
                      <span className="outline-tag">
                        {m.loaded
                          ? "Loaded"
                          : m.ready
                            ? "Installed"
                            : "Pending"}
                      </span>
                    </div>
                  ))}
                </section>
              </div>
              {ops.quality_evaluation && (
                <section className="panel p-4 mt-4">
                  <h2>Workflow quality evaluation</h2>
                  <p className="muted">{ops.quality_evaluation.scope}</p>
                  <div className="quality-metrics">
                    {Object.entries(ops.quality_evaluation.metrics || {}).map(
                      ([k, v]) => (
                        <div key={k}>
                          <small>{title(k)}</small>
                          <strong>
                            {typeof v === "number"
                              ? Number(v).toFixed(3)
                              : String(v)}
                          </strong>
                        </div>
                      ),
                    )}
                  </div>
                </section>
              )}
              <section className="panel p-4 mt-4">
                <h2>Evaluation scope</h2>
                <ul className="scope-list">
                  {(
                    ops.intent_experiment?.limitations || [
                      "No experiment results are available yet.",
                    ]
                  ).map((l: string) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
                <p className="muted mb-0">
                  Prompt version {ops.prompt_version} · Policy version{" "}
                  {ops.policy_version}. Offline benchmark results are not a
                  production service guarantee.
                </p>
              </section>
            </>
          )}
          {page === "Activity" && (
            <>
              <div className="stat-grid activity-stats">
                <div className="stat-card">
                  <div className="stat-top">
                    Recorded operations
                    <Icon name="activity" />
                  </div>
                  <strong>{ops.requests || 0}</strong>
                  <small>Latest 500 model and retrieval traces</small>
                </div>
                <div className="stat-card">
                  <div className="stat-top">
                    Successful operations
                    <Icon name="check2-circle" />
                  </div>
                  <strong>
                    {ops.success_rate == null ? "—" : pct(ops.success_rate)}
                  </strong>
                  <small>Measured in this workspace</small>
                </div>
                <div className="stat-card">
                  <div className="stat-top">
                    p95 operation latency
                    <Icon name="stopwatch" />
                  </div>
                  <strong>
                    {ops.p95_latency_ms == null
                      ? "—"
                      : (ops.p95_latency_ms / 1000).toFixed(2) + "s"}
                  </strong>
                  <small>Individual operations, including cold loads</small>
                </div>
              </div>
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Operation log</h2>
                    <p>Model calls, retrieval, timings, and errors</p>
                  </div>
                  <a
                    className="text-btn"
                    href="/api/operations"
                    target="_blank"
                    rel="noreferrer"
                  >
                    View API <Icon name="box-arrow-up-right" />
                  </a>
                </div>
                <div className="table-responsive">
                  <table className="table case-table">
                    <thead>
                      <tr>
                        <th>OPERATION</th>
                        <th>MODEL</th>
                        <th>DURATION</th>
                        <th>RESULT</th>
                        <th>TIME</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ops.traces?.map((t: Obj) => (
                        <tr key={t.id}>
                          <td>
                            {title(t.task)}
                            <small>{t.request_id}</small>
                          </td>
                          <td className="model-name">{t.model}</td>
                          <td>{(t.duration_ms / 1000).toFixed(2)}s</td>
                          <td>
                            <span
                              className={
                                "status " +
                                (t.success ? "status-resolved" : "status-error")
                              }
                            >
                              {t.success ? "Success" : "Failed"}
                            </span>
                          </td>
                          <td>{date(t.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {!ops.traces?.length && (
                  <Empty icon="activity">
                    Operations will be recorded as you use the workspace.
                  </Empty>
                )}
              </section>
            </>
          )}
          <footer>
            <span>Northstar Retail · Service workspace</span>
            <span>Human review at every decision.</span>
          </footer>
        </main>
      </div>
      {modal && (
        <div className="modal-backdrop-custom">
          <section
            className="case-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="new-case-title"
          >
            <div className="modal-heading">
              <div>
                <div className="eyebrow">CUSTOMER INTAKE</div>
                <h2 id="new-case-title">New support case</h2>
              </div>
              <button
                className="icon-btn"
                disabled={!!busy || recording}
                aria-label="Close new case"
                onClick={() => setModal(false)}
              >
                <Icon name="x-lg" />
              </button>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                act("Creating case", async () => {
                  const t = await api("/tickets", "POST", {
                    ...newCase,
                    customer: newCase.customer || "Guest customer",
                  });
                  setModal(false);
                  setSelected(await api("/tickets/" + t.id));
                  setPage("Cases");
                  await refresh();
                });
              }}
            >
              <div className="two-columns">
                <div>
                  <label htmlFor="case-customer" className="form-label">
                    Customer name
                  </label>
                  <input
                    id="case-customer"
                    autoFocus
                    className="form-control"
                    placeholder="e.g. Priya Sharma"
                    value={newCase.customer}
                    maxLength={100}
                    onChange={(e) =>
                      setNewCase({ ...newCase, customer: e.target.value })
                    }
                  />
                </div>
                <div>
                  <label htmlFor="case-subject" className="form-label">
                    Subject
                  </label>
                  <input
                    id="case-subject"
                    className="form-control"
                    required
                    minLength={3}
                    maxLength={160}
                    placeholder="e.g. Delivery update"
                    value={newCase.subject}
                    onChange={(e) =>
                      setNewCase({ ...newCase, subject: e.target.value })
                    }
                  />
                </div>
              </div>
              <div className="voice-upload">
                <div>
                  <Icon name="soundwave" />
                  <strong>Start with a voice note</strong>
                  <small>English audio · up to 3 minutes · 25 MB</small>
                </div>
                <div className="d-flex gap-2 flex-wrap">
                  <label
                    className={
                      "btn btn-outline-secondary mb-0 " +
                      (busy ? "disabled" : "")
                    }
                  >
                    <Icon name="upload" /> Upload audio
                    <input
                      className="visually-hidden"
                      aria-label="Upload customer audio"
                      type="file"
                      accept="audio/*"
                      disabled={!!busy || recording}
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) uploadAudio(f, f.name);
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    className={
                      "btn " +
                      (recording ? "btn-danger" : "btn-outline-secondary")
                    }
                    onClick={toggleRecording}
                    disabled={!!busy}
                  >
                    <Icon name={recording ? "stop-fill" : "mic"} />{" "}
                    {recording ? "Stop recording" : "Record"}
                  </button>
                </div>
              </div>
              {audioName && (
                <p className="audio-filename">
                  <Icon name="check2-circle" /> {audioName}
                </p>
              )}
              <label htmlFor="customer-message" className="form-label mt-3">
                Customer message{" "}
                {newCase.channel === "voice" && (
                  <span className="muted">· review the transcript</span>
                )}
              </label>
              <textarea
                id="customer-message"
                ref={msgRef}
                className="form-control"
                rows={6}
                required
                minLength={8}
                maxLength={6000}
                placeholder="Enter the customer’s enquiry, or upload an audio note above."
                value={newCase.message}
                onChange={(e) =>
                  setNewCase({ ...newCase, message: e.target.value })
                }
              />
              <div className="modal-footer-custom">
                <small className="muted">
                  Review personal details before saving.
                </small>
                <button
                  className="btn btn-primary"
                  disabled={!!busy || recording}
                >
                  Create case <Icon name="arrow-right" />
                </button>
              </div>
            </form>
          </section>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
