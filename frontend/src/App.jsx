import { useEffect, useMemo, useState } from "react";
import { fetchSources, indexSources, sendQuestion } from "./api";

function buildIntroMessage(activeUrls) {
  if (!activeUrls.length) {
    return {
      role: "assistant",
      content:
        "Add one or more URLs to start. I will answer only from the currently indexed sources.",
      sources: [],
    };
  }

  return {
    role: "assistant",
    content: `Your source set is ready. I will stay grounded to these URLs:\n\n${activeUrls
      .map((url) => `- ${url}`)
      .join("\n")}`,
    sources: [],
  };
}

function parseUrls(text) {
  const seen = new Set();
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .filter((line) => {
      if (seen.has(line)) {
        return false;
      }
      seen.add(line);
      return true;
    });
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "message-row-user" : ""}`}>
      <div className={`message-bubble ${isUser ? "message-bubble-user" : ""}`}>
        <div className="message-content">
          {message.content.split("\n").map((line, index) => (
            <p key={`${message.role}-${index}`}>{line || "\u00A0"}</p>
          ))}
        </div>

        {!isUser && message.sources?.length > 0 ? (
          <div className="message-sources">
            <div className="message-sources-label">Sources</div>
            <ul>
              {message.sources.map((source) => (
                <li key={source}>
                  <a href={source} target="_blank" rel="noreferrer">
                    {source}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function IndexStatus({ report }) {
  if (!report.length) {
    return null;
  }

  const successCount = report.filter((item) => item.state === "success").length;
  const noteCount = report.filter((item) => item.state === "note").length;
  const warningCount = report.filter((item) => item.state === "warning").length;
  const errorCount = report.filter((item) => item.state === "error").length;

  return (
    <div className="index-status">
      {successCount || noteCount ? (
        <div className="status-banner status-banner-success">
          Ready: {successCount} newly indexed, {noteCount} reused.
        </div>
      ) : null}
      {warningCount ? (
        <div className="status-banner status-banner-warning">
          Needs review: {warningCount} URL(s) did not produce usable chunks.
        </div>
      ) : null}
      {errorCount ? (
        <div className="status-banner status-banner-error">
          Failed: {errorCount} URL(s) could not be indexed.
        </div>
      ) : null}

      <details className="status-details">
        <summary>See indexing details</summary>
        <div className="status-detail-list">
          {report.map((item) => (
            <div key={`${item.state}-${item.url}`} className="status-detail-item">
              <div className={`status-detail-label status-detail-label-${item.state}`}>
                {item.state}
              </div>
              <div className="status-detail-url">{item.url}</div>
              <div className="status-detail-message">{item.message}</div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

export default function App() {
  const [activeSources, setActiveSources] = useState([]);
  const [messages, setMessages] = useState([buildIntroMessage([])]);
  const [question, setQuestion] = useState("");
  const [sourcesDraft, setSourcesDraft] = useState("");
  const [isSourcesOpen, setIsSourcesOpen] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexReport, setIndexReport] = useState([]);
  const [indexError, setIndexError] = useState("");
  const [appError, setAppError] = useState("");

  useEffect(() => {
    async function loadSources() {
      try {
        const data = await fetchSources();
        const urls = data.activeUrls || [];
        setActiveSources(urls);
        setSourcesDraft(urls.join("\n"));
        setMessages([buildIntroMessage(urls)]);
      } catch (error) {
        setAppError(error.message);
      }
    }

    loadSources();
  }, []);

  const activeSourceLabel = useMemo(() => {
    if (!activeSources.length) {
      return "No active sources yet.";
    }
    return `${activeSources.length} active source${activeSources.length === 1 ? "" : "s"}`;
  }, [activeSources]);

  async function handleIndexSources(event) {
    event.preventDefault();
    const urls = parseUrls(sourcesDraft);
    if (!urls.length) {
      setIndexError("Add at least one URL before refreshing sources.");
      return;
    }

    setIsIndexing(true);
    setIndexError("");
    setIndexReport([]);

    try {
      const data = await indexSources(urls);
      const nextActive = data.activeUrls || [];
      setActiveSources(nextActive);
      setSourcesDraft(nextActive.length ? nextActive.join("\n") : urls.join("\n"));
      setIndexReport(data.report || []);
      setMessages([buildIntroMessage(nextActive)]);
      if (nextActive.length) {
        setIsSourcesOpen(false);
      }
    } catch (error) {
      setIndexError(error.message);
    } finally {
      setIsIndexing(false);
    }
  }

  async function handleSendQuestion(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isSending) {
      return;
    }

    const userMessage = { role: "user", content: trimmedQuestion, sources: [] };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setAppError("");
    setIsSending(true);

    try {
      const data = await sendQuestion(trimmedQuestion, activeSources);
      setActiveSources(data.activeUrls || activeSources);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error.message,
          sources: [],
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Universal URL Research Tool</h1>
          <p>Add sources only when you need them, then ask questions in one simple chat.</p>
        </div>
      </header>

      <section className="source-strip">
        <div className="source-strip-heading">
          <div>
            <h2>Active Sources</h2>
            <p>{activeSourceLabel}</p>
          </div>
        </div>
        <div className="source-pill-row">
          {activeSources.length ? (
            activeSources.map((source) => (
              <span key={source} className="source-pill">
                {source}
              </span>
            ))
          ) : (
            <span className="empty-copy">No active sources yet.</span>
          )}
        </div>
      </section>

      <section className="chat-panel">
        <div className="chat-history">
          {messages.map((message, index) => (
            <MessageBubble key={`${message.role}-${index}`} message={message} />
          ))}
        </div>
      </section>

      <form className="composer" onSubmit={handleSendQuestion}>
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask anything about your sources..."
          disabled={isSending}
        />
        <button
          type="button"
          className="secondary-button"
          onClick={() => setIsSourcesOpen(true)}
        >
          Add Sources
        </button>
        <button type="submit" className="primary-button" disabled={isSending}>
          {isSending ? "Sending..." : "Send"}
        </button>
      </form>

      {appError ? <div className="page-error">{appError}</div> : null}

      {isSourcesOpen ? (
        <div className="modal-backdrop" onClick={() => setIsSourcesOpen(false)}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3>Manage Sources</h3>
                <p>Paste one URL per line, then refresh the source set.</p>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => setIsSourcesOpen(false)}
                aria-label="Close sources panel"
              >
                ×
              </button>
            </div>

            <form onSubmit={handleIndexSources} className="source-form">
              <textarea
                value={sourcesDraft}
                onChange={(event) => setSourcesDraft(event.target.value)}
                placeholder={"https://example.com/article-1\nhttps://example.com/article-2"}
                rows={8}
              />

              <div className="source-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => setIsSourcesOpen(false)}
                >
                  Cancel
                </button>
                <button type="submit" className="primary-button" disabled={isIndexing}>
                  {isIndexing ? "Refreshing..." : "Refresh Sources"}
                </button>
              </div>
            </form>

            {indexError ? <div className="status-banner status-banner-error">{indexError}</div> : null}
            <IndexStatus report={indexReport} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

