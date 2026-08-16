import { useEffect, useState } from "react";
import { fetchSources, indexSources, sendQuestion } from "./api";

function timestampLabel(date = new Date()) {
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function buildWelcomeMessage(activeUrls) {
  if (activeUrls.length) {
    return {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content:
        "Your sources are ready. Ask a question and I’ll answer only from the URLs currently indexed here.",
      sources: activeUrls,
      time: timestampLabel(),
    };
  }

  return {
    id: `assistant-${Date.now()}`,
    role: "assistant",
    content:
      "Hello! I can help you research topics using real-time information from your active sources. What would you like to explore today?",
    sources: [],
    time: timestampLabel(),
  };
}

function normalizeSourceInputs(values) {
  const seen = new Set();

  return values
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
    .filter((value) => {
      if (seen.has(value)) {
        return false;
      }
      seen.add(value);
      return true;
    });
}

function hostLabel(url) {
  try {
    const parsed = new URL(url);
    return parsed.hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "message-row-user" : ""}`}>
      {!isUser ? <div className="message-avatar" aria-hidden="true">✦</div> : null}

      <div className="message-stack">
        <div className={`message-bubble ${isUser ? "message-bubble-user" : ""}`}>
          <div className="message-content">
            {message.content.split("\n").map((line, index) => (
              <p key={`${message.id}-${index}`}>{line || "\u00A0"}</p>
            ))}
          </div>

          {!isUser && message.sources?.length > 0 ? (
            <div className="message-sources">
              <div className="message-sources-label">Sources used</div>
              <div className="message-source-tags">
                {message.sources.map((source) => (
                  <a
                    key={`${message.id}-${source}`}
                    className="message-source-tag"
                    href={source}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {hostLabel(source)}
                  </a>
                ))}
              </div>
            </div>
          ) : null}
        </div>

        <div className={`message-time ${isUser ? "message-time-user" : ""}`}>
          {message.time}
        </div>
      </div>
    </div>
  );
}

function IndexStatus({ report, error }) {
  if (!error && !report.length) {
    return null;
  }

  return (
    <div className="modal-feedback">
      {error ? <div className="status-banner status-banner-error">{error}</div> : null}

      {report.length ? (
        <div className="status-banner status-banner-success">
          Source set updated. Review the indexing notes below.
        </div>
      ) : null}

      {report.length ? (
        <div className="index-report-list">
          {report.map((item) => (
            <div className="index-report-item" key={`${item.state}-${item.url || "empty"}-${item.message}`}>
              <div className={`index-report-state index-report-state-${item.state}`}>
                {item.state}
              </div>
              {item.url ? <div className="index-report-url">{item.url}</div> : null}
              <div className="index-report-message">{item.message}</div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function reportHasIssues(report) {
  return report.some((item) => item.state === "error" || item.state === "warning");
}

export default function App() {
  const [activeSources, setActiveSources] = useState([]);
  const [messages, setMessages] = useState([buildWelcomeMessage([])]);
  const [question, setQuestion] = useState("");
  const [sourceInputs, setSourceInputs] = useState([""]);
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
        setSourceInputs(urls.length ? urls : [""]);
        setMessages([buildWelcomeMessage(urls)]);
      } catch (error) {
        setAppError(error.message);
      }
    }

    loadSources();
  }, []);

  function openSourcesModal() {
    setIndexError("");
    setIndexReport([]);
    setSourceInputs(activeSources.length ? activeSources : sourceInputs.length ? sourceInputs : [""]);
    setIsSourcesOpen(true);
  }

  function closeSourcesModal() {
    setIsSourcesOpen(false);
    setIndexError("");
  }

  function updateSourceInput(index, value) {
    setSourceInputs((current) =>
      current.map((entry, entryIndex) => (entryIndex === index ? value : entry)),
    );
  }

  function addSourceInput() {
    setSourceInputs((current) => [...current, ""]);
  }

  function removeSourceInput(index) {
    setSourceInputs((current) => current.filter((_, entryIndex) => entryIndex !== index));
  }

  async function handleIndexSources(event) {
    event.preventDefault();
    const urls = normalizeSourceInputs(sourceInputs);

    setIsIndexing(true);
    setIndexError("");
    setIndexReport([]);

    try {
      const data = await indexSources(urls);
      const nextActive = data.activeUrls || [];
      const nextReport = data.report || [];
      const hasIssues = reportHasIssues(nextReport);

      setActiveSources(nextActive);
      setIndexReport(nextReport);

      if (hasIssues) {
        return;
      }

      setSourceInputs(nextActive.length ? nextActive : []);
      setMessages([buildWelcomeMessage(nextActive)]);
      setTimeout(() => {
        setIsSourcesOpen(false);
      }, 250);
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

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmedQuestion,
      sources: [],
      time: timestampLabel(),
    };

    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setAppError("");
    setIsSending(true);

    try {
      const data = await sendQuestion(trimmedQuestion, activeSources);
      const nextActive = data.activeUrls || activeSources;

      setActiveSources(nextActive);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
          time: timestampLabel(),
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          content: error.message,
          sources: [],
          time: timestampLabel(),
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>Universal URL Research Tool</h1>
        <p>Research any topic with real-time insights from your sources.</p>
      </header>

      <section className="source-strip">
        <div className="source-strip-label">Active Sources</div>
        <div className="source-strip-row">
          {activeSources.length ? (
            activeSources.map((source) => (
              <span key={source} className="source-pill">
                {hostLabel(source)}
              </span>
            ))
          ) : (
            <span className="source-empty">No active sources yet.</span>
          )}

          <button type="button" className="add-sources-button" onClick={openSourcesModal}>
            + Add Sources
          </button>
        </div>
      </section>

      <section className="chat-card">
        <div className="chat-history">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>

        <form className="composer" onSubmit={handleSendQuestion}>
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask anything about your sources..."
            disabled={isSending}
          />
          <button type="submit" className="send-button" disabled={isSending}>
            {isSending ? "Sending..." : "Send"}
          </button>
        </form>
      </section>

      {appError ? <div className="page-error">{appError}</div> : null}

      {isSourcesOpen ? (
        <div className="modal-backdrop" onClick={closeSourcesModal}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h2>Add Sources</h2>
                <p>Add one or more URLs to include as sources for your research.</p>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={closeSourcesModal}
                aria-label="Close sources modal"
              >
                ×
              </button>
            </div>

            <form className="source-form" onSubmit={handleIndexSources}>
              {sourceInputs.length ? (
                <div className="source-input-list">
                  {sourceInputs.map((value, index) => (
                    <div className="source-input-row" key={`source-row-${index}`}>
                      <input
                        type="text"
                        value={value}
                        onChange={(event) => updateSourceInput(index, event.target.value)}
                        placeholder={`https://example.com/source-${index + 1}`}
                      />
                      <button
                        type="button"
                        className="source-delete-button"
                        onClick={() => removeSourceInput(index)}
                        aria-label={`Delete source ${index + 1}`}
                      >
                        🗑
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="source-empty-state">
                  No sources in this set right now. Add one below or click Save to clear everything.
                </div>
              )}

              <button type="button" className="add-source-row-button" onClick={addSourceInput}>
                + Add another source
              </button>

              <div className="modal-divider" />

              <div className="source-actions">
                <button type="button" className="cancel-button" onClick={closeSourcesModal}>
                  Cancel
                </button>
                <button type="submit" className="index-button" disabled={isIndexing}>
                  {isIndexing ? "Saving..." : "Save"}
                </button>
              </div>
            </form>

            <IndexStatus report={indexReport} error={indexError} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
