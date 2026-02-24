import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const ScaleIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{width:28,height:28}}>
    <path d="M12 3v18M3 12h18M6.5 6.5l11 11M17.5 6.5l-11 11" strokeLinecap="round"/>
    <circle cx="12" cy="12" r="9"/>
  </svg>
);

const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{width:18,height:18}}>
    <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{width:20,height:20}}>
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const CloseIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{width:14,height:14}}>
    <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round"/>
  </svg>
);

const BookIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{width:16,height:16}}>
    <path d="M4 19.5A2.5 2.5 0 016.5 17H20M4 19.5A2.5 2.5 0 004 17V5a1 1 0 011-1h14a1 1 0 011 1v12H6.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const suggestedQueries = [
  "What are my rights as a tenant facing eviction?",
  "Explain the elements of a valid contract",
  "What constitutes workplace discrimination?",
  "How does fair use apply in copyright law?",
];

export default function LegalQueryAssistant() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Welcome to Legal Query Assistant. I'm here to help you navigate complex legal questions with precision and clarity. How can I assist you today?",
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleQuery = async (queryText) => {
    const q = queryText || input.trim();
    if (!q || loading) return;
    setInput("");

    const userMsg = { role: "user", content: q, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      const answer = data.answer || data.response || data.result || JSON.stringify(data);
      const sources = data.sources || data.references || [];

      setMessages(prev => [...prev, {
        role: "assistant",
        content: answer,
        sources,
        timestamp: new Date(),
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `⚠️ Unable to reach the backend server. Please ensure it's running at \`${API_BASE}\`.\n\nError: ${err.message}`,
        timestamp: new Date(),
        isError: true,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (files) => {
    if (!files.length) return;
    const file = files[0];
    setUploadStatus({ state: "uploading", name: file.name });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      setUploadedFiles(prev => [...prev, { name: file.name, size: file.size }]);
      setUploadStatus({ state: "success", name: file.name });
      setTimeout(() => setUploadStatus(null), 3000);
    } catch (err) {
      setUploadStatus({ state: "error", name: file.name, message: err.message });
      setTimeout(() => setUploadStatus(null), 4000);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const formatTime = (date) =>
    date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div style={styles.root}>
      <style>{css}</style>

      {/* Sidebar */}
      <aside style={{ ...styles.sidebar, transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)" }}>
        <div style={styles.sidebarHeader}>
          <div style={styles.logoMark}>
            <ScaleIcon />
          </div>
          <div>
            <div style={styles.logoTitle}>Lex</div>
            <div style={styles.logoSub}>Legal Query Assistant</div>
          </div>
        </div>

        <div style={styles.sidebarSection}>
          <div style={styles.sectionLabel}>Suggested Queries</div>
          {suggestedQueries.map((q, i) => (
            <button key={i} style={styles.suggestionBtn} onClick={() => handleQuery(q)} className="suggestion-btn">
              <span style={styles.suggestionDot} />
              {q}
            </button>
          ))}
        </div>

        <div style={styles.sidebarSection}>
          <div style={styles.sectionLabel}>Document Upload</div>
          <div
            style={{ ...styles.dropzone, ...(dragOver ? styles.dropzoneActive : {}) }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFileUpload(e.dataTransfer.files); }}
            onClick={() => fileInputRef.current?.click()}
            className="dropzone"
          >
            <UploadIcon />
            <span style={styles.dropzoneText}>Drop PDF or click to upload</span>
            <span style={styles.dropzoneHint}>.pdf, .docx, .txt</span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            style={{ display: "none" }}
            onChange={(e) => handleFileUpload(e.target.files)}
          />

          {uploadStatus && (
            <div style={{
              ...styles.uploadStatus,
              background: uploadStatus.state === "success" ? "rgba(74,222,128,0.1)" :
                uploadStatus.state === "error" ? "rgba(248,113,113,0.1)" : "rgba(148,163,184,0.1)",
              borderColor: uploadStatus.state === "success" ? "rgba(74,222,128,0.3)" :
                uploadStatus.state === "error" ? "rgba(248,113,113,0.3)" : "rgba(148,163,184,0.2)",
            }}>
              {uploadStatus.state === "uploading" && <span style={styles.spinner} className="spin" />}
              <span style={styles.uploadStatusText}>
                {uploadStatus.state === "uploading" ? `Uploading ${uploadStatus.name}…` :
                  uploadStatus.state === "success" ? `✓ ${uploadStatus.name} uploaded` :
                    `✗ Upload failed`}
              </span>
            </div>
          )}

          {uploadedFiles.map((f, i) => (
            <div key={i} style={styles.fileChip}>
              <BookIcon />
              <span style={styles.fileChipName}>{f.name}</span>
              <button style={styles.fileChipRemove} onClick={() => setUploadedFiles(prev => prev.filter((_, j) => j !== i))}>
                <CloseIcon />
              </button>
            </div>
          ))}
        </div>

        <div style={styles.sidebarFooter}>
          <div style={styles.disclaimer}>
            ⚖️ Not legal advice. Consult a qualified attorney for your specific situation.
          </div>
        </div>
      </aside>

      {/* Main */}
      <main style={styles.main}>
        {/* Topbar */}
        <header style={styles.topbar}>
          <button style={styles.hamburger} onClick={() => setSidebarOpen(o => !o)} className="icon-btn">
            <div style={styles.ham1} />
            <div style={styles.ham2} />
            <div style={styles.ham3} />
          </button>
          <div style={styles.topbarTitle}>Legal Query Assistant</div>
          <div style={styles.statusDot} title="Backend status" />
        </header>

        {/* Messages */}
        <div style={styles.messages}>
          {messages.map((msg, i) => (
            <div key={i} style={{ ...styles.msgRow, justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}
              className="msg-row">
              {msg.role === "assistant" && (
                <div style={styles.avatarWrap}>
                  <div style={styles.avatar}><ScaleIcon /></div>
                </div>
              )}
              <div style={{
                ...styles.bubble,
                ...(msg.role === "user" ? styles.bubbleUser : styles.bubbleAssistant),
                ...(msg.isError ? styles.bubbleError : {}),
              }}>
                <div style={styles.bubbleContent}>{msg.content}</div>
                {msg.sources && msg.sources.length > 0 && (
                  <div style={styles.sources}>
                    <div style={styles.sourcesLabel}>Sources</div>
                    {msg.sources.map((s, si) => (
                      <div key={si} style={styles.sourceItem}>
                        <BookIcon />
                        <span>{typeof s === "string" ? s : s.title || s.source || JSON.stringify(s)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <div style={styles.timestamp}>{formatTime(msg.timestamp)}</div>
              </div>
            </div>
          ))}

          {loading && (
            <div style={{ ...styles.msgRow, justifyContent: "flex-start" }} className="msg-row">
              <div style={styles.avatarWrap}>
                <div style={styles.avatar}><ScaleIcon /></div>
              </div>
              <div style={{ ...styles.bubble, ...styles.bubbleAssistant }}>
                <div style={styles.typingDots}>
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div style={styles.inputArea}>
          <div style={styles.inputWrap}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a legal question…"
              rows={1}
              style={styles.textarea}
              className="chat-input"
            />
            <button
              onClick={() => handleQuery()}
              disabled={!input.trim() || loading}
              style={{
                ...styles.sendBtn,
                opacity: !input.trim() || loading ? 0.4 : 1,
                cursor: !input.trim() || loading ? "not-allowed" : "pointer",
              }}
              className="send-btn"
            >
              <SendIcon />
            </button>
          </div>
          <div style={styles.inputHint}>Press Enter to send · Shift+Enter for new line</div>
        </div>
      </main>
    </div>
  );
}

const styles = {
  root: {
    display: "flex",
    height: "100vh",
    width: "100vw",
    background: "#0c0f1a",
    color: "#e2e8f0",
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    overflow: "hidden",
    position: "relative",
  },
  sidebar: {
    width: 300,
    minWidth: 300,
    background: "linear-gradient(180deg, #0f1422 0%, #0c0f1a 100%)",
    borderRight: "1px solid rgba(148,163,184,0.08)",
    display: "flex",
    flexDirection: "column",
    transition: "transform 0.3s cubic-bezier(0.4,0,0.2,1)",
    zIndex: 10,
    flexShrink: 0,
  },
  sidebarHeader: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    padding: "28px 24px 20px",
    borderBottom: "1px solid rgba(148,163,184,0.08)",
  },
  logoMark: {
    width: 48,
    height: 48,
    background: "linear-gradient(135deg, #b8860b 0%, #daa520 100%)",
    borderRadius: 12,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#0c0f1a",
    flexShrink: 0,
    boxShadow: "0 4px 20px rgba(184,134,11,0.3)",
  },
  logoTitle: {
    fontFamily: "'Playfair Display', 'Georgia', serif",
    fontSize: 22,
    fontWeight: 700,
    letterSpacing: "0.05em",
    color: "#daa520",
    lineHeight: 1,
  },
  logoSub: {
    fontSize: 11,
    color: "#64748b",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    marginTop: 3,
    fontFamily: "'Crimson Pro', serif",
  },
  sidebarSection: {
    padding: "20px 24px",
    borderBottom: "1px solid rgba(148,163,184,0.08)",
  },
  sectionLabel: {
    fontSize: 10,
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    color: "#475569",
    marginBottom: 12,
    fontFamily: "'Crimson Pro', serif",
  },
  suggestionBtn: {
    display: "flex",
    alignItems: "flex-start",
    gap: 10,
    width: "100%",
    background: "transparent",
    border: "none",
    color: "#94a3b8",
    fontSize: 13,
    lineHeight: 1.5,
    padding: "8px 0",
    cursor: "pointer",
    textAlign: "left",
    transition: "color 0.2s",
    fontFamily: "'Crimson Pro', serif",
  },
  suggestionDot: {
    width: 4,
    height: 4,
    borderRadius: "50%",
    background: "#b8860b",
    flexShrink: 0,
    marginTop: 7,
  },
  dropzone: {
    border: "1.5px dashed rgba(148,163,184,0.2)",
    borderRadius: 10,
    padding: "18px 16px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
    cursor: "pointer",
    transition: "all 0.2s",
    marginBottom: 10,
  },
  dropzoneActive: {
    borderColor: "#daa520",
    background: "rgba(218,165,32,0.05)",
  },
  dropzoneText: {
    fontSize: 13,
    color: "#64748b",
    fontFamily: "'Crimson Pro', serif",
  },
  dropzoneHint: {
    fontSize: 11,
    color: "#334155",
  },
  uploadStatus: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid",
    marginBottom: 6,
  },
  uploadStatusText: {
    fontSize: 12,
    color: "#94a3b8",
    fontFamily: "'Crimson Pro', serif",
  },
  spinner: {
    width: 12,
    height: 12,
    borderRadius: "50%",
    border: "2px solid rgba(148,163,184,0.2)",
    borderTopColor: "#daa520",
    flexShrink: 0,
  },
  fileChip: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 10px",
    background: "rgba(148,163,184,0.06)",
    borderRadius: 8,
    marginBottom: 4,
    color: "#94a3b8",
  },
  fileChipName: {
    fontSize: 12,
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    fontFamily: "'Crimson Pro', serif",
  },
  fileChipRemove: {
    background: "transparent",
    border: "none",
    color: "#475569",
    cursor: "pointer",
    padding: 2,
    display: "flex",
  },
  sidebarFooter: {
    marginTop: "auto",
    padding: "16px 24px",
  },
  disclaimer: {
    fontSize: 11,
    color: "#334155",
    lineHeight: 1.6,
    fontStyle: "italic",
    fontFamily: "'Crimson Pro', serif",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    background: "radial-gradient(ellipse at 50% 0%, rgba(184,134,11,0.04) 0%, transparent 60%), #0c0f1a",
  },
  topbar: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "16px 28px",
    borderBottom: "1px solid rgba(148,163,184,0.08)",
    background: "rgba(12,15,26,0.8)",
    backdropFilter: "blur(10px)",
    position: "sticky",
    top: 0,
    zIndex: 5,
  },
  hamburger: {
    background: "transparent",
    border: "none",
    cursor: "pointer",
    display: "flex",
    flexDirection: "column",
    gap: 4,
    padding: 4,
  },
  ham1: { width: 20, height: 1.5, background: "#64748b", borderRadius: 2 },
  ham2: { width: 14, height: 1.5, background: "#64748b", borderRadius: 2 },
  ham3: { width: 20, height: 1.5, background: "#64748b", borderRadius: 2 },
  topbarTitle: {
    fontFamily: "'Playfair Display', 'Georgia', serif",
    fontSize: 18,
    color: "#e2e8f0",
    letterSpacing: "0.02em",
    flex: 1,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#22c55e",
    boxShadow: "0 0 8px rgba(34,197,94,0.5)",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "32px 28px",
    display: "flex",
    flexDirection: "column",
    gap: 20,
  },
  msgRow: {
    display: "flex",
    alignItems: "flex-end",
    gap: 12,
    animation: "fadeSlideIn 0.3s ease",
  },
  avatarWrap: {
    flexShrink: 0,
    marginBottom: 4,
  },
  avatar: {
    width: 36,
    height: 36,
    borderRadius: "50%",
    background: "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
    border: "1.5px solid rgba(184,134,11,0.3)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#b8860b",
  },
  bubble: {
    maxWidth: "68%",
    borderRadius: 16,
    padding: "14px 18px",
    lineHeight: 1.7,
    fontSize: 15,
    fontFamily: "'Crimson Pro', serif",
    position: "relative",
  },
  bubbleUser: {
    background: "linear-gradient(135deg, #1a2744 0%, #172033 100%)",
    border: "1px solid rgba(218,165,32,0.2)",
    borderBottomRightRadius: 4,
    color: "#e2e8f0",
  },
  bubbleAssistant: {
    background: "linear-gradient(135deg, #141824 0%, #111520 100%)",
    border: "1px solid rgba(148,163,184,0.1)",
    borderBottomLeftRadius: 4,
    color: "#cbd5e1",
  },
  bubbleError: {
    background: "rgba(127,29,29,0.2)",
    border: "1px solid rgba(248,113,113,0.2)",
    color: "#fca5a5",
  },
  bubbleContent: {
    whiteSpace: "pre-wrap",
  },
  sources: {
    marginTop: 12,
    paddingTop: 12,
    borderTop: "1px solid rgba(148,163,184,0.1)",
  },
  sourcesLabel: {
    fontSize: 10,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "#475569",
    marginBottom: 6,
  },
  sourceItem: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 12,
    color: "#64748b",
    padding: "3px 0",
  },
  timestamp: {
    fontSize: 10,
    color: "#334155",
    marginTop: 8,
    textAlign: "right",
  },
  typingDots: {
    display: "flex",
    gap: 5,
    padding: "4px 2px",
    alignItems: "center",
  },
  inputArea: {
    padding: "16px 28px 24px",
    borderTop: "1px solid rgba(148,163,184,0.08)",
    background: "rgba(12,15,26,0.9)",
    backdropFilter: "blur(10px)",
  },
  inputWrap: {
    display: "flex",
    alignItems: "flex-end",
    gap: 12,
    background: "rgba(15,20,34,0.8)",
    border: "1px solid rgba(148,163,184,0.12)",
    borderRadius: 14,
    padding: "10px 10px 10px 16px",
    transition: "border-color 0.2s",
  },
  textarea: {
    flex: 1,
    background: "transparent",
    border: "none",
    outline: "none",
    color: "#e2e8f0",
    fontSize: 15,
    fontFamily: "'Crimson Pro', serif",
    lineHeight: 1.6,
    resize: "none",
    maxHeight: 120,
    overflowY: "auto",
  },
  sendBtn: {
    width: 40,
    height: 40,
    borderRadius: 10,
    background: "linear-gradient(135deg, #b8860b 0%, #daa520 100%)",
    border: "none",
    color: "#0c0f1a",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
    flexShrink: 0,
  },
  inputHint: {
    fontSize: 11,
    color: "#334155",
    marginTop: 8,
    textAlign: "center",
    fontFamily: "'Crimson Pro', serif",
  },
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Crimson+Pro:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0c0f1a; }
  
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.15); border-radius: 2px; }

  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
  }

  .spin { animation: spin 0.8s linear infinite; display: inline-block; }
  
  .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #b8860b;
    animation: bounce 1.2s ease-in-out infinite;
  }
  .dot:nth-child(1) { animation-delay: 0s; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  .suggestion-btn:hover { color: #daa520 !important; }
  .dropzone:hover { border-color: rgba(218,165,32,0.4); background: rgba(218,165,32,0.03); }
  .send-btn:hover:not(:disabled) { transform: scale(1.05); box-shadow: 0 4px 16px rgba(184,134,11,0.4); }
  .chat-input:focus-visible ~ * { outline: none; }
  
  .inputWrap:focus-within {
    border-color: rgba(218,165,32,0.3) !important;
    box-shadow: 0 0 0 3px rgba(218,165,32,0.06);
  }
`;