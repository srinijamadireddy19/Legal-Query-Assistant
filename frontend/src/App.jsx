import { useState, useRef, useEffect } from "react";

import "./styles/global.css";
import "./styles/layout.css";

import Header       from "./components/Header";
import WelcomeScreen from "./components/WelcomeScreen";
import MessageBubble from "./components/MessageBubble";
import { TypingIndicator, ErrorBanner } from "./components/StatusWidgets";
import InputBox     from "./components/InputBox";
import { queryLegal } from "./services/api";

export default function App() {
  const [messages, setMessages]   = useState([]);
  const [input,    setInput]      = useState("");
  const [loading,  setLoading]    = useState(false);
  const [error,    setError]      = useState(null);

  // Stable session ID for the lifetime of the page
  const sessionId = useRef(`session_${Date.now()}`).current;
  const bottomRef = useRef(null);

  // Scroll to bottom whenever messages or loading changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (text) => {
    const query = (text ?? input).trim();
    if (!query || loading) return;

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const { answer, sources } = await queryLegal(query, sessionId);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: answer, sources },
      ]);
    } catch (err) {
      setError(err.message);
      // Roll back the optimistic user message so the user can retry
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="app">
      {/* ── Fixed header ─────────────────────────────── */}
      <Header showClear={!isEmpty} onClear={clearChat} />

      {/* ── Scrollable chat pane ─────────────────────── */}
      <div className="chat-pane">
        {isEmpty ? (
          <WelcomeScreen onSelectQuery={(q) => sendMessage(q)} />
        ) : (
          <div className="messages-area">
            <div className="messages-inner">
              {messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} />
              ))}

              {loading && <TypingIndicator />}

              <ErrorBanner message={error} />

              {/* Scroll anchor */}
              <div ref={bottomRef} />
            </div>
          </div>
        )}
      </div>

      {/* ── Sticky input bar ─────────────────────────── */}
      <div className="input-area">
        <div className="input-area__inner">
          <InputBox
            value={input}
            onChange={setInput}
            onSend={sendMessage}
            disabled={loading}
          />
          <p className="input-area__disclaimer">
            LEXIS COUNSEL · For informational purposes only · Not a substitute for professional legal advice
          </p>
        </div>
      </div>
    </div>
  );
}