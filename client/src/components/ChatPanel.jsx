import React, { useState, useRef, useEffect } from "react";
import { queryRAG } from "../api";
import ReactMarkdown from "react-markdown";

export default function ChatPanel({ ready }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const bottomRef = useRef(null);

  // auto scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim() || loading || !ready) return;

    const userMessage = { role: "user", text: input };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError("");

    try {
      const res = await queryRAG(input, conversationId);

      setConversationId(res.conversation_id);

      const assistantMessage = {
        role: "assistant",
        text: res.answer,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      setError("Failed to get response. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h3>Chat</h3>

      <div className="messages">
        {!ready && (
          <p style={{ color: "gray" }}>
            Upload and process a document to start chatting.
          </p>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`message ${m.role === "user" ? "user" : "assistant"}`}
          >
            <ReactMarkdown>{m.text}</ReactMarkdown>
          </div>
        ))}

        {loading && (
          <div className="message assistant">Assistant is thinking...</div>
        )}

        {error && (
          <div style={{ color: "red" }}>{error}</div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input">
        <input
          value={input}
          disabled={!ready || loading}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about the document..."
        />

        <button
          onClick={sendMessage}
          disabled={!input.trim() || loading}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}