import React, { useState } from "react";
import { queryRAG } from "../api";
import ReactMarkdown from "react-markdown";

export default function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    try {
      setLoading(true);
      setError("");

      // Add user message immediately (better UX)
      setMessages(prev => [
        ...prev,
        { role: "user", text: input }
      ]);

      const res = await queryRAG(input, conversationId);

      setConversationId(res.conversation_id);

      setMessages(prev => [
        ...prev,
        { role: "assistant", text: res.answer }
      ]);

      setInput("");
    } catch (err) {
      console.error(err);
      setError("Failed to get response. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h3>Chat</h3>

      <div style={{ marginBottom: 15 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 16 }}>
            <b>{m.role}:</b>
            <div style={{ marginTop: 6 }}>
              <ReactMarkdown>
                {m.text}
              </ReactMarkdown>
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ fontStyle: "italic", color: "gray" }}>
            Assistant is thinking...
          </div>
        )}

        {error && (
          <div style={{ color: "red" }}>
            {error}
          </div>
        )}
      </div>

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        disabled={loading}
      />

      <button
        onClick={sendMessage}
        disabled={!input.trim() || loading}
        style={{ marginLeft: 8 }}
      >
        {loading ? "Sending..." : "Send"}
      </button>
    </div>
  );
}