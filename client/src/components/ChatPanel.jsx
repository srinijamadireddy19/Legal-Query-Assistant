import React, { useState } from "react";
import { queryRAG } from "../api";
 

export default function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(null);

  const sendMessage = async () => {
    const res = await queryRAG(input, conversationId);

    setConversationId(res.conversation_id);

    setMessages([
      ...messages,
      { role: "user", text: input },
      { role: "assistant", text: res.answer },
    ]);

    setInput("");
  };

  return (
    <div>
      <h3>Chat</h3>

      {messages.map((m, i) => (
        <div key={i}>
          <b>{m.role}:</b> {m.text}
        </div>
      ))}

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />

      <button onClick={sendMessage}>Send</button>
    </div>
  );
}