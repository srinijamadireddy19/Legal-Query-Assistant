import { useRef } from "react";
import "../styles/inputbox.css";
import { SendIcon } from "./Icons";

export default function InputBox({ value, onChange, onSend, disabled }) {
  const textareaRef = useRef(null);

  const autoResize = () => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    }
  };

  const handleChange = (e) => {
    onChange(e.target.value);
    autoResize();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  };

  const canSend = !disabled && value.trim().length > 0;

  return (
    <div className="input-box">
        
      <textarea
        ref={textareaRef}
        className="input-box__textarea"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask a legal question… (Shift+Enter for new line)"
        rows={1}
      />
      <button
        className={`input-box__send-btn ${canSend ? "input-box__send-btn--active" : "input-box__send-btn--disabled"}`}
        onClick={() => canSend && onSend()}
        disabled={!canSend}
        aria-label="Send"
      >
        <SendIcon />
      </button>
    </div>
  );
}