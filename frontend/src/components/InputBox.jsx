import { useRef } from "react";
import "../styles/inputbox.css";
import { SendIcon } from "./Icons";

export default function InputBox({ value, onChange, onSend, onFileUpload, disabled }) {
    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);

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

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file && onFileUpload) {
            onFileUpload(file);
        }
        // Reset so same file can be re-uploaded
        e.target.value = "";
    };

    const canSend = !disabled && value.trim().length > 0;

    return (
        <div className="input-box">
            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                id="file-upload-input"
                className="input-box__file-input"
                accept=".pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg"
                onChange={handleFileChange}
                aria-label="Upload document"
            />

            {/* Paperclip / attach button */}
            <button
                className="input-box__attach-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
                aria-label="Attach file"
                title="Upload a document (PDF, DOCX, TXT, image…)"
                type="button"
            >
                {/* Paperclip SVG */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                    strokeLinecap="round" strokeLinejoin="round" width="18" height="18">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19
            a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                </svg>
            </button>

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