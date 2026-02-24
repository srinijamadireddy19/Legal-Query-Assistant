import "../styles/messages.css";

export function TypingIndicator() {
  return (
    <div className="typing-row">
      <span className="typing-label">Researching your query</span>
      <div className="typing-dots">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  );
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="error-banner">
      <strong className="error-banner__title">Connection Error</strong>
      <p className="error-banner__text">{message}</p>
    </div>
  );
}
