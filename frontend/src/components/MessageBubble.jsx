import "../styles/messages.css";
import { BalanceScaleIcon } from "./Icons";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const rowClass  = `message-row ${isUser ? "message-row--user" : "message-row--assistant"}`;
  const bubbleClass = `message-bubble ${isUser ? "message-bubble--user" : "message-bubble--assistant"}`;

  return (
    <div className={rowClass}>
      {/* Assistant avatar (left) */}
      {!isUser && (
        <div className="message-avatar message-avatar--assistant">
          <BalanceScaleIcon style={{ width: "100%", height: "100%" }} />
        </div>
      )}

      {/* Bubble */}
      <div className={bubbleClass}>
        {message.content}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <span className="message-sources__label">Sources</span>
            {message.sources.map((src, i) => (
              <div key={i} className="message-sources__item">
                <span className="message-sources__item-icon">§</span>
                <span>
                  {typeof src === "string"
                    ? src
                    : src.source || src.title || src.file || JSON.stringify(src)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* User avatar (right) */}
      {isUser && (
        <div className="message-avatar message-avatar--user">U</div>
      )}
    </div>
  );
}