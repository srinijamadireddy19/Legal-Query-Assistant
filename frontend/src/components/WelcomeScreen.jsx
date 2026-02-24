import "../styles/welcome.css";
import { BalanceScaleIcon } from "./Icons";

const SUGGESTED_QUERIES = [
  "What are my rights if my landlord refuses to return my security deposit?",
  "How does the statute of limitations work for personal injury cases?",
  "What constitutes wrongful termination under employment law?",
  "Explain the difference between civil and criminal liability.",
];

export default function WelcomeScreen({ onSelectQuery }) {
  return (
    <div className="welcome">
      <div className="welcome__emblem">
        <BalanceScaleIcon style={{ width: "100%", height: "100%" }} />
      </div>

      <h2 className="welcome__heading">Your Legal Counsel</h2>
      <p className="welcome__tagline">
        "The law is reason free from passion." — Aristotle
      </p>

      <div className="ornament">Suggested Questions</div>

      <div className="suggestions">
        {SUGGESTED_QUERIES.map((q, i) => (
          <button
            key={i}
            className="suggestion-chip"
            onClick={() => onSelectQuery(q)}
          >
            <span className="suggestion-chip__icon">§</span>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}