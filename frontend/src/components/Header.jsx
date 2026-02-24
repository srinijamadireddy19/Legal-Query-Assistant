import "../styles/layout.css";
import { BalanceScaleIcon, TrashIcon } from "./Icons";

export default function Header({ showClear, onClear }) {
  return (
    <header className="header">
      <div className="header__brand">
        <div className="header__logo">
          <BalanceScaleIcon style={{ width: "100%", height: "100%" }} />
        </div>
        <div>
          <h1 className="header__title">LEXIS COUNSEL</h1>
          <p className="header__subtitle">Legal Query Assistant</p>
        </div>
      </div>

      {showClear && (
        <button className="header__clear-btn" onClick={onClear}>
          <span className="header__clear-icon">
            <TrashIcon />
          </span>
          Clear
        </button>
      )}
    </header>
  );
}