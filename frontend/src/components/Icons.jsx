export function BalanceScaleIcon({ className, style }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={style}
    >
      <line x1="24" y1="4"  x2="24" y2="44" stroke="#bb9457" strokeWidth="2"   strokeLinecap="round" />
      <line x1="8"  y1="12" x2="40" y2="12" stroke="#bb9457" strokeWidth="2"   strokeLinecap="round" />
      <ellipse cx="10" cy="22" rx="8" ry="4"  stroke="#bb9457" strokeWidth="1.5" fill="none" />
      <ellipse cx="38" cy="22" rx="8" ry="4"  stroke="#bb9457" strokeWidth="1.5" fill="none" />
      <line x1="8"  y1="12" x2="10" y2="18" stroke="#bb9457" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="40" y1="12" x2="38" y2="18" stroke="#bb9457" strokeWidth="1.5" strokeLinecap="round" />
      <rect x="20" y="42" width="8" height="2" rx="1" fill="#bb9457" />
    </svg>
  );
}

export function SendIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ width: "100%", height: "100%" }}
    >
      <line x1="22" y1="2"  x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

export function TrashIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ width: "100%", height: "100%" }}
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4h6v2" />
    </svg>
  );
}