const API_BASE = "http://localhost:8000";

/**
 * Tries multiple common FastAPI RAG endpoint patterns and returns
 * { answer, sources } on success, or throws on total failure.
 */
export async function queryLegal(query, sessionId) {
  const attempts = [
    {
      url: `${API_BASE}/query`,
      body: { query, session_id: sessionId },
    },
    {
      url: `${API_BASE}/chat`,
      body: { message: query, session_id: sessionId },
    },
    {
      url: `${API_BASE}/ask`,
      body: { question: query },
    },
    {
      url: `${API_BASE}/api/query`,
      body: { query, session_id: sessionId },
    },
  ];

  let lastError;

  for (const attempt of attempts) {
    try {
      const res = await fetch(attempt.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(attempt.body),
      });

      if (!res.ok) continue;

      const data = await res.json();

      const answer =
        data.answer   ||
        data.response ||
        data.result   ||
        data.message  ||
        data.text     ||
        (typeof data === "string" ? data : JSON.stringify(data));

      const sources =
        data.sources    ||
        data.documents  ||
        data.references ||
        data.context    ||
        [];

      return { answer, sources };
    } catch (err) {
      lastError = err;
    }
  }

  throw new Error(
    lastError?.message ||
    "Unable to reach the server. Make sure the backend is running on http://localhost:8000"
  );
}