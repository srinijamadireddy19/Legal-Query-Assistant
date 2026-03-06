const api = "http://localhost:8000";

export async function uploadDocument(file) {
    const formData = new FormData();
    formData.append("file",file);

    const res = await fetch(`${api}/documents/upload`, {
        method: "POST",
        body: formData,
    });
    return res.json();
}

export async function processDocument(documentId) {
    const res = await fetch(`${api}/documents/${documentId}/process`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body : JSON.stringify({ collection_name: "legal_docs",
            model_version: "v1",
         }),
    });

    return res.json();
}

export async function queryRAG(query, conversationId) {
    const res = await fetch(`${api}/query/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      collection_name: "legal_docs",
      conversation_id: conversationId,
      k: 5,
    }),
  });

  return res.json();
}
