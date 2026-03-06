import React, { useState } from "react";
import { uploadDocument, processDocument } from "../api";

export default function UploadPanel({ setReady, documents, setDocuments }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) {
      setError("Select a file first");
      return;
    }

    try {
      setUploading(true);
      setError("");

      const upload = await uploadDocument(file);
      const docId = upload.document_id;

      await processDocument(docId);

      setDocuments((prev) => [...prev, file.name]);

      setReady(true);
      setFile(null);
    } catch (err) {
      setError("Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <h3>Documents</h3>

      <ul>
        {documents.map((doc, i) => (
          <li key={i} style={{ listStyleType: "none" }}>  {doc}</li>
        ))}
      </ul>

      <input
        type="file"
        disabled={uploading}
        onChange={(e) => setFile(e.target.files[0])}
      />

      {error && <p style={{ color: "red" }}>{error}</p>}
        <br /><br /><br />
      <button onClick={handleUpload} disabled={!file || uploading}>
        {uploading ? "Processing..." : "Upload"}
      </button>
    </div>
  );
}