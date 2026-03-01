import React, { useState } from "react";
import { uploadDocument, processDocument } from "../api";

export default function UploadPanel({ setReady }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }

    if (file.size > 2 * 1024 * 1024) {
      setError("File too large. Max 2MB.");
      return;
    }

    try {
      setUploading (true);
      setError("");
      const upload = await uploadDocument(file);
      const docId = upload.document_id;

      await processDocument(docId);
      setUploading(true);

      setReady(true);
    } catch (err) {
      setError("An error occurred during upload. Please try again.");
      console.error(err);
    } finally {
      setUploading(false);
    }

  };

  return (
    <div>
      <h3>Upload Document</h3>

      <input
        type="file"
        onChange={(e) => {
          setFile(e.target.files[0]);
          setError("");
        }}
      />

      {error && (
        <p style={{ color: "red", marginTop: 8 }}>
          {error}
        </p>
      )}

      <button onClick={handleUpload} > {uploading ? "Processing..." : "Upload & Process"}</button>
      
    </div>
  );
}