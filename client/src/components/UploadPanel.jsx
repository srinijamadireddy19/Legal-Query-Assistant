import React, { useState } from "react";
import { uploadDocument, processDocument } from "../api";

export default function UploadPanel({ setReady }) {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    const upload = await uploadDocument(file);
    const docId = upload.document_id;

    await processDocument(docId);

    alert("Document processed!");
    setReady(true);
  };

  return (
    <div>
      <h3>Upload Document</h3>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload}>Upload & Process</button>
    </div>
  );
}