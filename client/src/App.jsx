import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import UploadPanel from './components/UploadPanel'
import ChatPanel from './components/ChatPanel'

function App() {
  const [ready, setReady] = useState(false);

  return (
    <div style={{ padding: 20 }}>
      <h1>Legal RAG UI</h1>

      {!ready ? (
        <UploadPanel setReady={setReady} />
      ) : (
        <ChatPanel />
      )}
    </div>
  )
}

export default App
