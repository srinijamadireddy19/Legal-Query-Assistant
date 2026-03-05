import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import UploadPanel from './components/UploadPanel'
import ChatPanel from './components/ChatPanel'

function App() {
  const [ready, setReady] = useState(false);
  const [documents, setDocuments] = useState([]);

  return (
    <div className='main-container'>
      <div className='title'>
        <h1>Legal RAG UI</h1>
      </div>

      <div className='container'>
        <div className='sidebar'>
          <UploadPanel setReady={setReady} documents={documents} setDocuments={setDocuments} />
        </div>

        <div className='chat'>
          <ChatPanel ready={ready} />
        </div>
      </div>
    
      
    </div>
  )
}

export default App
