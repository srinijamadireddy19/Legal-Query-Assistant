import { useState } from 'react'
import logo from './assets/logo.jpg'
import viteLogo from '/vite.svg'
import './App.css'
import UploadPanel from './components/UploadPanel'
import ChatPanel from './components/ChatPanel'

function App() {
  const [ready, setReady] = useState(false);
  const [documents, setDocuments] = useState([]);

  return (
    <div className="main-container">
    <div className="title">
      <img src={logo} className="logo" />
    <div className="title-text">
      <h1>LEXORA-AI</h1>
      <h3>Legal Expert Retrieval & Explainable Reasoning Assistant</h3>
    </div>
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
