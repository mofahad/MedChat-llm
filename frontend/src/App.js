import React, { useState, useRef, useEffect } from 'react';
import './normal.css';
import './App.css';

const API_URL = (process.env.REACT_APP_API_URL || 'http://127.0.0.1:5005').replace(/\/$/, '');

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const logRef = useRef(null);
  const sequence = useRef(0);
  const pending = useRef(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [messages, isLoading]);
  useEffect(() => () => pending.current?.abort(), []);

  function start() {
    setMessages([{ id: ++sequence.current, sender: 'chatgpt', text: 'Ask about the MedChat model, retrieval, or limitations. I return project source excerpts; no LLM is running.' }]);
  }
  async function send() {
    const question = userInput.trim();
    if (pending.current || !question) return;
    const controller = new AbortController();
    pending.current = controller;
    setIsLoading(true); setError(''); setUserInput('');
    setMessages(rows => [...rows, { id: ++sequence.current, sender: 'user', text: question }]);
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }), signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Backend returned ${response.status}.`);
      if (typeof data.reply !== 'string') throw new Error('The backend returned an invalid reply.');
      setMessages(rows => [...rows, { id: ++sequence.current, sender: 'chatgpt', text: data.reply, sources: data.sources || [] }]);
    } catch (failure) {
      setError(failure.name === 'AbortError' ? 'Request timed out. Please try again.' : 'Cannot retrieve sources. Check that the local backend is running and try again.');
    } finally {
      clearTimeout(timeout); pending.current = null; setIsLoading(false);
    }
  }
  function clear() { if (!isLoading) { setMessages([]); setError(''); setUserInput(''); } }

  return <div className="App"><section className="chatbox" aria-label="MedChat source explorer">
    <header className="chat-title"><div>MedChat · Source explorer</div><button onClick={clear} disabled={isLoading} aria-label="Clear chat">Clear</button></header>
    <div className="line" /><p className="research-notice">Research demo · source excerpts, not medical advice · no LLM running</p>
    <div className="container">
      {messages.length === 0 ? <div className="empty-chat"><div><h2>Explore the project’s sources</h2><p>Free local retrieval. No keys, GPU, or external model downloads.</p><div className="outer-start-button"><button onClick={start} className="start-button">Get Started</button></div></div></div> : <>
        <div className="chat-log" ref={logRef} aria-live="polite">
          {messages.map(message => <div key={message.id} className={`chat-message ${message.sender}`}><div className="chat-message-center"><div className="message"><p>{message.text}</p>{message.sources?.length > 0 && <ul className="source-list">{message.sources.map(source => <li key={source.id}>[{source.id}] {source.source}</li>)}</ul>}</div></div></div>)}
          {isLoading && <p role="status">Retrieving sources…</p>}
        </div>
        <div className="chat-input-holder"><div className="chat-input-wrapper"><textarea rows="1" maxLength="2000" className="chat-input-textarea" aria-label="Question" placeholder="Ask about the model or project…" value={userInput} onChange={e => setUserInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} /><button className="chat-send-button" onClick={send} disabled={isLoading || !userInput.trim()} aria-label="Send question">↗</button></div></div>
      </>}
      {error && <p className="request-error" role="alert">{error}</p>}
    </div>
  </section></div>;
}
export default App;
