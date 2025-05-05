import React, { useState, useRef, useEffect } from 'react';
import './normal.css';
import './App.css';
import { TypeAnimation } from 'react-type-animation';


function App() {
  const [messages, setMessages] = useState([
  ]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const chatLogRef = useRef(null);

  const Backend_URL = "https://4354-34-16-130-203.ngrok-free.app"

  useEffect(() => {
    if (chatLogRef.current) {
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }
  }, [messages]);

  function handleStart() {
    const aiStart = {
      id: 0,
      text: "Hello! I'm Zoya, Your medical assistant. How can I assist you today?",
      sender: 'chatgpt'
    };
    setMessages((prevMessages) => [...prevMessages, aiStart]);
  }

  console.log(messages.length);

  async function handleSend() {
    if (isLoading) return; 
    if (!userInput.trim()) return; 

    const newMessage = {
      id: messages.length + 1,
      text: userInput,
      sender: 'user'
    };
    setMessages([...messages, newMessage]);

    setUserInput('');

    setIsLoading(true);

    try {
      console.log('here 1')
      const response = await fetch(`${Backend_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userInput })
      });
      console.log(response)
      const data = await response.json();
      console.log(data)
      const aiResponse = {
        id: messages.length + 2,
        text: data.reply,
        sender: 'chatgpt'
      };
      setMessages((prevMessages) => [...prevMessages, aiResponse]);
      console.log("Backend response:", data);
    
    } catch (error) {
      console.error("Error fetching from backend:", error);
    }
    setIsLoading(false);
    }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      console.log('Enter pressed, sending message...');
      handleSend();
    }
  }

  async function handleDelete(e) {
    try {
      const response = await fetch(`${Backend_URL}/clear-chat`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      setMessages([])
    
    } catch (error) {
      console.error("Error deleting chat", error);
    }
    
  }




  return (
    <div className="App"
    style={{
      backgroundImage: "url('/images/Background.png')",
      backgroundSize: "cover",
      backgroundRepeat: "no-repeat",
      backgroundPosition: "center center",
      height: "100vh",           // or any desired height
      width: "100%",             // optional
    }}>

      <div className="bg-container"></div>

      <section className="chatbox">

        <header className="chat-title">
          <div>Althea AI</div>
          <svg style={{ cursor: 'pointer' }} onClick={handleDelete}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            fill="none"
            stroke="#ff0000"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            viewBox="0 0 24 24"
          >
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
            <line x1="10" y1="11" x2="10" y2="17"></line>
            <line x1="14" y1="11" x2="14" y2="17"></line>
            <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path>
          </svg>

        </header>
        <div className="line" />
        <div className="container">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <div>
                <h2>Welcome to Althea AI</h2>
                <p style={{ fontSize: "15px" }}>Your trusted companion for clear, caring, and intelligent medical advice.</p>
                <div className='outer-start-button'>
                  <p onClick={() => handleStart()} className='start-button'>Get Started <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    fill="white"
                    viewBox="0 0 24 24">
                    <path d="M10 17l5-5-5-5v10z" />
                  </svg>
                  </p>
                </div>
              </div>
              <div></div>
            </div>
          ) : (
            <>
              <div className="chat-log" ref={chatLogRef}>
                {messages.map((msg) => (
                  <div key={msg.id} className={`chat-message ${msg.sender}`}>
                    <div className="chat-message-center">
                    {msg.sender === 'chatgpt' && (
                      <img src="/images/logo.webp" alt="Bot Avatar" className="avatar avatar-left" />
                    )}
                      <div className="message">
                        {msg.sender === 'chatgpt' ? (
                          <TypeAnimation
                            sequence={[
                              msg.text
                            ]}
                            speed={70}
                            cursor={false}
                            repeat={0}
                          />
                        ) : (
                          msg.text
                        )}
                      </div>

                      {msg.sender === 'user' && (
                        <img src="/images/user.png" alt="User Avatar" className="avatar avatar-right" />
                      )}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="chat-message chatgpt">
                    <div className="chat-message-center">
                      <img src="/images/logo.webp" alt="Bot Avatar" className="avatar avatar-left" />
                      <div className="reasoning-loader">
                        Thinking...
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="chat-input-holder">
                <div className="chat-input-wrapper">
                  <textarea
                    rows="1"
                    className="chat-input-textarea"
                    placeholder="Ask me anything :)"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />
                  <button className="chat-send-button" onClick={handleSend}>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="18"
                      height="18"
                      fill="none"
                      stroke="url(#gradient)"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      viewBox="0 0 24 24"
                    >
                      <defs>
                      <linearGradient id="gradient" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="24" y2="0" gradientTransform="rotate(325)">
                        <stop offset="0%" stop-color="#007bff" />       
                        <stop offset="50%" stop-color="#00bcd4" />      
                        <stop offset="100%" stop-color="#a3cfbb" />     
                      </linearGradient>

                      </defs>
                      <line x1="22" y1="2" x2="11" y2="13" />
                      <polygon points="22 2 15 22 11 13 2 9 22 2" />
                    </svg>

                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}

export default App;