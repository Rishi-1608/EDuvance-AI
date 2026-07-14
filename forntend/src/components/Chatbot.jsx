import React, { useState, useEffect, useRef } from 'react';
import { sendChatMessage } from '@/lib/api';
import { MessageSquare, X, Send } from 'lucide-react';
import './Chatbot.css';

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([{ role: 'assistant', content: 'Hi! Ask me how this app works, or what media you have uploaded.' }]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
        scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage = { role: 'user', content: inputValue.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send the entire chat history (backend limits it to last 5 internally)
      const data = await sendChatMessage(newMessages);
      setMessages([...newMessages, { role: 'assistant', content: data.response }]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages([...newMessages, { role: 'assistant', content: 'Sorry, I ran into an error getting the answer.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chatbot-container">
      {!isOpen && (
        <button className="chatbot-toggle-btn animate-in" onClick={() => setIsOpen(true)}>
          <MessageSquare size={24} />
        </button>
      )}

      {isOpen && (
        <div className="chatbot-window slide-up">
          <div className="chatbot-header">
            <div className="chatbot-header-info">
              <MessageSquare size={18} />
              <span>EDUvance AI Assistant</span>
            </div>
            <button className="chatbot-close-btn" onClick={() => setIsOpen(false)}>
              <X size={20} />
            </button>
          </div>

          <div className="chatbot-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chatbot-message ${msg.role}`}>
                {msg.content}
              </div>
            ))}
            {isLoading && (
              <div className="chatbot-message assistant loading">
                <span /><span /><span />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-area">
            <input
              type="text"
              placeholder="Ask a question..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
              disabled={isLoading}
            />
            <button className="chatbot-send-btn" onClick={handleSend} disabled={isLoading || !inputValue.trim()}>
              <Send size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
