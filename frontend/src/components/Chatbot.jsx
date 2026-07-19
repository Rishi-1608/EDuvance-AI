import React, { useState, useEffect, useRef, useCallback } from 'react';
import { sendChatMessage } from '@/lib/api';
import { MessageSquare, X, Send } from 'lucide-react';
import './Chatbot.css';

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([{ role: 'assistant', content: 'Hi! Ask me how this app works, or what media you have uploaded.' }]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // ── Drag state ──
  const [btnPos, setBtnPos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef({ startX: 0, startY: 0, origX: 0, origY: 0, moved: false });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) scrollToBottom();
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage = { role: 'user', content: inputValue.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      const data = await sendChatMessage(newMessages);
      setMessages([...newMessages, { role: 'assistant', content: data.response }]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages([...newMessages, { role: 'assistant', content: 'Sorry, I ran into an error getting the answer.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Drag handlers (mouse + touch) ──
  const handleDragStart = useCallback((clientX, clientY) => {
    setIsDragging(true);
    dragRef.current = {
      startX: clientX,
      startY: clientY,
      origX: btnPos.x,
      origY: btnPos.y,
      moved: false,
    };
  }, [btnPos]);

  const handleDragMove = useCallback((clientX, clientY) => {
    if (!isDragging) return;
    const dx = clientX - dragRef.current.startX;
    const dy = clientY - dragRef.current.startY;
    if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
      dragRef.current.moved = true;
    }
    setBtnPos({
      x: dragRef.current.origX + dx,
      y: dragRef.current.origY + dy,
    });
  }, [isDragging]);

  const handleDragEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Mouse events
  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    handleDragStart(e.clientX, e.clientY);

    const onMove = (ev) => {
      ev.preventDefault();
      handleDragMove(ev.clientX, ev.clientY);
    };
    const onUp = () => {
      handleDragEnd();
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [handleDragStart, handleDragMove, handleDragEnd]);

  // Touch events
  const onTouchStart = useCallback((e) => {
    const t = e.touches[0];
    handleDragStart(t.clientX, t.clientY);
  }, [handleDragStart]);

  const onTouchMove = useCallback((e) => {
    const t = e.touches[0];
    handleDragMove(t.clientX, t.clientY);
  }, [handleDragMove]);

  const onTouchEnd = useCallback(() => {
    handleDragEnd();
  }, [handleDragEnd]);

  // Click handler — suppress if drag occurred
  const handleToggleClick = useCallback(() => {
    if (dragRef.current.moved) {
      dragRef.current.moved = false;
      return;
    }
    setIsOpen(true);
  }, []);

  return (
    <div className="chatbot-container">
      {!isOpen && (
        <button
          className="chatbot-toggle-btn animate-in"
          style={{
            transform: `translate(${btnPos.x}px, ${btnPos.y}px)`,
            cursor: isDragging ? 'grabbing' : 'grab',
            transition: isDragging ? 'none' : 'transform 0.15s ease, box-shadow 0.2s ease',
            touchAction: 'none',
          }}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
          onClick={handleToggleClick}
        >
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

