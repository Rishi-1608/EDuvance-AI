import React from 'react';
import { Sun, Moon, ArrowLeft } from 'lucide-react';
import './Navbar.css';

export function Navbar({ theme, onToggleTheme, onBack, historyLength, activeTab }) {
  // Convert tab ID to pretty title
  const getTabTitle = (id) => {
    const titles = {
      home: 'Home',
      dashboard: 'Pipeline Status',
      library: 'Media Library',
      details: 'Lecture Analytics',
      notes: 'Study Notes',
      flashcards: 'Smart Flashcards',
      quiz: 'Self-Assessment Quiz',
      audio: 'Audio Insights',
      activity: 'Intelligence Dashboard',
      diagnostics: 'System Health'
    };
    return titles[id] || id;
  };

  return (
    <nav className="navbar glass">
      <div className="navbar-left">
        {historyLength > 1 && (
          <button className="nav-btn-navbar" onClick={onBack} title="Go Back">
            <ArrowLeft size={18} />
          </button>
        )}
      </div>

      <div className="navbar-center">
        <h1 className="navbar-title glass-text">{getTabTitle(activeTab)}</h1>
      </div>

      <div className="navbar-right">
        <div className="navbar-actions">
          <button className="nav-btn-navbar theme-btn-navbar" onClick={onToggleTheme} title="Toggle Theme">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </div>
    </nav>
  );
}
