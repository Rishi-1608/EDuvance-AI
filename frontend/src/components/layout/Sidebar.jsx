import { Sun, Moon, Home, LayoutDashboard, Library, BookOpen, Layers, HelpCircle, Mic, Cpu, ChevronLeft, ChevronRight, Zap, LogOut, User as UserIcon, BarChart3, Radio } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import { useState } from 'react';
import './Sidebar.css';

const NAV_ITEMS = [
  { id: 'home', label: 'Home', icon: Home, section: 'main' },
  { id: 'activity', label: 'Insights', icon: BarChart3, section: 'main' },
  { id: 'dashboard', label: 'Processing', icon: Zap, section: 'main' },
  { id: 'library', label: 'Library', icon: Library, section: 'main' },
  { id: 'notes', label: 'Study Notes', icon: BookOpen, section: 'results' },
  { id: 'flashcards', label: 'Flashcards', icon: Layers, section: 'results' },
  { id: 'quiz', label: 'Quiz', icon: HelpCircle, section: 'results' },
  { id: 'audio', label: 'Audio', icon: Mic, section: 'results' },
  { id: 'live', label: 'Live Lecture', icon: Radio, section: 'main' },
  { id: 'diagnostics', label: 'System', icon: Cpu, section: 'system' },
];

const MOBILE_ITEMS = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'activity', label: 'Insights', icon: BarChart3 },
  { id: 'library', label: 'Library', icon: Library },
  { id: 'notes', label: 'Notes', icon: BookOpen },
  { id: 'flashcards', label: 'Cards', icon: Layers },
  { id: 'quiz', label: 'Quiz', icon: HelpCircle },
  { id: 'live', label: 'Live', icon: Radio },
];

export function Sidebar({ activeTab, onTabChange, pipelineRunning, user, onLogout }) {
  const { theme, toggle } = useTheme();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      {/* ── Desktop Sidebar ── */}
      <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-brand-clickable" onClick={() => onTabChange('home')}>
            <div className="sidebar-logo">
              <img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 'inherit' }} />
            </div>
            <div className="sidebar-brand-text">
              <span className="sidebar-brand-name gradient-text">EDuvance AI</span>

            </div>
          </div>

          <button
            className="sidebar-toggle-btn"
            onClick={() => setCollapsed(c => !c)}
            aria-label={collapsed ? 'Expand' : 'Collapse'}
            title={collapsed ? 'Expand' : 'Collapse'}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        {/* User Info Section */}
        <div className="sidebar-user-pill glass">
          <div className="sidebar-user-avatar">
            <UserIcon size={14} />
          </div>
          {!collapsed && (
            <div className="sidebar-user-details animate-in">
              <span className="sidebar-user-name mono">{user?.username}</span>
              <span className="sidebar-user-role">Student</span>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {NAV_ITEMS
            .map(item => (
              <button
                key={item.id}
                className={`sidebar-item ${activeTab === item.id ? 'active' : ''}`}
                onClick={() => onTabChange(item.id)}
                title={collapsed ? item.label : undefined}
              >
                <div className="sidebar-item-icon">
                  <item.icon size={18} />
                </div>
                <span className="sidebar-item-label">{item.label}</span>
                {activeTab === item.id && <div className="sidebar-active-indicator" />}
              </button>
            ))}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer">
          {/* Pipeline status */}
          {pipelineRunning && (
            <div className="sidebar-status">
              <div className="pulse-dot green" />
              <span className="sidebar-status-text">Processing…</span>
            </div>
          )}

          {/* Logout button */}
          <button
            className="sidebar-item sidebar-logout-btn"
            onClick={onLogout}
            aria-label="Sign out"
            title={collapsed ? 'Sign Out' : undefined}
          >
            <div className="sidebar-item-icon">
              <LogOut size={18} />
            </div>
            <span className="sidebar-item-label">Sign Out</span>
          </button>
        </div>
      </aside>

      {/* ── Mobile Bottom Bar ── */}
      <nav className="mobile-bar">
        {MOBILE_ITEMS.map(item => (
          <button
            key={item.id}
            className={`mobile-bar-item ${activeTab === item.id ? 'active' : ''}`}
            onClick={() => onTabChange(item.id)}
          >
            <item.icon size={18} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </>
  );
}
