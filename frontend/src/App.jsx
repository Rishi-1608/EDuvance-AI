import { useState, useEffect } from 'react';
import { useTheme } from '@/hooks/useTheme';
import { Sidebar } from '@/components/layout/Sidebar';
import { UploadPage } from '@/pages/UploadPage';
import { DashboardPage } from '@/pages/DashboardPage';
import { LibraryPage } from '@/pages/LibraryPage';
import { VideoDetailsPage } from '@/pages/VideoDetailsPage';
import { NotesPage } from '@/pages/NotesPage';
import { FlashcardsPage } from '@/pages/FlashcardsPage';
import { QuizPage } from '@/pages/QuizPage';
import { DiagnosticsPage } from '@/pages/DiagnosticsPage';
import { AudioResultsPage } from '@/pages/AudioResultsPage';
import { ActivityPage } from '@/pages/ActivityPage';
import { LoginPage } from '@/pages/LoginPage';
import { LiveLecturePage } from '@/pages/LiveLecturePage';
import { useStatus } from '@/hooks/useStatus';
import { getMe } from '@/lib/api';
import './App.css';
import { Navbar } from '@/components/layout/Navbar';
import { ArrowLeft } from 'lucide-react';
import { Chatbot } from '@/components/Chatbot';


export default function App() {
  const { theme, toggle } = useTheme();
  const [user, setUser] = useState(null);
  const [tab, setTab] = useState('home');
  const [history, setHistory] = useState(['home']);
  const [selectedStem, setSelectedStem] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const { status } = useStatus(8000, !!user);

  useEffect(() => {
    // Check if we have a token and fetch user info
    const token = localStorage.getItem('eduvance_ai_token');
    if (token) {
      getMe()
        .then(setUser)
        .catch(() => localStorage.removeItem('eduvance_ai_token'));
    }
  }, []);

  const handleTabChange = (newTab) => {
    if (newTab !== tab) {
      setHistory(prev => [...prev, newTab]);
      setTab(newTab);
    }
  };

  const handleBack = () => {
    if (history.length > 1) {
      const newHistory = [...history];
      newHistory.pop(); // Remove current
      const prevTab = newHistory[newHistory.length - 1];
      setHistory(newHistory);
      setTab(prevTab);
    }
  };

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem('eduvance_ai_token');
    setUser(null);
  };

  const navigateToDetails = (stem) => {
    setSelectedStem(stem);
    handleTabChange('details');
  };

  const navigateToResult = (resultTab, stem) => {
    setSelectedStem(stem);
    handleTabChange(resultTab);
  };

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  const pages = {
    home: <UploadPage 
      onStart={(types) => {
        setIsUploading(true);
        if (types.some(t => ['video', 'image', 'document', 'pdf'].includes(t))) {
          handleTabChange('dashboard');
        } else if (types.includes('audio')) {
          handleTabChange('audio');
        }
      }}
      onUploadComplete={() => {
        setIsUploading(false);
      }} 
    />,
    dashboard: <DashboardPage onTabChange={handleTabChange} isUploading={isUploading} />,
    library: <LibraryPage onSelectVideo={navigateToDetails} />,
    details: <VideoDetailsPage
      stem={selectedStem}
      onBack={handleBack}
      onTabChange={navigateToResult}
    />,
    notes: <NotesPage preselectedStem={selectedStem} />,
    flashcards: <FlashcardsPage preselectedStem={selectedStem} />,
    quiz: <QuizPage preselectedStem={selectedStem} />,
    audio: <AudioResultsPage />,
    live: <LiveLecturePage onResult={(stem) => navigateToResult('notes', stem)} />,
    activity: <ActivityPage user={user} />,
    diagnostics: <DiagnosticsPage />,
  };

  return (
    <div className="app">
      <Sidebar
        activeTab={tab}
        onTabChange={handleTabChange}
        pipelineRunning={status?.pipeline_running}
        user={user}
        onLogout={handleLogout}
      />
      <main className="app-main" key={tab}>
        <Navbar
          theme={theme}
          onToggleTheme={toggle}
          onBack={handleBack}
          historyLength={history.length}
          activeTab={tab}
        />
        <div className="app-content-area">
          {pages[tab]}
        </div>
      </main>
      <Chatbot />
    </div>
  );

}