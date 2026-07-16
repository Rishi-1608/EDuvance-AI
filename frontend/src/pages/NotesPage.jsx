import { useState, useEffect, useRef } from 'react';
import { FileText, Download, RefreshCw, AlertCircle, Search, Zap, CheckCircle, Layers, Volume2, VolumeX, Mic, Video } from 'lucide-react';
import { useStatus } from '@/hooks/useStatus';
import { getStudyNotes, getPdfUrl, downloadPdf, generateFlashcardsAndQuiz } from '@/lib/api';
import { toStem, shortStem } from '@/lib/utils';
import './NotesPage.css';

function MarkdownViewer({ content }) {
  const render = (md) => {
    if (!md) return '';
    let html = md
      // Headers (order matters — longest prefix first)
      .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      // Inline formatting
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      // Horizontal rules
      .replace(/^---$/gm, '<hr />')
      // Blockquotes
      .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
      // Unordered list items
      .replace(/^\- (.+)$/gm, '<li>$1</li>')
      .replace(/^\* (.+)$/gm, '<li>$1</li>')
      // Numbered list items
      .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
      // Paragraphs
      .replace(/\n\n/g, '</p><p>');

    // Wrap consecutive <li> groups in <ul>
    html = html.replace(/(<li>.*?<\/li>(\s*)?)+/gs, (match) => `<ul>${match}</ul>`);

    return html;
  };
  return (
    <div
      className="markdown-viewer"
      dangerouslySetInnerHTML={{ __html: render(content) }}
    />
  );
}

function DraggableAction({ children }) {
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, elX: 0, elY: 0 });
  const hasMoved = useRef(false);

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    hasMoved.current = false;
    dragStart.current = {
      x: e.clientX,
      y: e.clientY,
      elX: pos.x,
      elY: pos.y,
    };
    
    // We bind to window/document to keep dragging even if cursor leaves the button
    const handleMouseMove = (e) => {
      e.preventDefault();
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        hasMoved.current = true;
      }
      setPos({
        x: dragStart.current.elX + dx,
        y: dragStart.current.elY + dy,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleCaptureClick = (e) => {
    if (hasMoved.current) {
      e.stopPropagation();
      e.preventDefault();
    }
  };

  return (
    <div
      className="notes-floating-action animate-in fade-in slide-in-from-bottom-4 duration-500"
      style={{
        transform: `translate(${pos.x}px, ${pos.y}px)`,
        cursor: isDragging ? 'grabbing' : 'grab',
        transition: isDragging ? 'none' : 'transform 0.05s linear',
        userSelect: 'none'
      }}
      onMouseDown={handleMouseDown}
      onClickCapture={handleCaptureClick}
    >
      {children}
    </div>
  );
}

export function NotesPage({ preselectedStem }) {
  const { status } = useStatus(5000);
  const [selectedStem, setSelectedStem] = useState(preselectedStem || '');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [generating, setGenerating] = useState(false);
  const [genMsg, setGenMsg] = useState(null);
  const [genError, setGenError] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const videos = status?.videos || {};
  const allKeys = Object.keys(videos);
  const allStems = allKeys.map(toStem);
  const stemToKey = Object.fromEntries(allKeys.map(k => [toStem(k), k]));
  const readyStems = allKeys.filter(k => videos[k].study_notes_ready).map(toStem);

  useEffect(() => {
    if (preselectedStem) {
      setSelectedStem(preselectedStem);
    } else if (!selectedStem && readyStems.length > 0) {
      setSelectedStem(readyStems[readyStems.length - 1]);
    }
  }, [preselectedStem, readyStems.join(',')]);

  const fetchNotes = async (stem) => {
    setLoading(true);
    setError(null);
    setNotes('');
    setGenMsg(null);
    setGenError(null);
    try {
      const text = await getStudyNotes(stem);
      setNotes(text);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedStem) return;
    setGenerating(true);
    setGenMsg(null);
    setGenError(null);
    try {
      const res = await generateFlashcardsAndQuiz(selectedStem);
      setGenMsg(res?.message || 'Flashcard & quiz generation started!');
    } catch (e) {
      setGenError(e.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleToggleSpeak = () => {
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    if (!notes) return;

    // Clean markdown for better speech
    const cleanText = notes
      .replace(/[#*`]/g, '')
      .replace(/\[.+?\]\(.+?\)/g, '')
      .replace(/\n+/g, ' ');

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  };

  // Stop speaking if we switch stems
  useEffect(() => {
    return () => window.speechSynthesis.cancel();
  }, []);

  useEffect(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [selectedStem]);

  useEffect(() => {
    if (selectedStem) fetchNotes(selectedStem);
  }, [selectedStem]);

  const filtered = notes && search
    ? notes.split('\n').filter(l => l.toLowerCase().includes(search.toLowerCase())).join('\n')
    : notes;

  const selectedVideoData = selectedStem ? videos[stemToKey[selectedStem]] : null;
  const flashcardsAlreadyReady = selectedVideoData?.flashcards_ready && selectedVideoData?.quiz_ready;
  const notesReady = selectedVideoData?.study_notes_ready;

  return (
    <div className="notes-layout-container animate-in fade-in zoom-in-95 duration-300">
      {/* ── Main Content Panel (Left) ── */}
      <div className="notes-main flex-1 overflow-hidden flex flex-col relative z-0">

        <div className="notes-top-bar flex items-center justify-between px-10 py-6 border-b border-glass flex-shrink-0">
          <div className="flex flex-col flex-1 truncate mr-8">
            <h2 className="text-2xl font-display font-semibold m-0 text-white flex items-center gap-3 truncate">
              <FileText size={24} className="text-accent flex-shrink-0" />
              <span className="truncate">{selectedStem ? (selectedVideoData?.lecture_title || selectedVideoData?.display_name || selectedStem) : "Study Notes"}</span>
            </h2>
            <div className="flex items-center gap-4 mt-2">
              {notes && (
                <span className="text-xs font-mono text-muted tracking-wide uppercase opacity-70">
                  ~{Math.ceil(notes.split(' ').length / 200)} min read
                </span>
              )}
              {flashcardsAlreadyReady && (
                <span className="text-xs font-mono text-success uppercase tracking-widest bg-success/10 px-2 py-0.5 rounded border border-success/20">
                  Deck Ready
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-4">
            {notes && (
              <div className="search-container flex items-center gap-2 mr-4">
                <div className="search-box-wrapper relative flex items-center">
                  <Search size={14} className="absolute left-3.5 text-muted pointer-events-none" />
                  <input
                    type="text"
                    placeholder="Search note content..."
                    className="search-input-field pl-10 pr-10 py-2.5 bg-black/30 border border-white/10 rounded-xl text-sm focus:ring-1 focus:ring-accent/40 focus:bg-black/50 focus:border-accent/40 focus:w-64 w-44 transition-all duration-500 outline-none text-white placeholder:text-white/30"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                  />
                  {search && (
                    <button
                      onClick={() => setSearch('')}
                      className="absolute right-3 text-muted/30 hover:text-white transition-colors"
                    >
                      <AlertCircle size={14} />
                    </button>
                  )}
                </div>
              </div>
            )}
            <div className="flex items-center gap-2">
              {selectedStem && (
                <>
                  <button className="btn btn-ghost btn-sm bg-black/20 hover:bg-black/40" onClick={() => fetchNotes(selectedStem)}>
                    <RefreshCw size={14} /> Reload
                  </button>
                  <button className="btn btn-ghost btn-sm bg-black/20 hover:bg-black/40" onClick={handleToggleSpeak}>
                    {isSpeaking ? <VolumeX size={14} className="text-error" /> : <Volume2 size={14} />}
                    <span className="ml-1.5">{isSpeaking ? 'Stop' : 'Listen'}</span>
                  </button>
                  <button className="btn btn-primary btn-sm" onClick={() => downloadPdf(selectedStem).catch(e => alert(e.message))}>
                    <Download size={14} /> PDF
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="notes-content-area flex-1 overflow-y-auto px-10 py-8 relative z-0">
          {genMsg && (
            <div className="mb-6 animate-in fade-in slide-in-from-top-4">
              <div className="glass p-4 rounded-xl border-success/30 bg-success/5 text-success flex items-center justify-between">
                <div className="flex items-center gap-3 text-sm">
                  <CheckCircle size={16} /> {genMsg}
                </div>
                <button onClick={() => setGenMsg(null)} className="text-success/60 hover:text-success"><AlertCircle size={16} /></button>
              </div>
            </div>
          )}

          {error ? (
            <div className="flex flex-col items-center justify-center p-20 text-error">
              <AlertCircle size={48} className="opacity-40 mb-4" />
              <p className="font-semibold">{error}</p>
            </div>
          ) : loading ? (
            <div className="flex flex-col items-center justify-center p-20">
              <div className="spinner mb-4" />
              <p className="text-muted mono text-sm animate-pulse">Syncing with Intelligence Database...</p>
            </div>
          ) : notes ? (
            <div className="notes-viewer-container max-w-4xl mx-auto pb-20">
              {search && (
                <div className="mb-6 text-xs font-mono text-accent uppercase tracking-widest opacity-60">
                  Showing results for: "{search}"
                </div>
              )}
              <MarkdownViewer content={filtered} />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center p-20 text-muted opacity-40">
              <FileText size={64} className="mb-4" />
              <p className="font-display text-lg">Select a note from the switcher to begin.</p>
            </div>
          )}
        </div>

        {!flashcardsAlreadyReady && notesReady && (
          <DraggableAction>
            <button
              className="btn btn-primary btn-lg shadow-2xl flex items-center gap-3 px-8 rounded-2xl border border-white/10"
              disabled={generating}
              onClick={handleGenerate}
            >
              {generating ? <div className="spinner w-4 h-4" /> : <Zap size={18} />}
              {generating ? 'Engine Processing...' : 'Generate Flashcard Deck'}
            </button>
          </DraggableAction>
        )}
      </div>

      {/* ── Right Panel: Note Switcher ── */}
      <div className="notes-sidebar z-20 relative bg-black/40 border-l border-glass w-[340px] flex flex-col flex-shrink-0">
        <div className="sidebar-header p-6 border-b border-glass flex items-center justify-between shadow-sm bg-black/20">
          <h3 className="text-lg font-display font-semibold m-0 text-white flex items-center gap-2">
            <Layers size={18} className="text-accent" />
            Select Note Deck
          </h3>
        </div>
        <div className="sidebar-list overflow-y-auto flex-1 p-4 space-y-3">
          {allStems.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted">No media indexed yet.</div>
          ) : (
            allStems.map(s => {
              const key = stemToKey[s];
              const vid = videos[key];
              const isReady = vid?.study_notes_ready;
              const isSelected = selectedStem === s;
              const isAudioOnly = vid?.audio_ready && !vid?.summary_ready;

              return (
                <div
                  key={s}
                  className={`sidebar-item flex items-center gap-4 p-4 rounded-xl cursor-pointer transition-all border ${isSelected ? 'bg-active border-glow shadow-glow text-primary block-active scale-[1.02]' : (isReady ? 'border-glass hover:bg-hover hover:border-glass text-secondary shadow-sm bg-black/20' : 'opacity-40 cursor-not-allowed border-transparent bg-black/10')}`}
                  onClick={() => isReady && setSelectedStem(s)}
                >
                  <div className={`icon w-12 h-12 flex items-center justify-center rounded-xl flex-shrink-0 border border-white/5 ${isSelected ? 'bg-black/40 text-accent shadow-inner' : 'bg-black/60 text-muted'}`}>
                    {vid?.input_type === 'document' ? <Layers size={20} /> : (isAudioOnly ? <Mic size={20} /> : <Video size={20} />)}
                  </div>
                  <div className="info overflow-hidden flex-1">
                    <div className="filename truncate text-sm font-semibold mb-1" title={vid?.lecture_title || vid?.display_name || s}>{vid?.lecture_title || vid?.display_name || s}</div>
                    <div className="meta text-xs font-mono opacity-80 flex items-center gap-1.5">
                      {isReady ? (
                        <span className="text-success flex items-center gap-1"><CheckCircle size={12} /> Notes Ready</span>
                      ) : (
                        <span className="text-warning flex items-center gap-1 animate-pulse"><RefreshCw size={12} className="animate-spin-slow" /> Indexing</span>
                      )}
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  );
}
