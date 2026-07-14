import { useState, useEffect, useRef, useMemo } from 'react';
import { Mic, RefreshCw, AlertCircle, Clock, Volume2, FileText, Layers, Video, Search, Globe, AlignLeft } from 'lucide-react';
import { getAudioResults } from '@/lib/api';
import './AudioResultsPage.css';

function StatCard({ title, value, icon, className = '' }) {
  return (
    <div className={`bg-black/30 border border-glass rounded-xl p-4 flex flex-col justify-center items-start shadow-inner ${className}`}>
      <div className="text-accent mb-2">{icon}</div>
      <div className="text-xl font-display font-semibold text-white leading-none mb-1">{value}</div>
      <div className="text-xs text-muted uppercase tracking-wider">{title}</div>
    </div>
  );
}

function SegmentRow({ seg, index, isActive, onClick }) {
  const fmt = (s) => {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1).padStart(4, '0');
    return `${String(m).padStart(2, '0')}:${sec}`;
  };

  return (
    <div
      className={`seg-row flex items-start gap-4 p-3 rounded-md cursor-pointer transition-colors border border-transparent ${isActive ? 'bg-active border-glow' : 'hover:bg-hover hover:border-glass'}`}
      onClick={() => onClick(index)}
    >
      <div className="seg-time mono w-16 text-muted flex-shrink-0 text-xs mt-0.5">
        <span>{fmt(seg.start)}</span>
      </div>
      <div className={`seg-text flex-1 text-sm leading-relaxed ${isActive ? 'text-primary font-medium' : 'text-secondary'}`}>
        {seg.text?.trim()}
      </div>
    </div>
  );
}

export function AudioResultsPage() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);

  const fetch = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAudioResults();
      const validData = Array.isArray(data) ? data : [];
      setResults(validData);
      
      if (validData.length > 0) {
        if (!selectedResult || !validData.find(r => (r.audio_path || r.video_path) === (selectedResult.audio_path || selectedResult.video_path))) {
           setSelectedResult(validData[0]);
        }
      } else {
        setSelectedResult(null);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetch(); }, []);

  return (
    <div className="audio-layout-container animate-in">
      {/* Sidebar for Selection */}
      <div className="audio-sidebar z-10 relative">
        <div className="sidebar-header p-5 border-b border-glass flex items-center justify-between">
          <h2 className="text-lg font-display font-semibold flex items-center gap-2 m-0">
            <Mic size={18} className="text-accent" />
            Media Library
          </h2>
          <button className="p-2 rounded-lg hover:bg-hover text-muted hover:text-primary transition-colors" onClick={fetch} disabled={loading} title="Refresh">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
        
        <div className="sidebar-list overflow-y-auto flex-1 p-3 space-y-2">
          {loading ? (
            <div className="flex items-center justify-center p-8"><div className="spinner" /></div>
          ) : results.length === 0 ? (
            <div className="empty-msg text-muted p-6 text-center text-sm bg-black/20 rounded-lg border border-dashed border-glass m-2">
              No transcriptions available.<br/><span className="text-xs opacity-60 mt-2 block">Upload content to get started.</span>
            </div>
          ) : (
            results.map((r, i) => {
              const stem = r.video_stem || r.batch_stem || (r.audio_path ? r.audio_path.split(/[\\/]/).pop().replace(/\.[^/.]+$/, "") : null);
              const filename = r.lecture_title || r.display_name || stem || `Media ${i + 1}`;
              const isSelected = selectedResult === r;
              
              return (
                <div 
                  key={r.audio_path || r.video_path || i} 
                  className={`sidebar-item flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all border ${isSelected ? 'bg-active border-glow shadow-glow text-primary block-active' : 'border-transparent hover:bg-hover hover:border-glass text-secondary'}`}
                  onClick={() => setSelectedResult(r)}
                >
                  <div className={`icon w-10 h-10 flex items-center justify-center rounded-lg flex-shrink-0 ${isSelected ? 'bg-black/20 text-accent' : 'bg-black/40 text-muted'}`}>
                    {r.from_video ? <Video size={18} /> : <Mic size={18} />}
                  </div>
                  <div className="info overflow-hidden flex-1">
                    <div className="filename truncate text-sm font-semibold">{filename}</div>
                    <div className="meta text-xs opacity-70 flex items-center gap-1.5 mt-1">
                       <Layers size={10} /> {r.from_video ? 'Video' : 'Audio'}
                       <span>•</span>
                       {r.transcription?.segments?.length || 0} segments
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="audio-main">
        {selectedResult ? (
          <TranscriptViewer result={selectedResult} />
        ) : (
             <div className="flex items-center justify-center h-full text-muted flex-col gap-4">
                 <Mic size={48} className="opacity-30" />
                 <p className="text-sm font-medium">Select a transcription from the library to view.</p>
             </div>
        )}
      </div>
    </div>
  );
}

function TranscriptViewer({ result }) {
  const [activeSegment, setActiveSegment] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const playerRef = useRef(null);

  const t = result.transcription || {};
  const segments = t.segments || [];
  const duration = segments.length > 0 ? segments[segments.length - 1]?.end : 0;
  
  // Calculate word count quickly
  const fullText = t.text || segments.map(s => s.text).join(' ') || '';
  const wordCount = fullText.split(/\s+/).filter(Boolean).length;

  const stem = result.video_stem || result.batch_stem || (result.audio_path ? result.audio_path.split(/[\\/]/).pop().replace(/\.[^/.]+$/, "") : null);
  const mediaUrl = stem ? `${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/video/${stem}?token=${localStorage.getItem('eduvance_ai_token')}` : null;

  const filteredSegments = useMemo(() => {
    if (!searchTerm) return segments;
    return segments.filter(s => s.text?.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [segments, searchTerm]);

  const handleSegmentClick = (index) => {
    setActiveSegment(index);
    if (playerRef.current && segments[index]) {
      playerRef.current.currentTime = segments[index].start;
      playerRef.current.play().catch(e => console.log("Auto-play prevented:", e));
    }
  };

  const fmtDuration = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div className="transcript-viewer-container flex flex-col h-full w-full relative p-4 gap-4 animate-in fade-in zoom-in-95 duration-300">
      
      {/* ── TOP ROW: Video & Stats ── */}
      <div className="top-dashboard-row flex gap-4 flex-shrink-0">
        
        {/* Selected Video Player */}
        <div className="media-player-box flex-1 bg-black/50 border border-glass rounded-2xl flex justify-center items-center shadow-lg relative overflow-hidden">
          {result.from_video && mediaUrl ? (
            <video
              ref={playerRef}
              controls
              className="absolute inset-0 w-full h-full object-contain"
              src={mediaUrl}
            />
          ) : mediaUrl ? (
            <div className="audio-player-wrapper w-full p-6 flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center animate-pulse">
                <Mic size={32} className="text-accent" />
              </div>
              <audio
                ref={playerRef}
                controls
                className="w-full accent-player"
                src={mediaUrl}
              />
            </div>
          ) : (
            <div className="text-error flex items-center gap-2"><AlertCircle size={16}/> Media not available</div>
          )}
        </div>

        {/* Info Cards Grid (2x2) */}
        <div className="stats-cards-grid grid grid-cols-2 grid-rows-2 gap-3 flex-shrink-0">
          <StatCard 
             title="Duration" 
             value={duration > 0 ? fmtDuration(duration) : "—"} 
             icon={<Clock size={20}/>} 
          />
          <StatCard 
             title="Segments" 
             value={segments.length} 
             icon={<AlignLeft size={20}/>} 
          />
          <StatCard 
             title="Language" 
             value={t.language?.toUpperCase() || result.detected_language?.toUpperCase() || (result.from_video ? 'MIX' : 'EN')} 
             icon={<Globe size={20}/>} 
          />
          <StatCard 
             title="Word Count" 
             value={wordCount > 0 ? wordCount.toLocaleString() : "—"} 
             icon={<FileText size={20}/>} 
          />
        </div>

      </div>

      {/* ── BOTTOM ROW: Transcriptions ── */}
      <div className="transcript-content bg-black/30 rounded-2xl border border-glass flex flex-col flex-1 min-h-[300px] shadow-lg overflow-hidden relative">
        <div className="transcript-header p-4 bg-black/40 border-b border-white/5 flex items-center justify-between gap-4 backdrop-blur-md sticky top-0 z-20">
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-lg bg-accent/20 text-accent flex items-center justify-center">
                <FileText size={16} />
             </div>
             <div>
                <h3 className="text-sm font-display font-semibold m-0 text-white leading-tight">
                  {result.lecture_title || result.display_name || stem || 'Transcription Segments'}
                </h3>
             </div>
          </div>
          
          <div className="search-box relative w-full max-w-xs transition-transform hover:scale-[1.02]">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <input 
              type="text" 
              placeholder="Search transcript..." 
              className="w-full bg-black/50 border border-glass rounded-full py-1.5 pl-9 pr-4 text-sm text-white outline-none focus:border-accent transition-colors shadow-inner"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* Timeline representation */}
        {duration > 0 && (
          <div className="timeline-container px-6 pt-4 pb-2">
            <div className="timeline-track h-2 bg-black/60 rounded-full relative overflow-hidden border border-white/5 shadow-inner">
              {segments.map((seg, i) => (
                <div
                  key={i}
                  className={`absolute top-0 bottom-0 rounded-full cursor-pointer transition-all hover:-translate-y-px ${i === activeSegment ? 'bg-accent shadow-glow z-10' : 'bg-accent/50 hover:bg-accent'}`}
                  style={{
                    left: `${(seg.start / duration) * 100}%`,
                    width: `${Math.max(0.2, ((seg.end - seg.start) / duration) * 100)}%`,
                  }}
                  onClick={() => handleSegmentClick(i)}
                  title={seg.text?.trim()}
                />
              ))}
            </div>
            <div className="flex justify-between mt-2 text-[10px] text-muted font-mono uppercase tracking-wider opacity-60">
               <span>00:00</span>
               <span>{fmtDuration(duration).replace('m', ':').replace('s', '').replace(' ', '')}</span>
            </div>
          </div>
        )}

        {/* Segment list body */}
        <div className="segments-scrollable flex-1 overflow-y-auto p-4 space-y-1 scroll-smooth">
          {filteredSegments.length === 0 ? (
            <div className="p-12 text-center text-muted flex flex-col items-center gap-3">
               <Search size={32} className="opacity-20" />
               <p>No transcription matching "{searchTerm}"</p>
            </div>
          ) : (
            filteredSegments.map(seg => {
              const originalIndex = segments.indexOf(seg);
              return (
                <SegmentRow
                  key={seg.id || originalIndex}
                  seg={seg}
                  index={originalIndex}
                  isActive={originalIndex === activeSegment}
                  onClick={handleSegmentClick}
                />
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
