import { useState, useEffect, useRef } from 'react';
import { ArrowLeft, BookOpen, Layers, HelpCircle, FileText, Download, Zap, CheckCircle, Clock, Image as ImageIcon, Mic, Play, Radio } from 'lucide-react';
import { useStatus } from '@/hooks/useStatus';
import { getStudyNotes, getPdfUrl, downloadPdf, generateFlashcardsAndQuiz } from '@/lib/api';
import './VideoDetailsPage.css';

export function VideoDetailsPage({ stem, onBack, onTabChange }) {
  const { status, refetch } = useStatus(2000);  // Poll every 2 seconds (faster updates)
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [msg, setMsg] = useState(null);

  // Find the video data in status.videos
  const videos = status?.videos || {};
  const videoKey = Object.keys(videos).find(k =>
    k === stem ||
    k.split(' (')[0] === stem ||
    k.split('.')[0] === stem ||
    videos[k]?.video_stem === stem  // ← ADD THIS: match by video_stem field
  ) || stem;
  const videoData = videos[videoKey] || {};

  const getMediaConfig = (type) => {
    switch (type) {
      case 'video':
        return { icon: <Play size={28} />, label: 'Video Lecture', color: 'video' };
      case 'images':
        return { icon: <ImageIcon size={28} />, label: 'Image Analysis', color: 'images' };
      case 'audio':
        return { icon: <Mic size={28} />, label: 'Audio Lecture', color: 'audio' };
      case 'document':
      case 'pdf':
        return { icon: <FileText size={28} />, label: 'Document Analysis', color: 'document' };
      case 'live':
        return { icon: <Radio size={28} />, label: 'Live Lecture', color: 'live' };
      default:
        return { icon: <BookOpen size={28} />, label: 'Lecture Analysis', color: 'video' };
    }
  };

  const mediaConfig = getMediaConfig(videoData.input_type || (videoData.media_type));

  const handleGenerate = async () => {
    if (!stem) return;
    setGenerating(true);
    setMsg(null);
    try {
      const res = await generateFlashcardsAndQuiz(stem);
      setMsg(res?.message || 'Flashcard generation started!');
      refetch();
    } catch (e) {
      setMsg(`Error: ${e.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const isReady = videoData.study_notes_ready;
  const flashcardsReady = videoData.flashcards_ready && videoData.quiz_ready;
  const prevStateRef = useRef(videoData.flashcards_generation_state);

  // Auto-refresh when flashcard generation completes
  useEffect(() => {
    const currentState = videoData.flashcards_generation_state;
    if (prevStateRef.current === 'running' && currentState === 'done') {
      refetch(); // Immediate refresh when generation completes
    }
    prevStateRef.current = currentState;
  }, [videoData.flashcards_generation_state, refetch]);

  return (
    <div className="details-page">
      <div className="details-header animate-up stagger-1">
        <div className={`header-icon-box type-${mediaConfig.color}`}>
          {mediaConfig.icon}
        </div>
        <div className="header-info">
          <h2 className="details-title">{videoData.lecture_title || videoData.display_name || stem}</h2>
          <p className="details-subtitle">
            {videoData.subject_area || 'Library Lecture'} • {videoData.difficulty || 'Analysis Ready'}
          </p>
        </div>

        <div className="header-actions">
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => downloadPdf(stem).catch(e => alert(e.message))}
            disabled={!videoData.pdf_ready}
          >
            <Download size={14} /> Download PDF
          </button>

          {!flashcardsReady && isReady && (
            <button
              className="btn btn-primary btn-sm"
              onClick={handleGenerate}
              disabled={generating}
            >
              {generating ? <div className="spinner" /> : <Zap size={14} />}
              {generating ? 'Generating...' : 'Unlock Quiz & Flashcards'}
            </button>
          )}
        </div>
      </div>

      {msg && (
        <div className="status-msg animate-up">
          <CheckCircle size={14} /> {msg}
        </div>
      )}

      <div className="details-grid">
        {/* Notes Preview / Link */}
        <div
          className={`details-card glass animate-up stagger-2 ${!isReady ? 'disabled' : ''}`}
          onClick={() => isReady && onTabChange('notes', stem)}
        >
          <div className="card-top">
            <div className="card-icon notes-icon"><FileText size={20} /></div>
            <div className={`status-badge ${isReady ? 'ready' : 'pending'}`}>
              {isReady ? 'Ready' : 'Pending'}
            </div>
          </div>
          <h3>Study Notes</h3>
          <p>Read the comprehensive summary and key concepts extracted from your lecture.</p>
          <div className="card-action">Open Notes <ChevronRight size={14} /></div>
        </div>

        {/* Flashcards Preview / Link */}
        <div
          className={`details-card glass animate-up stagger-3 ${!videoData.flashcards_ready ? 'disabled' : ''}`}
          onClick={() => videoData.flashcards_ready && onTabChange('flashcards', stem)}
        >
          <div className="card-top">
            <div className="card-icon cards-icon"><Layers size={20} /></div>
            <div className={`status-badge ${videoData.flashcards_ready ? 'ready' : 'pending'}`}>
              {videoData.flashcards_ready ? 'Ready' : 'Pending'}
            </div>
          </div>
          <h3>Flashcards</h3>
          <p>Master the material with {videoData.flashcard_count || 0} tailored Q&A cards.</p>
          <div className="card-action">Review Cards <ChevronRight size={14} /></div>
        </div>

        {/* Quiz Preview / Link */}
        <div
          className={`details-card glass animate-up stagger-4 ${!videoData.quiz_ready ? 'disabled' : ''}`}
          onClick={() => videoData.quiz_ready && onTabChange('quiz', stem)}
        >
          <div className="card-top">
            <div className="card-icon quiz-icon"><HelpCircle size={20} /></div>
            <div className={`status-badge ${videoData.quiz_ready ? 'ready' : 'pending'}`}>
              {videoData.quiz_ready ? 'Ready' : 'Pending'}
            </div>
          </div>
          <h3>MCQ Quiz</h3>
          <p>Test your knowledge with {videoData.quiz_count || 0} challenging questions.</p>
          <div className="card-action">Take Quiz <ChevronRight size={14} /></div>
        </div>

        {/* View Photo Card (New) */}
        {videoData.input_type === 'images' && (
          <div
            className="details-card glass animate-up stagger-5"
            onClick={() => {
              const el = document.getElementById('media-render-section');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            <div className="card-top">
              <div className="card-icon cards-icon"><ImageIcon size={20} /></div>
              <div className="status-badge ready">Ready</div>
            </div>
            <h3>View Photo</h3>
            <p>Inspect the original lecture slide retrieved securely from storage.</p>
            <div className="card-action">View Image <ChevronRight size={14} /></div>
          </div>
        )}
      </div>

      <div className="video-stats-panel glass animate-up stagger-5" style={{ marginTop: 30 }}>
        <div className="panel-header">
          <h4>{mediaConfig.label}</h4>
          <div className={`status-badge ${videoData.study_notes_ready ? 'ready' : 'pending'}`}>
            {videoData.study_notes_ready ? 'Processed' : 'Processing...'}
          </div>
        </div>

        {/* Video Player / Image Viewer */}
        <div className="video-player-container" id="media-render-section">
          {stem ? (
            videoData.input_type === 'images' ? (
              <div className="image-viewer-container" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#000', borderRadius: '12px', minHeight: '350px', overflow: 'hidden' }}>
                {videoData.frames_index?.[0]?.frame_url ? (
                  <img
                    src={videoData.frames_index[0].frame_url.startsWith('http')
                      ? videoData.frames_index[0].frame_url
                      : `${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}${videoData.frames_index[0].frame_url}`}
                    alt={videoData.lecture_title || 'Analyzed Image'}
                    style={{ maxWidth: '100%', maxHeight: '600px', objectFit: 'contain' }}
                  />
                ) : (
                  <div style={{ textAlign: 'center', color: 'rgba(255,255,255,0.5)' }}>
                    <ImageIcon size={48} style={{ opacity: 0.6, marginBottom: '15px' }} />
                    <p>Image Batch Rendered</p>
                    <span style={{ fontSize: '13px' }}>{videoData.total_frames_analysed || 0} image(s) processed in this batch</span>
                  </div>
                )}
              </div>
            ) : videoData.input_type === 'audio' ? (
              <div className="audio-viewer-container" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', background: 'var(--accent-gradient)', borderRadius: '12px', minHeight: '200px', padding: '40px' }}>
                <div className="audio-icon-pulse" style={{ marginBottom: '20px' }}>
                  <Mic size={48} color="#fff" />
                </div>
                <audio
                  controls
                  style={{ width: '100%', maxWidth: '500px' }}
                >
                  <source src={`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/video/${stem}?token=${localStorage.getItem('eduvance_ai_token')}`} type="audio/mpeg" />
                  Your browser does not support the audio element.
                </audio>
              </div>
            ) : videoData.input_type === 'document' || videoData.input_type === 'pdf' ? (
              <div className="doc-viewer-container" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', minHeight: '300px', padding: '40px', border: '2px dashed var(--glass-border)' }}>
                <FileText size={64} style={{ color: 'var(--nebula-violet)', marginBottom: '20px', opacity: 0.8 }} />
                <h3 style={{ marginBottom: '10px' }}>Document Processed</h3>
                <p style={{ color: 'var(--text-muted)', marginBottom: '20px' }}>{videoData.total_frames_analysed || 0} pages analysed from this PDF.</p>
                <button
                  className="btn btn-primary"
                  onClick={() => downloadPdf(stem).catch(e => alert(e.message))}
                  disabled={!videoData.pdf_ready}
                >
                  <Download size={16} /> Download Academic Report
                </button>
              </div>
            ) : videoData.input_type === 'live' ? (
              <div className="audio-viewer-container" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', background: 'var(--accent-gradient)', borderRadius: '12px', minHeight: '200px', padding: '40px' }}>
                <div className="audio-icon-pulse" style={{ marginBottom: '20px' }}>
                  <Radio size={48} color="#fff" />
                </div>
                <h3 style={{ color: '#fff', marginBottom: '10px' }}>Live Audio Archive</h3>
                <audio
                  controls
                  style={{ width: '100%', maxWidth: '500px', marginTop: '15px' }}
                >
                  <source src={`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/video/${stem}?token=${localStorage.getItem('eduvance_ai_token')}`} type="audio/wav" />
                  Your browser does not support the audio element.
                </audio>
              </div>
            ) : (
              <video
                controls
                className="lecture-video-player"
                poster={videoData.frames_index?.[0]?.frame_url}
                style={{ width: '100%', borderRadius: '12px' }}
              >
                <source src={`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/video/${stem}?token=${localStorage.getItem('eduvance_ai_token')}`} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )
          ) : (
            <div className="video-placeholder">
              <Play size={48} />
              <p>Decoding media source...</p>
            </div>
          )}
        </div>

        <div className="meta-grid" style={{ marginTop: 20 }}>
          <div className="meta-item">
            <span className="meta-label">Topic Summary</span>
            <p className="meta-value">{videoData.lecture_summary?.summary || 'N/A'}</p>
          </div>
          {/* ... rest of the meta ... */}
          <div className="meta-item">
            <span className="meta-label">Total Frames Analysed</span>
            <p className="meta-value mono">{videoData.total_frames_analysed || '—'}</p>
          </div>
          <div className="meta-item">
            <span className="meta-label">Detected Language</span>
            <p className="meta-value">{videoData.detected_language || 'Auto-detected'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChevronRight({ size, className }) {
  return (
    <svg
      width={size} height={size}
      viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round"
      className={className}
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}
