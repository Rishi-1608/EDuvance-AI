import { useState, useRef, useEffect } from 'react';
import { Mic, Square, Loader2, Play, AlertCircle, FileText, Activity, Clock, Radio, Waves, Zap, Timer, Hash } from 'lucide-react';
import { startLiveSession, endLiveSession, getLiveStatus, cancelLiveSession, BASE_URL } from '@/lib/api';
import { ConfirmModal } from '@/components/ConfirmModal';
import './LiveLecturePage.css';

export function LiveLecturePage({ onResult }) {
  const [session, setSession] = useState(null); // Current session object
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);

  const [title, setTitle] = useState('My Live Lecture');
  const [transcript, setTranscript] = useState('');
  const [segments, setSegments] = useState([]);
  const [stats, setStats] = useState({ wordCount: 0, segmentCount: 0 });
  const [elapsed, setElapsed] = useState(0);

  const mediaRecorderRef = useRef(null);
  const websocketRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const streamRef = useRef(null);
  const segmentsEndRef = useRef(null);
  const timerRef = useRef(null);

  // Auto-scroll transcript
  useEffect(() => {
    if (segmentsEndRef.current) {
      segmentsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [segments]);

  // Timer
  useEffect(() => {
    if (isRecording) {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed(prev => prev + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRecording]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopRecordingLocally();
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const formatTime = (s) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const stopRecordingLocally = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (websocketRef.current) {
      if (websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'stop' }));
      }
      websocketRef.current.close();
    }
  };

  const handleStart = async () => {
    try {
      setError(null);
      // 1. Get Mic permission first
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // 2. Init session via API
      const res = await startLiveSession(title);
      setSession(res);
      setSegments([]);
      setTranscript('');
      setStats({ wordCount: 0, segmentCount: 0 });

      // 3. Open WebSocket
      const token = localStorage.getItem('eduvance_ai_token');
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = BASE_URL.replace(/^https?:\/\//, ''); // e.g. 127.0.0.1:8000
      const wsUrl = `${wsProtocol}//${wsHost}/live/lecture/${res.session_id}?token=${token}`;

      const ws = new WebSocket(wsUrl);
      websocketRef.current = ws;

      ws.onopen = () => {
        setIsRecording(true);
        // 4. Start MediaRecorder
        const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };

        // Send data every 15 seconds (15000ms)
        mediaRecorder.start(15000);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'segment') {
          setSegments(prev => {
            // Check if segment already exists
            if (prev.find(s => s.id === data.id && s.start === data.start)) return prev;
            const updated = [...prev, data];
            updated.sort((a, b) => a.start - b.start);
            setTranscript(updated.map(s => s.text).join(' '));
            return updated;
          });
        } else if (data.type === 'heartbeat') {
          setStats({
            wordCount: data.word_count,
            segmentCount: data.segment_count
          });
        } else if (data.type === 'error') {
          setError(data.message);
        } else if (data.type === 'done') {
          startPolling(res.session_id);
        }
      };

      ws.onerror = (e) => {
        setError('WebSocket error occurred.');
        stopRecordingLocally();
        setIsRecording(false);
      };

      ws.onclose = () => {
        setIsRecording(false);
        if (!isProcessing) {
          startPolling(res.session_id);
        }
      };

    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to start live session. Check microphone permissions.');
      stopRecordingLocally();
    }
  };

  const handleStop = () => {
    setIsRecording(false);
    stopRecordingLocally();
    setShowConfirmModal(true);
  };

  const handleProcess = async () => {
    setShowConfirmModal(false);
    setIsProcessing(true);

    if (session?.session_id) {
      try {
        await endLiveSession(session.session_id);
        startPolling(session.session_id);
      } catch (err) {
        console.error("Failed to end session cleanly:", err);
        setError("Failed to start processing.");
        setIsProcessing(false);
      }
    }
  };

  const handleCancel = async () => {
    setShowConfirmModal(false);
    
    if (session?.session_id) {
      try {
        await cancelLiveSession(session.session_id);
      } catch (err) {
        console.error("Failed to cancel session cleanly:", err);
      }
    }
    
    // Reset state completely
    setSession(null);
    setSegments([]);
    setTranscript('');
    setStats({ wordCount: 0, segmentCount: 0 });
    setElapsed(0);
    setError(null);
  };

  const startPolling = (sessionId) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    setIsProcessing(true);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const st = await getLiveStatus(sessionId);
        if (st.state === 'done') {
          clearInterval(pollIntervalRef.current);
          setIsProcessing(false);
          if (onResult && st.pipeline_stem) {
            onResult(st.pipeline_stem);
          }
        } else if (st.state === 'error') {
          clearInterval(pollIntervalRef.current);
          setIsProcessing(false);
          setError(st.pipeline_error || 'Pipeline failed.');
        }
      } catch (e) {
        // keep polling
      }
    }, 2000);
  };

  const statusLabel = isRecording ? 'RECORDING' : isProcessing ? 'PROCESSING' : 'READY';
  const statusColor = isRecording ? 'live' : isProcessing ? 'processing' : 'idle';

  return (
    <div className="ll-page">
      {/* ── Ambient background blobs ── */}
      <div className="ll-ambient">
        <div className={`ll-blob ll-blob-1 ${isRecording ? 'active' : ''}`} />
        <div className={`ll-blob ll-blob-2 ${isRecording ? 'active' : ''}`} />
        <div className={`ll-blob ll-blob-3 ${isRecording ? 'active' : ''}`} />
      </div>

      {/* ── Header ── */}
      <header className="ll-header">
        <div className="ll-header-left">
          <div className="ll-header-icon">
            <Radio size={20} />
          </div>
          <div>
            <h1 className="ll-page-title">Live Lecture</h1>
            <p className="ll-page-sub">Real-time AI-powered transcription engine</p>
          </div>
        </div>
        <div className={`ll-status-chip ${statusColor}`}>
          <span className="ll-status-dot" />
          <span>{statusLabel}</span>
        </div>
      </header>

      {/* ── Main Grid ── */}
      <div className="ll-grid">
        {/* ═══ Left Panel — Controls ═══ */}
        <div className="ll-left-panel">
          {/* ── Visualizer Orb ── */}
          <div className={`ll-orb-container ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}>
            <div className="ll-orb-glow" />
            <div className="ll-orb">
              <div className="ll-orb-inner">
                {isRecording && (
                  <div className="ll-orb-rings">
                    <div className="ll-ring ll-ring-1" />
                    <div className="ll-ring ll-ring-2" />
                    <div className="ll-ring ll-ring-3" />
                  </div>
                )}
                {isProcessing ? (
                  <Loader2 size={32} className="ll-orb-icon spin" />
                ) : isRecording ? (
                  <Waves size={32} className="ll-orb-icon" />
                ) : (
                  <Mic size={32} className="ll-orb-icon" />
                )}
              </div>
            </div>
            {isRecording && (
              <div className="ll-orb-bars">
                {[...Array(12)].map((_, i) => (
                  <div key={i} className="ll-bar" style={{ '--i': i, '--delay': `${i * 0.08}s` }} />
                ))}
              </div>
            )}
          </div>

          {/* ── Timer ── */}
          <div className={`ll-timer ${isRecording ? 'active' : ''}`}>
            <Timer size={14} />
            <span className="ll-timer-value mono">{formatTime(elapsed)}</span>
          </div>

          {/* ── Title Input ── */}
          <div className="ll-input-wrap">
            <label className="ll-input-label">Lecture Title</label>
            <input
              id="lecture-title-input"
              type="text"
              className="ll-input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={isRecording || isProcessing}
              placeholder="e.g. Intro to Machine Learning"
            />
          </div>

          {/* ── Action Button ── */}
          <div className="ll-action-area">
            {!isRecording && !isProcessing ? (
              <button id="start-recording-btn" className="ll-btn-start" onClick={handleStart}>
                <div className="ll-btn-glow" />
                <Mic size={20} />
                <span>Start Recording</span>
              </button>
            ) : isRecording ? (
              <button id="stop-recording-btn" className="ll-btn-stop" onClick={handleStop}>
                <Square size={18} />
                <span>End Session</span>
              </button>
            ) : (
              <button className="ll-btn-processing" disabled>
                <Loader2 size={18} className="spin" />
                <span>Processing...</span>
              </button>
            )}
          </div>

          {/* ── Error ── */}
          {error && (
            <div className="ll-error">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* ── Stats Grid ── */}
          <div className="ll-stats-grid">
            <div className="ll-stat-card">
              <div className="ll-stat-icon words"><Zap size={16} /></div>
              <div className="ll-stat-info">
                <span className="ll-stat-value">{stats.wordCount}</span>
                <span className="ll-stat-label">Words</span>
              </div>
            </div>
            <div className="ll-stat-card">
              <div className="ll-stat-icon segments"><Hash size={16} /></div>
              <div className="ll-stat-info">
                <span className="ll-stat-value">{stats.segmentCount}</span>
                <span className="ll-stat-label">Segments</span>
              </div>
            </div>
            <div className="ll-stat-card">
              <div className="ll-stat-icon duration"><Clock size={16} /></div>
              <div className="ll-stat-info">
                <span className="ll-stat-value">{formatTime(elapsed)}</span>
                <span className="ll-stat-label">Duration</span>
              </div>
            </div>
            <div className="ll-stat-card">
              <div className="ll-stat-icon rate"><Activity size={16} /></div>
              <div className="ll-stat-info">
                <span className="ll-stat-value">{elapsed > 0 ? Math.round(stats.wordCount / (elapsed / 60)) : 0}</span>
                <span className="ll-stat-label">WPM</span>
              </div>
            </div>
          </div>
        </div>

        {/* ═══ Right Panel — Transcript ═══ */}
        <div className="ll-right-panel">
          <div className="ll-transcript-card">
            <div className="ll-transcript-header">
              <div className="ll-transcript-title">
                <FileText size={18} />
                <h2>Live Transcript</h2>
                {isRecording && <span className="ll-live-badge"><span className="ll-live-dot" />LIVE</span>}
              </div>
            </div>

            <div className="ll-transcript-body">
              {segments.length === 0 ? (
                <div className="ll-empty-state">
                  {isRecording ? (
                    <div className="ll-listening">
                      <div className="ll-eq">
                        {[...Array(5)].map((_, i) => (
                          <div key={i} className="ll-eq-bar" style={{ '--delay': `${i * 0.15}s` }} />
                        ))}
                      </div>
                      <span className="ll-listening-text">Listening for speech...</span>
                      <span className="ll-listening-hint">Speak clearly into your microphone</span>
                    </div>
                  ) : isProcessing ? (
                    <div className="ll-listening">
                      <Loader2 size={28} className="spin" style={{ color: 'var(--nebula-violet)' }} />
                      <span className="ll-listening-text">Processing transcript...</span>
                      <span className="ll-listening-hint">Generating notes and study materials</span>
                    </div>
                  ) : (
                    <div className="ll-idle-state">
                      <div className="ll-idle-icon">
                        <Mic size={36} />
                      </div>
                      <span className="ll-idle-text">Ready to Record</span>
                      <span className="ll-idle-hint">Start recording to see your transcript appear here in real-time</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="ll-segments">
                  {segments.map((seg, idx) => (
                    <div key={idx} className="ll-segment animate-in" style={{ '--stagger': `${idx * 0.05}s` }}>
                      <div className="ll-segment-timeline">
                        <div className="ll-segment-dot" />
                        {idx < segments.length - 1 && <div className="ll-segment-line" />}
                      </div>
                      <div className="ll-segment-content">
                        <span className="ll-segment-time mono">{seg.start.toFixed(1)}s</span>
                        <p className="ll-segment-text">{seg.text}</p>
                      </div>
                    </div>
                  ))}
                  <div ref={segmentsEndRef} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <ConfirmModal
        isOpen={showConfirmModal}
        onClose={() => {}} // Don't allow closing without choosing
        onConfirm={handleCancel}
        onProcess={handleProcess}
        title="Session Ended"
        message="Recording stopped. What would you like to do with this lecture?"
        confirmText="Cancel Lecture"
        processText="Start Processing"
      />
    </div>
  );
}
