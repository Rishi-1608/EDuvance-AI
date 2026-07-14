import { useState, useEffect, useRef } from 'react';
import {
  Zap,
  CheckCircle2,
  Clock,
  AlertCircle,
  Terminal,
  RotateCcw,
  ArrowRight,
  Loader2,
  Cpu
} from 'lucide-react';
import { useStatus } from '@/hooks/useStatus';
import { stopPipeline, clearResults } from '@/lib/api';
import './DashboardPage.css';

export function DashboardPage({ onTabChange }) {
  const { status, error, refetch } = useStatus(1500); // Poll faster during processing
  const [logs, setLogs] = useState([]);
  const lastStepRef = useRef(null);

  // Maintain a "log" of steps
  useEffect(() => {
    if (status?.pipeline_status && status.pipeline_status !== lastStepRef.current) {
      const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setLogs(prev => [{ time: timestamp, msg: status.pipeline_status }, ...prev].slice(0, 5));
      lastStepRef.current = status.pipeline_status;
    }
  }, [status?.pipeline_status]);

  const isRunning = status?.pipeline_running;
  const isDone = !isRunning && status?.task_done && status?.videos_in_pipeline > 0;
  const isEmpty = !isRunning && !isDone && Object.keys(status?.videos || {}).length === 0;

  const handleStop = async () => {
    if (confirm('Stop the current process?')) {
      await stopPipeline();
      refetch();
    }
  };

  return (
    <div className="processing-view">
      <div className="proc-container">
        {error && (
          <div className="status-msg error animate-up" style={{ marginBottom: '2rem' }}>
            <AlertCircle size={16} />
            <span>Connection Lost: {error}</span>
          </div>
        )}

        <div className="proc-card animate-up">
          {isRunning ? (
            <>
              <div className="loader-wrapper">
                <div className="main-loader">
                  <div className="loader-ring"></div>
                  <div className="loader-ring"></div>
                  <div className="loader-ring"></div>
                  <Cpu size={40} className="loader-icon" />
                </div>
              </div>

              <div className="proc-progress-section">
                <div className="proc-label-row">
                  <span className="proc-status-text">
                    {status?.pipeline_status || 'Initializing...'}
                  </span>
                  <span className="proc-pct">{status?.pipeline_progress || 0}%</span>
                </div>
                <div className="proc-bar-track">
                  <div
                    className="proc-bar-fill"
                    style={{ width: `${status?.pipeline_progress || 0}%` }}
                  />
                </div>
              </div>

              <div className="proc-steps-log">
                {logs.map((log, i) => (
                  <div key={i} className={`log-entry ${i === 0 ? 'active' : ''}`}>
                    <span className="log-time">[{log.time}]</span>
                    <span className="log-msg">{log.msg}</span>
                  </div>
                ))}
              </div>

              <div className="proc-actions">
                <button className="btn btn-ghost" onClick={handleStop}>
                  Stop Pipeline
                </button>
              </div>
            </>
          ) : isDone ? (
            <div className="proc-done-state">
              <CheckCircle2 size={64} className="proc-done-icon" />
              <h2 className="proc-status-text" style={{ fontSize: '1.75rem', marginBottom: '1rem' }}>
                Processing Complete
              </h2>
              <p style={{ color: '#94a3b8', marginBottom: '2rem' }}>
                All lecture materials are ready in your library.
              </p>
              <div className="proc-actions">
                <button className="btn btn-primary" onClick={() => onTabChange('library')}>
                  Go to Library <ArrowRight size={16} style={{ marginLeft: 8 }} />
                </button>
                <button className="btn btn-ghost" onClick={() => onTabChange('home')}>
                  Upload More
                </button>
              </div>
            </div>
          ) : (
            <div className="proc-empty-state">
              <Zap size={64} style={{ color: '#94a3b8', marginBottom: '1.5rem', opacity: 0.5 }} />
              <h2 className="proc-status-text" style={{ fontSize: '1.75rem', marginBottom: '1rem' }}>
                System Idle
              </h2>
              <p style={{ color: '#94a3b8', marginBottom: '2rem' }}>
                No active processing tasks. Upload a lecture to begin.
              </p>
              <div className="proc-actions">
                <button className="btn btn-primary" onClick={() => onTabChange('home')}>
                  Upload Lecture <Zap size={16} style={{ marginLeft: 8 }} />
                </button>
              </div>
            </div>
          )}
        </div>

        {isRunning && (
          <p className="animate-pulse-slow" style={{ marginTop: '2rem', color: '#64748b', fontSize: '0.875rem' }}>
            The AI is processing multiple modalities. This may take several minutes depending on video length.
          </p>
        )}
      </div>
    </div>
  );
}
