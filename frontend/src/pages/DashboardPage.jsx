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

export function DashboardPage({ onTabChange, isUploading }) {
  const { status, error, refetch } = useStatus(5000);
  const [logs, setLogs] = useState([]);
  const [fakeProgress, setFakeProgress] = useState(0);
  const lastStepRef = useRef(null);

  // Maintain a "log" of steps
  useEffect(() => {
    if (status?.pipeline_status && status.pipeline_status !== lastStepRef.current) {
      const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setLogs(prev => [{ time: timestamp, msg: status.pipeline_status }, ...prev].slice(0, 5));
      lastStepRef.current = status.pipeline_status;
    }
  }, [status?.pipeline_status]);

  const actualIsRunning = status?.pipeline_running;
  const isRunning = actualIsRunning || isUploading;
  const isDone = !isRunning && status?.task_done && status?.videos_in_pipeline > 0;
  const isEmpty = !isRunning && !isDone && Object.keys(status?.videos || {}).length === 0;

  // Fake slow progress logic to ensure UI never looks stuck
  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(() => {
        setFakeProgress(prev => {
          const remaining = 99 - prev;
          const step = Math.max(0.1, remaining * 0.05);
          return prev + step;
        });
      }, 1000);
    } else {
      setFakeProgress(0);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const actualProgress = status?.pipeline_progress || 0;
  const displayProgress = Math.min(99, Math.max(Math.round(fakeProgress), actualProgress));
  const displayStatusText = isUploading && !actualIsRunning 
    ? 'Uploading files to server...' 
    : (status?.pipeline_status || 'Initializing...');

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
                    {displayStatusText}
                  </span>
                  <span className="proc-pct">{displayProgress}%</span>
                </div>
                <div className="proc-bar-track">
                  <div
                    className="proc-bar-fill"
                    style={{ width: `${displayProgress}%`, transition: 'width 1s linear' }}
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
              <h2 className="proc-status-text proc-title-text">
                Processing Complete
              </h2>
              <p className="proc-desc-text">
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
              <Zap size={64} className="proc-empty-icon" style={{ color: '#94a3b8', opacity: 0.5 }} />
              <h2 className="proc-status-text proc-title-text">
                System Idle
              </h2>
              <p className="proc-desc-text">
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