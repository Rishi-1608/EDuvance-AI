import { useState, useEffect } from 'react';
import { Cpu, RefreshCw, CheckCircle2, XCircle, AlertCircle, Server, Folder, Zap } from 'lucide-react';
import { getDiagnostics } from '@/lib/api';
import './DiagnosticsPage.css';

function StatusRow({ label, value, ok }) {
  return (
    <div className="diag-row">
      <span className="diag-label mono">{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {ok !== undefined && (
          ok
            ? <CheckCircle2 size={14} style={{ color: 'var(--success)' }} />
            : <XCircle size={14} style={{ color: 'var(--error)' }} />
        )}
        <span className="diag-value">{String(value)}</span>
      </div>
    </div>
  );
}

export function DiagnosticsPage() {
  const [diag, setDiag] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetch = async () => {
    setLoading(true); setError(null);
    try { setDiag(await getDiagnostics()); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetch(); }, []);

  return (
    <div className="diag-page container">
      <div className="diag-header animate-up">
        <div />
        <button className="btn btn-ghost btn-sm" onClick={fetch} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
        </button>
      </div>

      {error && (
        <div className="diag-error animate-up">
          <AlertCircle size={15} />
          <div>
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Cannot reach API</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.85 }}>{error}</div>
          </div>
        </div>
      )}

      {loading && !diag && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}>
          <div className="spinner" style={{ width: 36, height: 36 }} />
        </div>
      )}

      {diag && (
        <div className="diag-grid">
          {/* Core System */}
          <div className="diag-card glass animate-up">
            <div className="diag-card-title">
              <Server size={16} style={{ color: 'var(--nebula-blue)' }} />
              System Core
            </div>
            <StatusRow label="System" value={diag.system} />
            <StatusRow label="Reasoning" value={diag.reasoning} />
            <StatusRow label="Auth Type" value={diag.auth} />
          </div>

          {/* Backends */}
          <div className="diag-card glass animate-up stagger-1">
            <div className="diag-card-title">
              <Cpu size={16} style={{ color: 'var(--nebula-violet)' }} />
              Backends & Models
            </div>
            <StatusRow label="EasyOCR" value={diag.backends?.easyocr ? 'Ready' : 'Missing'} ok={diag.backends?.easyocr} />
            <StatusRow label="Tesseract" value={diag.backends?.tesseract ? 'Ready' : 'Missing'} ok={diag.backends?.tesseract} />
            <StatusRow label="Whisper" value={diag.backends?.whisper_active_backend} ok={diag.backends?.whisper_active_backend !== 'none'} />
            <StatusRow label="W Model" value={diag.backends?.whisper_model} />
            <StatusRow label="Skimage" value={diag.backends?.skimage_ssim ? 'Ready' : 'Missing'} ok={diag.backends?.skimage_ssim} />
          </div>

          {/* Long Video Config */}
          <div className="diag-card glass animate-up stagger-2">
            <div className="diag-card-title">
              <AlertCircle size={16} style={{ color: 'var(--nebula-teal)' }} />
              Video Pipeline Config
            </div>
            {Object.entries(diag.v310_long_video_config || {}).map(([k, v]) => (
              <StatusRow key={k} label={k.replace(/_/g, ' ')} value={v} />
            ))}
          </div>

          {/* Phase 2 Strategy */}
          <div className="diag-card glass animate-up stagger-3">
            <div className="diag-card-title">
              <Zap size={16} style={{ color: 'var(--nebula-gold)' }} />
              Intelligence Strategy
            </div>
            {Object.entries(diag.phase2_strategy || {}).map(([k, v]) => (
              <StatusRow key={k} label={k.replace(/_/g, ' ')} value={v} />
            ))}
          </div>

          {/* v321 Additions */}
          <div className="diag-card diag-card-wide glass animate-up stagger-4">
            <div className="diag-card-title theme-text-nebula">
              <CheckCircle2 size={16} style={{ color: 'var(--success)' }} />
              Architecture: Multi-User Persistent (v3.2.x)
            </div>
            <div style={{ padding: '0 16px 12px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px 24px' }}>
              {[
                ...(diag.v321_additions?.db_persistence || "").split(','),
                diag.v321_additions?.user_isolation,
                ...(diag.v321_additions?.auth_endpoints || [])
              ].filter(Boolean).map((text, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  <CheckCircle2 size={12} style={{ color: 'var(--success)', opacity: 0.7 }} />
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{text.trim()}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Endpoints */}
          <div className="diag-card diag-card-wide glass animate-up stagger-5">
            <div className="diag-card-title">
              <Server size={16} style={{ color: 'var(--nebula-rose)' }} />
              API Reference
            </div>
            <div className="endpoints-grid">
              {Object.entries(diag.all_endpoints || {}).map(([k, v]) => (
                <div key={k} className="endpoint-row">
                  <span className="endpoint-key mono">{k}</span>
                  <code className="endpoint-val">{v}</code>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
