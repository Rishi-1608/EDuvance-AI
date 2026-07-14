import { useState, useEffect } from 'react';
import { getDashboardStats } from '@/lib/api';
import {
  BarChart3,
  BrainCircuit,
  Film,
  Image as ImageIcon,
  Mic,
  FileText,
  Target,
  Zap,
  CheckCircle2,
  TrendingUp,
  Activity as ActivityIcon,
  Sparkles,
  Layers
} from 'lucide-react';
import './ActivityPage.css';

function GlassCard({ children, className = '', delay = 0 }) {
  return (
    <div className={`glass-card animate-up ${className}`} style={{ animationDelay: `${delay}ms` }}>
      {children}
      <div className="glass-card-glow"></div>
    </div>
  );
}

function MetricWidget({ label, value, icon: Icon, color, trend, delay }) {
  return (
    <GlassCard className="metric-widget" delay={delay}>
      <div className="metric-header">
        <div className="metric-icon-wrap" style={{ '--widget-color': color }}>
          <Icon size={20} className="stroke-glow" />
        </div>
        {trend && (
          <div className="metric-trend">
            <TrendingUp size={14} />
            <span>{trend}</span>
          </div>
        )}
      </div>
      <div className="metric-body">
        <h4 className="metric-value mono">{value ?? 0}</h4>
        <p className="metric-label">{label}</p>
      </div>
    </GlassCard>
  );
}

function IntelligenceRing({ val1, val2 }) {
  const r1 = 54;
  const r2 = 38;
  const c1 = 2 * Math.PI * r1;
  const c2 = 2 * Math.PI * r2;
  const off1 = c1 - (val1 / 100) * c1;
  const off2 = c2 - (val2 / 100) * c2;

  return (
    <div className="intelligence-ring-wrapper">
      <svg className="intelligence-ring" width="140" height="140">
        <defs>
          <linearGradient id="g-primary" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--nebula-teal)" />
            <stop offset="100%" stopColor="var(--nebula-blue)" />
          </linearGradient>
          <linearGradient id="g-secondary" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--nebula-rose)" />
            <stop offset="100%" stopColor="var(--nebula-violet)" />
          </linearGradient>
        </defs>
        {/* Outer Ring */}
        <circle cx="70" cy="70" r={r1} className="ring-bg" strokeWidth="8" />
        <circle
          cx="70" cy="70" r={r1}
          className="ring-fill"
          strokeWidth="8"
          stroke="url(#g-primary)"
          strokeDasharray={c1}
          strokeDashoffset={val1 > 0 ? off1 : c1}
        />
        {/* Inner Ring */}
        <circle cx="70" cy="70" r={r2} className="ring-bg" strokeWidth="8" />
        <circle
          cx="70" cy="70" r={r2}
          className="ring-fill"
          strokeWidth="8"
          stroke="url(#g-secondary)"
          strokeDasharray={c2}
          strokeDashoffset={val2 > 0 ? off2 : c2}
        />
      </svg>
      <div className="ring-center">
        <BrainCircuit size={32} className="text-secondary" />
      </div>
    </div>
  );
}

export function ActivityPage({ user }) {
  const [dbStats, setDbStats] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchStats = () => {
    getDashboardStats()
      .then(setDbStats)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchStats();
    // Poll every 10 seconds for live feeling
    const intv = setInterval(fetchStats, 10000);
    return () => clearInterval(intv);
  }, []);

  if (loading && !dbStats) {
    return (
      <div className="activity-page container center-content">
        <div className="spinner" style={{ width: 32, height: 32, opacity: 0.5 }}></div>
      </div>
    );
  }

  const { engagement = {}, last_48h = {}, recent_lectures = [] } = dbStats || {};

  const totalMedia =
    (dbStats?.total_lectures || 0) +
    (dbStats?.total_images || 0) +
    (dbStats?.total_audios || 0) +
    (dbStats?.total_docs || 0);

  const recentUploadsCount =
    (last_48h.videos || 0) +
    (last_48h.images || 0) +
    (last_48h.audios || 0) +
    (last_48h.docs || 0);

  const accuracy = engagement.quiz_accuracy_pct || 0;
  const confidence = engagement.avg_confidence_pct || 0;

  return (
    <div className="activity-page">
      {/* HEADER SECTION */}
      <div style={{ height: 20 }} />

      {/* METRICS ROW */}
      <div className="metrics-grid">
        <MetricWidget label="Total Media" value={totalMedia} icon={Layers} color="var(--accent-primary)" trend={`+${recentUploadsCount} recent`} delay={100} />
        <MetricWidget label="Videos" value={dbStats?.total_lectures || 0} icon={Film} color="var(--nebula-blue)" delay={200} />
        <MetricWidget label="Documents" value={dbStats?.total_docs || 0} icon={FileText} color="var(--nebula-teal)" delay={300} />
        <MetricWidget label="Images" value={dbStats?.total_images || 0} icon={ImageIcon} color="var(--nebula-rose)" delay={400} />
      </div>

      {/* BENTO GRID (DASHBOARD HIGHLIGHTS) */}
      <div className="bento-grid">
        {/* Core Intelligence / Brain Card */}
        <GlassCard className="bento-card col-span-2 brain-card flex-row" delay={500}>
          <div className="bento-content">
            <div className="bento-head">
              <Sparkles size={18} className="text-secondary" />
              <h3>Cognitive Profile</h3>
            </div>
            <p className="text-muted">Extracted from {engagement.total_interactions || 0} interactions</p>

            <div className="cognitive-stats mt-auto">
              <div className="cog-stat">
                <Target size={16} className="text-success" />
                <span>Quiz Accuracy</span>
                <span className="mono bold ml-auto">{accuracy}%</span>
              </div>
              <div className="cog-stat">
                <Zap size={16} className="text-warning" />
                <span>Confidence Rating</span>
                <span className="mono bold ml-auto">{confidence}%</span>
              </div>
            </div>
          </div>
          <div className="bento-visual">
            <IntelligenceRing val1={accuracy} val2={confidence} />
            <div className="legend">
              <div className="legend-item"><span className="dot dot-outer"></span> Accuracy</div>
              <div className="legend-item"><span className="dot dot-inner"></span> Confidence</div>
            </div>
          </div>
        </GlassCard>

        {/* Breakdown Card */}
        <GlassCard className="bento-card breakdown-card" delay={600}>
          <div className="bento-head">
            <BarChart3 size={18} className="text-secondary" />
            <h3>Processing Volume</h3>
          </div>
          <div className="volume-list mt-auto">
            <div className="v-item">
              <span className="v-label text-muted">Video Hours</span>
              <span className="v-val mono bold text-primary">{dbStats?.total_lectures || 0}</span>
            </div>
            <div className="v-item">
              <span className="v-label text-muted">Pages Parsed</span>
              <span className="v-val mono bold text-secondary">{dbStats?.total_docs || 0}</span>
            </div>
            <div className="v-item">
              <span className="v-label text-muted">Flashcards Created</span>
              <span className="v-val mono bold text-success">{engagement.flashcards_reviewed || 0}</span>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* RECENT ACTIVITY TABLE */}
      <GlassCard className="history-section" delay={700}>
        <div className="bento-head mb-4">
          <ActivityIcon size={18} className="text-secondary" />
          <h3>Recent Database Commits</h3>
        </div>

        <div className="table-responsive">
          <table className="nexus-table">
            <thead>
              <tr>
                <th>Asset Identity</th>
                <th width="120">Type</th>
                <th width="150">Status</th>
                <th width="150" className="text-right">Commit Time</th>
              </tr>
            </thead>
            <tbody>
              {recent_lectures.map((lec, i) => (
                <tr key={lec.stem || i}>
                  <td className="mono identity-cell">{lec.lecture_title || lec.display_name || lec.title || lec.stem}</td>
                  <td>
                    <span className={`pill type-${lec.type || 'unknown'}`}>
                      {lec.type || 'Video'}
                    </span>
                  </td>
                  <td>
                    <div className="flex-row gap-2">
                      <CheckCircle2 size={14} className="text-success" />
                      <span>Indexed</span>
                    </div>
                  </td>
                  <td className="text-muted text-right mono small">
                    {lec.date ? new Date(lec.date).toLocaleDateString() : '--'}
                  </td>
                </tr>
              ))}
              {recent_lectures.length === 0 && (
                <tr>
                  <td colSpan="4" className="text-center text-muted" style={{ padding: '40px 0' }}>
                    No assets indexed in the database yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
}
