import { useState, useEffect, useRef } from 'react';
import { getDashboardStats, generateStudyPlan, generateMindMap } from '@/lib/api';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: 'var(--font-mono)',
});

function Mermaid({ chart }) {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState(false);

  useEffect(() => {
    const renderChart = async () => {
      try {
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        const { svg: renderedSvg } = await mermaid.render(id, chart);
        setSvg(renderedSvg);
        setError(false);
      } catch (err) {
        console.error('Mermaid render error:', err);
        setError(true);
      }
    };
    if (chart) renderChart();
  }, [chart]);

  if (error) {
    return <div className="mermaid-error">Error rendering mind map structure.</div>;
  }

  return <div className="mermaid-rendered" dangerouslySetInnerHTML={{ __html: svg }} />;
}

import {
  BarChart3,
  BrainCircuit,
  Play,
  Image as ImageIcon,
  Mic,
  FileText,
  Target,
  Zap,
  CheckCircle2,
  TrendingUp,
  Activity as ActivityIcon,
  Sparkles,
  Layers,
  Radio,
  PieChart,
  AlertTriangle,
  BookOpen,
  GraduationCap,
  ArrowRight,
  Crosshair,
  Loader2,
  Sparkle,
  Network
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

function AccuracyGraph({ data = [] }) {
  const [hovered, setHovered] = useState(null);

  if (!data || data.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-muted opacity-40 py-10">
        <ActivityIcon size={32} className="mb-2" />
        <p className="text-sm font-mono tracking-tight">No quiz sessions recorded yet</p>
      </div>
    );
  }

  const width = 600;
  const height = 180;
  const padding = 30;

  const chartData =
    data.length === 1
      ? [{ ...data[0], score: data[0].score - 0.01, title: 'Starting Point' }, data[0]]
      : data;

  const points = chartData.map((d, i) => ({
    x: padding + (i * (width - 2 * padding)) / (chartData.length - 1),
    y: height - padding - (d.score * (height - 2 * padding)) / 100,
    score: d.score,
    title: d.title,
    date: d.date,
  }));

  const pathD = points.reduce((acc, p, i) => 
    i === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`, 
  "");

  const TOOLTIP_W = 140;
  const TOOLTIP_H = 54;

  return (
    <div className="accuracy-graph-container flex-1 w-full mt-2 relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full overflow-visible"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="traj-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--nebula-rose)" />
            <stop offset="50%" stopColor="var(--nebula-violet)" />
            <stop offset="100%" stopColor="var(--nebula-teal)" />
          </linearGradient>
          <filter id="traj-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="4" />
            <feOffset dx="0" dy="6" result="offsetblur" />
            <feComponentTransfer><feFuncA type="linear" slope="0.3"/></feComponentTransfer>
            <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Grid lines */}
        {[0, 50, 100].map((val) => {
          const y = height - padding - (val * (height - 2 * padding)) / 100;
          return (
            <g key={val}>
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="rgba(255,255,255,0.05)"
                strokeWidth="1"
                strokeDasharray="4,4"
              />
              <text
                x="5"
                y={y + 4}
                fill="rgba(255,255,255,0.2)"
                fontSize="10"
                fontFamily="var(--font-mono)"
              >
                {val}%
              </text>
            </g>
          );
        })}

        {/* Single Path with Gradient */}
        <path 
          d={pathD} 
          fill="none" 
          stroke="url(#traj-grad)" 
          strokeWidth="4" 
          strokeLinecap="round" 
          strokeLinejoin="round"
          filter="url(#traj-shadow)"
          className="animate-draw"
        />

        {/* Dots */}
        {points.map((p, i) => {
          return (
            <g
              key={`dot-${i}`}
              className="graph-point"
              onMouseEnter={() => setHovered({ ...p, index: i })}
              onMouseLeave={() => setHovered(null)}
            >
              <circle cx={p.x} cy={p.y} r={12} fill="transparent" />
              <circle
                cx={p.x}
                cy={p.y}
                r={5}
                fill="#000"
                stroke="var(--nebula-teal)"
                strokeWidth="3"
              />
            </g>
          );
        })}

        {/* Tooltip rendered inside SVG via foreignObject */}
        {hovered && (() => {
          const rawX = hovered.x - TOOLTIP_W / 2;
          const tx = Math.min(
            Math.max(rawX, padding),
            width - TOOLTIP_W - padding
          );
          const ty = hovered.y - TOOLTIP_H - 14 < 0
            ? hovered.y + 14
            : hovered.y - TOOLTIP_H - 14;

          return (
            <foreignObject
              x={tx}
              y={ty}
              width={TOOLTIP_W}
              height={TOOLTIP_H + 10}
              style={{ overflow: 'visible', pointerEvents: 'none' }}
            >
              <div
                className="tooltip-box"
                style={{ padding: '6px 10px', minWidth: TOOLTIP_W }}
              >
                <div
                  className="tooltip-title"
                  style={{
                    maxWidth: TOOLTIP_W - 20,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {hovered.title}
                </div>
                <div className="tooltip-score">{hovered.score}%</div>
              </div>
            </foreignObject>
          );
        })()}
      </svg>
    </div>
  );
}

const DONUT_COLORS = [
  '#8b5cf6', '#3b82f6', '#14b8a6', '#ec4899',
  '#f59e0b', '#22c55e', '#ef4444', '#06b6d4',
  '#a855f7', '#f97316',
];

function SubjectDonut({ data = {} }) {
  const [hovered, setHovered] = useState(null);

  const entries = Object.entries(data)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  const total = entries.reduce((s, e) => s + e.count, 0);

  if (entries.length === 0) {
    return (
      <div className="donut-empty">
        <PieChart size={32} className="mb-2" />
        <p className="text-sm font-mono tracking-tight">No subject data yet</p>
      </div>
    );
  }

  const cx = 100, cy = 100, r = 72, strokeW = 28;
  const circumference = 2 * Math.PI * r;
  let offset = 0;

  const arcs = entries.map((entry, i) => {
    const pct = entry.count / total;
    const dash = pct * circumference;
    const gap = circumference - dash;
    const currentOffset = offset;
    offset += dash;
    return { ...entry, pct, dash, gap, offset: currentOffset, color: DONUT_COLORS[i % DONUT_COLORS.length] };
  });

  return (
    <div className="donut-wrapper">
      <div className="donut-chart-area">
        <svg viewBox="0 0 200 200" className="donut-svg">
          {/* Background ring */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth={strokeW} />
          {/* Segments */}
          {arcs.map((arc, i) => (
            <circle
              key={arc.name}
              cx={cx} cy={cy} r={r}
              fill="none"
              stroke={arc.color}
              strokeWidth={hovered === i ? strokeW + 6 : strokeW}
              strokeDasharray={`${arc.dash} ${arc.gap}`}
              strokeDashoffset={-arc.offset}
              className="donut-segment"
              style={{
                filter: hovered === i ? `drop-shadow(0 0 8px ${arc.color}80)` : 'none',
                transform: 'rotate(-90deg)',
                transformOrigin: '100px 100px',
              }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            />
          ))}
          {/* Center label */}
          <text x={cx} y={cy - 8} textAnchor="middle" fill="var(--text-primary)" fontSize="22" fontWeight="800" fontFamily="var(--font-display)">
            {total}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontWeight="600" letterSpacing="0.1em" textTransform="uppercase" fontFamily="var(--font-mono)">
            SUBJECTS
          </text>
        </svg>
      </div>
      <div className="donut-legend">
        {arcs.map((arc, i) => (
          <div
            key={arc.name}
            className={`donut-legend-item ${hovered === i ? 'active' : ''}`}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          >
            <span className="donut-dot" style={{ background: arc.color }} />
            <span className="donut-legend-name">{arc.name}</span>
            <span className="donut-legend-count mono">{arc.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const URGENCY_CONFIG = {
  critical: { color: 'var(--error)',   bg: 'rgba(239, 68, 68, 0.08)',  border: 'rgba(239, 68, 68, 0.2)',  label: 'Critical' },
  high:     { color: 'var(--warning)',  bg: 'rgba(245, 158, 11, 0.08)', border: 'rgba(245, 158, 11, 0.2)', label: 'High' },
  medium:   { color: 'var(--nebula-blue)', bg: 'rgba(59, 130, 246, 0.08)', border: 'rgba(59, 130, 246, 0.2)', label: 'Medium' },
  low:      { color: 'var(--nebula-teal)', bg: 'rgba(20, 184, 166, 0.08)', border: 'rgba(20, 184, 166, 0.2)', label: 'Review' },
};

const REC_ICONS = {
  alert: AlertTriangle,
  book: BookOpen,
  cards: Layers,
};

function TodaysFocus({ recommendations = [] }) {
  const [plans, setPlans] = useState({}); // { stem: "plan text" }
  const [mindmaps, setMindmaps] = useState({}); // { stem: "mermaid code" }
  const [loading, setLoading] = useState({}); // { stem: true/false }
  const [mindMapLoading, setMindMapLoading] = useState({});

  const handleCreatePlan = async (stem) => {
    if (loading[stem]) return;
    setLoading(prev => ({ ...prev, [stem]: true }));
    try {
      const res = await generateStudyPlan(stem);
      setPlans(prev => ({ ...prev, [stem]: res.plan }));
    } catch (err) {
      console.error("Failed to generate plan:", err);
    } finally {
      setLoading(prev => ({ ...prev, [stem]: false }));
    }
  };

  const handleGenerateMindMap = async (stem) => {
    if (mindMapLoading[stem]) return;
    setMindMapLoading(prev => ({ ...prev, [stem]: true }));
    try {
      const res = await generateMindMap(stem);
      setMindmaps(prev => ({ ...prev, [stem]: res.mindmap }));
    } catch (err) {
      console.error("Failed to generate mind map:", err);
    } finally {
      setMindMapLoading(prev => ({ ...prev, [stem]: false }));
    }
  };


  if (recommendations.length === 0) {
    return (
      <div className="focus-empty">
        <CheckCircle2 size={28} />
        <p>You're all caught up! No urgent reviews today.</p>
      </div>
    );
  }

  return (
    <div className="focus-list">
      {recommendations.map((rec, i) => {
        const cfg = URGENCY_CONFIG[rec.urgency] || URGENCY_CONFIG.low;
        const Icon = REC_ICONS[rec.icon] || BookOpen;
        const stem = rec.stem;
        const hasPlan = !!plans[stem];
        const hasMindMap = !!mindmaps[stem];
        const isGenerating = !!loading[stem];
        const isGeneratingMindMap = !!mindMapLoading[stem];

        return (
          <div
            key={stem || i}
            className={`focus-item-container animate-up ${hasPlan || hasMindMap ? 'has-plan' : ''}`}
            style={{ animationDelay: `${i * 80}ms` }}
          >

            <div
              className="focus-item"
              style={{
                '--focus-color': cfg.color,
                '--focus-bg': cfg.bg,
                '--focus-border': cfg.border,
              }}
            >
              <div className="focus-icon-wrap">
                <Icon size={16} />
              </div>
              <div className="focus-body">
                <div className="focus-title">{rec.title}</div>
                <div className="focus-reason">{rec.reason}</div>
              </div>
              
              {!hasPlan ? (
                <button 
                  className="create-plan-btn" 
                  onClick={() => handleCreatePlan(stem)}
                  disabled={isGenerating}
                >
                  {isGenerating ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <>
                      <Sparkle size={14} />
                      <span>Plan</span>
                    </>
                  )}
                </button>
              ) : (
                <span className="plan-ready-badge">
                  <Sparkles size={12} />
                  Plan Active
                </span>
              )}

              {!hasMindMap ? (
                <button 
                  className="mindmap-btn" 
                  onClick={() => handleGenerateMindMap(stem)}
                  disabled={isGeneratingMindMap}
                >
                  {isGeneratingMindMap ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <>
                      <Network size={14} />
                      <span>Map</span>
                    </>
                  )}
                </button>
              ) : (
                <span className="mindmap-ready-badge">
                  <Network size={12} />
                  Map Ready
                </span>
              )}
              
              <span className="focus-urgency-badge">{cfg.label}</span>
            </div>

            <div className="focus-details-expansion">
              {hasPlan && (
                <div className="focus-plan-box animate-slide-down">
                  <div className="plan-header">
                    <GraduationCap size={14} />
                    <span>Personalized Strategy</span>
                  </div>
                  <p className="plan-text">{plans[stem]}</p>
                </div>
              )}

              {hasMindMap && (
                <div className="focus-mindmap-box animate-slide-down">
                  <div className="plan-header">
                    <Network size={14} />
                    <span>Mind Map Structure</span>
                  </div>
                  <div className="mindmap-render-wrap">
                    <Mermaid chart={mindmaps[stem]} />
                  </div>
                </div>
              )}
            </div>
          </div>

        );
      })}
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
    (dbStats?.total_docs || 0) +
    (dbStats?.total_live || 0);

  const recentUploadsCount =
    (last_48h.videos || 0) +
    (last_48h.images || 0) +
    (last_48h.audios || 0) +
    (last_48h.docs || 0);

  const accuracy = engagement.quiz_accuracy_pct || 0;
  const quizHistory = engagement.quiz_history || [];
  const subjectDistribution = dbStats?.subject_distribution || {};
  const studyRecs = dbStats?.study_recommendations || [];

  return (
    <div className="activity-page">
      {/* HEADER SECTION */}
      <div style={{ height: 20 }} />

      {/* METRICS ROW */}
      <div className="metrics-grid">
        <MetricWidget
          label="Total Media"
          value={totalMedia}
          icon={Layers}
          color="var(--accent-primary)"
          trend={`+${recentUploadsCount} recent`}
          delay={100}
        />
        <MetricWidget
          label="Videos"
          value={dbStats?.total_lectures || 0}
          icon={Play}
          color="var(--nebula-blue)"
          delay={200}
        />
        <MetricWidget
          label="Documents"
          value={dbStats?.total_docs || 0}
          icon={FileText}
          color="var(--nebula-teal)"
          delay={300}
        />
        <MetricWidget
          label="Images"
          value={dbStats?.total_images || 0}
          icon={ImageIcon}
          color="var(--nebula-rose)"
          delay={400}
        />
        <MetricWidget
          label="Live Lectures"
          value={dbStats?.total_live || 0}
          icon={Radio}
          color="var(--success)"
          delay={500}
        />
      </div>

      {/* BENTO GRID (DASHBOARD HIGHLIGHTS) */}
      <div className="bento-grid">
        {/* Accuracy Trend Card */}
        <GlassCard className="bento-card col-span-2 brain-card" delay={500}>
          <div className="bento-content w-full h-full">
            <div className="bento-head">
              <TrendingUp size={18} className="text-secondary" />
              <h3>Academic Trajectory</h3>
            </div>
            <p className="text-muted">
              Dynamic analysis of quiz performance across{' '}
              {engagement.total_interactions || 0} commit sessions
            </p>
            <AccuracyGraph data={quizHistory} />
          </div>
        </GlassCard>

        {/* Global Accuracy Card */}
        <GlassCard className="bento-card breakdown-card" delay={600}>
          <div className="bento-head">
            <Target size={18} className="text-secondary" />
            <h3>Cumulative Accuracy</h3>
          </div>

          <div className="accuracy-display">
            <div className="accuracy-value font-display font-bold gradient-text">
              {accuracy}
              <span className="accuracy-unit">%</span>
            </div>
            <div className="text-xs mono uppercase tracking-widest text-muted opacity-80">
              Global Quiz Accuracy
            </div>
          </div>

          <div className="volume-list mt-auto pt-4 border-t border-glass">
            <div className="v-item">
              <span className="v-label text-muted">Quizzes Attempted</span>
              <span className="v-val mono bold text-primary">
                {engagement.total_quizzes ?? quizHistory.length}
              </span>
            </div>
          </div>
        </GlassCard>
        {/* Subject Breakdown Donut */}
        <GlassCard className="bento-card col-span-3 subject-donut-card" delay={700}>
          <div className="bento-head">
            <PieChart size={18} className="text-secondary" />
            <h3>Subject Breakdown</h3>
          </div>
          <SubjectDonut data={subjectDistribution} />
        </GlassCard>
      </div>

      {/* TODAY'S FOCUS — Study Recommendations */}
      <GlassCard className="focus-section" delay={750}>
        <div className="bento-head mb-4">
          <Crosshair size={18} className="text-secondary" />
          <h3>Today's Focus</h3>
          <span className="focus-subtitle">AI-powered study recommendations</span>
        </div>
        <TodaysFocus recommendations={studyRecs} />
      </GlassCard>

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
                <th width="150" className="text-right">
                  Commit Time
                </th>
              </tr>
            </thead>
            <tbody>
              {recent_lectures.map((lec, i) => (
                <tr key={lec.stem || i}>
                  <td className="mono identity-cell">
                    {lec.lecture_title || lec.display_name || lec.title || lec.stem}
                  </td>
                  <td>
                    <span className={`pill type-${lec.type || 'unknown'}`}>
                      {lec.type === 'live'
                        ? 'Live Lecture'
                        : lec.type === 'images'
                          ? 'Image Batch'
                          : lec.type === 'video'
                            ? 'Video'
                            : lec.type || 'Video'}
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
                  <td
                    colSpan="4"
                    className="text-center text-muted"
                    style={{ padding: '40px 0' }}
                  >
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