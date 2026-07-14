import { useState, useEffect } from 'react';
import { HelpCircle, CheckCircle2, RefreshCw, AlertCircle, Trophy, Layers, Video, Mic, ChevronLeft } from 'lucide-react';
import { useStatus } from '@/hooks/useStatus';
import { getQuiz } from '@/lib/api';
import { toStem, shortStem } from '@/lib/utils';
import './QuizPage.css';

// ── Error boundary ────────────────────────────────────────────────────────────
import { Component } from 'react';
class QuizErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(e) { return { error: e?.message || String(e) }; }
  render() {
    if (this.state.error) {
      return (
        <div className="quiz-error" style={{ margin: '24px 0', borderRadius: 'var(--radius-md)', padding: '16px 20px' }}>
          <AlertCircle size={16} />
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Something went wrong rendering the quiz</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>{this.state.error}</div>
            <button
              className="btn btn-ghost btn-sm"
              style={{ marginTop: 10 }}
              onClick={() => this.setState({ error: null })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Single question ───────────────────────────────────────────────────────────
function QuizQuestion({ question, index, total, onAnswer }) {
  const [selected, setSelected] = useState(null);
  const [revealed, setRevealed] = useState(false);

  if (!question || typeof question !== 'object') return null;

  // ── Normalise options ──────────────────────────────────────────────────────
  // Backend may return options as:
  //   Array:  ["text A", "text B", ...]          → use as-is
  //   Object: {"A": "text A", "B": "text B", ...} → convert to [{key, label}]
  const rawOpts = question.options || question.choices || [];
  const opts = Array.isArray(rawOpts)
    ? rawOpts.map((label, i) => ({ key: String.fromCharCode(65 + i), label: String(label) }))
    : typeof rawOpts === 'object' && rawOpts !== null
      ? Object.entries(rawOpts).map(([key, label]) => ({ key, label: String(label) }))
      : [];

  // ── Normalise correct answer ───────────────────────────────────────────────
  // Could be a letter key "D", an index, or the full answer text
  const rawCorrect = String(question.correct_answer ?? question.answer ?? question.correct ?? '').trim();
  // Resolve to the matching opt key (handles both "D" letter and full-text match)
  const correctKey = (() => {
    // Direct key match e.g. "D"
    const byKey = opts.find(o => o.key === rawCorrect);
    if (byKey) return byKey.key;
    // Full text match
    const byLabel = opts.find(o => o.label.trim() === rawCorrect);
    if (byLabel) return byLabel.key;
    // Numeric index "3" → key "D"
    const idx = parseInt(rawCorrect, 10);
    if (!isNaN(idx) && opts[idx]) return opts[idx].key;
    return rawCorrect; // fallback
  })();

  const qText = question.question || question.text || question.stem || '(no question text)';

  const handleSelect = (key) => {
    if (revealed) return;
    setSelected(key);
    setRevealed(true);
    onAnswer(key === correctKey);
  };

  const optClass = (key) => {
    if (!revealed) return '';
    if (key === correctKey) return 'correct';
    if (key === selected) return 'wrong';
    return '';
  };

  const answerIsCorrect = revealed && selected === correctKey;

  return (
    <div className="quiz-question glass animate-up">
      <div className="q-header">
        <span className="mono q-number">{String(index + 1).padStart(2, '0')}</span>
        <span className="mono q-total">/ {total}</span>
        {revealed && (
          <span className={'badge ' + (answerIsCorrect ? 'badge-success' : 'badge-error')}>
            {answerIsCorrect ? 'Correct' : 'Wrong'}
          </span>
        )}
      </div>

      <div className="q-text">{qText}</div>

      <div className="q-options">
        {opts.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '8px 0' }}>
            No answer options available for this question.
          </div>
        ) : (
          opts.map(({ key, label }) => (
            <button
              key={key}
              className={'quiz-option ' + (selected === key ? 'selected ' : '') + optClass(key)}
              onClick={() => handleSelect(key)}
              disabled={revealed}
            >
              <span className="opt-letter mono">{key}</span>
              <span>{label}</span>
            </button>
          ))
        )}
      </div>

      {revealed && question.explanation && (
        <div className="q-explanation">
          <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--accent-primary)', marginBottom: 4, letterSpacing: '0.08em' }}>
            EXPLANATION
          </div>
          {String(question.explanation)}
        </div>
      )}
    </div>
  );
}

// ── Score screen ──────────────────────────────────────────────────────────────
function ScoreBoard({ score, total, onRetry }) {
  if (!total || total < 1) {
    return (
      <div className="scoreboard glass animate-up">
        <HelpCircle size={40} style={{ color: 'var(--text-muted)', marginBottom: 12 }} />
        <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, marginBottom: 16 }}>
          No scoreable questions
        </div>
        <button className="btn btn-primary" onClick={onRetry}>
          <RefreshCw size={14} /> Try Again
        </button>
      </div>
    );
  }
  const pct = Math.round((score / total) * 100);
  const grade = pct >= 90 ? 'A' : pct >= 80 ? 'B' : pct >= 70 ? 'C' : pct >= 60 ? 'D' : 'F';
  const msg = pct >= 80 ? 'Excellent work!' : pct >= 60 ? 'Good effort!' : 'Keep studying!';

  return (
    <div className="scoreboard glass animate-up">
      <Trophy size={48} style={{ color: 'var(--nebula-gold)', marginBottom: 12 }} />
      <div className="score-grade gradient-text">{grade}</div>
      <div className="score-pct">{pct}%</div>
      <div style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>{msg}</div>
      <div className="mono" style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: 24 }}>
        {score} / {total} correct
      </div>
      <div className="progress-track" style={{ width: 200, margin: '0 auto 24px' }}>
        <div className="progress-fill" style={{ width: pct + '%' }} />
      </div>
      <button className="btn btn-primary" onClick={onRetry}>
        <RefreshCw size={14} /> Retake Quiz
      </button>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export function QuizPage({ preselectedStem }) {
  const { status } = useStatus(5000);
  const [selectedStem, setSelectedStem] = useState(preselectedStem || '');
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);

  const videos = status?.videos || {};
  const allKeys = Object.keys(videos);
  const allStems = allKeys.map(toStem);
  const stemToKey = Object.fromEntries(allKeys.map(k => [toStem(k), k]));
  const readyStems = allKeys.filter(k => videos[k].quiz_ready).map(toStem);

  // Auto-select first ready stem or use preselected
  useEffect(() => {
    if (preselectedStem) {
      setSelectedStem(preselectedStem);
    } else if (!selectedStem && readyStems.length > 0) {
      setSelectedStem(readyStems[readyStems.length - 1]);
    }
  }, [preselectedStem, readyStems.join(',')]);

  const fetchQuiz = async (stem) => {
    setLoading(true);
    setError(null);
    setQuestions([]);
    setAnswers({});
    setDone(false);
    try {
      const data = await getQuiz(stem);
      // Normalise — accept array or object with a questions key
      const arr = Array.isArray(data)
        ? data
        : Array.isArray(data?.questions)
          ? data.questions
          : [];
      setQuestions(arr);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedStem) fetchQuiz(selectedStem);
  }, [selectedStem]);

  const handleAnswer = (idx, correct) => setAnswers(a => ({ ...a, [idx]: correct }));

  const validQs = questions.filter(q => q && typeof q === 'object');
  const answeredN = Object.keys(answers).length;
  const allAnswered = validQs.length > 0 && answeredN >= validQs.length;
  const score = Object.values(answers).filter(Boolean).length;

  const retry = () => {
    setAnswers({});
    setDone(false);
    if (selectedStem) fetchQuiz(selectedStem);
  };

  // Explicit render function — no ambiguous conditional chains
  const renderBody = () => {
    return (
      <div className="quiz-content-wrapper flex-1 overflow-y-auto px-10 py-8 flex flex-col relative z-0">
        {allAnswered && !done && (
          <div className="mb-8 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="glass p-6 rounded-2xl border-glow flex items-center justify-between bg-accent/5">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                  <CheckCircle2 size={24} />
                </div>
                <div>
                  <h3 className="text-white font-display font-semibold m-0">Quiz Complete!</h3>
                  <p className="text-muted text-sm m-0">You've answered all questions. Ready to see how you did?</p>
                </div>
              </div>
              <button className="btn btn-primary btn-lg" onClick={() => setDone(true)}>
                See Results
              </button>
            </div>
          </div>
        )}
        <div className="questions-list space-y-6">
          {validQs.map((q, i) => (
            <QuizQuestion
              key={i}
              question={q}
              index={i}
              total={validQs.length}
              onAnswer={(correct) => handleAnswer(i, correct)}
            />
          ))}
        </div>
      </div>
    );
  };

  const currentDisplayName = selectedStem && stemToKey[selectedStem]
    ? videos[stemToKey[selectedStem]]?.lecture_title || videos[stemToKey[selectedStem]]?.display_name || selectedStem
    : "Select a Quiz Deck";

  return (
    <div className="quiz-layout-container animate-in fade-in zoom-in-95 duration-300">

      {/* ── Main Panel (Left) ── */}
      <div className="quiz-main flex-1 overflow-hidden flex flex-col relative z-0">

        <div className="quiz-top-bar flex items-center justify-between px-10 py-6 border-b border-glass flex-shrink-0">
          <div className="flex flex-col">
            <h2 className="text-2xl font-display font-semibold m-0 text-white flex items-center gap-3">
              <HelpCircle size={24} className="text-accent" />
              {currentDisplayName}
            </h2>
            {validQs.length > 0 && !done && (
              <div className="text-sm font-mono text-muted mt-2 tracking-wide uppercase opacity-70">
                Question {answeredN + 1} of {validQs.length}
              </div>
            )}
          </div>

          <div className="flex items-center gap-4">
            {validQs.length > 0 && (
              <button className="btn btn-ghost btn-sm bg-black/20 hover:bg-black/40" onClick={retry}>
                <RefreshCw size={14} /> Reset
              </button>
            )}
          </div>
        </div>

        {!done && validQs.length > 0 && (
          <div className="quiz-progress-bar h-1.5 w-full bg-black/40 relative">
            <div
              className="h-full bg-accent transition-all duration-500 ease-out shadow-[0_0_10px_rgba(var(--accent-primary-rgb),0.5)]"
              style={{ width: (answeredN / validQs.length * 100) + '%' }}
            />
          </div>
        )}

        <QuizErrorBoundary>
          {loading ? (
            <div className="flex-1 flex flex-col items-center justify-center p-20">
              <div className="spinner mb-4" />
              <p className="text-muted mono text-sm animate-pulse">Accessing Intelligence Database...</p>
            </div>
          ) : (
            renderBody()
          )}
        </QuizErrorBoundary>
      </div>

      {/* ── Right Panel: Media Selection ── */}
      <div className="quiz-sidebar z-20 relative bg-black/40 border-l border-glass w-[340px] flex flex-col flex-shrink-0">
        <div className="sidebar-header p-6 border-b border-glass flex items-center justify-between shadow-sm bg-black/20">
          <h3 className="text-lg font-display font-semibold m-0 text-white flex items-center gap-2">
            <HelpCircle size={18} className="text-accent" />
            Select Quiz Deck
          </h3>
        </div>
        <div className="sidebar-list overflow-y-auto flex-1 p-4 space-y-3">
          {allStems.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted">No media available yet.</div>
          ) : (
            allStems.map(s => {
              const key = stemToKey[s];
              const vid = videos[key];
              const isReady = vid?.quiz_ready;
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
                        <span className="text-success flex items-center gap-1"><CheckCircle2 size={12} /> Ready</span>
                      ) : (
                        <span className="text-warning flex items-center gap-1 animate-pulse"><RefreshCw size={12} className="animate-spin-slow" /> Generating</span>
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