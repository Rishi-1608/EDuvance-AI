import { useState, useEffect } from 'react';
import { Layers, ChevronLeft, ChevronRight, RefreshCw, Shuffle, AlertCircle, Video, Mic, CheckCircle2 } from 'lucide-react';
import { useStatus } from '@/hooks/useStatus';
import { getFlashcards } from '@/lib/api';
import { toStem, shortStem } from '@/lib/utils';
import './FlashcardsPage.css';

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function FlashcardsPage({ preselectedStem }) {
  const { status } = useStatus(5000);
  const [selectedStem, setSelectedStem] = useState(preselectedStem || '');
  const [cards, setCards] = useState([]);
  const [current, setCurrent] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [known, setKnown] = useState(new Set());

  const videos = status?.videos || {};
  const allKeys = Object.keys(videos).sort((a, b) => videos[b].created_at - videos[a].created_at);
  const allStems = allKeys.map(toStem);
  const stemToKey = Object.fromEntries(allKeys.map(k => [toStem(k), k]));
  const readyStems = allKeys.filter(k => videos[k].flashcards_ready).map(toStem);

  useEffect(() => {
    if (preselectedStem) {
      setSelectedStem(preselectedStem);
    } else if (!selectedStem && readyStems.length > 0) {
      setSelectedStem(readyStems[0]); // Select newest ready by default
    }
  }, [preselectedStem, readyStems.join(',')]);

  const fetchCards = async (stem) => {
    setLoading(true);
    setError(null);
    setCards([]);
    setCurrent(0);
    setFlipped(false);
    setKnown(new Set());
    try {
      const data = await getFlashcards(stem);
      setCards(Array.isArray(data) ? data : []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { if (selectedStem) fetchCards(selectedStem); }, [selectedStem]);

  const card = cards[current];
  const progress = cards.length ? Math.round((current / cards.length) * 100) : 0;

  const next = () => {
    setFlipped(false);
    setTimeout(() => setCurrent(c => Math.min(c + 1, cards.length - 1)), 150);
  };
  const prev = () => {
    setFlipped(false);
    setTimeout(() => setCurrent(c => Math.max(c - 1, 0)), 150);
  };
  const reset = () => { setCurrent(0); setFlipped(false); setKnown(new Set()); };
  const doShuffle = () => { setCards(s => shuffle(s)); setCurrent(0); setFlipped(false); };
  const markKnown = () => {
    setKnown(k => { const n = new Set(k); n.add(current); return n; });
    if (current < cards.length - 1) next();
  };

  const currentDisplayTitle = selectedStem && stemToKey[selectedStem]
    ? videos[stemToKey[selectedStem]]?.lecture_title || videos[stemToKey[selectedStem]]?.display_name || selectedStem
    : "Flashcards";

  return (
    <div className="flashcards-layout-container glass animate-in fade-in zoom-in-95 duration-300">

      {/* ── Main Panel (Left) ── */}
      <div className="flashcards-main flex-1 overflow-hidden px-10 py-8 flex flex-col relative z-0">

        <div className="fc-header flex items-center justify-between mb-6 pb-5 border-b border-glass flex-shrink-0">
          <div className="flex flex-col">
            <h2 className="text-2xl font-display font-semibold m-0 text-white flex items-center gap-3 shadow-text">
              <Layers size={24} className="text-accent" />
              {currentDisplayTitle}
            </h2>
            {cards.length > 0 && <div className="text-sm font-mono text-muted mt-2 tracking-wide uppercase opacity-70 block">{cards.length} cards in this deck</div>}
          </div>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
            {cards.length > 0 && (
              <>
                <button className="btn btn-ghost btn-sm bg-black/20 hover:bg-black/40" onClick={doShuffle}>
                  <Shuffle size={14} /> Shuffle
                </button>
                <button className="btn btn-ghost btn-sm bg-black/20 hover:bg-black/40" onClick={reset}>
                  <RefreshCw size={14} /> Reset
                </button>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="fc-error"><AlertCircle size={15} /> {error}</div>
        )}

        {loading && (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 80, height: '100%' }}>
            <div className="spinner" style={{ width: 44, height: 44 }} />
          </div>
        )}

        {!loading && cards.length === 0 && !error && (
          <div className="empty-state glass animate-up my-auto border border-white/5 shadow-2xl">
            <Layers size={48} style={{ color: 'var(--text-muted)', marginBottom: 16, opacity: 0.5 }} className="mx-auto" />
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, marginBottom: 8, fontSize: '1.2rem' }} className="text-white">
              {allStems.length === 0 ? 'No media found' : 'Select media to view flashcards'}
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              {allStems.length === 0
                ? 'Upload media to automatically generate flashcards.'
                : 'Select an available media file from the right panel to study its flashcards.'}
            </p>
          </div>
        )}

        {/* The Card Stage Area */}
        {!loading && cards.length > 0 && (
          <div className="max-w-4xl w-full mx-auto flex flex-col flex-1 min-h-0 pb-2">
            <div className="fc-progress glass animate-up flex-shrink-0 w-full border border-white/5 bg-black/40 mb-8">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                <span className="mono bg-black/40 px-2 py-0.5 rounded-md border border-white/5 shadow-inner">Card {current + 1} / {cards.length}</span>
                <span className="mono text-success flex items-center gap-1.5"><CheckCircle2 size={14} /> {known.size} Mastered</span>
              </div>
              <div className="progress-track h-2 bg-black/50 rounded-full border border-white/5 shadow-inner overflow-hidden">
                <div className="progress-fill h-full bg-accent transition-all duration-300" style={{ width: Math.max(progress, 2) + '%' }} />
              </div>
            </div>

            <div className="card-stage animate-up stagger-1 w-full flex-1 flex flex-col justify-center items-center min-h-0 mb-8">
              <div
                className={'flashcard ' + (flipped ? 'flipped' : '')}
                onClick={() => setFlipped(f => !f)}
                role="button"
                tabIndex={0}
                onKeyDown={e => e.key === 'Enter' && setFlipped(f => !f)}
              >
                <div className="flashcard-inner">
                  <div className="flashcard-front glass border border-white/10 shadow-xl">
                    <div className="card-side-label mono">Question</div>
                    <div className="card-text text-lg md:text-xl lg:text-2xl font-display">
                      {card && (card.question || card.front || Object.values(card)[0])}
                    </div>
                    <div className="flip-hint mono shimmer-text">click or press enter to flip</div>
                  </div>
                  <div className="flashcard-back glass border border-accent/20 shadow-glow">
                    <div className="card-side-label mono" style={{ color: 'var(--accent-primary)' }}>Answer</div>
                    <div className="card-text text-base md:text-lg lg:text-xl text-accent font-medium bg-black/20 p-4 md:p-6 rounded-xl border border-white/5 overflow-y-auto max-h-[85%]">
                      {card && (card.answer || card.back || Object.values(card)[1])}
                    </div>
                  </div>
                </div>
              </div>
              {known.has(current) && (
                <div className="known-badge absolute -top-4 -right-4 z-20 animate-bounce">
                  <span className="badge badge-success shadow-lg px-3 py-1.5 font-bold uppercase tracking-wider bg-green-500 text-black border-transparent">Known</span>
                </div>
              )}
            </div>

            <div className="fc-controls flex-shrink-0 animate-up stagger-2 w-full flex justify-between gap-4 mb-6">
              <button className="btn btn-ghost hover:bg-black/20 hover:scale-105 transition-transform" onClick={prev} disabled={current === 0}>
                <ChevronLeft size={20} /> <span className="hidden sm:inline">Previous</span>
              </button>
              <div style={{ display: 'flex', gap: 12 }}>
                <button
                  className="btn btn-ghost border hover:bg-black/20 transition-all hover:scale-105"
                  style={{ color: 'var(--error)', borderColor: 'rgba(248,113,113,0.3)' }}
                  onClick={next}
                  disabled={current === cards.length - 1}
                >
                  Skip
                </button>
                <button className="btn btn-primary font-bold shadow-glow hover:scale-105 transition-transform px-6 py-2" onClick={markKnown}>Got it !</button>
              </div>
              <button className="btn btn-ghost hover:bg-black/20 hover:scale-105 transition-transform" onClick={next} disabled={current === cards.length - 1}>
                <span className="hidden sm:inline">Next</span> <ChevronRight size={20} />
              </button>
            </div>

            <div className="fc-overview flex-shrink-0 animate-up stagger-3 w-full bg-black/20 p-4 rounded-xl border border-white/5 mt-2">
              <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Quick Navigate ({cards.length})
              </div>
              <div className="card-dots">
                {cards.map((_, i) => (
                  <button
                    key={i}
                    className={'card-dot hover:scale-125 transition-transform' + (i === current ? ' current' : '') + (known.has(i) ? ' known' : '')}
                    onClick={() => { setCurrent(i); setFlipped(false); }}
                    aria-label={'Card ' + (i + 1)}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>


      {/* ── Right Panel: Media Selection ── */}
      <div className="flashcards-sidebar z-20 relative bg-black/40 border-l border-glass w-[340px] flex flex-col flex-shrink-0">
        <div className="sidebar-header p-6 border-b border-glass flex items-center justify-between shadow-sm bg-black/20">
          <h3 className="text-lg font-display font-semibold m-0 text-white flex items-center gap-2">
            <Layers size={18} className="text-accent" />
            Select Media Deck
          </h3>
        </div>
        <div className="sidebar-list overflow-y-auto flex-1 p-4 space-y-3">
          {allStems.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted">No media available yet.</div>
          ) : (
            allStems.map(s => {
              const key = stemToKey[s];
              const vid = videos[key];
              const isReady = vid?.flashcards_ready;
              const isSelected = selectedStem === s;
              const isAudioOnly = vid?.audio_ready && !vid?.summary_ready && vid?.input_type !== 'document' && vid?.input_type !== 'image'; // proxy logic for icon

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
                        <span className="text-success flex items-center gap-1 opacity-90"><CheckCircle2 size={12} /> {vid?.flashcard_count || 'Ready'} Cards</span>
                      ) : (
                        <span className="text-warning flex items-center gap-1 opacity-90"><RefreshCw size={10} className="animate-spin" /> Generating...</span>
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
