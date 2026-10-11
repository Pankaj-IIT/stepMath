import React, { useState, useEffect, useRef } from 'react';
import katex from 'katex';

// ---------------------------------------------------------------------------
// What the Python server sends back
// ---------------------------------------------------------------------------
interface Step {
  name: string;
  latex: string;
}

interface SolveResponse {
  steps: Step[];
  final: string;
}

interface Entry {
  id: number;
  kind: 'identity' | 'method';
  name: string;
  statement: string; // LaTeX, may be empty
  steps: string | null; // methods only: their steps as JSON text
  verified: number; // 1 = trusted, 0 = waiting for review
  origin: string; // 'ai' | 'hand'
  source: string | null;
  usage_count: number;
}

interface EntryList {
  entries: Entry[];
  counts: { total: number; verified: number; needs_review: number };
}

interface ErrorResponse {
  error?: string;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Draws one line of maths (e.g. "5 x + 4") using KaTeX.
function MathBlock({ math }: { math: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      katex.render(math, containerRef.current, { displayMode: true, throwOnError: false });
    }
  }, [math]);

  return <div ref={containerRef} />;
}

// One call to the Python server. Throws a readable message on failure.
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const data = await response.json();

  if (!response.ok) {
    throw new Error((data as ErrorResponse).error ?? 'Something went wrong.');
  }
  return data as T;
}

// A one-line summary of a method's steps, for the formula list.
function methodSummary(stepsJson: string | null): string {
  if (!stepsJson) return '';
  try {
    const steps = JSON.parse(stepsJson);
    return steps
      .map((step: any) => (step.op ? step.op : `${step.left} -> ${step.right}`))
      .join('   |   ');
  } catch {
    return '';
  }
}

// Style for the little round labels (the text goes in the element's children).
const BADGE = (color: string): React.CSSProperties => ({
  fontSize: '0.7rem',
  padding: '2px 8px',
  borderRadius: '999px',
  background: color,
  color: '#0f172a',
  fontWeight: 700,
  letterSpacing: '0.03em',
});

const BUTTON = (background: string, color = '#0f172a'): React.CSSProperties => ({
  padding: '7px 14px',
  borderRadius: '8px',
  border: 'none',
  background,
  color,
  cursor: 'pointer',
  fontSize: '0.85rem',
  fontWeight: 600,
});

const CARD = (borderColor: string): React.CSSProperties => ({
  background: '#0f172a',
  padding: '16px',
  borderRadius: '8px',
  borderLeft: `4px solid ${borderColor}`,
});

// ---------------------------------------------------------------------------
// The app
// ---------------------------------------------------------------------------
export default function App() {
  // --- which page ---------------------------------------------------------
  const [tab, setTab] = useState<'solve' | 'formulas'>('solve');

  // --- solve page ---------------------------------------------------------
  const [inputStr, setInputStr] = useState('2*x + 3*x + 5 - 1');
  const [mode, setMode] = useState<'simplify' | 'solve'>('simplify');
  const [steps, setSteps] = useState<Step[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [methodTitle, setMethodTitle] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [methods, setMethods] = useState<Entry[]>([]);

  // --- formulas page ------------------------------------------------------
  const [query, setQuery] = useState('');
  const [reviewOnly, setReviewOnly] = useState(false);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [counts, setCounts] = useState({ total: 0, verified: 0, needs_review: 0 });
  const [formulaError, setFormulaError] = useState<string | null>(null);

  // Ask the server for automatic steps whenever the text/mode changes.
  useEffect(() => {
    const controller = new AbortController();

    // Wait 300ms after the last keystroke, so we do not send a request
    // for every single letter.
    const timer = setTimeout(async () => {
      try {
        const data = await api<SolveResponse>('/api/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ expression: inputStr, mode }),
          signal: controller.signal,
        });
        setSteps(data.steps);
        setError(null);
        setMethodTitle(null);
      } catch (err: any) {
        if (err?.name === 'AbortError') return; // user kept typing, ignore
        setSteps([]);
        setError(err?.message ?? 'Cannot reach the Python server. Start it with:  npm run api');
      }
    }, 300);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [inputStr, mode, refreshKey]);

  // Which complete methods could start on this expression?
  useEffect(() => {
    if (tab !== 'solve' || mode !== 'simplify') return;

    const timer = setTimeout(async () => {
      try {
        const data = await api<EntryList>(
          `/api/methods/applicable?expression=${encodeURIComponent(inputStr)}`
        );
        setMethods(data.entries);
      } catch {
        setMethods([]); // no database / bad expression: just show nothing
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [inputStr, mode, tab]);

  // Load formulas whenever the search page needs them.
  useEffect(() => {
    if (tab !== 'formulas') return;

    const timer = setTimeout(async () => {
      try {
        const verified = reviewOnly ? '&verified=0' : '';
        const data = await api<EntryList>(
          `/api/entries?q=${encodeURIComponent(query)}${verified}&limit=50`
        );
        setEntries(data.entries);
        setCounts(data.counts);
        setFormulaError(null);
      } catch (err: any) {
        setFormulaError(err?.message ?? 'Could not load the formula list.');
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [tab, query, reviewOnly, refreshKey]);

  // --- actions ------------------------------------------------------------
  const runMethod = async (entry: Entry) => {
    try {
      const data = await api<SolveResponse>('/api/methods/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: inputStr, id: entry.id }),
      });
      setSteps(data.steps);
      setMethodTitle(entry.name);
      setError(null);
      setTab('solve');
    } catch (err: any) {
      setError(err?.message ?? 'The method could not run.');
    }
  };

  const review = async (entryId: number, approve: boolean) => {
    try {
      await api(`/api/entries/${entryId}/${approve ? 'approve' : 'reject'}`, {
        method: 'POST',
      });
      setRefreshKey((key) => key + 1); // reload the list
    } catch (err: any) {
      setFormulaError(err?.message ?? 'That did not work.');
    }
  };

  // --- shared bits --------------------------------------------------------
  const modeButton = (value: 'simplify' | 'solve', label: string) => (
    <button
      onClick={() => setMode(value)}
      style={{
        ...BUTTON(mode === value ? '#38bdf8' : '#0f172a', mode === value ? '#0f172a' : '#cbd5e1'),
        border: '1px solid #475569',
      }}
    >
      {label}
    </button>
  );

  const tabButton = (value: 'solve' | 'formulas', label: string) => (
    <button
      onClick={() => setTab(value)}
      style={BUTTON(tab === value ? '#22c55e' : '#334155', tab === value ? '#0f172a' : '#cbd5e1')}
    >
      {label}
    </button>
  );

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', padding: '24px', background: '#1e293b', borderRadius: '12px' }}>
      <h1 style={{ fontSize: '1.6rem', marginBottom: '16px' }}>Interactive Math Step Solver</h1>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
        {tabButton('solve', 'Solve')}
        {tabButton('formulas', `Formulas${counts.needs_review ? ` (${counts.needs_review} to check)` : ''}`)}
      </div>

      {tab === 'solve' ? (
        <>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            {modeButton('simplify', 'Simplify')}
            {modeButton('solve', 'Solve equation')}
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '8px', color: '#94a3b8' }}>
              {mode === 'simplify' ? 'Type an Expression:' : 'Type an Equation (with =):'}
            </label>
            <input
              type="text"
              value={inputStr}
              onChange={(e) => setInputStr(e.target.value)}
              style={{
                width: '100%',
                padding: '12px 16px',
                borderRadius: '8px',
                border: '1px solid #475569',
                background: '#0f172a',
                color: '#f8fafc',
                fontSize: '1.1rem',
                boxSizing: 'border-box',
              }}
            />
            {error && <p style={{ color: '#ef4444', marginTop: '6px' }}>{error}</p>}
          </div>

          {/* Methods that could start on this expression, ready to click. */}
          {mode === 'simplify' && methods.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '8px' }}>
                Methods you can run on this expression:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {methods.map((entry) => (
                  <button key={entry.id} onClick={() => runMethod(entry)} style={BUTTON('#facc15')}>
                    {entry.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* When a method ran, say so and offer the automatic steps back. */}
          {methodTitle && (
            <div
              style={{
                ...CARD('#facc15'),
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '14px',
              }}
            >
              <span style={{ color: '#fde68a', fontWeight: 600 }}>Method: {methodTitle}</span>
              <button
                onClick={() => {
                  setMethodTitle(null);
                  setRefreshKey((key) => key + 1);
                }}
                style={BUTTON('#334155', '#cbd5e1')}
              >
                Show automatic steps
              </button>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {steps.map((step, index) => (
              <div key={index} style={CARD(index === 0 ? '#38bdf8' : '#22c55e')}>
                <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '4px' }}>
                  {index === 0 ? 'Initial Expression' : `Step ${index}: "${step.name}"`}
                </div>
                <MathBlock math={step.latex} />
              </div>
            ))}
          </div>
        </>
      ) : (
        <>
          {/* --- the formula book / review queue --- */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
            <input
              type="text"
              placeholder="Search formulas, e.g. square or sin"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{
                flex: 1,
                padding: '10px 14px',
                borderRadius: '8px',
                border: '1px solid #475569',
                background: '#0f172a',
                color: '#f8fafc',
                fontSize: '1rem',
              }}
            />
            <button
              onClick={() => setReviewOnly((value) => !value)}
              style={BUTTON(reviewOnly ? '#f97316' : '#334155', reviewOnly ? '#0f172a' : '#cbd5e1')}
            >
              {reviewOnly ? 'Showing: to check' : 'Needs review'}
            </button>
          </div>

          <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '14px' }}>
            {counts.total} formulas · {counts.needs_review} waiting for review
          </div>

          {formulaError && <p style={{ color: '#ef4444' }}>{formulaError}</p>}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {entries.map((entry) => (
              <div key={entry.id} style={CARD(entry.verified ? '#22c55e' : '#f97316')}>
                <div
                  style={{
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'center',
                    marginBottom: '6px',
                    flexWrap: 'wrap',
                  }}
                >
                  <strong style={{ color: '#f8fafc' }}>{entry.name}</strong>
                  <span style={BADGE(entry.kind === 'method' ? '#a78bfa' : '#38bdf8')}>
                    {entry.kind}
                  </span>
                  <span style={BADGE(entry.verified ? '#22c55e' : '#f97316')}>
                    {entry.verified ? 'verified' : 'to check'}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    {entry.origin} · used {entry.usage_count}×
                    {entry.source ? ` · ${entry.source}` : ''}
                  </span>
                </div>

                {entry.statement && <MathBlock math={entry.statement} />}

                {entry.kind === 'method' && methodSummary(entry.steps) && (
                  <div
                    style={{
                      fontSize: '0.75rem',
                      color: '#94a3b8',
                      fontFamily: 'monospace',
                      marginTop: '6px',
                    }}
                  >
                    {methodSummary(entry.steps)}
                  </div>
                )}

                {!entry.verified && (
                  <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                    <button onClick={() => review(entry.id, true)} style={BUTTON('#22c55e')}>
                      Approve
                    </button>
                    <button onClick={() => review(entry.id, false)} style={BUTTON('#ef4444', '#fff')}>
                      Reject
                    </button>
                  </div>
                )}
              </div>
            ))}

            {entries.length === 0 && !formulaError && (
              <p style={{ color: '#94a3b8' }}>Nothing found.</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
