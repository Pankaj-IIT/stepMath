import React, { useState, useEffect, useRef } from 'react';
import katex from 'katex';

// ---------------------------------------------------------------------------
// What the Python server sends back
// ---------------------------------------------------------------------------
interface Step {
  name: string;
  latex: string;
  formula?: string | null; // the formula that was applied, if it came from the database
}

interface SolveResponse {
  steps: Step[];
  final: string;
  alternatives?: AlternativeWay[]; // other routes to the same answer
}

interface AskResponse extends SolveResponse {
  intent?: string; // simplify | prove | solve | ...
  explanation?: string; // why the app read the question that way
  proven?: boolean | null; // only for prove
}

interface AlternativeWay {
  name: string; // the first move of that route
  steps: Step[];
}

interface Entry {
  id: number;
  kind: string; // 'identity' | 'method' | 'strategy' | 'note' | 'solution'
  name: string;
  statement: string; // LaTeX, may be empty
  steps: string | null; // methods only: their steps as JSON text
  verified: number; // 1 = trusted, 0 = waiting for review
  origin: string; // 'ai' | 'hand'
  source: string | null;
  usage_count: number;
  groups?: string[]; // formula families, e.g. ['two-angle-sum']
}

interface EntryList {
  entries: Entry[];
  counts: { total: number; verified: number; needs_review: number };
}

interface ErrorResponse {
  error?: string;
}

interface StrategyPhase {
  do: string; // "walk" or a known operation name
  scope: string; // "whole" | "term"
  until: string; // "stuck" | "once"
}

interface Strategy {
  id: number;
  name: string;
  form: string;
  phases: StrategyPhase[];
  verified: number;
  usage_count: number;
}

interface StrategyVocabulary {
  forms: string[];
  dos: string[];
  scopes: string[];
  untils: string[];
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

// One solution for now: the "other ways" block below is held for a later
// version (the JSX stays, this switch just keeps it off screen).
const SHOW_ALTERNATIVE_WAYS = false;

// Style for the little round labels (the text goes in the element's children).
// Colour of the little round label for each kind of entry.
const KIND_COLOR: Record<string, string> = {
  identity: '#38bdf8',
  method: '#a78bfa',
  note: '#94a3b8',
  solution: '#f472b6',
};

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

// Text fields for the Strategies add/edit panel (same look as the
// search box and the Solve input).
const INPUT: React.CSSProperties = {
  width: '100%',
  padding: '10px 14px',
  borderRadius: '8px',
  border: '1px solid #475569',
  background: '#0f172a',
  color: '#f8fafc',
  fontSize: '1rem',
  boxSizing: 'border-box',
};

const SELECT: React.CSSProperties = {
  padding: '8px 10px',
  borderRadius: '8px',
  border: '1px solid #475569',
  background: '#0f172a',
  color: '#f8fafc',
  fontSize: '0.85rem',
};

// ---------------------------------------------------------------------------
// The app
// ---------------------------------------------------------------------------
export default function App() {
  // --- which page ---------------------------------------------------------
  const [tab, setTab] = useState<'solve' | 'formulas' | 'strategies'>('solve');

  // --- solve page ---------------------------------------------------------
  const [inputStr, setInputStr] = useState('2*x + 3*x + 5 - 1');
  const [mode, setMode] = useState<'simplify' | 'solve'>('simplify');
  const [steps, setSteps] = useState<Step[]>([]);
  const [alternatives, setAlternatives] = useState<AlternativeWay[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  // How the app understood the question ("PROVE -- it starts with ...").
  const [reading, setReading] = useState<{ intent: string; explanation: string } | null>(null);

  // --- formulas page ------------------------------------------------------
  const [query, setQuery] = useState('');
  const [reviewOnly, setReviewOnly] = useState(false);
  const [selectedFamily, setSelectedFamily] = useState(''); // '' = all
  const [families, setFamilies] = useState<
    { name: string; description: string; count: number }[]
  >([]);
  // Simplifications SymPy found that we cannot reach with steps yet --
  // the "strategies to plug" list for whoever supplies formulas.
  const [gaps, setGaps] = useState<
    { target: string; example: string; tried: string; hits: number }[]
  >([]);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [counts, setCounts] = useState({ total: 0, verified: 0, needs_review: 0 });
  const [formulaError, setFormulaError] = useState<string | null>(null);

  // --- strategies page ----------------------------------------------------
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [vocab, setVocab] = useState<StrategyVocabulary | null>(null);
  const [strategyError, setStrategyError] = useState<string | null>(null);
  // The add/edit panel: null = closed; id null = adding a new strategy.
  const [draft, setDraft] = useState<{
    id: number | null;
    name: string;
    form: string;
    phases: StrategyPhase[];
  } | null>(null);
  // Bumped after every successful save/delete -> the list refetches.
  const [strategyKey, setStrategyKey] = useState(0);

  // Ask the server for automatic steps whenever the text/mode changes.
  useEffect(() => {
    const controller = new AbortController();

    // Wait 300ms after the last keystroke, so we do not send a request
    // for every single letter.
    const timer = setTimeout(async () => {
      try {
        // Simplify mode goes through the QUESTION layer (it can spot
        // "prove that ... = ..." inside the box); the Solve toggle keeps
        // the direct equation endpoint.
        const isEquation = mode === 'solve';
        const data = await api<AskResponse>(isEquation ? '/api/solve' : '/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(isEquation
            ? { expression: inputStr, mode }
            : { question: inputStr }),
          signal: controller.signal,
        });
        setSteps(data.steps);
        setAlternatives(data.alternatives ?? []);
        // Only show the banner when it recognised a QUESTION FORM --
        // a plain expression gets no banner (that would be noise).
        setReading(
          data.intent && data.intent !== 'simplify'
            ? { intent: data.intent, explanation: data.explanation ?? '' }
            : null
        );
        setError(null);
      } catch (err: any) {
        if (err?.name === 'AbortError') return; // user kept typing, ignore
        setSteps([]);
        setAlternatives([]);
        setReading(null);
        setError(err?.message ?? 'Cannot reach the Python server. Start it with:  npm run api');
      }
    }, 300);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [inputStr, mode, refreshKey]);

  // Load formulas whenever the search page needs them.
  useEffect(() => {
    if (tab !== 'formulas') return;

    const timer = setTimeout(async () => {
      try {
        const verified = reviewOnly ? '&verified=0' : '';
        const family = selectedFamily ? `&group=${encodeURIComponent(selectedFamily)}` : '';
        const data = await api<EntryList>(
          `/api/entries?q=${encodeURIComponent(query)}${verified}${family}&limit=500`
        );
        setEntries(data.entries);
        setCounts(data.counts);
        setFormulaError(null);
      } catch (err: any) {
        setFormulaError(err?.message ?? 'Could not load the formula list.');
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [tab, query, reviewOnly, selectedFamily, refreshKey]);

  // The formula families (filter chips) for the search page.
  useEffect(() => {
    if (tab !== 'formulas') return;

    let cancelled = false;
    (async () => {
      try {
        const data = await api<{ groups: { name: string; description: string; count: number }[] }>(
          '/api/groups'
        );
        if (!cancelled) setFamilies(data.groups);
      } catch {
        if (!cancelled) setFamilies([]); // no database: no chips
      }
      try {
        const data = await api<{ gaps: { target: string; example: string; tried: string; hits: number }[] }>(
          '/api/gaps'
        );
        if (!cancelled) setGaps(data.gaps);
      } catch {
        if (!cancelled) setGaps([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, refreshKey]);

  // Load the strategy list (and the phases vocabulary) when the tab
  // opens -- and again after every save/delete.
  useEffect(() => {
    if (tab !== 'strategies') return;

    let cancelled = false;
    (async () => {
      try {
        const data = await api<{ strategies: Strategy[] }>('/api/strategies');
        if (!cancelled) {
          setStrategies(data.strategies);
          setStrategyError(null);
        }
      } catch (err: any) {
        if (!cancelled) {
          setStrategyError(err?.message ?? 'Could not load the strategies.');
        }
      }
      try {
        const data = await api<StrategyVocabulary>('/api/strategy-meta');
        if (!cancelled) setVocab(data);
      } catch {
        if (!cancelled) setVocab(null); // no server: the selects stay empty
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, strategyKey]);

  // --- actions ------------------------------------------------------------
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

  // --- strategies: open the panel, save, delete ---------------------------
  const startAddStrategy = () => {
    setStrategyError(null);
    setDraft({
      id: null,
      name: '',
      form: vocab?.forms[0] ?? '',
      phases: [{ do: 'walk', scope: 'whole', until: 'stuck' }],
    });
  };

  const startEditStrategy = (strategy: Strategy) => {
    setStrategyError(null);
    setDraft({
      id: strategy.id,
      name: strategy.name,
      form: strategy.form,
      phases: strategy.phases.map((phase) => ({ ...phase })),
    });
  };

  const closeStrategyPanel = () => {
    setDraft(null);
    setStrategyError(null);
  };

  // Instant save: POST when adding, PUT when editing; on an API error the
  // server's message is shown (same pattern as the other pages).
  const saveStrategy = async () => {
    if (!draft) return;
    try {
      await api(draft.id === null ? '/api/strategies' : `/api/strategies/${draft.id}`, {
        method: draft.id === null ? 'POST' : 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: draft.name, form: draft.form, phases: draft.phases }),
      });
      setDraft(null);
      setStrategyKey((key) => key + 1); // reload the list
    } catch (err: any) {
      setStrategyError(err?.message ?? 'That did not work.');
    }
  };

  const deleteStrategy = async (strategy: Strategy) => {
    if (!window.confirm(`Delete the strategy "${strategy.name}"?`)) return;
    try {
      await api(`/api/strategies/${strategy.id}`, { method: 'DELETE' });
      setStrategyKey((key) => key + 1); // reload the list
    } catch (err: any) {
      setStrategyError(err?.message ?? 'That did not work.');
    }
  };

  const setPhaseField = (
    index: number,
    field: 'do' | 'scope' | 'until',
    value: string
  ) => {
    if (!draft) return;
    const phases = draft.phases.map((phase, i) => {
      if (i !== index) return phase;
      const next = { ...phase };
      next[field] = value;
      return next;
    });
    setDraft({ ...draft, phases });
  };

  const removePhase = (index: number) => {
    if (!draft) return;
    setDraft({ ...draft, phases: draft.phases.filter((_, i) => i !== index) });
  };

  const addPhase = () => {
    if (!draft) return;
    setDraft({
      ...draft,
      phases: [...draft.phases, { do: 'walk', scope: 'whole', until: 'stuck' }],
    });
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

  const tabButton = (value: 'solve' | 'formulas' | 'strategies', label: string) => (
    <button
      onClick={() => setTab(value)}
      style={BUTTON(tab === value ? '#22c55e' : '#334155', tab === value ? '#0f172a' : '#cbd5e1')}
    >
      {label}
    </button>
  );

  // One step card. Used by the main route AND by every alternative route,
  // so both look and behave the same.
  const stepCard = (step: Step, index: number, key: string) => (
    <div key={key} style={CARD(index === 0 ? '#38bdf8' : '#22c55e')}>
      <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '4px' }}>
        {index === 0 ? 'Initial Expression' : `Step ${index}: "${step.name}"`}
      </div>
      <MathBlock math={step.latex} />

      {/* When a database formula was applied, show it too, so the student
          can see WHICH rule of maths was used. */}
      {step.formula && (
        <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px dashed #334155' }}>
          <div style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '2px' }}>
            Formula used:
          </div>
          <MathBlock math={step.formula} />
        </div>
      )}
    </div>
  );

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', padding: '24px', background: '#1e293b', borderRadius: '12px' }}>
      <h1 style={{ fontSize: '1.6rem', marginBottom: '16px' }}>Interactive Math Step Solver</h1>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
        {tabButton('solve', 'Solve')}
        {tabButton('formulas', `Formulas${counts.needs_review ? ` (${counts.needs_review} to check)` : ''}`)}
        {tabButton('strategies', 'Strategies')}
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

          {/* How the question was understood -- only for question forms. */}
          {reading && (
            <div style={{ ...CARD('#38bdf8'), marginBottom: '14px' }}>
              <span style={{ color: '#7dd3fc', fontWeight: 600 }}>
                Reading this as: {reading.intent.toUpperCase()}
              </span>
              <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                {' '}— because {reading.explanation}
              </span>
            </div>
          )}

          {/* Methods live in the flow now: the main route, or under
              "Other ways" below -- there is no button bar. */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {steps.map((step, index) => stepCard(step, index, String(index)))}
          </div>

          {/* Other routes that finish at exactly the same answer, so a
              student can compare ways of doing the same question. */}
          {SHOW_ALTERNATIVE_WAYS && alternatives.length > 0 && (
            <div style={{ marginTop: '30px' }}>
              <div style={{ color: '#94a3b8', marginBottom: '12px', fontWeight: 600 }}>
                Other ways to reach the same answer:
              </div>

              {alternatives.map((way, wayIndex) => (
                <div key={wayIndex} style={{ marginBottom: '22px' }}>
                  <div style={{ color: '#facc15', fontWeight: 600, marginBottom: '8px' }}>
                    Way {wayIndex + 2} — starts with "{way.name}"
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {way.steps.map((step, index) =>
                      stepCard(step, index, `${wayIndex}-${index}`)
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      ) : tab === 'formulas' ? (
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

          {/* Formula families: click one to see only its members. */}
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '6px',
              marginBottom: '12px',
            }}
          >
            <button
              onClick={() => setSelectedFamily('')}
              title="Show every formula"
              style={BUTTON(
                selectedFamily === '' ? '#38bdf8' : '#334155',
                selectedFamily === '' ? '#0f172a' : '#cbd5e1'
              )}
            >
              All ({counts.total})
            </button>
            {families.map((family) => (
              <button
                key={family.name}
                onClick={() => setSelectedFamily(family.name)}
                title={family.description}
                style={BUTTON(
                  selectedFamily === family.name ? '#a78bfa' : '#334155',
                  selectedFamily === family.name ? '#0f172a' : '#cbd5e1'
                )}
              >
                {family.name} ({family.count})
              </button>
            ))}
          </div>

          <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '14px' }}>
            {counts.total} formulas · {counts.needs_review} waiting for review
            {selectedFamily ? ` · showing family "${selectedFamily}"` : ''}
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
                  <span style={BADGE(KIND_COLOR[entry.kind] ?? '#38bdf8')}>
                    {entry.kind}
                  </span>
                  <span style={BADGE(entry.verified ? '#22c55e' : '#f97316')}>
                    {entry.verified ? 'verified' : 'to check'}
                  </span>
                  {/* The formula families this row belongs to */}
                  {(entry.groups ?? []).map((groupName) => (
                    <span
                      key={groupName}
                      onClick={() => setSelectedFamily(groupName)}
                      title="Click to see this whole family"
                      style={{ ...BADGE('#64748b'), cursor: 'pointer' }}
                    >
                      {groupName}
                    </span>
                  ))}
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

          {/* "Plug the strategy here": simplifications SymPy found that
              our step-by-step strategies cannot demonstrate yet. Each row
              is a formula or strategy we could add. */}
          {gaps.length > 0 && (
            <div style={{ marginTop: '28px' }}>
              <div style={{ color: '#facc15', fontWeight: 600, marginBottom: '6px' }}>
                Strategies we are missing ({gaps.length})
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '10px' }}>
                SymPy found these simpler forms, but no formula of ours reaches
                them yet -- each is a candidate for the next formula batch.
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {gaps.map((gap) => (
                  <div key={gap.target} style={CARD('#f97316')}>
                    <MathBlock math={gap.target} />
                    <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
                      from: <MathBlock math={gap.example} /> tried: {gap.tried || '—'} · seen {gap.hits}×
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          {/* --- the strategies page: list + add/edit panel --- */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '14px' }}>
            <button onClick={startAddStrategy} style={BUTTON('#22c55e')}>
              Add strategy
            </button>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8', alignSelf: 'center' }}>
              {strategies.length} strategies · instant save
            </span>
          </div>

          {strategyError && <p style={{ color: '#ef4444' }}>{strategyError}</p>}

          {/* Add / Edit panel -- appears under the tab while open. */}
          {draft && (
            <div style={{ ...CARD('#38bdf8'), marginBottom: '14px' }}>
              <strong style={{ color: '#f8fafc' }}>
                {draft.id === null ? 'New strategy' : 'Edit strategy'}
              </strong>

              <div style={{ marginTop: '10px' }}>
                <label style={{ display: 'block', marginBottom: '4px', color: '#94a3b8', fontSize: '0.85rem' }}>
                  Name
                </label>
                <input
                  type="text"
                  value={draft.name}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                  style={INPUT}
                />
              </div>

              <div style={{ marginTop: '10px' }}>
                <label style={{ display: 'block', marginBottom: '4px', color: '#94a3b8', fontSize: '0.85rem' }}>
                  Form (the shape it matches)
                </label>
                <select
                  value={draft.form}
                  onChange={(e) => setDraft({ ...draft, form: e.target.value })}
                  style={SELECT}
                >
                  {(vocab?.forms ?? []).map((form) => (
                    <option key={form} value={form}>
                      {form}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ marginTop: '10px' }}>
                <label style={{ display: 'block', marginBottom: '4px', color: '#94a3b8', fontSize: '0.85rem' }}>
                  Phases (run in this order)
                </label>
                {draft.phases.map((phase, index) => (
                  <div
                    key={index}
                    style={{ display: 'flex', gap: '6px', marginTop: '6px', alignItems: 'center' }}
                  >
                    <span style={{ color: '#94a3b8', fontSize: '0.8rem', width: '18px' }}>
                      {index + 1}.
                    </span>
                    <select
                      value={phase.do}
                      onChange={(e) => setPhaseField(index, 'do', e.target.value)}
                      style={SELECT}
                    >
                      {(vocab?.dos ?? []).map((op) => (
                        <option key={op} value={op}>
                          {op}
                        </option>
                      ))}
                    </select>
                    <select
                      value={phase.scope}
                      onChange={(e) => setPhaseField(index, 'scope', e.target.value)}
                      style={SELECT}
                    >
                      {(vocab?.scopes ?? []).map((scope) => (
                        <option key={scope} value={scope}>
                          {scope}
                        </option>
                      ))}
                    </select>
                    <select
                      value={phase.until}
                      onChange={(e) => setPhaseField(index, 'until', e.target.value)}
                      style={SELECT}
                    >
                      {(vocab?.untils ?? []).map((until) => (
                        <option key={until} value={until}>
                          {until}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => removePhase(index)}
                      title="Remove this phase"
                      style={BUTTON('#ef4444', '#fff')}
                    >
                      −
                    </button>
                  </div>
                ))}
                <button
                  onClick={addPhase}
                  style={{ ...BUTTON('#334155', '#cbd5e1'), marginTop: '8px' }}
                >
                  Add phase
                </button>
              </div>

              <div style={{ display: 'flex', gap: '8px', marginTop: '14px' }}>
                <button onClick={saveStrategy} style={BUTTON('#22c55e')}>
                  Save
                </button>
                <button onClick={closeStrategyPanel} style={BUTTON('#334155', '#cbd5e1')}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {strategies.map((strategy) => (
              <div key={strategy.id} style={CARD('#a78bfa')}>
                <div
                  style={{
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'center',
                    marginBottom: '6px',
                    flexWrap: 'wrap',
                  }}
                >
                  <strong style={{ color: '#f8fafc' }}>{strategy.name}</strong>
                  <span style={BADGE('#38bdf8')}>{strategy.form}</span>
                  <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    used {strategy.usage_count}×
                  </span>
                </div>

                <ol
                  style={{
                    margin: '6px 0',
                    paddingLeft: '20px',
                    color: '#cbd5e1',
                    fontSize: '0.85rem',
                  }}
                >
                  {strategy.phases.map((phase, index) => (
                    <li key={index} style={{ marginBottom: '2px' }}>
                      {phase.do} · {phase.scope} · until {phase.until}
                    </li>
                  ))}
                </ol>

                <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                  <button onClick={() => startEditStrategy(strategy)} style={BUTTON('#38bdf8')}>
                    Edit
                  </button>
                  <button
                    onClick={() => deleteStrategy(strategy)}
                    style={BUTTON('#ef4444', '#fff')}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}

            {strategies.length === 0 && !strategyError && (
              <p style={{ color: '#94a3b8' }}>No strategies yet.</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
