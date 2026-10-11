import React, { useState, useEffect, useRef } from 'react';
import katex from 'katex';

// ---------------------------------------------------------------------------
// What the Python server sends back: a list of steps, each with a plain
// English name and a LaTeX string that KaTeX can draw.
// ---------------------------------------------------------------------------
interface Step {
  name: string;
  latex: string;
}

interface SolveResponse {
  steps: Step[];
  final: string;
}

interface ErrorResponse {
  error?: string;
}

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

export default function App() {
  const [inputStr, setInputStr] = useState('2*x + 3*x + 5 - 1');
  // "simplify" = tidy an expression, "solve" = find x in an equation
  const [mode, setMode] = useState<'simplify' | 'solve'>('simplify');
  const [steps, setSteps] = useState<Step[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Ask the Python server whenever the text or the mode changes.
  useEffect(() => {
    // Ignore in-flight answers when the user keeps typing.
    const controller = new AbortController();

    // Wait 300ms after the last keystroke, so we do not send a request
    // for every single letter.
    const timer = setTimeout(async () => {
      try {
        const response = await fetch('/api/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ expression: inputStr, mode }),
          signal: controller.signal,
        });

        const data: SolveResponse & ErrorResponse = await response.json();

        if (!response.ok) {
          setSteps([]);
          setError(data.error ?? 'Something went wrong.');
          return;
        }

        setSteps(data.steps);
        setError(null);
      } catch (err: any) {
        if (err?.name === 'AbortError') return; // user kept typing, ignore
        setSteps([]);
        setError('Cannot reach the Python server. Run: uvicorn api.main:app --reload');
      }
    }, 300);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [inputStr, mode]);

  const modeButton = (value: 'simplify' | 'solve', label: string) => (
    <button
      onClick={() => setMode(value)}
      style={{
        padding: '8px 16px',
        borderRadius: '8px',
        border: '1px solid #475569',
        background: mode === value ? '#38bdf8' : '#0f172a',
        color: mode === value ? '#0f172a' : '#cbd5e1',
        cursor: 'pointer',
        fontSize: '0.95rem',
        fontWeight: 600,
      }}
    >
      {label}
    </button>
  );

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', padding: '24px', background: '#1e293b', borderRadius: '12px' }}>
      <h1 style={{ fontSize: '1.6rem', marginBottom: '16px' }}>Interactive Math Step Solver</h1>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        {modeButton('simplify', 'Simplify')}
        {modeButton('solve', 'Solve equation')}
      </div>

      <div style={{ marginBottom: '24px' }}>
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
            boxSizing: 'border-box'
          }}
        />
        {error && <p style={{ color: '#ef4444', marginTop: '6px' }}>{error}</p>}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {steps.map((step, index) => (
          <div
            key={index}
            style={{
              background: '#0f172a',
              padding: '16px',
              borderRadius: '8px',
              borderLeft: index === 0 ? '4px solid #38bdf8' : '4px solid #22c55e',
            }}
          >
            {/* First line is what you typed, the rest are the moves we made. */}
            <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '4px' }}>
              {index === 0 ? 'Initial Expression' : `Step ${index}: "${step.name}"`}
            </div>
            <MathBlock math={step.latex} />
          </div>
        ))}
      </div>
    </div>
  );
}
