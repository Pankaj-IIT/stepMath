import React, { useState, useEffect, useRef } from 'react';
import katex from 'katex';
import { parse } from './engine/parser';
import { solve, CORE_RULES } from './engine/solver';
import { toLaTeX } from './engine/ast';

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
  const [inputStr, setInputStr] = useState('sin(x)^2 + cos(x)^2 + (0 * y) + 4 + 6');
  const [error, setError] = useState<string | null>(null);

  let steps: any[] = [];
  try {
    const parsed = parse(inputStr);
    steps = solve(parsed, CORE_RULES);
    if (error) setError(null);
  } catch (err: any) {
    steps = [];
    if (!error) setError(err.message);
  }

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', padding: '24px', background: '#1e293b', borderRadius: '12px' }}>
      <h1 style={{ fontSize: '1.6rem', marginBottom: '16px' }}>Interactive Math Step Solver</h1>

      <div style={{ marginBottom: '24px' }}>
        <label style={{ display: 'block', marginBottom: '8px', color: '#94a3b8' }}>Type an Expression:</label>
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
          <div key={index} style={{ background: '#0f172a', padding: '16px', borderRadius: '8px', borderLeft: index === 0 ? '4px solid #38bdf8' : '4px solid #22c55e' }}>
            <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '4px' }}>
              {index === 0 ? 'Initial Expression' : `Step ${index}: Applied "${step.ruleName}"`}
            </div>
            <MathBlock math={toLaTeX(step.expr)} />
          </div>
        ))}
      </div>
    </div>
  );
}