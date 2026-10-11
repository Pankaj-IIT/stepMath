import { Expr, exprEquals } from './ast';

export type Bindings = Record<string, Expr>;

export function match(pattern: Expr, expr: Expr, env: Bindings = {}): Bindings | null {
  const currentEnv = { ...env };

  // 1. Wildcard matching
  if (pattern.type === 'Wildcard') {
    if (pattern.name in currentEnv) {
      return exprEquals(currentEnv[pattern.name], expr) ? currentEnv : null;
    }
    currentEnv[pattern.name] = expr;
    return currentEnv;
  }

  // 2. Constants & Variables
  if (pattern.type === 'Const' && expr.type === 'Const') {
    return pattern.val === expr.val ? currentEnv : null;
  }
  if (pattern.type === 'Var' && expr.type === 'Var') {
    return pattern.name === expr.name ? currentEnv : null;
  }

  if (pattern.type !== expr.type) return null;

  // 3. Commutative matching for Add & Mul
  if (pattern.type === 'Add' || pattern.type === 'Mul') {
    const e = expr as typeof pattern;
    // Direct match: pattern.left -> e.left, pattern.right -> e.right
    const directLeft = match(pattern.left, e.left, currentEnv);
    if (directLeft) {
      const directRight = match(pattern.right, e.right, directLeft);
      if (directRight) return directRight;
    }

    // Commuted match: pattern.left -> e.right, pattern.right -> e.left
    const flippedLeft = match(pattern.left, e.right, currentEnv);
    if (flippedLeft) {
      const flippedRight = match(pattern.right, e.left, flippedLeft);
      if (flippedRight) return flippedRight;
    }

    return null;
  }

  // 4. Non-commutative operators
  if (pattern.type === 'Pow') {
    const e = expr as typeof pattern;
    const baseMatch = match(pattern.base, e.base, currentEnv);
    if (!baseMatch) return null;
    return match(pattern.exp, e.exp, baseMatch);
  }

  if (pattern.type === 'Div') {
    const e = expr as typeof pattern;
    const numMatch = match(pattern.num, e.num, currentEnv);
    if (!numMatch) return null;
    return match(pattern.den, e.den, numMatch);
  }

  if (pattern.type === 'Func') {
    const e = expr as typeof pattern;
    if (pattern.name !== e.name) return null;
    return match(pattern.arg, e.arg, currentEnv);
  }

  return null;
}