import { Expr, Const, Add, Mul, Pow, Div, Func, exprEquals } from './ast';
import { match, Bindings } from './matcher';
import { substitute } from './substituter';
import { parse } from './parser';

export interface Rule {
  name: string;
  pattern: Expr;
  target?: Expr;
  guard?: (env: Bindings) => boolean;
  transform?: (env: Bindings) => Expr | null;
}

export interface SolveStep {
  expr: Expr;
  ruleName?: string;
}

// Convenient helper: create rules using strings!
export function createRule(
  name: string,
  patternStr: string,
  targetStr: string,
  guard?: (env: Bindings) => boolean
): Rule {
  return {
    name,
    pattern: parse(patternStr),
    target: parse(targetStr),
    guard,
  };
}

export const CORE_RULES: Rule[] = [
  // Constant arithmetic evaluations (Automatic computation)
  {
    name: 'Add Numbers',
    pattern: parse('?a + ?b'),
    transform: (env) => {
      if (env.a.type === 'Const' && env.b.type === 'Const') {
        return Const(env.a.val + env.b.val);
      }
      return null;
    },
  },
  {
    name: 'Multiply Numbers',
    pattern: parse('?a * ?b'),
    transform: (env) => {
      if (env.a.type === 'Const' && env.b.type === 'Const') {
        return Const(env.a.val * env.b.val);
      }
      return null;
    },
  },

  // Arithmetic Identities
  createRule('Multiplication by Zero', '0 * ?a', '0'),
  createRule('Multiplication by One', '1 * ?a', '?a'),
  createRule('Addition with Zero', '0 + ?a', '?a'),
  createRule('Exponent of Zero', '?a ^ 0', '1'),
  createRule('Exponent of One', '?a ^ 1', '?a'),

  // Trigonometric Identities
  createRule('Pythagorean Identity', 'sin(?a)^2 + cos(?a)^2', '1'),
  createRule('Tangent Quotient', 'tan(?a)', 'sin(?a) / cos(?a)'),

  // Algebraic Expansions
  createRule('Square of Sum', '(?a + ?b)^2', '?a^2 + 2*?a*?b + ?b^2'),
];

function rewriteNode(expr: Expr, rules: Rule[]): { rewritten: Expr; ruleName?: string } {
  // Recurse bottom-up
  if (expr.type === 'Add' || expr.type === 'Mul') {
    const lRes = rewriteNode(expr.left, rules);
    if (lRes.ruleName) return { rewritten: expr.type === 'Add' ? Add(lRes.rewritten, expr.right) : Mul(lRes.rewritten, expr.right), ruleName: lRes.ruleName };
    const rRes = rewriteNode(expr.right, rules);
    if (rRes.ruleName) return { rewritten: expr.type === 'Add' ? Add(expr.left, rRes.rewritten) : Mul(expr.left, rRes.rewritten), ruleName: rRes.ruleName };
  } else if (expr.type === 'Pow') {
    const bRes = rewriteNode(expr.base, rules);
    if (bRes.ruleName) return { rewritten: Pow(bRes.rewritten, expr.exp), ruleName: bRes.ruleName };
    const eRes = rewriteNode(expr.exp, rules);
    if (eRes.ruleName) return { rewritten: Pow(expr.base, eRes.rewritten), ruleName: eRes.ruleName };
  } else if (expr.type === 'Div') {
    const nRes = rewriteNode(expr.num, rules);
    if (nRes.ruleName) return { rewritten: Div(nRes.rewritten, expr.den), ruleName: nRes.ruleName };
    const dRes = rewriteNode(expr.den, rules);
    if (dRes.ruleName) return { rewritten: Div(expr.num, dRes.rewritten), ruleName: dRes.ruleName };
  } else if (expr.type === 'Func') {
    const aRes = rewriteNode(expr.arg, rules);
    if (aRes.ruleName) return { rewritten: Func(expr.name, aRes.rewritten), ruleName: aRes.ruleName };
  }

  // Check rules at this node
  for (const rule of rules) {
    const env = match(rule.pattern, expr);
    if (env) {
      if (rule.guard && !rule.guard(env)) continue;

      let nextExpr: Expr | null = null;
      if (rule.transform) {
        nextExpr = rule.transform(env);
      } else if (rule.target) {
        nextExpr = substitute(rule.target, env);
      }

      if (nextExpr && !exprEquals(nextExpr, expr)) {
        return { rewritten: nextExpr, ruleName: rule.name };
      }
    }
  }

  return { rewritten: expr };
}

export function solve(initialExpr: Expr, rules: Rule[] = CORE_RULES, maxSteps = 20): SolveStep[] {
  const steps: SolveStep[] = [{ expr: initialExpr }];
  let current = initialExpr;

  for (let i = 0; i < maxSteps; i++) {
    const { rewritten, ruleName } = rewriteNode(current, rules);
    if (!ruleName) break;
    steps.push({ expr: rewritten, ruleName });
    current = rewritten;
  }

  return steps;
}