export type Expr =
  | { type: 'Const'; val: number }
  | { type: 'Var'; name: string }
  | { type: 'Add'; left: Expr; right: Expr }
  | { type: 'Mul'; left: Expr; right: Expr }
  | { type: 'Pow'; base: Expr; exp: Expr }
  | { type: 'Div'; num: Expr; den: Expr }
  | { type: 'Func'; name: string; arg: Expr }
  | { type: 'Wildcard'; name: string };

export const Const = (val: number): Expr => ({ type: 'Const', val });
export const Var = (name: string): Expr => ({ type: 'Var', name });
export const Add = (left: Expr, right: Expr): Expr => ({ type: 'Add', left, right });
export const Mul = (left: Expr, right: Expr): Expr => ({ type: 'Mul', left, right });
export const Pow = (base: Expr, exp: Expr): Expr => ({ type: 'Pow', base, exp });
export const Div = (num: Expr, den: Expr): Expr => ({ type: 'Div', num, den });
export const Func = (name: string, arg: Expr): Expr => ({ type: 'Func', name, arg });
export const Wildcard = (name: string): Expr => ({ type: 'Wildcard', name });

export function exprEquals(a: Expr, b: Expr): boolean {
  if (a.type !== b.type) return false;
  switch (a.type) {
    case 'Const': return a.val === (b as typeof a).val;
    case 'Var': return a.name === (b as typeof a).name;
    case 'Wildcard': return a.name === (b as typeof a).name;
    case 'Add':
    case 'Mul':
      return exprEquals(a.left, (b as typeof a).left) && exprEquals(a.right, (b as typeof a).right);
    case 'Pow':
      return exprEquals(a.base, (b as typeof a).base) && exprEquals(a.exp, (b as typeof a).exp);
    case 'Div':
      return exprEquals(a.num, (b as typeof a).num) && exprEquals(a.den, (b as typeof a).den);
    case 'Func':
      return a.name === (b as typeof a).name && exprEquals(a.arg, (b as typeof a).arg);
  }
}

export function toLaTeX(expr: Expr): string {
  switch (expr.type) {
    case 'Const': return `${expr.val}`;
    case 'Var': return expr.name;
    case 'Wildcard': return `?${expr.name}`;
    case 'Add': return `${toLaTeX(expr.left)} + ${toLaTeX(expr.right)}`;
    case 'Mul': {
      const leftStr = expr.left.type === 'Add' ? `(${toLaTeX(expr.left)})` : toLaTeX(expr.left);
      const rightStr = expr.right.type === 'Add' ? `(${toLaTeX(expr.right)})` : toLaTeX(expr.right);
      return `${leftStr} \\cdot ${rightStr}`;
    }
    case 'Pow': return `{(${toLaTeX(expr.base)})}^{${toLaTeX(expr.exp)}}`;
    case 'Div': return `\\frac{${toLaTeX(expr.num)}}{${toLaTeX(expr.den)}}`;
    case 'Func': return `\\${expr.name}\\left(${toLaTeX(expr.arg)}\\right)`;
  }
}