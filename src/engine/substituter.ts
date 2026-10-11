import { Expr, Add, Mul, Pow, Div, Func } from './ast';
import { Bindings } from './matcher';

export function substitute(template: Expr, env: Bindings): Expr {
  switch (template.type) {
    case 'Wildcard':
      if (template.name in env) return env[template.name];
      throw new Error(`Unbound wildcard: ${template.name}`);
    case 'Const':
    case 'Var':
      return template;
    case 'Add':
      return Add(substitute(template.left, env), substitute(template.right, env));
    case 'Mul':
      return Mul(substitute(template.left, env), substitute(template.right, env));
    case 'Pow':
      return Pow(substitute(template.base, env), substitute(template.exp, env));
    case 'Div':
      return Div(substitute(template.num, env), substitute(template.den, env));
    case 'Func':
      return Func(template.name, substitute(template.arg, env));
  }
}