import { Expr, Const, Var, Wildcard, Add, Mul, Div, Pow, Func } from './ast';

type TokenType = 'NUMBER' | 'IDENT' | 'WILDCARD' | 'PLUS' | 'MINUS' | 'MUL' | 'DIV' | 'POW' | 'LPAREN' | 'RPAREN' | 'EOF';

interface Token {
  type: TokenType;
  value: string;
}

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < input.length) {
    const ch = input[i];

    if (/\s/.test(ch)) {
      i++;
      continue;
    }

    if (/\d/.test(ch)) {
      let numStr = '';
      while (i < input.length && /[\d.]/.test(input[i])) {
        numStr += input[i++];
      }
      tokens.push({ type: 'NUMBER', value: numStr });
      continue;
    }

    if (ch === '?') {
      let name = '';
      i++;
      while (i < input.length && /[a-zA-Z0-9_]/.test(input[i])) {
        name += input[i++];
      }
      tokens.push({ type: 'WILDCARD', value: name });
      continue;
    }

    if (/[a-zA-Z_]/.test(ch)) {
      let ident = '';
      while (i < input.length && /[a-zA-Z0-9_]/.test(input[i])) {
        ident += input[i++];
      }
      tokens.push({ type: 'IDENT', value: ident });
      continue;
    }

    switch (ch) {
      case '+': tokens.push({ type: 'PLUS', value: '+' }); break;
      case '-': tokens.push({ type: 'MINUS', value: '-' }); break;
      case '*': tokens.push({ type: 'MUL', value: '*' }); break;
      case '/': tokens.push({ type: 'DIV', value: '/' }); break;
      case '^': tokens.push({ type: 'POW', value: '^' }); break;
      case '(': tokens.push({ type: 'LPAREN', value: '(' }); break;
      case ')': tokens.push({ type: 'RPAREN', value: ')' }); break;
      default:
        throw new Error(`Unexpected character: ${ch}`);
    }
    i++;
  }

  tokens.push({ type: 'EOF', value: '' });
  return tokens;
}

export class Parser {
  private tokens: Token[];
  private pos = 0;

  constructor(input: string) {
    this.tokens = tokenize(input);
  }

  private peek(): Token {
    return this.tokens[this.pos];
  }

  private consume(expected?: TokenType): Token {
    const t = this.peek();
    if (expected && t.type !== expected) {
      throw new Error(`Expected ${expected}, but got ${t.type}`);
    }
    this.pos++;
    return t;
  }

  public parse(): Expr {
    const expr = this.parseExpression(0);
    if (this.peek().type !== 'EOF') {
      throw new Error(`Unexpected token remaining: ${this.peek().value}`);
    }
    return expr;
  }

  private getPrecedence(type: TokenType): number {
    switch (type) {
      case 'PLUS':
      case 'MINUS': return 1;
      case 'MUL':
      case 'DIV': return 2;
      case 'POW': return 3;
      default: return 0;
    }
  }

  private parsePrimary(): Expr {
    const token = this.peek();

    if (token.type === 'NUMBER') {
      this.consume();
      return Const(parseFloat(token.value));
    }

    if (token.type === 'WILDCARD') {
      this.consume();
      return Wildcard(token.value);
    }

    if (token.type === 'IDENT') {
      this.consume();
      // Check if it's a function call like sin(...)
      if (this.peek().type === 'LPAREN') {
        this.consume('LPAREN');
        const arg = this.parseExpression(0);
        this.consume('RPAREN');
        return Func(token.value, arg);
      }
      return Var(token.value);
    }

    if (token.type === 'MINUS') {
      this.consume();
      return Mul(Const(-1), this.parsePrimary());
    }

    if (token.type === 'LPAREN') {
      this.consume('LPAREN');
      const expr = this.parseExpression(0);
      this.consume('RPAREN');
      return expr;
    }

    throw new Error(`Unexpected token in parsePrimary: ${token.value || token.type}`);
  }

  private parseExpression(minPrecedence: number): Expr {
    let left = this.parsePrimary();

    while (true) {
      const token = this.peek();
      const prec = this.getPrecedence(token.type);
      if (prec <= minPrecedence) break;

      this.consume();

      // Right-associative exponentiation: 2^3^4 = 2^(3^4)
      const nextMinPrec = token.type === 'POW' ? prec - 1 : prec;
      const right = this.parseExpression(nextMinPrec);

      switch (token.type) {
        case 'PLUS': left = Add(left, right); break;
        case 'MINUS': left = Add(left, Mul(Const(-1), right)); break;
        case 'MUL': left = Mul(left, right); break;
        case 'DIV': left = Div(left, right); break;
        case 'POW': left = Pow(left, right); break;
      }
    }

    return left;
  }
}

export function parse(input: string): Expr {
  return new Parser(input).parse();
}