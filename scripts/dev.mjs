/**
 * scripts/dev.mjs  --  so you only need ONE command:  npm run dev
 *
 * It starts both servers at the same time:
 *     [api]  the Python maths server  (port 8000)
 *     [web]  the React dev server     (port 5173)
 *
 * Their output is mixed into one terminal, each line labelled so you can
 * tell them apart.  Press Ctrl+C once and BOTH servers stop.
 */

import { spawn } from 'node:child_process';
import readline from 'node:readline';

const isWindows = process.platform === 'win32';
const npm = isWindows ? 'npm.cmd' : 'npm';

const SERVERS = [
  { name: 'api', args: ['run', 'api'] },
  { name: 'web', args: ['run', 'dev:web'] },
];

const children = [];
let stopping = false;

/** Stop everything (used for Ctrl+C and when one server dies). */
function stopAll() {
  if (stopping) return;
  stopping = true;

  for (const child of children) {
    if (child.exitCode !== null || child.killed) continue;

    if (isWindows) {
      // /T = take the whole tree with it (uvicorn --reload has children too)
      spawn('taskkill', ['/pid', String(child.pid), '/T', '/F']);
    } else {
      child.kill('SIGTERM');
    }
  }
}

for (const server of SERVERS) {
  const child = spawn(npm, server.args, {
    stdio: ['ignore', 'pipe', 'pipe'], // we print the output ourselves
    shell: isWindows, // Windows needs a shell to run npm.cmd
  });

  children.push(child);

  // Label every line, e.g.   [api] INFO:     Uvicorn running on port 8000
  for (const stream of [child.stdout, child.stderr]) {
    readline.createInterface({ input: stream }).on('line', (line) => {
      console.log(`[${server.name}] ${line}`);
    });
  }

  // If one server dies, stop the other one too -- a half-running app is
  // confusing, and the error message would be buried anyway.
  child.on('exit', (code) => {
    console.log(`[${server.name}] stopped (code ${code})`);
    stopAll();
    process.exitCode = code ?? 0;
  });
}

console.log('Starting both servers... press Ctrl+C to stop them both.');

process.on('SIGINT', () => {
  console.log('\nStopping both servers...');
  stopAll();
  setTimeout(() => process.exit(0), 500);
});
