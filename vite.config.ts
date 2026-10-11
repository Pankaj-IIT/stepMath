import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // In development, send every /api request to the Python server,
    // so the browser does not complain about cross-origin calls.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
});