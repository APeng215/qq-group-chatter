import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/console-assets/',
  build: { outDir: '../qq_group_chatter/static/console', emptyOutDir: true, chunkSizeWarningLimit: 1100 },
  server: { proxy: { '/api': 'http://127.0.0.1:8080' } },
  test: { environment: 'jsdom', setupFiles: ['./src/test-setup.ts'], include: ['src/**/*.test.{ts,tsx}'] },
});
