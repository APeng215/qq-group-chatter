import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  use: {baseURL: 'http://127.0.0.1:4173', screenshot: 'only-on-failure', trace: 'retain-on-failure'},
  webServer: {command: 'npm run dev -- --port 4173', url: 'http://127.0.0.1:4173/console-assets/', reuseExistingServer: false},
  projects: [
    {name: 'desktop', use: {...devices['Desktop Chrome'], viewport: {width: 1440, height: 1000}}},
    {name: 'mobile', use: {...devices['Desktop Chrome'], viewport: {width: 390, height: 844}}},
  ],
});
