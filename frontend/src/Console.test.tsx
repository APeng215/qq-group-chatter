import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import Console from './Console';
import { App } from './App';
import type { Trace } from './types';

const calls: Trace[] = [
  {trace_id: 'retry-1', round_id: 'a', component: 'chat_agent', operation: 'decision', attempt: 1, created_at: '2026-10-26T01:00:00Z', status: 'success', parsed_action: 'retry', validation_error: 'whitespace_content', response_text: '   ', current_user_message: '测试问题'},
  {trace_id: 'success-2', round_id: 'a', component: 'chat_agent', operation: 'decision', attempt: 2, created_at: '2026-10-26T01:00:01Z', status: 'success', parsed_action: 'reply', final_reply: '有效回复', response_text: '{"action":"reply","content":"有效回复"}', current_user_message: '测试问题'},
  {trace_id: 'planner-3', round_id: 'a', component: 'memory_planner', operation: 'plan_memory', status: 'success'},
];
function mockFetch() {
  return vi.spyOn(global, 'fetch').mockImplementation(async path => ({ok: true, json: async () => String(path).includes('/api/memory') ? {memories: [], summary: {total: 0}} : {traces: calls}} as Response));
}
it('shows retry chain, preserves selected tab on refresh, copies complete round', async () => {
  mockFetch();
  const write = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: write}});
  render(<Console/>);
  await screen.findByText('有效回复');
  expect(screen.getByText('第1步', {exact: false})).toBeInTheDocument();
  fireEvent.click(screen.getByText('第1步', {exact: false}));
  expect(await screen.findByText(/纯空白：3 字符/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', {name: '原始 response'}));
  fireEvent.click(screen.getByRole('button', {name: '刷新'}));
  await waitFor(() => expect(screen.getByRole('tab', {name: '原始 response'})).toHaveAttribute('aria-selected', 'true'));
  fireEvent.click(screen.getByRole('button', {name: '复制本轮完整 trace'}));
  expect(write).toHaveBeenCalledWith(expect.stringContaining('planner-3'));
});
it('defaults to console and loads memories only on entering the library', async () => {
  const fetch = mockFetch(); render(<App/>);
  await screen.findByText('有效回复');
  expect(fetch.mock.calls.some(([path]) => String(path).includes('/api/memory'))).toBe(false);
  fireEvent.click(screen.getByRole('tab', {name: '记忆库'}));
  await waitFor(() => expect(fetch.mock.calls.some(([path]) => String(path).includes('/api/memory'))).toBe(true));
});
it('shows API failure without blanking the entire page', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ok: false, status: 503} as Response);
  render(<Console/>); expect(await screen.findByText(/HTTP 503/)).toBeInTheDocument();
});
