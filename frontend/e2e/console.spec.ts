import { test, expect } from '@playwright/test';
import type { Trace } from '../src/types';

const resource = 'a'.repeat(64);
const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';
const traces: Trace[] = [
  {trace_id: 'failed-call', round_id: 'round-a', component: 'chat_agent', operation: 'decision', attempt: 1, status: 'success', parsed_action: 'retry', validation_error: 'whitespace_content', response_text: '   \n\t', created_at: '2026-10-26T01:00:00Z', current_user_message: '请看这张图'},
  {trace_id: 'good-call', round_id: 'round-a', component: 'chat_agent', operation: 'decision', attempt: 2, status: 'success', parsed_action: 'reply', final_reply: '这是测试图片。', response_text: '{"action":"reply","content":"这是测试图片。"}', created_at: '2026-10-26T01:00:01Z', current_user_message: '请看这张图', usage: {total_tokens: 20}, duration_ms: 80,
    reasoning_content: '隔离测试思考内容', raw_response: {choices: [{message: {content: '{"action":"reply","content":"这是测试图片。"}'}}], long_field: '超长内容'.repeat(1000)},
    messages: [{role: 'system', content: '<script>alert("unsafe")</script>'}, {role: 'user', content: [{type: 'text', text: '图片上下文\n' + '长正文'.repeat(500)}, {type: 'text', text: '图1'}, {type: 'image_ref', resource_id: resource, input_index: 1}, {type: 'image_ref', resource_id: 'b'.repeat(64), input_index: 2}]}],
    image_context: {enabled: true, max_images: 4, sent_count: 1, attachments: [
      {status: 'sent', input_index: 1, resource_id: resource, message_id: 'm1', position: 1},
      {status: 'sent_duplicate', input_index: 1, resource_id: resource, message_id: 'm0', position: 1},
      {status: 'unavailable', resource_id: 'b'.repeat(64), message_id: 'm2', position: 2}]}},
  {trace_id: 'planner-call', round_id: 'round-a', component: 'memory_planner', operation: 'plan_memory', status: 'success', created_at: '2026-10-26T01:00:02Z'},
];
test.beforeEach(async ({page}) => {
  await page.route('**/api/llm-traces', route => route.fulfill({json: {traces, image_understanding: {enabled: true, max_images_per_request: 4}}}));
  await page.route('**/api/memory', route => route.fulfill({json: {summary: {total: 1, user: 1, queue_size: 0}, memories: [{id: 'test-memory', content: '喜欢咖啡', scope: 'user', kind: 'preference', metadata: {source: 'fake'}, history: [{event: 'ADD'}]}]}}));
  await page.route('**/api/images?*', route => route.request().url().includes(resource) ? route.fulfill({contentType: 'image/png', body: Buffer.from(png, 'base64')}) : route.fulfill({status: 404, body: '图片已不可用'}));
});
test('console round, retry, raw content, preview, export, refresh and library', async ({page}, info) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  const requests: string[] = [];
  page.on('request', r => requests.push(r.url()));
  await page.goto('/console-assets/');
  await expect(page.getByRole('tab', {name: 'LLM 控制台', exact: true})).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', {name: /请看这张图/})).toBeVisible();
  expect(requests.some(url => url.includes('/api/memory'))).toBe(false);
  await page.getByRole('button', {name: /请看这张图/}).click();
  await expect(page.getByText('这是测试图片。', {exact: true})).toBeVisible();
  await page.screenshot({path: info.outputPath('console.png'), fullPage: true, animations: 'disabled'});
  await page.getByRole('button', {name: /第1\/3次/}).click();
  await expect(page.getByText(/纯空白：5 字符/)).toBeVisible();
  await page.getByText('查看转义文本').click();
  await expect(page.locator('.response-content pre')).toContainText('\\n\\t');
  await page.getByRole('button', {name: /第2\/3次/}).click();
  await page.locator('figure .ant-image-mask').first().click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await page.getByRole('tab', {name: '请求上下文', exact: true}).click();
  await expect(page.getByRole('tab', {name: '请求上下文', exact: true})).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('图片已不可用', {exact: true})).toBeVisible();
  await expect(page.getByText('<script>alert("unsafe")</script>', {exact: true})).toBeVisible();
  await page.getByRole('tab', {name: '原始 response', exact: true}).click();
  await page.getByRole('button', {name: '刷新', exact: true}).click();
  await expect(page.getByRole('tab', {name: '原始 response', exact: true})).toHaveAttribute('aria-selected', 'true');
  await page.screenshot({path: info.outputPath('raw-response.png'), fullPage: true, animations: 'disabled'});
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', {name: '导出当前筛选'}).click();
  const download = await downloadEvent;
  const stream = await download.createReadStream();
  const chunks = []; for await (const chunk of stream!) chunks.push(chunk);
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  expect(exported.traces).toHaveLength(2);
  expect(JSON.stringify(exported)).not.toContain('data:image');
  await page.getByRole('tab', {name: '记忆库', exact: true}).click();
  await page.getByRole('button', {name: '喜欢咖啡', exact: true}).click();
  await expect(page.getByText('记忆详情', {exact: true})).toBeVisible();
  await page.screenshot({path: info.outputPath('memory.png'), fullPage: true, animations: 'disabled'});
  expect(errors).toEqual([]);
});
test('empty, request failure, narrow navigation and keyboard focus', async ({page}, info) => {
  await page.route('**/api/llm-traces', route => route.fulfill({json: {traces: []}}));
  await page.goto('/console-assets/');
  await expect(page.getByText('没有匹配的 LLM trace')).toBeVisible();
  await page.route('**/api/llm-traces', route => route.fulfill({status: 503}));
  await page.getByRole('button', {name: '刷新', exact: true}).click();
  await expect(page.getByText(/HTTP 503/)).toBeVisible();
  await page.keyboard.press('Tab');
  await page.screenshot({path: info.outputPath('empty-error.png'), fullPage: true, animations: 'disabled'});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
