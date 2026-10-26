import { describe, expect, it } from 'vitest';
import { aggregate, business, chainWarning, describeContent, filteredPayload, groupRounds, originalContent } from './trace';
import type { Filters, Trace } from './types';

const filters: Filters = {search: '', component: 'chat_agent', apiStatus: '', business: ''};
const trace = (id: string, extra: Partial<Trace> = {}): Trace => ({trace_id: id, round_id: 'a', component: 'chat_agent', operation: 'decision', attempt: 1, status: 'success', current_user_message: '相同提问', ...extra});
describe('rounds and filtering', () => {
  it('groups explicit IDs only; identical old or new questions never merge', () => {
    const groups = groupRounds([trace('1'), trace('2', {round_id: 'b'}), trace('3', {round_id: undefined}), trace('4', {round_id: undefined})], filters);
    expect(groups).toHaveLength(4);
  });
  it('includes full chain but exports and aggregates matched calls only', () => {
    const traces = [trace('1', {attempt: 1, parsed_action: 'retry', usage: {total_tokens: 10}}), trace('2', {attempt: 2, parsed_action: 'reply', usage: {total_tokens: 20}}), trace('3', {component: 'memory_planner', usage: {total_tokens: 30}})];
    const group = groupRounds(traces, filters)[0];
    expect(group.calls).toHaveLength(3); expect(group.matched).toHaveLength(2);
    expect(filteredPayload(traces, filters).traces).toHaveLength(2);
    expect(aggregate(group.matched, 'tokens')).toBe('30');
    expect(business(traces[0])).toBe('重试'); expect(business(traces[1])).toBe('已生成回复');
  });
  it('detects missing attempts and old records', () => {
    expect(chainWarning(groupRounds([trace('1', {attempt: 2})], filters)[0])).toContain('不完整');
    expect(chainWarning(groupRounds([trace('1', {round_id: undefined})], filters)[0])).toContain('旧记录');
    expect(chainWarning(groupRounds([trace('1')], filters)[0])).toBeNull();
  });
});
describe('raw content and usage', () => {
  it('preserves null versus empty and whitespace', () => {
    expect(originalContent(trace('1', {raw_response: {choices: [{message: {content: null}}]}, response_text: ''}))).toBeNull();
    expect(describeContent(null)).toContain('null'); expect(describeContent('')).toContain('0 字符');
    expect(describeContent('  \n\t')).toBe('纯空白：4 字符 · 空格 2 · 换行 1 · 制表符 1');
    expect(describeContent('正常')).toBeNull();
  });
  it('never silently turns missing usage into zero', () => {
    expect(aggregate([trace('1')], 'tokens')).toBe('未知');
    expect(aggregate([trace('1'), trace('2', {usage: {total_tokens: 12}})], 'tokens')).toContain('部分缺失');
    expect(business(trace('1'))).toBe('未知');
  });
});
