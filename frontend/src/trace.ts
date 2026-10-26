import type { Filters, Trace } from './types';

export const validationLabels: Record<string, string> = {
  empty_content: '空字符串', whitespace_content: '纯空白', invalid_json: '非法 JSON', invalid_decision: '决策字段不合规',
};
export function business(trace: Trace): string {
  if (trace.status === 'error') return '调用异常';
  if (trace.parsed_action === 'retry') return '重试';
  if (trace.parsed_action === 'fallback') return '最终兜底';
  if (trace.parsed_action === 'web_search') return '触发搜索';
  if (trace.parsed_action === 'reply') return '已生成回复';
  if (trace.status === 'running') return '运行中';
  return '未知';
}
export function statusColor(value: string): string {
  if (['error', '调用异常', '最终兜底'].includes(value)) return 'red';
  if (value === '重试') return 'orange';
  if (['running', '运行中', '触发搜索'].includes(value)) return 'blue';
  if (['success', '已生成回复'].includes(value)) return 'green';
  return 'default';
}
export function matches(trace: Trace, filters: Filters): boolean {
  return (!filters.component || trace.component === filters.component)
    && (!filters.apiStatus || trace.status === filters.apiStatus)
    && (!filters.business || business(trace) === filters.business)
    && (!filters.search.trim() || JSON.stringify(trace).toLowerCase().includes(filters.search.trim().toLowerCase()));
}
export interface Round {id: string; calls: Trace[]; matched: Trace[]; legacy: boolean}
export function groupRounds(traces: Trace[], filters: Filters): Round[] {
  const groups = new Map<string, Round>();
  for (const trace of traces) {
    const id = trace.round_id ? `round:${trace.round_id}` : `legacy:${trace.trace_id}`;
    const group = groups.get(id) ?? {id, calls: [], matched: [], legacy: !trace.round_id};
    group.calls.push(trace);
    if (matches(trace, filters)) group.matched.push(trace);
    groups.set(id, group);
  }
  return [...groups.values()].filter(group => group.matched.length).map(group => ({...group,
    calls: [...group.calls].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? '')),
  })).sort((a, b) => (b.calls.at(-1)?.created_at ?? '').localeCompare(a.calls.at(-1)?.created_at ?? ''));
}
export function aggregate(traces: Trace[], key: 'tokens' | 'duration'): string {
  const values = traces.map(trace => key === 'tokens' ? trace.usage?.total_tokens : trace.duration_ms);
  const known = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (!known.length) return '未知';
  const value = known.reduce((sum, n) => sum + n, 0);
  const formatted = key === 'tokens' ? value.toLocaleString() : `${(value / 1000).toFixed(2)} s`;
  return known.length === values.length ? formatted : `${formatted}（已知合计，部分缺失）`;
}
export function originalContent(trace: Trace): string | null | undefined {
  const message = trace.raw_response?.choices?.[0]?.message;
  return message && Object.hasOwn(message, 'content') ? message.content : trace.response_text;
}
export function describeContent(value: unknown): string | null {
  if (value === null) return 'content 为 null';
  if (value === undefined) return '未记录 content';
  if (value === '') return '空字符串（0 字符）';
  if (typeof value === 'string' && !value.trim()) {
    return `纯空白：${Array.from(value).length} 字符 · 空格 ${(value.match(/ /g) ?? []).length} · 换行 ${(value.match(/\n/g) ?? []).length} · 制表符 ${(value.match(/\t/g) ?? []).length}`;
  }
  return null;
}
export function chainWarning(round: Round): string | null {
  if (round.legacy) return '旧记录无轮次标识，作为独立调用展示。';
  const attempts = round.calls.filter(call => call.operation === 'decision' && call.attempt).map(call => call.attempt!);
  if (!attempts.length || !attempts.includes(1) || Math.max(...attempts) !== new Set(attempts).size)
    return '调用链可能不完整：部分记录未保留，或该轮仍在进行中。';
  return null;
}
export function filteredPayload(traces: Trace[], filters: Filters) {
  return {generated_at: new Date().toISOString(), filters, traces: traces.filter(trace => matches(trace, filters))};
}
