import { useEffect, useMemo, useState } from 'react';
import { Alert, Button, Card, Dropdown, Empty, Input, Modal, Select, Space, Spin, Tabs, Tag, Timeline, Typography } from 'antd';
import type { Filters, Trace, TraceSnapshot } from './types';
import { aggregate, business, chainWarning, filteredPayload, groupRounds, statusColor, validationLabels } from './trace';
import { copy, JsonPanel, Messages, QuestionAnswer, Status } from './components';

export async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {cache: 'no-store', ...init});
  if (!response.ok) throw new Error(`请求失败：HTTP ${response.status}`);
  return response.json() as Promise<T>;
}
export function downloadJson(value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = `llm-traces-${Date.now()}.json`;
  link.click(); URL.revokeObjectURL(url);
}
const defaults: Filters = {search: '', component: 'chat_agent', apiStatus: '', business: ''};
export default function Console() {
  const [snapshot, setSnapshot] = useState<TraceSnapshot>({traces: []});
  const [filters, setFilters] = useState<Filters>(defaults);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState('');
  const [selectedCall, setSelectedCall] = useState('');
  const [tab, setTab] = useState('qa');
  const [mobileDetail, setMobileDetail] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const rounds = useMemo(() => groupRounds(snapshot.traces, filters), [snapshot.traces, filters]);
  const round = rounds.find(item => item.id === selected) ?? rounds[0];
  const matched = useMemo(() => filteredPayload(snapshot.traces, filters).traces, [snapshot.traces, filters]);
  const calls = round ? (showAll ? round.calls : round.calls.filter(call => round.matched.some(m => m.trace_id === call.trace_id))) : [];
  const call = calls.find(item => item.trace_id === selectedCall) ?? [...calls].reverse().find(item => item.component === 'chat_agent') ?? calls.at(-1);
  async function refresh() {
    setLoading(true); setError('');
    try { setSnapshot(await getJson<TraceSnapshot>('/api/llm-traces')); }
    catch (e) {setError(String(e));} finally {setLoading(false);}
  }
  useEffect(() => {void refresh();}, []);
  useEffect(() => {if (round) setSelected(round.id);}, [round?.id]);
  const change = (key: keyof Filters, value: string) => setFilters(old => ({...old, [key]: value}));
  const components = [...new Set(['chat_agent', ...snapshot.traces.map(t => t.component).filter((v): v is string => Boolean(v))])];
  const options = (values: string[], all: string) => [{value: '', label: all}, ...values.map(value => ({value, label: value}))];
  const warning = round ? chainWarning(round) : null;
  return <div>
    <div className="toolbar">
      <Input.Search aria-label="搜索 trace" placeholder="搜索 prompt / response / error" allowClear value={filters.search} onChange={e => change('search', e.target.value)}/>
      <Select aria-label="组件筛选" value={filters.component} options={options(components, '全部组件')} onChange={v => change('component', v)}/>
      <Select aria-label="API状态筛选" value={filters.apiStatus} options={options(['running', 'success', 'error'], '全部 API 状态')} onChange={v => change('apiStatus', v)}/>
      <Select aria-label="业务状态筛选" value={filters.business} options={options(['重试', '已生成回复', '触发搜索', '最终兜底', '调用异常', '运行中', '未知'], '全部业务状态')} onChange={v => change('business', v)}/>
      <Button aria-label="刷新" onClick={() => void refresh()} loading={loading}>刷新</Button>
      <Button onClick={() => downloadJson(filteredPayload(snapshot.traces, filters))}>导出当前筛选</Button>
      <Dropdown menu={{items: [{key: 'clear', label: '清空所有 trace', danger: true}], onClick: () => Modal.confirm({title: '确定清空当前所有 LLM trace？', content: '建议先导出当前筛选留存排障证据。', okButtonProps: {danger: true}, onOk: async () => {try {await getJson('/api/llm-traces/clear', {method: 'POST'}); await refresh();} catch (e) {setError(String(e)); throw e;}}})}}><Button>更多操作</Button></Dropdown>
    </div>
    {error && <Alert type="error" showIcon message={error} className="notice"/>}
    {(snapshot.errors ?? []).map((e, i) => <Alert key={i} type="warning" message={e} className="notice"/>)}
    <div className="stats">
      <Card size="small"><span>匹配调用 / 轮次</span><strong>{matched.length} / {rounds.length}</strong></Card>
      <Card size="small"><span>Token 合计</span><strong>{aggregate(matched, 'tokens')}</strong></Card>
      <Card size="small"><span>调用耗时合计</span><strong>{aggregate(matched, 'duration')}</strong></Card>
      <Card size="small"><span>图片理解</span><strong>{snapshot.image_understanding?.enabled ? '开启' : '关闭'} · 最多 {snapshot.image_understanding?.max_images_per_request ?? 4} 张</strong></Card>
    </div>
    <Typography.Paragraph type="secondary">当前显示 {matched.length} / {snapshot.traces.length} 条调用；统计及导出仅包含匹配调用。</Typography.Paragraph>
    <Spin spinning={loading}><div className={`console-grid ${mobileDetail ? 'show-detail' : ''}`}>
      <aside className="round-list" aria-label="轮次列表">{rounds.length ? rounds.map(item => {
        const last = [...item.calls].reverse().find(t => t.component === 'chat_agent') ?? item.calls.at(-1)!;
        return <button key={item.id} className={`round-row ${round?.id === item.id ? 'selected' : ''}`} onClick={() => {setSelected(item.id); setSelectedCall(''); setMobileDetail(true);}}>
          <div className="section-heading"><time>{formatTime(last.created_at)}</time><Tag color={statusColor(business(last))}>{business(last)}</Tag></div>
          <div className="question-summary">{item.calls.find(t => t.current_user_message)?.current_user_message ?? '未记录用户提问'}</div>
          <div className="list-meta">{item.matched.length} 条匹配 / {item.calls.length} 次调用 · {aggregate(item.matched, 'tokens')} tokens</div>
          <div className="list-meta">调用耗时合计 {aggregate(item.matched, 'duration')}</div>
        </button>;
      }) : <Empty description="没有匹配的 LLM trace"/>}</aside>
      <Card className="detail-pane" styles={{body: {padding: 20}}}>{round && call ? <>
        <Button className="mobile-back" onClick={() => setMobileDetail(false)}>返回列表</Button>
        <div className="section-heading"><Typography.Title level={4}>调用详情</Typography.Title><Button onClick={() => copy({round_id: round.id, traces: round.calls})}>复制本轮完整 trace</Button></div>
        <Typography.Paragraph type="secondary" ellipsis={{tooltip: round.id}}>轮次 {round.id}</Typography.Paragraph>
        <Space wrap className="notice"><Tag>{round.calls.length} 次调用</Tag><span>Token {aggregate(round.calls, 'tokens')}</span><span>调用耗时合计 {aggregate(round.calls, 'duration')}</span></Space>
        {warning && <Alert message={warning} type="info" className="notice"/>}
        {round.calls.length !== round.matched.length && <Button size="small" className="notice" onClick={() => setShowAll(!showAll)}>{showAll ? '仅看匹配调用' : `展开完整链路（${round.calls.length - round.matched.length} 条未匹配）`}</Button>}
        <Timeline className="call-timeline" items={calls.map(t => ({color: statusColor(business(t)) === 'default' ? 'gray' : statusColor(business(t)), children:
          <button className={`timeline-call ${call.trace_id === t.trace_id ? 'selected-call' : ''}`} onClick={() => setSelectedCall(t.trace_id)}>
            <span>{t.component} / {t.operation} {t.attempt ? `第${t.attempt}/3次` : ''}</span><Status trace={t}/>
            {t.validation_error && <small>{validationLabels[t.validation_error] ?? t.validation_error}</small>}
            {!round.matched.some(m => m.trace_id === t.trace_id) && <Tag>未匹配筛选</Tag>}
          </button>}))}/>
        <Space wrap className="notice"><Status trace={call}/><Typography.Text copyable>{call.trace_id}</Typography.Text></Space>
        <Tabs activeKey={tab} onChange={setTab} items={[
          {key: 'qa', label: '问答', children: <QuestionAnswer trace={call}/>},
          {key: 'reasoning', label: '思考', children: <pre className="prose">{call.reasoning_content ?? '未记录思考内容'}</pre>},
          {key: 'context', label: '请求上下文', children: <Messages trace={call}/>},
          {key: 'raw', label: '原始 response', children: <JsonPanel label="完整 API response（SDK 对象）" value={call.raw_response ?? {response_text: call.response_text}}/>},
          {key: 'usage', label: '用量与参数', children: <JsonPanel value={{usage: call.usage, model: call.model, thinking: call.thinking, response_format: call.response_format, temperature: call.temperature, duration_ms: call.duration_ms}}/>},
        ]}/>
      </> : <Empty description="选择一轮调用查看详情"/>}</Card>
    </div></Spin>
  </div>;
}
export function formatTime(value?: string) {
  if (!value) return '时间未知';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('zh-CN', {hour12: false});
}
