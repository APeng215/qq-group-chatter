import { useEffect, useState } from 'react';
import { Alert, Button, Card, Drawer, Input, Select, Space, Table, Tag } from 'antd';
import { getJson, formatTime } from './Console';
import { JsonPanel } from './components';
import type { Memory, MemorySnapshot } from './types';

export default function MemoryLibrary() {
  const [snapshot, setSnapshot] = useState<MemorySnapshot>({memories: [], summary: {}});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [scope, setScope] = useState('');
  const [kind, setKind] = useState('');
  const [selected, setSelected] = useState<Memory>();
  async function refresh() {
    setLoading(true); setError('');
    try {setSnapshot(await getJson<MemorySnapshot>('/api/memory'));}
    catch (e) {setError(String(e));} finally {setLoading(false);}
  }
  useEffect(() => {void refresh();}, []);
  const memories = snapshot.memories.filter(m => (!scope || m.scope === scope) && (!kind || m.kind === kind)
    && (!search.trim() || JSON.stringify(m).toLowerCase().includes(search.trim().toLowerCase())));
  return <div>
    <div className="stats">{[['total', '记忆总量'], ['user', '用户记忆'], ['conversation', '会话记忆'], ['queue_size', '记忆队列']].map(([key, label]) =>
      <Card key={key} size="small"><span>{label}</span><strong>{snapshot.summary[key] ?? '未知'}</strong></Card>)}</div>
    <div className="toolbar"><Input.Search aria-label="搜索记忆" placeholder="搜索正文、ID、metadata" value={search} onChange={e => setSearch(e.target.value)}/>
      <Select aria-label="记忆范围" value={scope} onChange={setScope} options={[{value: '', label: '全部 scope'}, ...['user', 'conversation', 'other'].map(value => ({value, label: value}))]}/>
      <Select aria-label="记忆类别" value={kind} onChange={setKind} options={[{value: '', label: '全部类别'}, ...[...new Set(snapshot.memories.map(m => m.kind))].filter(Boolean).map(value => ({value, label: value}))]}/>
      <Button loading={loading} onClick={() => void refresh()}>刷新记忆</Button></div>
    {error && <Alert type="error" message={error} className="notice"/>}
    {(snapshot.errors ?? []).map((e, i) => <Alert type="warning" message={e} key={i} className="notice"/>)}
    <Card><Table<Memory> rowKey="id" loading={loading} dataSource={memories} pagination={{pageSize: 20, showSizeChanger: false}} scroll={{x: 700}}
      columns={[
        {title: '记忆正文', dataIndex: 'content', render: (text: string, m) => <Button type="link" className="memory-content" onClick={() => setSelected(m)}>{text}</Button>},
        {title: '范围', dataIndex: 'scope', width: 120, render: (value: string) => <Tag>{value}</Tag>},
        {title: '类别', dataIndex: 'kind', width: 150},
        {title: '更新时间', width: 180, render: (_, m) => formatTime(m.updated_at ?? m.created_at)},
      ]}/></Card>
    <Drawer title="记忆详情" open={Boolean(selected)} onClose={() => setSelected(undefined)} width="min(680px, 95vw)">
      {selected && <Space direction="vertical" className="full-width" size="large"><pre className="prose">{selected.content}</pre>
        <JsonPanel label="基本信息与 metadata" value={{...selected, history: undefined}}/>
        <JsonPanel label="变更历史" value={selected.history ?? []}/></Space>}
    </Drawer>
  </div>;
}
