import { useEffect, useState } from 'react';
import { Alert, Button, Card, Empty, Select, Space, Spin, Switch, Tabs, Tag, Typography } from 'antd';
import { copy } from './components';

type Group = {key: string; title: string; system: string; user: string};
type Snapshot = {groups: Group[]; source: Record<string, string>; errors?: string[]};

export default function Prompts({onOpenCalls}: {onOpenCalls: () => void}) {
  const [data, setData] = useState<Snapshot>({groups: [], source: {}});
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [view, setView] = useState('runtime');
  const [selected, setSelected] = useState('chat');
  const [source, setSource] = useState('');
  async function refresh() {
    setLoading(true);
    try {
      const response = await fetch('/api/prompts', {cache: 'no-store'});
      if (!response.ok) throw Error(`HTTP ${response.status}`);
      setData(await response.json()); setError('');
    } catch (e) {setError(String(e));} finally {setLoading(false);}
  }
  useEffect(() => {void refresh();}, []);
  useEffect(() => {if (!auto) return; const timer = setInterval(() => {void refresh();}, 5000); return () => clearInterval(timer);}, [auto]);
  const group = data.groups.find(g => g.key === selected) ?? data.groups[0];
  const name = source in data.source ? source : Object.keys(data.source)[0];
  function textCard(title: string, text: string, role?: string) {
    return <Card className="notice" title={<Space>{role && <Tag color="blue">{role}</Tag>}{title}</Space>}
      extra={<Button size="small" onClick={() => copy(text, true)}>复制正文</Button>}>
      <pre className="prose prompt-text">{text}</pre></Card>;
  }
  return <div>
    <div className="toolbar"><Button loading={loading} onClick={() => {void refresh();}}>刷新提示词</Button>
      <Space><Switch checked={auto} onChange={setAuto} aria-label="自动刷新提示词"/>每 5 秒刷新</Space>
      <Button onClick={onOpenCalls}>查看真实调用上下文</Button></div>
    <Alert className="notice" showIcon type="info" message="运行中模板与文件内容分别展示"
      description="运行中模板是当前进程已加载的提示词，system 已展开身份和上下文规则；文件区块读取当前 prompts.md，编辑后需重启机器人。动态历史、记忆、图片和工具消息请到 LLM 控制台的“请求上下文”查看。"/>
    {error && <Alert className="notice" type="error" showIcon message={error}/>}
    {data.errors?.map(e => <Alert key={e} className="notice" type="warning" message={e}/>)}
    <Tabs activeKey={view} onChange={setView} items={[{key: 'runtime', label: '运行中 · 完整模板'}, {key: 'source', label: '文件 · 原始区块'}]}/>
    <Spin spinning={loading}>
      {view === 'runtime' ? <>
        <div className="toolbar"><Select aria-label="提示词用途" value={group?.key} onChange={setSelected}
          options={data.groups.map(g => ({value: g.key, label: g.title}))}/>
          <Typography.Text type="secondary">动态占位符在请求时填入，不是已发送的完整会话</Typography.Text></div>
        {group ? <>{textCard('系统提示词', group.system, 'system')}{textCard('用户上下文模板', group.user, 'user')}</> : <Empty description="暂无提示词"/>}
      </> : <><div className="toolbar"><Select aria-label="提示词文件区块" value={name} onChange={setSource}
        options={Object.keys(data.source).map(n => ({value: n, label: n}))}/></div>
        {name ? textCard(name, data.source[name]) : <Empty description="未读取到文件区块"/>}</>}
    </Spin>
  </div>;
}
