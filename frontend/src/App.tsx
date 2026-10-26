import { useState } from 'react';
import { ConfigProvider, Tabs, Typography } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import Console from './Console';
import MemoryLibrary from './MemoryLibrary';
import './style.css';

export function App() {
  const [page, setPage] = useState('console');
  return <ConfigProvider locale={zhCN} theme={{token: {colorPrimary: '#2563eb', borderRadius: 10, fontSize: 14, colorBgLayout: '#f5f7fb'}}}>
    <header className="app-header"><div className="brand-mark">神</div><div><Typography.Title level={3}>神奈 · 运行控制台</Typography.Title>
      <Typography.Text type="secondary">模型调用与记忆，一处查看</Typography.Text></div></header>
    <main className="app-main"><Tabs activeKey={page} onChange={setPage} items={[
      {key: 'console', label: 'LLM 控制台'}, {key: 'memory', label: '记忆库'},
    ]}/><div hidden={page !== 'console'}><Console/></div>{page === 'memory' && <MemoryLibrary/>}</main>
  </ConfigProvider>;
}
