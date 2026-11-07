import { useEffect, useState } from 'react';
import { Alert, Button, Card, Empty, Space, Switch, Typography } from 'antd';
import { copy } from './components';

export default function Todo() {
  const [content, setContent] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  async function refresh() {
    setLoading(true);
    try {
      const response = await fetch('/api/todo', {cache: 'no-store'});
      if (!response.ok) throw Error(`HTTP ${response.status}`);
      const data = await response.json();
      setContent(data.content); setError('');
    } catch (e) {setError(String(e));} finally {setLoading(false);}
  }
  useEffect(() => {void refresh();}, []);
  useEffect(() => {
    if (!auto) return;
    const timer = setInterval(() => {void refresh();}, 5000);
    return () => clearInterval(timer);
  }, [auto]);
  return <div>
    <div className="toolbar"><Button loading={loading} onClick={() => {void refresh();}}>刷新 TODO</Button>
      <Space><Switch checked={auto} onChange={setAuto} aria-label="自动刷新 TODO"/>每 5 秒刷新</Space></div>
    <Typography.Paragraph type="secondary">来自 todo/todo.md，修改文件后自动更新。此处只读，待办及附件仍保存在被 Git 忽略的 todo/ 目录。</Typography.Paragraph>
    {error && <Alert className="notice" showIcon type="error" message={error}/>}
    <Card title="项目待办" extra={<Button size="small" disabled={!content} onClick={() => copy(content, true)}>复制 Markdown</Button>}>
      {content ? <pre className="prose">{content}</pre> : <Empty description="暂无 TODO，请在 todo/todo.md 中记录"/>}
    </Card>
  </div>;
}
