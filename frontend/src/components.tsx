import { useEffect, useState } from 'react';
import { Alert, Button, Collapse, Image, Space, Tag, Typography, message } from 'antd';
import JsonView from '@uiw/react-json-view';
import type { Attachment, Trace } from './types';
import { business, describeContent, originalContent, statusColor, validationLabels } from './trace';

export async function copy(value: unknown, raw = false) {
  const text = raw && typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? 'undefined';
  try { await navigator.clipboard.writeText(text); message.success('已复制'); }
  catch { message.error('复制失败，请使用安全上下文或手动复制'); }
}
export function JsonPanel({value, label = '原始数据'}: {value: unknown; label?: string}) {
  return <section className="json-panel"><div className="section-heading"><strong>{label}</strong>
    <Button size="small" onClick={() => copy(value)}>复制原始数据</Button></div>
    <div className="json-scroll">{value && typeof value === 'object'
      ? <JsonView value={value as object} collapsed={2} displayDataTypes={false} enableClipboard={false}/>
      : <pre>{JSON.stringify(value) ?? '未记录'}</pre>}</div></section>;
}
export function Content({value}: {value: string | null | undefined}) {
  const description = describeContent(value);
  return <div className="response-content">{description ? <Alert type="warning" showIcon message={description}/> : <pre className="prose">{value}</pre>}
    <Collapse size="small" ghost items={[{key: 'escaped', label: '查看转义文本', children: <pre>{JSON.stringify(value) ?? 'undefined'}</pre>}]}/>
  </div>;
}
export function Status({trace}: {trace: Trace}) {
  return <Space size={4} wrap><Tag color={statusColor(trace.status ?? '')}>API {trace.status ?? '未知'}</Tag>
    <Tag color={statusColor(business(trace))}>{business(trace)}</Tag></Space>;
}
const attachmentLabels: Record<string, string> = {
  sent: '已发送', sent_duplicate: '复用', disabled: '功能关闭', request_limit: '超限省略', message_limit: '接收超限',
  missing_url: '缺少下载地址', unavailable: '图片已不可用', timeout: '下载超时', download_failed: '下载失败',
  http_error: '下载失败', too_large: '文件超限', unsupported_format: '格式不支持', private_address: '地址不允许', invalid_url: '无效地址',
};
export function Preview({resourceId, title}: {resourceId?: string; title?: string}) {
  const [failed, setFailed] = useState(false);
  useEffect(() => {setFailed(false);}, [resourceId]);
  if (failed || !/^[a-f0-9]{64}$/.test(resourceId ?? '')) return <span className="unavailable">图片已不可用</span>;
  return <Image width={112} height={88} style={{objectFit: 'contain'}} alt={title ?? '图片'}
    src={`/api/images?resource_id=${encodeURIComponent(resourceId!)}`} onError={() => setFailed(true)} preview={{}}/>;
}
export function Images({trace}: {trace: Trace}) {
  const context = trace.image_context;
  if (!context) return null;
  const sent = context.attachments.filter(a => a.status === 'sent');
  return <section className="image-section"><Typography.Text type="secondary">实际发送 {context.sent_count} / {context.max_images} 张 · {context.enabled ? '开启' : '关闭'}</Typography.Text>
    <Image.PreviewGroup><div className="thumbnails">{sent.map((a, i) => <figure key={`${a.resource_id}-${i}`}>
      <Preview resourceId={a.resource_id} title={`图${a.input_index}`}/><figcaption>图{a.input_index}</figcaption></figure>)}</div></Image.PreviewGroup>
    <Collapse size="small" items={[{key: 'attachments', label: '附件来源与下载详情', children: context.attachments.map((a: Attachment, i) =>
      <div className="attachment" key={i}><strong>{a.input_index ? `${a.status === 'sent_duplicate' ? '复用' : ''}图${a.input_index}` : '未发送图片'}</strong>
        <Tag>{attachmentLabels[a.status ?? ''] ?? a.status}</Tag><div>来源消息：{a.message_id ?? '未知'} · {a.nickname ?? a.user_id} · 第{a.position}张</div>
        <Typography.Text type="secondary">{a.media_type} · {a.size_bytes ?? 0} bytes · {a.duration_ms ?? 0} ms</Typography.Text>
        <div className="resource-id">{a.resource_id}</div></div>)}]}/>
  </section>;
}
export function Messages({trace}: {trace: Trace}) {
  return <div><Typography.Paragraph type="secondary">请求上下文（图片以引用展示），不是原始 HTTP 请求字节。</Typography.Paragraph>
    {(trace.messages ?? []).map((item, i) => <section className="context-message" key={i}><Tag>{item.role}</Tag>
      {typeof item.content === 'string' ? <pre className="prose">{item.content}</pre> : item.content.map((block, j) =>
        block.type === 'text' ? <pre className="prose" key={j}>{block.text}</pre> : block.type === 'image_ref'
          ? <div key={j}><span>{block.input_index ? `图${block.input_index}` : '图片引用'} </span><Preview resourceId={block.resource_id}/></div>
          : <JsonPanel key={j} value={block}/>)}</section>)}
  </div>;
}
export function QuestionAnswer({trace}: {trace: Trace}) {
  const failure = validationLabels[trace.validation_error ?? ''] ?? trace.validation_error ?? trace.fallback_reason;
  return <Space direction="vertical" size="middle" className="full-width">
    <section><h3>用户提问</h3><pre className="prose">{trace.current_user_message ?? '未记录用户提问'}</pre></section>
    <section><div className="section-heading"><h3>神奈回复</h3><Button size="small" onClick={() => copy(originalContent(trace), true)}>复制 content</Button></div>
      <Typography.Paragraph type="secondary">模型生成结果，不代表 QQ 已发送成功。</Typography.Paragraph>
      {failure && <Alert type="warning" showIcon message={failure}/>}
      {trace.error_message && <Alert type="error" showIcon message={trace.error_message}/>}
      {trace.final_reply || trace.search_notice ? <pre className="prose">{trace.final_reply ?? trace.search_notice}</pre> : null}
      {(failure || (!trace.final_reply && !trace.search_notice)) && <Content value={originalContent(trace)}/>}</section><Images trace={trace}/>
  </Space>;
}
