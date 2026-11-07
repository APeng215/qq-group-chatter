import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import Prompts from './Prompts';

it('separates loaded templates from disk blocks and opens real calls', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ok: true, json: async () => ({
    groups: [{key: 'chat', title: '普通聊天', system: '已加载的身份正文', user: '{user_message}'}],
    source: {'chat_agent_system.txt': '编辑后的文件正文'},
  })} as Response);
  const open = vi.fn(); render(<Prompts onOpenCalls={open}/>);
  expect(await screen.findByText('已加载的身份正文')).toBeInTheDocument();
  expect(screen.queryByText('编辑后的文件正文')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', {name: '文件 · 原始区块'}));
  expect(await screen.findByText('编辑后的文件正文')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', {name: '查看真实调用上下文'}));
  expect(open).toHaveBeenCalledOnce();
});
