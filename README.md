# QQ Group Chatter

NoneBot2 + OneBot v11 的 QQ 群聊/私聊机器人，使用 DeepSeek 生成回复，使用 Mem0 + 本地 Qdrant 保存长期记忆，并可通过 Tavily 做联网搜索。

## 功能

- 支持 QQ 群聊和私聊。
- 群聊中只响应 `@机器人` 的消息，私聊直接响应。
- 使用短期记忆保留最近对话上下文。
- 使用 Mem0 保存长期记忆，包括用户偏好、约束、身份信息和会话规则。
- 聊天 Agent 可在需要实时资料时自动触发联网搜索。
- 提供长期记忆面板和本地 LLM 调用控制台：`/memory`。
- 运行日志默认写入 `logs/`，日志目录不会提交到 Git。

## 环境要求

- Python 3.11 或更高版本。
- 一个 OneBot v11 兼容的 QQ 连接端，例如 NapCat、Lagrange、go-cqhttp 等。
- `DEEPSEEK_API_KEY`。聊天模型和 Mem0 内部 LLM 都需要它。
- `DEEPSEEK_THINKING` 可选，默认 `true`；如需关闭 DeepSeek thinking，设为 `false`。也兼容 `enabled` / `disabled`，但示例统一使用 `true` / `false`。
- 如果启用联网搜索，还需要 `TAVILY_API_KEY`。

## 安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

如果只运行机器人、不跑测试，也可以安装主依赖：

```powershell
python -m pip install -e .
```

首次初始化 Mem0 时，`fastembed` 可能会下载 embedding 模型。

## 配置

复制配置示例：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，至少填写：

```env
DEEPSEEK_API_KEY=你的 DeepSeek Key
DEEPSEEK_THINKING=true
```

如果保留默认的联网搜索开启状态，还需要：

```env
TAVILY_API_KEY=你的 Tavily Key
```

如果暂时不用联网搜索，可以关闭：

```env
WEB_SEARCH_ENABLED=false
```

常用配置：

```env
# Mem0 本地 embedding 模型
MEM0_FASTEMBED_MODEL=BAAI/bge-small-zh-v1.5

# DeepSeek thinking 开关，默认开启；推荐 true / false，也兼容 enabled / disabled
DEEPSEEK_THINKING=true

# 搜索结果数量
WEB_SEARCH_MAX_RESULTS=3

# 日志级别
QQ_GROUP_CHATTER_LOG_LEVEL=INFO
QQ_GROUP_CHATTER_FRAMEWORK_LOG_LEVEL=INFO

# 文件日志
QQ_GROUP_CHATTER_FILE_LOG_ENABLED=true
QQ_GROUP_CHATTER_LOG_DIR=logs

# 本地 LLM 调用 trace，供 /memory 里的 LLM 控制台查看
QQ_GROUP_CHATTER_LLM_TRACE_ENABLED=true
QQ_GROUP_CHATTER_LLM_TRACE_PATH=logs/llm-traces.jsonl
QQ_GROUP_CHATTER_LLM_TRACE_MAX_RECORDS=500
```

`.env`、`.mem0/`、`logs/` 都是本地运行数据，不要提交。

## 启动

启动机器人：

```powershell
python bot.py
```

`bot.py` 会：

1. 初始化 NoneBot。
2. 注册 OneBot v11 adapter。
3. 加载聊天插件 `qq_group_chatter.plugins.chat`。
4. 创建默认应用 `create_default_application()`。
5. 随 NoneBot 启停长期记忆后台 worker。
6. 注册长期记忆只读面板。

OneBot v11 连接端需要连接到 NoneBot 的监听地址。具体连接方式取决于你使用的 QQ 连接端，请按对应项目的 OneBot v11 反向 WebSocket / HTTP 配置填写。

## 使用

### 普通聊天

私聊机器人时，直接发送消息即可。

群聊中需要 `@机器人`，例如：

```text
@机器人 今天帮我总结一下这个方案
```

普通聊天链路会读取短期记忆和长期记忆，再生成回复。有效用户消息会投递给长期记忆后台 worker，长期记忆写入不会阻塞本轮回复。

### 联网搜索

联网搜索是普通聊天链路里的补充资料步骤。所有消息都会先进入普通聊天链路，由 ChatAgent 判断是直接回复，还是先联网搜索。

示例：

```text
搜一下 Python 3.13 有哪些变化
搜索 DeepSeek v3.1 发布信息
查一下 今天的 AI 新闻
Python 3.13 有哪些变化？
```

默认行为：

- ChatAgent 先返回 `reply` 或 `web_search` 决策；如果是 `web_search`，机器人会先发送一条搜索提示，再调用 Tavily。
- 使用 Tavily 搜索。
- 最多读取 3 条结果。
- 基于网页 `raw_content` 生成回答。
- 默认不向 prompt 暴露 URL，最终回答通常不展示 URL。
- 搜索上下文不会写入长期记忆。
- 搜索提示不会写入短期记忆。
- 搜索回复发送成功后，会作为 assistant 回复进入短期记忆。

### 长期记忆和 LLM 控制台

机器人启动后，可以访问：

```text
http://127.0.0.1:8080/memory
```

这个页面包含两个视图：

- 长期记忆：只读查看当前 Mem0 本地存储里的长期记忆。
- LLM 控制台：查看项目自己发给 DeepSeek 的原始 messages、返回内容、耗时、状态和错误摘要。

LLM 控制台默认读取 `logs/llm-traces.jsonl`，最多保留最近 500 次调用。它会保存原始聊天上下文，可能包含 QQ 号、昵称、当前消息、短期历史、长期记忆片段和联网搜索网页正文片段；这是本地调试数据，不要提交、公开或发给无关人员。普通日志仍会尽量避免输出原始 QQ 号、API key 和敏感消息内容。

具体端口以 NoneBot 运行配置为准。

也可以访问 JSON API：

```text
http://127.0.0.1:8080/api/memory
http://127.0.0.1:8080/api/llm-traces
```

## 记忆说明

短期记忆：

- 存在内存里。
- 默认每个会话保留最近 30 条。
- prompt 默认读取最近 20 条。
- 重启后丢失。

长期记忆：

- 使用 Mem0 真实存储。
- 本地向量库默认在 `.mem0/qdrant`。
- 用户记忆 ID：`qq_user:{conversation_id}:{user_id}`。
- 会话记忆 ID：`qq_conversation:{conversation_id}`。
- 只处理用户消息，不处理 assistant 回复。
- 写入 Mem0 时使用 `infer=False`，由项目自己的 planner 决定 add/update/skip。
- Mem0 查询必须带 `user_id`、`agent_id`、`run_id` 之一；同会话全局相关记忆查询使用 `{"user_id": "*", "conversation_id": context.conversation_id}`，再由项目代码过滤和去重。
- 不提取手机号、密码、token、API key、地址等敏感内容。

## 验证

运行测试：

```powershell
python -m pytest -q
```

验证真实 Mem0 能初始化：

```powershell
python -c "from qq_group_chatter.app import create_default_mem0_client; c=create_default_mem0_client(); print(type(c).__name__); c.close()"
```

这条命令需要本地 `.env` 中有 `DEEPSEEK_API_KEY`，并且可能下载 fastembed 模型。

## 日志

默认文件日志写入：

```text
logs/qq-group-chatter.log
```

可通过 `.env` 调整：

```env
QQ_GROUP_CHATTER_FILE_LOG_ENABLED=true
QQ_GROUP_CHATTER_FILE_LOG_LEVEL=DEBUG
QQ_GROUP_CHATTER_LOG_DIR=logs
QQ_GROUP_CHATTER_FILE_LOG_ROTATION=10 MB
QQ_GROUP_CHATTER_FILE_LOG_RETENTION=5
```

日志会尽量避免输出原始 QQ 号、API key 和敏感消息内容。

LLM trace 不是普通日志。它默认写入：

```text
logs/llm-traces.jsonl
```

这个文件用于复现 LLM 调用问题，会保存原始 prompt 和 response；`logs/` 已经被 Git 忽略，不要手动提交或公开它。

## 常见问题

### 启动时报 `DEEPSEEK_API_KEY is required`

检查 `.env` 是否存在，且是否填写了：

```env
DEEPSEEK_API_KEY=...
```

### 不想启用搜索

在 `.env` 中设置：

```env
WEB_SEARCH_ENABLED=false
```

### 搜索提示没有配置

如果 `WEB_SEARCH_ENABLED=true`，需要填写：

```env
TAVILY_API_KEY=...
```

### Mem0 / Qdrant 初始化失败

先确认依赖已安装：

```powershell
python -m pip install -e .
```

如果曾经改过 embedding 模型，旧的本地 Qdrant collection 可能和新模型维度不匹配。停止机器人后，再检查 `.mem0/qdrant` 是否需要迁移或清理。不要在机器人运行时删除 `.mem0/`。

## 开发入口

- 启动入口：`bot.py`
- NoneBot 插件：`qq_group_chatter/plugins/chat.py`
- 主编排：`qq_group_chatter/orchestrator.py`
- 聊天 Agent：`qq_group_chatter/agent/chat_agent.py`
- DeepSeek LLM：`qq_group_chatter/agent/deepseek_llm.py`
- LLM trace：`qq_group_chatter/llm_tracing.py`
- 短期记忆：`qq_group_chatter/services/short_term_memory.py`
- 长期记忆：`qq_group_chatter/services/long_term_memory.py`
- 长期记忆 planner：`qq_group_chatter/services/long_term_memory_planner.py`
- 联网搜索：`qq_group_chatter/services/web_search.py`
- Prompt 文件：`qq_group_chatter/prompts/`
