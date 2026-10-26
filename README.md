# QQ Group Chatter

NoneBot2 + OneBot v11 的 QQ 群聊/私聊机器人，使用 DeepSeek 生成回复，使用 Mem0 + 本地 Qdrant 保存长期记忆，并可通过 Tavily 做联网搜索。

## 功能

- 支持 QQ 群聊和私聊。
- 群聊中只响应 `@机器人` 的消息，私聊直接响应。
- 使用短期记忆保留最近对话上下文。
- 支持图片理解与短期上下文看图，默认开启，单次模型请求最多携带 4 张图片。
- 使用 Mem0 保存长期记忆，包括用户偏好、约束、身份信息和会话规则。
- 使用独立的原始对话语义归档召回更早的历史对话片段。
- 聊天 Agent 可在需要实时资料时自动触发联网搜索。
- 提供运行控制台：`/console`，包含记忆库和本地 LLM 调用控制台。
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
MEM0_TELEMETRY=false
MEM0_FASTEMBED_MODEL=BAAI/bge-small-zh-v1.5
LONG_TERM_MEMORY_TOP_K=10
LONG_TERM_MEMORY_CANDIDATE_K=30
LONG_TERM_MEMORY_SEMANTIC_WEIGHT=0.85
LONG_TERM_MEMORY_RECENCY_WEIGHT=0.15
LONG_TERM_MEMORY_TIME_DECAY_DAYS=180

# 短期记忆持久化；prompt 仍只读取最近 30 条
SHORT_TERM_MEMORY_MAX_MESSAGES=300
SHORT_TERM_MEMORY_PATH=.mem0/short-term-memory.jsonl

# 图片理解；修改后重启生效，false 会同时关闭聊天和 planner 的图片输入
IMAGE_UNDERSTANDING_ENABLED=true
IMAGE_UNDERSTANDING_MAX_IMAGES_PER_REQUEST=4
IMAGE_UNDERSTANDING_MAX_IMAGES_PER_MESSAGE=8
IMAGE_UNDERSTANDING_MAX_BYTES_PER_IMAGE=8388608
IMAGE_UNDERSTANDING_TIMEOUT_SECONDS=8
IMAGE_UNDERSTANDING_DOWNLOAD_CONCURRENCY=3

# 更久以前的原始对话语义召回；和长期记忆分开存储
CONVERSATION_ARCHIVE_ENABLED=true
CONVERSATION_ARCHIVE_MAX_MESSAGES_PER_CONVERSATION=5000
CONVERSATION_ARCHIVE_TOP_K=5
CONVERSATION_ARCHIVE_CANDIDATE_K=20
CONVERSATION_ARCHIVE_SEMANTIC_WEIGHT=0.85
CONVERSATION_ARCHIVE_RECENCY_WEIGHT=0.15
CONVERSATION_ARCHIVE_TIME_DECAY_DAYS=90

# DeepSeek thinking 开关，默认开启；推荐 true / false，也兼容 enabled / disabled
DEEPSEEK_THINKING=true

# 搜索结果数量
WEB_SEARCH_MAX_RESULTS=5

# 日志级别
QQ_GROUP_CHATTER_LOG_LEVEL=INFO
QQ_GROUP_CHATTER_FRAMEWORK_LOG_LEVEL=INFO

# 文件日志
QQ_GROUP_CHATTER_FILE_LOG_ENABLED=true
QQ_GROUP_CHATTER_LOG_DIR=logs

# 本地 LLM 调用 trace，供 /console 里的 LLM 控制台查看
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
5. 随 NoneBot 启停长期记忆后台 worker 和原始对话语义归档 worker。
6. 注册运行控制台。

OneBot v11 连接端需要连接到 NoneBot 的监听地址。具体连接方式取决于你使用的 QQ 连接端，请按对应项目的 OneBot v11 反向 WebSocket / HTTP 配置填写。

## 使用

### 普通聊天

私聊机器人时，直接发送消息即可。

群聊中需要 `@机器人`，例如：

```text
@机器人 今天帮我总结一下这个方案
```

普通聊天链路会读取短期记忆、长期记忆和相关历史对话片段，再生成回复。有效用户消息和发送成功后的 assistant 回复会进入原始对话语义归档；长期记忆写入走后台 worker，不会阻塞下一轮消息处理。

### 图片理解

NapCat 推送的图片消息会携带图片信息。机器人只使用事件中的 HTTP(S) URL 下载图片，不调用 `get_image` 或 `get_msg` 补取资源。URL 缺失或下载失败时会明确标记未读取，需要时请重新发送。

图片缓存在 `.mem0/images/`，短期消息保存本地引用；重启后仍能在保留的短期上下文中继续追问。未 @ 机器人的群图片也会按现有记录规则缓存，但不会触发模型回复。超出短期记忆且无进行中请求引用的缓存会定期清理。

聊天决策、搜索增强回复和后台记忆 planner 共用本轮图片快照，默认最多 4 张：先选当前消息，再选被引用消息，最后选 prompt 内最近历史图片；相同图片去重。每个模型调用分别计算图片输入 token，4 张是单次请求上限，不是整轮所有调用的合计上限。

长期记忆只保存文字事实，不保存图片或图片引用。本轮图片可以结合本轮文字提供事实；历史图片和 assistant 回复只用于理解指代，不能独立产生新事实。原始对话语义归档只保留文字和图片占位。

`IMAGE_UNDERSTANDING_ENABLED=false` 会停止下载新图片，并停止向所有模型发送当前或历史图片；文字聊天继续运行，已有缓存保留。修改 `.env` 后重启生效。

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

### 运行控制台

机器人启动后，可以访问：

```text
http://127.0.0.1:8080/console
```

这个页面包含两个视图：

- LLM 控制台（默认）：按轮次查看模型调用及重试，默认筛选 `chat_agent`；区分 API 状态与业务状态，查看空白响应、上下文、完整 API response 和累计用量。请求中的图片以资源引用展示，不是原始 HTTP 请求字节。
- 记忆库：进入页面后才加载 Mem0 长期记忆，只读查看正文、metadata、变更历史和后台队列数量。
- 图片输入：查看开关、单次图片上限、每次实际发送的图片、来源及失败/省略状态；支持缩略图和原图预览。图片 Base64 不写入 trace，复制和导出仅包含附件元数据及资源引用，缓存清理后显示“图片已不可用”。

旧入口 `http://127.0.0.1:8080/memory` 仍保留为兼容别名。

控制台使用 React + TypeScript + Vite + Ant Design。仓库及 Python 包自带 `qq_group_chatter/static/console/` 构建产物，运行机器人不需要 Node 服务或 CDN。修改前端后，使用 Node.js 22.12+ 在项目根目录执行统一检查与构建：

```powershell
python scripts/build_console.py
```

该命令执行 `npm ci`、类型检查、Vitest 测试和生产构建；提交前端源码时应同步提交锁文件及静态产物。构建产物缺失时控制台返回明确的503提示，不影响机器人其他功能。

前端开发与浏览器验收（API 默认代理到 `127.0.0.1:8080`；浏览器测试全部使用 fake API）：

```powershell
cd frontend
npm run dev
# 首次浏览器验收先安装 Chromium
npx playwright install chromium
npm run test:e2e
```

默认不轮询，手动刷新保留当前轮次、调用及详情标签。筛选导出只包含匹配调用，详情可复制整轮调用。旧 trace 缺少轮次标识时独立展示；调用耗时合计不代表端到端等待时间，模型生成成功也不代表 QQ 已发送。

LLM 控制台默认读取 `logs/llm-traces.jsonl`，最多保留最近 500 次调用。它会保存原始聊天上下文，可能包含 QQ 号、昵称、当前消息、短期历史、长期记忆片段和联网搜索网页正文片段；这是本地调试数据，不要提交、公开或发给无关人员。普通日志仍会尽量避免输出原始 QQ 号、API key 和敏感消息内容。

具体端口以 NoneBot 运行配置为准。

也可以访问 JSON API：

```text
http://127.0.0.1:8080/api/memory
http://127.0.0.1:8080/api/llm-traces
```

## 记忆说明

短期记忆：

- 存在内存里，并默认持久化到 `.mem0/short-term-memory.jsonl`。
- 默认每个会话在磁盘和内存里保留最近 300 条。
- prompt 默认读取最近 30 条。
- 重启后会从本地文件恢复最近的短期历史。
- 可通过 `SHORT_TERM_MEMORY_MAX_MESSAGES` 和 `SHORT_TERM_MEMORY_PATH` 调整保留上限和路径。

原始对话语义归档：

- 用于召回更早的原始 user / assistant 对话片段，不是长期记忆事实库。
- 默认启用：`CONVERSATION_ARCHIVE_ENABLED=true`。
- 使用独立 Mem0 命名空间，默认本地向量库在 `.mem0/qdrant-archive`，历史库在 `.mem0/archive-history.db`。
- 归档 ID 使用 `qq_archive:{conversation_id}`，并用 `conversation_id` 和 `archive_type="conversation_message"` 过滤当前会话。
- 默认每个会话最多保留 5000 条，可通过 `CONVERSATION_ARCHIVE_MAX_MESSAGES_PER_CONVERSATION` 调整。
- 默认召回 5 条结果，可通过 `CONVERSATION_ARCHIVE_TOP_K`、`CONVERSATION_ARCHIVE_CANDIDATE_K` 和权重配置调整。
- 它只作为聊天上下文和长期记忆 planner 的指代参考；planner 不能仅凭旧归档创建、更新或删除长期记忆。

长期记忆：

- 使用 Mem0 真实存储。
- 本地向量库默认在 `.mem0/qdrant`。
- 用户记忆 ID：`qq_user:{conversation_id}:{user_id}`。
- 会话记忆 ID：`qq_conversation:{conversation_id}`。
- 只处理用户消息，不处理 assistant 回复。
- 写入 Mem0 时使用 `infer=False`，由项目自己的 planner 决定 add/update/skip。
- user 记忆可以属于当前发言者，也可以属于同一会话内可唯一确定的其他目标用户；planner 会通过 `target_user_id` 指定目标用户。
- 如果提到的是他人个人事实且能唯一确定 QQ，会写入该目标用户的 user 记忆；如果无法唯一确定目标用户但有长期价值，可作为 conversation 背景记忆。
- 召回时先从 Mem0 获取候选，再按语义分和长期记忆最近使用/更新时间做时间衰减重排。
- 回复发送成功后的长期记忆 planner 仍只调用一次；会携带本轮用户消息、短期上下文、长期记忆快照、原始对话语义归档快照和 assistant 回复。同一次结果里可用 `usage_updates` 标记本轮明显用到的记忆，后台异步刷新 `last_recalled_at` 和 `recall_count`。
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
- 原始对话语义归档：`qq_group_chatter/services/conversation_archive.py`
- 联网搜索：`qq_group_chatter/services/web_search.py`
- Prompt 文件：`qq_group_chatter/prompts/`
