# AGENTS.md

## 项目硬约束

- 本项目是 NoneBot2 + OneBot v11 的 QQ 群聊/私聊机器人。
- Agent 调用走普通 LangChain/LLM 封装；不要引入 LangGraph、checkpointer，或把 Mem0 暴露成 Agent tool。
- 默认聊天模型是 `deepseek-flash`，可通过 `DEEPSEEK_MODEL` 配置；`thinking=enabled`，可通过 `DEEPSEEK_THINKING=false` 关闭；除非用户明确要求，不要改默认模型。
- 密钥只允许来自环境变量或本地 `.env`；不要写进代码、测试快照、日志或提交内容。

## 入口和链路

- 启动入口：`bot.py`。
- 生产装配必须用 `create_default_application()`，让长期记忆 worker 和会话归档 worker 随 `application.start()` / `application.stop()` 管理生命周期。
- `create_default_orchestrator()` 只用于测试或显式自定义装配；直接用它不会自动启动长期记忆 worker，也不会装配会话归档。
- NoneBot 插件入口：`qq_group_chatter/plugins/chat.py`。
- 主编排：`qq_group_chatter/orchestrator.py`。
- 联网搜索服务：`qq_group_chatter/services/deepseek_search.py`，由 `create_default_application()` 注册为 Agent 工具。
- 原始对话语义归档：`qq_group_chatter/services/conversation_archive.py`，由 `create_default_application()` 装配后注入 `ChatOrchestrator`。
- 群聊和私聊都先构造成 `ConversationContext`。

主链路顺序保持：

1. 收到消息并构造 `ConversationContext`
2. 过滤空消息等无效输入
3. 写入 `ShortTermMemoryService`
4. 读取短期记忆
5. 查询用户长期记忆和会话长期记忆
6. 查询原始对话语义归档，作为更早历史对话参考
7. 调用 `ChatAgent` 生成回复
8. 发送回复
9. 把 assistant 回复写入短期记忆和原始对话语义归档
10. 携带本轮长期记忆查询快照和原始对话语义归档快照，投递用户消息到长期记忆后台 ingestion

聊天使用单 Agent 工具调用循环；ChatAgent 自主决定直接回复或调用 Python/MCP 工具，程序执行并回传结果后继续判断。搜索通过 DeepSeek Search/Fetch 工具完成，每轮最多发送一次搜索提示。默认最多 8 次模型请求、12 次工具执行、单工具 20 秒、整轮 120 秒。

长期记忆 planner 的事实来源只看本轮用户消息（文字及实际发送的本轮图片）；短期上下文及历史图片、原始对话语义归档和 assistant 回复只允许作为指代、确认、拒绝、限定上下文，不能作为独立记忆事实来源。长期记忆只保存文字，不保存图片或图片引用。

## 图片理解

- 默认 `IMAGE_UNDERSTANDING_ENABLED=true`，每次模型请求最多 `IMAGE_UNDERSTANDING_MAX_IMAGES_PER_REQUEST=4` 张；聊天、搜索增强和 planner 复用本轮图片快照。
- 关闭后不下载新图，也不向任何模型发送图片，文字聊天照常运行；修改配置后重启生效。
- 只使用 NapCat 推送的图片事件和 URL，不通过 `get_image`、`get_msg` 回查图片或历史。
- 图片缓存在 `.mem0/images/`，短期记忆保存附件引用；原始归档不保存图片引用。不要提交缓存或在 trace 中写入图片 Base64。

## 记忆边界

- 短期记忆：`qq_group_chatter/services/short_term_memory.py`，按 `conversation_id` 保存最近消息；默认每会话在内存和 `.mem0/short-term-memory.jsonl` 持久化文件里保留 300 条，prompt 默认只读最近 30 条。
- 短期主键：
  - 群聊：`qq_group:{group_id}`
  - 私聊：`qq_private:{user_id}`
- 长期记忆：`qq_group_chatter/services/long_term_memory.py`，使用 Mem0 真实存储；默认不能静默退回 Noop。
- `NoopMem0Client` 只允许在测试或显式注入时使用。
- 默认 Mem0 初始化失败应抛 `MemoryConfigurationError`，不要让机器人假装长期记忆可用。
- 长期记忆写入走后台 worker，不能阻塞用户回复；写入前必须做候选校验和重复抑制。
- 长期记忆 scope 只保留 `user` 和 `conversation`；不要引入 group-user 组合记忆。
- 长期 ID：
  - 用户：`qq_user:{conversation_id}:{user_id}`
  - 会话：`qq_conversation:{conversation_id}`
- 长期记忆 planner：`qq_group_chatter/services/long_term_memory_planner.py`，默认 `deepseek-flash` + `thinking=enabled`；模型跟随 `DEEPSEEK_MODEL`，可通过 `DEEPSEEK_THINKING=false` 关闭思考。一次调用内完成提取判断和 add/update/skip 决策，不直接给聊天 Agent。
- user scope 记忆可以属于当前发言者或同 conversation 内可唯一确定的目标用户；他人个人事实应写入目标用户的 user 记忆，无法唯一确定目标但有长期价值时可降级写入 conversation。
- user scope 操作可带 `target_user_id` / `target_nickname`；规则层必须校验 `target_user_id` 来自当前用户、短期上下文或原始对话语义归档的已知用户，不能把未知目标写到当前发言者名下。
- 长期记忆写入 Mem0 时使用 `infer=False`；记忆提取、合并和跳过决策由项目自己的 `LongTermMemoryPlanner` 负责，不依赖 Mem0 内部 LLM 自动推理。
- Mem0 `search()` / `get_all()` 的 `filters` 必须至少包含 `user_id`、`agent_id`、`run_id` 之一；不要只用 `conversation_id` 调 Mem0。
- 查询当前用户和当前会话长期记忆时用 `filters={"user_id": ...}`；查询同会话全局相关记忆时用 `filters={"user_id": "*", "conversation_id": context.conversation_id}`，再由项目代码过滤 owner 和去重。
- 不要提取手机号、密码、token、api key、地址等敏感内容。

## 原始对话语义归档

- 原始对话语义归档不是长期记忆事实库；它保存较早的 user / assistant 原始消息片段，用于普通聊天和搜索增强回复里的历史对话召回。
- 服务在 `qq_group_chatter/services/conversation_archive.py`；默认由 `create_default_application()` 创建，随应用生命周期启动/停止后台 worker。
- 默认启用：`CONVERSATION_ARCHIVE_ENABLED=true`。关闭后普通聊天仍可运行，但不会召回更早原始对话。
- 默认使用独立 Mem0 命名空间和本地存储：collection 前缀 `qq_group_chatter_archive`，向量库 `.mem0/qdrant-archive`，历史库 `.mem0/archive-history.db`。不要和长期记忆的 `.mem0/qdrant` 混为同一个逻辑库。
- 归档写入 `infer=False`，只存原始消息和元数据，不依赖 Mem0 内部 LLM 自动推理。
- 归档 user_id 使用 `qq_archive:{conversation_id}`；Mem0 查询仍必须带 `user_id`，并同时带 `conversation_id` 和 `archive_type="conversation_message"` 过滤。
- 默认每会话最多保留 `CONVERSATION_ARCHIVE_MAX_MESSAGES_PER_CONVERSATION=5000` 条；超过后按最旧消息裁剪。不要在机器人运行时直接删除 `.mem0/`。
- 归档召回默认 `CONVERSATION_ARCHIVE_TOP_K=5`、`CONVERSATION_ARCHIVE_CANDIDATE_K=20`，再按语义分和时间衰减重排。
- 归档可以包含 assistant 原始回复；这只用于聊天上下文和 planner 指代理解，不能让 planner 仅凭历史归档创建、更新或删除长期记忆。

## DeepSeek 和 Mem0

- `DEEPSEEK_API_KEY` 必填：聊天 Agent 和 Mem0 内部 LLM 都需要。
- `DEEPSEEK_THINKING` 可选：默认 `true`；设为 `false` 可关闭项目自己发起的 DeepSeek 调用的 thinking；env 示例统一用 `true` / `false`，解析层兼容 `enabled` / `disabled`。
- `MEM0_FASTEMBED_MODEL` 可选：默认 `BAAI/bge-small-zh-v1.5`。
- 当前默认：
  - ChatAgent：`deepseek-flash` + `thinking=enabled`，模型跟随 `DEEPSEEK_MODEL`
  - LongTermMemoryPlanner：`deepseek-flash` + `thinking=enabled`，模型跟随 `DEEPSEEK_MODEL`
  - Mem0 内部 LLM provider：`deepseek`，模型跟随 `DEEPSEEK_MODEL`，默认 `deepseek-flash`
  - Mem0 embedding：本地 `fastembed`
  - 本地向量库：`.mem0/qdrant`
- `.env` 本地使用且已 ignore；提交示例只改 `.env.example`。
- `.mem0/` 是运行数据、本地长期记忆存储、原始对话语义归档和短期记忆持久化存储，可以保留用于加快下次运行或保留本地记忆；不要提交。

## 工具与联网搜索

- 工具注册在 `qq_group_chatter/agent/tools.py`，循环在 `agent/tool_loop.py`；不引入 LangGraph/checkpointer，不把 Mem0 注册为工具。
- `tools.json` 为本地配置，使用 `AGENT_TOOLS_CONFIG` 指定路径；提交示例为 `tools.example.json`，凭据只允许 `${ENV_VAR}` 引用。
- Python 工厂 `module:factory` 返回 AgentTool 或其列表；MCP 支持 stdio / streamable_http，随应用生命周期管理。安装和启用由管理员负责。
- DeepSeek Search/Fetch 分别调用 `/v1/tools/search` 和 `/v1/tools/fetch`，复用 `DEEPSEEK_API_KEY`；生产链路不依赖 Tavily、不自动回退。
- `WEB_SEARCH_ENABLED=true` 默认注册 web_search/web_fetch；search count 默认 5、need_content=true；查询 1–128 字符。
- 搜索正文默认每条 8000 字符，Fetch 正文默认 20000 字符，截断需标记；工具结果保留 URL 便于抓取，最终回复默认不展示 URL。
- 搜索和工具结果不作为长期记忆事实来源；过程提示不入聊天记忆；最终回复发送成功后才记录。
- 群聊默认支持文字开头呼唤“神奈”；`CHAT_KEYWORD_TRIGGER_ENABLED=false` 关闭关键词响应，保留 @、机器人引用和私聊行为；重启生效。

## 可观测性

- 可观测性在 `qq_group_chatter/observability.py`。
- 保持结构化日志和 Prometheus 风格指标：消息处理数量/结果、端到端回复耗时、chat agent / memory planner LLM 耗时、Mem0 search/add 耗时、长期记忆队列长度、候选 add/skip/error、duplicate skip、stage + error_type 错误计数。
- 不要在日志中输出原始 QQ 号、API key 或敏感消息内容。
- LLM 调用 trace 在 `qq_group_chatter/llm_tracing.py`，默认写入 `logs/llm-traces.jsonl`，用于 `/console` 运行控制台里的 LLM 控制台；`/memory` 保留为兼容入口。
- LLM trace 是本地调试数据，不是普通日志；它会保存原始 messages 和 response，可能包含 QQ 号、昵称、聊天正文、长期记忆片段和搜索正文片段。不要提交、公开或转发该文件。

## 测试和验证

常用验证：

```powershell
python -m pytest -q
```

真实 Mem0 初始化验证需要本地 `.env` 有 `DEEPSEEK_API_KEY`，且可能下载 fastembed 模型：

```powershell
python -c "from qq_group_chatter.app import create_default_mem0_client; c=create_default_mem0_client(); print(type(c).__name__); c.close()"
```

- pytest / Python 缓存可以保留用于加快本地开发和下次验证；不要提交。
- 测试或手动验证如果会写入长期记忆，必须使用隔离的测试 Mem0 目录、`NoopMem0Client` 或 fake client。
- 测试创建的记忆不应该保留到真实 `.mem0/`，也不能误删用户真实长期记忆或模型/向量缓存。
- 测试里如果要避免真实网络或真实 Mem0，必须显式注入 `NoopMem0Client` 或 fake client，不要依赖默认工厂自动降级。

## 开发注意

- 不要跳过 `create_default_application()` 的生产装配；否则长期记忆 worker、会话归档 worker、联网搜索和 LLM trace 可能不会按生产链路注入或随生命周期管理。
- 不要把长期记忆 ingestion 放到缺少上下文快照的位置；发送回复成功后，应携带本轮用户消息、短期上下文、长期记忆快照、原始对话语义归档快照和 assistant 回复投递。
- 不要把 assistant 回复用于长期记忆 planner。
- 不要提交 `.env`、`.mem0/`、`logs/llm-traces.jsonl`、`*.egg-info/`、缓存目录或 API key。
- 修改默认模型、记忆 scope、持久化策略前，先确认用户是否真的改变了需求。
