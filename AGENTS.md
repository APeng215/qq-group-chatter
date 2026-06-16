# AGENTS.md

## 项目硬约束

- 本项目是 NoneBot2 + OneBot v11 的 QQ 群聊/私聊机器人。
- Agent 调用走普通 LangChain/LLM 封装；不要引入 LangGraph、checkpointer，或把 Mem0 暴露成 Agent tool。
- 默认聊天模型是 `deepseek-v4-pro`，`thinking=disabled`。这是速度测试后的用户选择；除非用户明确要求，不要改模型或开启 thinking。
- 密钥只允许来自环境变量或本地 `.env`；不要写进代码、测试快照、日志或提交内容。

## 入口和链路

- 启动入口：`bot.py`。
- 生产装配必须用 `create_default_application()`，让长期记忆 worker 随 `application.start()` / `application.stop()` 管理生命周期。
- `create_default_orchestrator()` 只用于测试或显式自定义装配；直接用它不会自动启动长期记忆 worker。
- NoneBot 插件入口：`qq_group_chatter/plugins/chat.py`。
- 主编排：`qq_group_chatter/orchestrator.py`。
- 联网搜索服务：`qq_group_chatter/services/web_search.py`，由 `create_default_application()` 装配后通过 `setup_search_service()` 注入插件。
- 群聊和私聊都先构造成 `ConversationContext`。

主链路顺序保持：

1. 收到消息并构造 `ConversationContext`
2. 过滤空消息等无效输入
3. 写入 `ShortTermMemoryService`
4. 读取短期记忆
5. 查询用户长期记忆和会话长期记忆
6. 携带本轮长期记忆查询快照，投递用户消息到长期记忆后台 ingestion
7. 调用 `ChatAgent` 生成回复
8. 发送回复
9. 把 assistant 回复写入短期记忆

搜索消息是普通聊天前的插件级分支：`plugins/chat.py` 会先用 `parse_search_command()` 判断是否为搜索触发词；如果是，调用 `WebSearchService.search_reply()` 并发送搜索回复，不进入 `ChatOrchestrator.handle_message()` 的普通聊天链路。

长期记忆 planner 只看用户消息；不要把 assistant 回复喂给长期记忆处理链路。

## 记忆边界

- 短期记忆：`qq_group_chatter/services/short_term_memory.py`，内存版，按 `conversation_id` 保存最近消息；默认每会话 30 条，prompt 默认读最近 20 条。重启丢失是当前 MVP 可接受行为。
- 短期主键：
  - 群聊：`qq_group:{group_id}`
  - 私聊：`qq_private:{user_id}`
- 长期记忆：`qq_group_chatter/services/long_term_memory.py`，使用 Mem0 真实存储；默认不能静默退回 Noop。
- `NoopMem0Client` 只允许在测试或显式注入时使用。
- 默认 Mem0 初始化失败应抛 `MemoryConfigurationError`，不要让机器人假装长期记忆可用。
- 长期记忆写入走后台 worker，不能阻塞用户回复；写入前必须做候选校验和重复抑制。
- 长期记忆 scope 只保留 `user` 和 `conversation`；不要引入 group-user 组合记忆，除非用户重新提出。
- 长期 ID：
  - 用户：`qq_user:{user_id}`
  - 会话：`qq_conversation:{conversation_id}`
- 长期记忆 planner：`qq_group_chatter/services/long_term_memory_planner.py`，默认 `deepseek-v4-pro` + `thinking=disabled`；一次调用内完成提取判断和 add/update/skip 决策，不直接给聊天 Agent。
- 长期记忆写入 Mem0 时使用 `infer=False`；记忆提取、合并和跳过决策由项目自己的 `LongTermMemoryPlanner` 负责，不依赖 Mem0 内部 LLM 自动推理。
- 不要提取手机号、密码、token、api key、地址等敏感内容。

## DeepSeek 和 Mem0

- `DEEPSEEK_API_KEY` 必填：聊天 Agent 和 Mem0 内部 LLM 都需要。
- `MEM0_FASTEMBED_MODEL` 可选：默认 `BAAI/bge-small-zh-v1.5`。
- 当前默认：
  - ChatAgent：`deepseek-v4-pro` + `thinking=disabled`
  - LongTermMemoryPlanner：`deepseek-v4-pro` + `thinking=disabled`
  - Mem0 内部 LLM provider：`deepseek`，模型 `deepseek-v4-pro`
  - Mem0 embedding：本地 `fastembed`
  - 本地向量库：`.mem0/qdrant`
- `.env` 本地使用且已 ignore；提交示例只改 `.env.example`。
- `.mem0/` 是运行数据和本地长期记忆存储，可以保留用于加快下次运行或保留本地记忆；不要提交。

## 联网搜索

- 搜索入口在 `qq_group_chatter/services/web_search.py`；插件层在普通聊天前拦截搜索触发词。
- 只使用低风险中文触发词：`搜一下`、`搜索`、`查一下`。不要新增 slash 命令，避免和 QQ/NoneBot 命令体系或风控行为混在一起。
- 默认搜索 provider 是 Tavily；`WEB_SEARCH_ENABLED=true` 时需要 `TAVILY_API_KEY`，否则搜索服务初始化会失败。
- 默认 Tavily 参数：`WEB_SEARCH_DEPTH=basic`、`WEB_SEARCH_MAX_RESULTS=3`、`WEB_SEARCH_INCLUDE_RAW_CONTENT=markdown`、`WEB_SEARCH_INCLUDE_ANSWER=false`、`WEB_SEARCH_TIMEOUT_SECONDS=8`。
- 搜索回答由 `SearchAnswerAgent` 基于网页 `raw_content` 生成，不直接转发 Tavily 的 `answer`，也不默认暴露 URL。
- 搜索回答模型默认 `WEB_SEARCH_ANSWER_MODEL=deepseek-v4-pro`、`WEB_SEARCH_ANSWER_THINKING=enabled`；不要重新加默认 `max_tokens` 上限，除非用户明确要求。
- `WEB_SEARCH_INCLUDE_URLS=false` 时 prompt 和回复都不应包含 URL；如需展示 URL，必须显式打开该环境变量。
- 每条来源正文默认按 `WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT=3000` 截断；prompt 里看到的是网页正文片段，不保证是完整页面。
- 搜索上下文不写入长期记忆；搜索回复只在发送成功后作为 assistant 回复进入短期记忆。

## 可观测性

- 可观测性在 `qq_group_chatter/observability.py`。
- 保持结构化日志和 Prometheus 风格指标：消息处理数量/结果、端到端回复耗时、chat agent / memory planner LLM 耗时、Mem0 search/add 耗时、长期记忆队列长度、候选 add/skip/error、duplicate skip、stage + error_type 错误计数。
- 不要在日志中输出原始 QQ 号、API key 或敏感消息内容。

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

- 不要把长期记忆 ingestion 延后到生成回复之后；收到有效用户消息后，应在读取短期记忆和查询长期记忆后，携带本轮长期记忆快照投递。
- 不要把 assistant 回复用于长期记忆 planner。
- 不要提交 `.env`、`.mem0/`、`*.egg-info/`、缓存目录或 API key。
- 修改默认模型、记忆 scope、持久化策略前，先确认用户是否真的改变了需求。
