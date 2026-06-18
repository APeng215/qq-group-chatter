# AGENTS.md

## 项目硬约束

- 本项目是 NoneBot2 + OneBot v11 的 QQ 群聊/私聊机器人。
- Agent 调用走普通 LangChain/LLM 封装；不要引入 LangGraph、checkpointer，或把 Mem0 暴露成 Agent tool。
- 默认聊天模型是 `deepseek-v4-pro`，`thinking=enabled`。可通过 `DEEPSEEK_THINKING=false` 关闭；除非用户明确要求，不要改模型。
- 密钥只允许来自环境变量或本地 `.env`；不要写进代码、测试快照、日志或提交内容。

## 入口和链路

- 启动入口：`bot.py`。
- 生产装配必须用 `create_default_application()`，让长期记忆 worker 随 `application.start()` / `application.stop()` 管理生命周期。
- `create_default_orchestrator()` 只用于测试或显式自定义装配；直接用它不会自动启动长期记忆 worker。
- NoneBot 插件入口：`qq_group_chatter/plugins/chat.py`。
- 主编排：`qq_group_chatter/orchestrator.py`。
- 联网搜索服务：`qq_group_chatter/services/web_search.py`，由 `create_default_application()` 装配后注入 `ChatOrchestrator`。
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

联网搜索是普通聊天链路里的补充资料步骤：`ChatAgent` 先用 DeepSeek JSON Output 返回 `reply` 或 `web_search` 决策；如果是 `web_search`，`ChatOrchestrator` 先发送模型生成的 `notice`，再调用 `WebSearchService.search_sources()` 获取 Tavily 来源，最后回到 `ChatAgent.generate_grounded_search_reply()` 用神奈口吻回答。

长期记忆 planner 的事实来源仍然只看用户消息；assistant 回复只允许作为确认/拒绝/限定上下文，不能作为独立记忆事实来源。

## 记忆边界

- 短期记忆：`qq_group_chatter/services/short_term_memory.py`，按 `conversation_id` 保存最近消息；默认每会话在内存和 `.mem0/short-term-memory.jsonl` 持久化文件里保留 300 条，prompt 默认只读最近 30 条。
- 短期主键：
  - 群聊：`qq_group:{group_id}`
  - 私聊：`qq_private:{user_id}`
- 长期记忆：`qq_group_chatter/services/long_term_memory.py`，使用 Mem0 真实存储；默认不能静默退回 Noop。
- `NoopMem0Client` 只允许在测试或显式注入时使用。
- 默认 Mem0 初始化失败应抛 `MemoryConfigurationError`，不要让机器人假装长期记忆可用。
- 长期记忆写入走后台 worker，不能阻塞用户回复；写入前必须做候选校验和重复抑制。
- 长期记忆 scope 只保留 `user` 和 `conversation`；不要引入 group-user 组合记忆，除非用户重新提出。
- 长期 ID：
  - 用户：`qq_user:{conversation_id}:{user_id}`
  - 会话：`qq_conversation:{conversation_id}`
- 长期记忆 planner：`qq_group_chatter/services/long_term_memory_planner.py`，默认 `deepseek-v4-pro` + `thinking=enabled`；可通过 `DEEPSEEK_THINKING=false` 关闭。一次调用内完成提取判断和 add/update/skip 决策，不直接给聊天 Agent。
- 长期记忆写入 Mem0 时使用 `infer=False`；记忆提取、合并和跳过决策由项目自己的 `LongTermMemoryPlanner` 负责，不依赖 Mem0 内部 LLM 自动推理。
- Mem0 `search()` / `get_all()` 的 `filters` 必须至少包含 `user_id`、`agent_id`、`run_id` 之一；不要只用 `conversation_id` 调 Mem0。
- 查询当前用户和当前会话长期记忆时用 `filters={"user_id": ...}`；查询同会话全局相关记忆时用 `filters={"user_id": "*", "conversation_id": context.conversation_id}`，再由项目代码过滤 owner 和去重。
- 不要提取手机号、密码、token、api key、地址等敏感内容。

## DeepSeek 和 Mem0

- `DEEPSEEK_API_KEY` 必填：聊天 Agent 和 Mem0 内部 LLM 都需要。
- `DEEPSEEK_THINKING` 可选：默认 `true`；设为 `false` 可关闭项目自己发起的 DeepSeek 调用的 thinking；env 示例统一用 `true` / `false`，解析层兼容 `enabled` / `disabled`。
- `MEM0_FASTEMBED_MODEL` 可选：默认 `BAAI/bge-small-zh-v1.5`。
- 当前默认：
  - ChatAgent：`deepseek-v4-pro` + `thinking=enabled`
  - LongTermMemoryPlanner：`deepseek-v4-pro` + `thinking=enabled`
  - Mem0 内部 LLM provider：`deepseek`，模型 `deepseek-v4-pro`
  - Mem0 embedding：本地 `fastembed`
  - 本地向量库：`.mem0/qdrant`
- `.env` 本地使用且已 ignore；提交示例只改 `.env.example`。
- `.mem0/` 是运行数据、本地长期记忆存储和短期记忆持久化存储，可以保留用于加快下次运行或保留本地记忆；不要提交。

## 联网搜索

- 搜索入口在 `qq_group_chatter/services/web_search.py`；插件层不再拦截搜索触发词，所有消息都先进入普通聊天链路。
- 即使用户说“搜一下/搜索/查一下”，也只是普通聊天内容；是否联网由 `ChatAgent` 的结构化 `web_search` 决策决定。不要重新新增 slash 命令或插件级硬搜索分支。
- 默认搜索 provider 是 Tavily；`WEB_SEARCH_ENABLED=true` 时需要 `TAVILY_API_KEY`，否则搜索服务初始化会失败。
- 默认 Tavily 参数：`WEB_SEARCH_DEPTH=basic`、`WEB_SEARCH_MAX_RESULTS=3`、`WEB_SEARCH_INCLUDE_RAW_CONTENT=markdown`、`WEB_SEARCH_INCLUDE_ANSWER=false`、`WEB_SEARCH_TIMEOUT_SECONDS=8`。
- Tavily 只提供网页来源和 `raw_content`；自动搜索最终回复由 `ChatAgent` 基于 `chat_search_grounded.txt` 生成。
- 自动搜索默认不向 prompt 暴露 URL；如需展示 URL，应先确认需求并更新 grounded prompt 与来源格式化逻辑。
- 每条来源正文默认按 `WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT=5000` 截断；prompt 里看到的是网页正文片段，不保证是完整页面。
- 搜索上下文不写入长期记忆；搜索提示不写入短期记忆；最终搜索增强回复只在发送成功后作为 assistant 回复进入短期记忆。

## 可观测性

- 可观测性在 `qq_group_chatter/observability.py`。
- 保持结构化日志和 Prometheus 风格指标：消息处理数量/结果、端到端回复耗时、chat agent / memory planner LLM 耗时、Mem0 search/add 耗时、长期记忆队列长度、候选 add/skip/error、duplicate skip、stage + error_type 错误计数。
- 不要在日志中输出原始 QQ 号、API key 或敏感消息内容。
- LLM 调用 trace 在 `qq_group_chatter/llm_tracing.py`，默认写入 `logs/llm-traces.jsonl`，用于 `/memory` 里的 LLM 控制台。
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

- 不要把长期记忆 ingestion 延后到生成回复之后；收到有效用户消息后，应在读取短期记忆和查询长期记忆后，携带本轮长期记忆快照投递。
- 不要把 assistant 回复用于长期记忆 planner。
- 不要提交 `.env`、`.mem0/`、`logs/llm-traces.jsonl`、`*.egg-info/`、缓存目录或 API key。
- 修改默认模型、记忆 scope、持久化策略前，先确认用户是否真的改变了需求。
