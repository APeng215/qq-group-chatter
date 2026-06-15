# AGENTS.md

## 项目定位

这是一个基于 NoneBot2 + OneBot v11 的 QQ 群聊/私聊机器人项目。Agent 调用走普通 LangChain/LLM 封装，不使用 LangGraph，不使用 checkpointer，不把 Mem0 暴露成 Agent tool。

默认回复模型是 `deepseek-v4-pro`，`thinking=disabled`。这个默认值来自本项目的速度测试和用户选择，除非用户明确要求，不要改成别的模型或开启 thinking。

## 入口与主流程

- 启动入口是 `bot.py`。
- 生产入口必须通过 `create_default_application()` 创建应用对象，让长期记忆 worker 随 `application.start()` / `application.stop()` 管理生命周期。
- `create_default_orchestrator()` 只用于测试或显式自定义装配；直接用它不会自动启动长期记忆后台 worker。
- NoneBot 插件入口是 `qq_group_chatter/plugins/chat.py`。
- 主编排在 `qq_group_chatter/orchestrator.py`。
- 群聊和私聊都先构造成 `ConversationContext`。
- `conversation_id` 是短期记忆主键：
  - 群聊：`qq_group:{group_id}`
  - 私聊：`qq_private:{user_id}`
- 长期记忆 ID：
  - 用户：`qq_user:{user_id}`
  - 会话：`qq_conversation:{conversation_id}`

主链路顺序应保持：

1. 收到消息并构造 `ConversationContext`
2. 过滤空消息等无效输入
3. 写入 `ShortTermMemoryService`
4. 读取短期记忆
5. 查询用户长期记忆和会话长期记忆
6. 携带本轮长期记忆查询快照，投递用户消息到长期记忆后台 ingestion
7. 调用 `ChatAgent` 生成回复
8. 发送回复
9. 把 assistant 回复写入短期记忆

长期记忆 planner 只看用户消息，不要把 assistant 回复喂给长期记忆处理链路。

## 记忆架构约束

短期记忆：

- 实现在 `qq_group_chatter/services/short_term_memory.py`。
- 当前是内存版，按 `conversation_id` 保存最近消息。
- 默认每个会话保存 30 条，prompt 默认读取最近 20 条。
- 程序重启后短期记忆丢失是当前 MVP 可接受行为。

长期记忆：

- 实现在 `qq_group_chatter/services/long_term_memory.py`。
- 使用 Mem0 做真实存储，默认不应静默退回 Noop。
- `NoopMem0Client` 只允许在测试或显式注入时使用。
- 默认 Mem0 初始化失败应抛 `MemoryConfigurationError`，不要让机器人假装长期记忆可用。
- 写入走后台 worker，不能阻塞用户回复。
- 写入前必须做候选校验和重复抑制。
- scope 只保留 `user` 和 `conversation`，不要引入 group-user 组合记忆，除非用户重新提出这个需求。

长期记忆规划：

- 实现在 `qq_group_chatter/services/long_term_memory_planner.py`。
- 默认 planner LLM 用 `deepseek-v4-flash`，`thinking=disabled`。
- planner 一次调用内完成是否提取长期记忆以及 add/update/skip 决策，不直接给聊天 Agent。
- 不要提取手机号、密码、token、api key、地址等敏感内容。

## DeepSeek 与 Mem0 配置

密钥只能来自环境变量或本地 `.env`，不要写进代码、测试快照或提交内容。

`.env` 本地使用，已被 ignore。提交示例只改 `.env.example`。

关键配置：

- DEEPSEEK_API_KEY：必填。聊天 Agent 和 Mem0 内部 LLM 都需要。
- MEM0_FASTEMBED_MODEL：可选。本地 fastembed 模型名，默认 BAAI/bge-small-zh-v1.5。

当前默认：

- ChatAgent：`deepseek-v4-pro` + `thinking=disabled`
- LongTermMemoryPlanner：`deepseek-v4-flash`
- Mem0 内部 LLM provider：`deepseek`
- Mem0 embedding：只使用本地 `fastembed`，推荐 `BAAI/bge-small-zh-v1.5`
- 本地向量库：`.mem0/qdrant`

`.mem0/` 是运行数据，不要提交。

## 可观测性

可观测实现在 `qq_group_chatter/observability.py`。

保持结构化日志和 Prometheus 风格指标：

- 消息处理数量和结果
- 端到端回复耗时
- chat agent / memory planner LLM 耗时
- Mem0 search/add 耗时
- 长期记忆队列长度
- 候选 add/skip/error
- duplicate skip
- stage + error_type 错误计数

不要在日志中输出原始 QQ 号、API key 或敏感消息内容。

## 测试与验证

常用验证命令：

```powershell
python -m pytest -q
```

真实 Mem0 初始化验证需要本地 `.env` 里有 `DEEPSEEK_API_KEY`，且可能需要下载 fastembed 模型：

```powershell
python -c "from qq_group_chatter.app import create_default_mem0_client; c=create_default_mem0_client(); print(type(c).__name__); c.close()"
```

如果真实初始化创建了 `.mem0/`，这是运行数据，应清理且不要提交。

每次在本仓库运行 pytest 后，最终回复或提交前都必须清理 pytest 生成物：`.pytest_cache/`、`tests/.tmp/`、`pytest-cache-files-*` 和 `__pycache__/`。

测试里如果要避免真实网络或真实 Mem0，必须显式注入 `NoopMem0Client` 或 fake client，不要依赖默认工厂自动降级。

## 开发注意事项

- 不要引入 LangGraph/checkpointer。
- 不要把 Mem0 做成 Agent tool。
- 不要把长期记忆 ingestion 延后到生成回复之后；收到有效用户消息后，应在读取短期记忆和查询长期记忆后，携带本轮长期记忆快照投递。
- 不要把 assistant 回复用于长期记忆 planner。
- 不要提交 `.env`、`.mem0/`、`*.egg-info/`、缓存目录或 API key。
- 修改默认模型、记忆 scope、持久化策略前，先确认用户是否真的改变了需求。
