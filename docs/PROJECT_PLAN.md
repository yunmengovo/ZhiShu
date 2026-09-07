# 通用文档型 Agentic KG-RAG：产品需求与实施记录

> 当前状态：下述六个阶段均已落地。最新实现、接口和使用方法以
> [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) 为准。

## 1. 系统定位

本系统不依赖固定数据集或预先存在的知识图谱。用户上传自己的 PDF、DOCX、Markdown、TXT 或 HTML 文档，系统自动完成解析、分块、知识抽取、索引、检索和带来源回答。

首版只围绕一条主链路开发：

```text
上传文档
  -> 解析与标准化
  -> 文本分块
  -> 向量索引 + 知识图谱构建
  -> 用户提问
  -> 向量/图谱混合检索
  -> 证据检查
  -> 生成带引用回答
```

## 2. 用户可以做什么

### 知识库管理

- 创建、重命名和删除知识库。
- 批量上传 PDF、Word、Markdown、TXT、HTML。
- 查看文档处理状态、页数、片段数、实体数和关系数。
- 重新解析、更新或删除文档。
- 同一知识库内跨文档检索和推理。

### 智能问答

- 针对选定知识库提问。
- 查看回答引用的文件、页码、章节和原文片段。
- 查看命中的实体和相关子图。
- 本地证据不足时，由 Agent 判断是否改写问题或使用网络搜索。
- 支持连续对话，但每轮事实回答仍需重新绑定证据。

### 运行观察

- 查看解析、Embedding、建图和问答任务的运行状态。
- 通过 LangSmith 查看 LangGraph 节点、工具调用、耗时、Token 和错误。

## 3. 统一文档模型

无论输入格式是什么，解析后都转成统一结构：

```text
Document
  document_id
  knowledge_base_id
  filename / file_type / checksum
  title / source / created_at
  parse_status / parse_error

Section
  section_id / heading / level / order

Chunk
  chunk_id / document_id / section_id
  text / page_start / page_end / order
  metadata
```

解析器只负责把不同文件转成该结构，后面的分块、建图和索引不关心原文件格式。

## 4. 文件解析策略

| 格式 | 首版解析器 | 说明 |
|---|---|---|
| PDF | PyMuPDF | 提取普通文本、页码和基础元数据 |
| DOCX | python-docx | 提取标题、段落、列表和表格文本 |
| Markdown/TXT | Python原生读取 | 保留标题层级和段落 |
| HTML | BeautifulSoup + markdownify | 去除导航等噪声并转成Markdown |
| 扫描或复杂PDF | MinerU，可选 | 已接入官方批次 API，不作为基础依赖 |

解析失败不能阻塞整个批次；每份文档独立记录错误。

## 5. 分阶段开发

### 阶段一：文档处理

实现上传、格式识别、解析、统一文档结构、分块和处理状态。

验收标准：

- Word和PDF测试文档均能解析。
- 标题、正文、列表、表格和页码信息尽量保留。
- 每个 Chunk 可以追溯到文件及页码或章节。
- 同一文件重复上传可通过校验和识别。

### 阶段二：普通 RAG 基线

实现 Embedding、Qdrant 入库、相似度检索和基于证据生成回答。

验收标准：

- 能回答测试文档中的直接事实问题。
- 回答展示文件名和原文片段。
- 记录召回耗时、生成耗时和 Token 使用量。

这一阶段先证明文档解析与向量检索稳定，不接知识图谱和Agent循环。

### 阶段三：知识图谱增强

从 Chunk 中抽取实体与关系，进行基础名称归一化后写入图存储；关系必须保存来源
`chunk_id`。系统提供内嵌 SQLite 和 Neo4j 两个可切换适配器，开发演示可使用 SQLite，
需要原生图遍历和可视化时切换到 Neo4j。

查询时同时执行：

- Qdrant语义检索；
- SQLite 或 Neo4j 中的实体定位和关系检索；
- 两路证据去重、融合和排序。

验收标准：能够回答跨段落或跨文档关系问题，并展示支撑关系的原文。

### 阶段四：Agentic RAG

LangGraph首版流程：

```text
START
  -> analyze_question
  -> retrieve_vector + retrieve_graph
  -> merge_evidence
  -> check_evidence
       -> 充分: generate_answer
       -> 不充分: rewrite_query -> retrieve
       -> 需要最新信息且允许联网: tavily_search -> merge_evidence
  -> verify_citations
  -> END
```

限制最多两次补充检索。Tavily默认关闭，只有本地证据不足并且问题具有时效性时才启用。

验收标准：能够展示检索路径；证据不足时明确说明；最终引用必须来自本轮 Evidence。

### 阶段五：会话记忆

- 使用 `session_id` 隔离和恢复不同会话；
- 检索前利用历史对话消解当前问题中的指代；
- 使用近似 token 计数控制上下文长度；
- 超过阈值后把较早消息合并为滚动摘要，并保留最近若干条原始消息；
- 摘要模型失败时保留原消息，不影响本轮回答。

### 阶段六：界面和监控

Gradio首版包含三个页面：

- 知识库：上传和管理文档、查看处理状态。
- 智能问答：选择知识库、提问、查看答案和引用。
- 调试视图：查看命中片段、相关子图、检索方式和 LangSmith Trace。

LangSmith记录：

- 每个LangGraph节点输入和输出；
- LLM、Embedding和工具调用；
- 检索数量、分数、重试原因；
- 延迟、Token、异常和当前代码版本标签。

## 6. 核心数据流

### 离线入库

```text
文件 -> Parser -> NormalizedDocument -> Chunker
                                  ├-> Embedding -> Qdrant
                                  └-> Entity/Relation Extraction -> SQLite / Neo4j
```

### 在线问答

```text
问题 + 会话记忆 -> LangGraph -> Contextualizer -> VectorRetriever + GraphRetriever
                -> EvidenceMerger -> EvidenceChecker
                -> AnswerGenerator -> CitationVerifier
```

向量结果、图谱结果和网络结果统一转换为：

```text
Evidence
  evidence_id
  source_type: vector | graph | web
  content
  document_id / chunk_id
  filename / page / section
  score
```

## 7. 首版技术边界

首版使用 LangGraph、OpenAI兼容接口、Qdrant、SQLite/Neo4j 图存储、FastAPI 和 Gradio。
暂时不加入 Deep Agents、多智能体、GraphRAG 社区摘要、本地模型训练和复杂权限系统。

## 8. 实现状态

- [x] PDF、Word、Markdown/MDX、TXT、HTML 统一解析与 Chunk 输出；
- [x] 多知识空间、批量导入、重复检测与文档生命周期；
- [x] Qdrant 向量 RAG 与知识空间范围校验；
- [x] SQLite/Neo4j 图谱构建、关系检索与子图可视化；
- [x] LangGraph 证据检查、查询改写、融合和引用校验；
- [x] 持久会话、256K 预算策略和自动滚动摘要；
- [x] Tavily、MinerU、LangSmith 可选集成及故障降级；
- [x] 后台重建任务、运行状态和完整 Gradio 工作台。
