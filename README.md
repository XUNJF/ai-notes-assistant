# 个人笔记助手（RAG + 多智能体）

一个能在我自己的学习笔记里检索并回答问题的 AI 助手。**多智能体协作 + 检索增强生成（RAG）+ 流式输出**，Docker 部署在云服务器上，可公网访问。

**在线演示**：`http://1.14.71.104:8000/?key=<访问口令>` （需要口令，请私下索取）

---

## 解决的问题

软件工程专业背景，转向 AI 应用开发方向。几个月学下来积累了 40 多天的技术笔记，散落在 20 多个 Markdown 文件里。

想查「我之前是怎么理解 RAG 的」时，只能靠记忆和全文搜索——而全文搜索不认语义：我问「怎么让页面自动滚到底部」，它找不到那篇讲 `scrollTo` 的笔记，因为字面上没有重合。

所以我做了这个东西：把笔记变成可语义检索的知识库，用自然语言直接问。

---

## 功能

- **语义检索**：基于 `BAAI/bge-m3`（1024 维）向量，检索笔记内容而非关键词匹配
- **多智能体协作**：主管（supervisor）根据问题类型，派研究员查笔记 / 派计算员算数 / 派撰稿人直接回答
- **流式输出**：答案逐字返回，同时把「主管派了谁」的决策过程实时显示在界面上
- **精确计算**：算术题交给 Python 算，模型只负责把中文题翻译成算式，避免大模型算错
- **多轮记忆**：基于 LangGraph checkpointer 持久化到 SQLite，关掉页面再打开还能接着聊
- **历史压缩**：对话变长时自动把旧消息压成摘要，控制 context 长度
- **会话历史**：侧边抽屉列出历史会话，点开可回看并继续该会话
- **访问控制**：轻量口令中间件，防止公开部署后被滥用 API 额度

---

## 技术架构

```
START
  ↓
compress（历史压缩：超过阈值就把旧消息压成摘要）
  ↓
receive（接收新问题，清空上一轮的中间状态）
  ↓
supervisor（主管：根据当前进度决定下一步）
  ├── research → researcher（研究员：向量检索笔记，只交资料不写答案）
  ├── calc     → calculator（计算员：模型转算式 → AST 白名单求值 → Python 计算）
  └── write    → writer（撰稿人：拿到资料/计算结果后写答案）
                        ↓
                       END
```

**每个专员完成后回到 supervisor**，由主管判断信息是否足够，不够就继续派别人。

### 技术栈

| 层 | 用了什么 |
|---|---|
| **LLM** | DeepSeek（`deepseek-flash`） |
| **Embedding** | 硅基流动 `BAAI/bge-m3`（1024 维） |
| **Agent 编排** | LangGraph（StateGraph + 条件边 + 循环） |
| **向量数据库** | ChromaDB（cosine 空间，242 条向量） |
| **对话持久化** | SQLite（LangGraph `SqliteSaver`） |
| **后端** | FastAPI + `StreamingResponse`（NDJSON 流式） |
| **前端** | 原生 HTML/CSS/JS（`fetch` + `getReader()` 手工解析流） |
| **部署** | Docker + 阿里云轻量应用服务器（Ubuntu 22.04） |

---

## 几个关键的技术决策

### 1. 为什么用多智能体，而不是一个大提示词

一个模型兼做「判断该不该查资料」和「写答案」时，容易在该查的时候不查、或者查了不用。拆成三个角色后，每个角色的提示词可以高度专一：

- **主管**只回答一个词（`research` / `calc` / `write`），不用它写任何正文
- **研究员**只交资料，不写答案——避免它在检索阶段就开始编
- **撰稿人**只根据已有资料写，没有资料就走「闲聊」分支

代价是每次请求要多调几次模型，延迟高一些。

### 2. 计算交给 Python，不交给模型

模型负责把「8472加3951再乘27减1364」翻译成 `(8472+3951)*27-1364`，然后由 `safe_eval()` 用 AST 白名单求值：

```python
# 只放行三种节点，其余一律 raise
allowed = (ast.Constant, ast.BinOp, ast.UnaryOp)
```

**为什么不用 `eval()`**：`eval` 会把字符串当代码执行，`__import__('os').system(...)` 这类输入可以直接拿到 shell。白名单的好处是「列不全也只是多拒绝，不会漏放」。

### 3. 向量检索的距离度量必须显式指定

Chroma 默认用 **L2 平方距离**，不是余弦。而 `bge-m3` 输出的向量已经归一化（长度为 1），两者在这个前提下单调等价——**排序结果一样，但绝对分数差 4 倍**。

如果按默认配置却用 `1 - distance` 当相似度，数值就是错的（设阈值会全部失效）。所以建集合时必须显式写：

```python
metadata={"hnsw:space": "cosine"}
```

这个参数**只在创建集合时生效**，建错了只能删库重建。

### 4. 摘要会丢信息，要用提示词约束

对话压缩是有损的。实测同一段输入跑 6 次，用「保留关键信息」这种笼统的提示词，有 1 次把用户的姓名压掉了（约 17%）。改成显式列出必须保留的项之后，丢 0 次：

```
【必须保留】
1. 用户的个人信息：姓名、身份、偏好、习惯、约定过的事（一个字都不能丢）
2. 用户问了什么、结论是什么
3. 关键概念
```

### 5. 索引可增量更新，不必全量重建

最早用 `index.json` 存全部向量，每次加一篇笔记都要重新切分、重新调 embedding 接口。换成 Chroma 后：

```python
col.upsert(ids=[...], documents=[...], embeddings=[...], metadatas=[...])
```

—— 注意是 `upsert` 不是 `add`：`add` 遇到已存在的 id 会**静默丢弃新数据**，不报错。

---

## 本地运行

```bash
pip install -r requirements.txt
```

需要两个密钥，通过环境变量传（也可以用 `key.txt` / `key_sf.txt` 文件，代码会回退读取）：

```bash
# Linux / macOS
export DEEPSEEK_API_KEY=sk-xxx
export SILICONFLOW_API_KEY=sk-yyy

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-xxx"
$env:SILICONFLOW_API_KEY="sk-yyy"
```

启动：

```bash
python -m uvicorn server:app --reload
```

打开 `http://127.0.0.1:8000`

**关于索引**：仓库里已经包含构建好的 `chroma_db/`（242 条向量），直接就能用。`build_chroma.py` 是重建索引用的，它需要原始笔记文件——那些是个人的，不在本仓库里，所以**这个脚本在本仓库无法直接运行**，保留它只是为了说明索引用什么方式构建。

---

## 部署

```bash
docker build -t notes-agent:v1 .
docker run -d --name notes --restart unless-stopped \
  -p 8000:8000 --env-file .env notes-agent:v1
```

`--restart unless-stopped` 是必须的——否则服务器一重启服务就没了。

密钥通过 `.env` 注入，不进代码库（记得 `chmod 600 .env`）：

```
DEEPSEEK_API_KEY=sk-xxx
SILICONFLOW_API_KEY=sk-yyy
APP_ACCESS_KEY=你的访问口令
```

**国内服务器的一个坑**：`docker.io` 通常不可达，`docker build` 拉基础镜像会超时。把 Dockerfile 第一行换成国内镜像源：

```dockerfile
FROM docker.m.daocloud.io/library/python:3.12-slim
```

---

## 已知限制

**说清楚比藏着好：**

- **知识库只到第 18 天**。第 19 天之后的笔记没写进索引，问后期内容它会回答「资料里没有相关内容」
- **单机、无并发优化**。SQLite 和 Chroma 都是本地文件，不适合多实例横向扩展
- **摘要会累积变长**，长期使用需要做二级压缩
- **没有用户体系**，只有一层口令，不区分用户，历史记录是全局的
- **中文问句的「句式撞车」问题**：笔记标题大量使用「XXX 是什么」的格式，导致任何「X 是什么」的问句都会拿到偏高的相似度分数。目前通过路由层（不让闲聊/问答类问题触发检索）规避，没有从检索层解决
- **前端是原生 JS**，没有组件化，再复杂就需要上框架

---

## 项目结构

```
.
├── multi_agent.py        # LangGraph 多智能体图（主管+三个专员）
├── rag_agent.py          # 单 Agent 版本（较早的实现，保留作对照）
├── search_chroma.py      # 向量检索：query + 拼装上下文
├── build_chroma.py       # 建索引：切分笔记 → 转向量 → 写入 Chroma
├── db.py                 # 会话历史存取（SQLite，参数化查询）
├── server.py             # FastAPI 接口 + 访问口令中间件
├── index.html            # 前端（流式解析 + 步骤显示 + 历史抽屉）
├── Dockerfile            # 容器化
└── chroma_db/            # 向量库（已纳入版本管理，3MB）
```

**接口：**

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/` | 前端页面 |
| `POST` | `/chat/stream` | 流式对话（NDJSON，含 `step` / `text` / `meta` 三种消息） |
| `GET` | `/sessions` | 会话列表（按 `thread_id` 分组） |
| `GET` | `/sessions/{thread_id}` | 某个会话的全部消息 |
| `GET` | `/history` | 最近 N 条问答记录 |

---

## 说明

这个项目在我 21 周转向 AI 应用开发的学习过程中写成，最初是学习仓库 `python-practice` 里的一个目录，代码是每天实际用、边用边改出来的，不是教程示例。后来为了便于展示单独抽成了这个仓库（保留了提交历史）。
