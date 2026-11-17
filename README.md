# 个人笔记助手（RAG + 多智能体）

一个能在我自己的学习笔记里检索并回答问题的 AI 助手。**多智能体协作 + 检索增强生成（RAG）+ 流式输出**，Docker 部署在云服务器上，可公网访问。

**在线演示**：`http://1.15.79.167:8000/?key=<访问口令>` （需要口令，请私下索取）

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

**国内服务器的一个坑**：`docker.io` 通常不可达，`docker build` 拉基础镜像会超时。所以 Dockerfile 第一行直接指向国内镜像源：

```dockerfile
FROM docker.m.daocloud.io/library/python:3.12-slim
```

（在能访问 Docker Hub 的环境里，换回 `python:3.12-slim` 也一样。）

---
## 评测

`eval.py` 跑 16 道题：3 道闲聊、3 道计算、10 道检索。量三件事 ——
路由是否正确、该查笔记时是否命中、端到端耗时。

| 阶段 | 路由正确率 | 检索 top-5 命中率 | 平均耗时 |
|---|---|---|---|
| 纯向量检索 | 16/16 | 9/10 | 3.5 秒 |
| + BM25 混合检索（RRF 融合） | 16/16 | 9/10 | 3.6 秒 |
| + 停用词过滤 + 按笔记聚合 | 16/16 | 10/10 | 3.7 秒 |

**关于最后一行的 10/10：** 样本只有 10 题，这个数字不构成统计结论。
两处改动的依据是两个结构性问题，不是对测试集调参：

1. **疑问词在 BM25 中无区分度却仍贡献分数。** 「是」出现在 242 个片段中的
   131 个里，但因为 idf 公式取 `log(... + 1)` 恒为正，它照样给每个片段加分。
   实测「变量是什么」的分词结果 `['变量','是','什么']` 中，真正有区分度的
   只有「变量」。
2. **检索粒度与评测粒度不一致。** 检索按「片段」排序，而用户（和评测）关心的是
   「哪份笔记」。一份有很多片段的笔记可以靠多个片段挤满前 5，而真正讲这个知识点
   的笔记只有一个片段，反而进不去。改为先按笔记聚合（每份取最高分），
   再取前 5 份，每份保底出一个片段。

唯一曾失败的用例是「变量是什么」—— 第 11 天的示例代码里恰好有
`print(bot.chat("什么是变量？"))` 出现了四次，字面与问题几乎一致但语义无关。
这是「字面相似 ≠ 语义相关」的典型例子。

## MCP 接入

`notes_mcp.py` 是一个 MCP（Model Context Protocol）server，
把检索能力暴露给任意支持 MCP 的客户端（Claude Desktop、Cursor 等）。

这样 AI 客户端可以**自己判断**「先搜再读」，不需要我提供界面，
也不需要它知道我的接口长什么样 —— 它只看得见「有哪些工具可用」。

| 工具 | 作用 |
|---|---|
| `list_notes` | 列出知识库里的笔记文件 |
| `search_notes` | 混合检索，返回最相关的片段 |
| `read_note` | 读取某份笔记的全文 |

实测：在 Claude Desktop 里问「我笔记里 RAG 是怎么写的」，
它会先 `search_notes` 定位到第 17、18 天，再 `read_note` 读全文，
最后把两天的内容合起来答 —— 这个「先搜再读」的策略不是我写死的，
是它自己根据工具说明决定的。

### 远程部署

除了本地 stdio，还支持 HTTP 模式，带 Bearer token 鉴权：

```bash
MCP_TOKEN=<随机串> python notes_mcp.py --http
curl -H "Authorization: Bearer <token>" https://<域名>/mcp
```

生产环境用 Cloudflare Tunnel 暴露，不需要开服务器公网端口：

```bash
cloudflared tunnel --url http://127.0.0.1:9000
```

**踩到的三个坑：**

1. **`Invalid Host header`** —— MCP 自带 DNS 重绑定防护，只允许 localhost。
   隧道域名不在白名单里会被拒。关掉 `enable_dns_rebinding_protection`
   可以放行，此时真正的防线是 Bearer token。
2. **`MCP_TOKEN` 改了不生效** —— `--env-file` 只在容器启动时读一次，
   改完 `.env` 必须 `docker restart`。
3. **隧道域名每次重启都变** —— quick tunnel 的固有行为。
   想固定域名要用 named tunnel，那需要自有域名。

### 两个必须处理的坑

**1. 工作目录**
`search_chroma` 用相对路径找 `chroma_db` 和 `key_sf.txt`。
而 MCP server 是被客户端启动的，启动目录不可控（可能是 `C:\Windows\System32`）。
所以脚本开头要 `os.chdir(脚本所在目录)`。

**2. stdout 污染**
MCP over stdio 是**用标准输入输出传协议报文**的。
任何库往 stdout 打一行日志，协议就废了，客户端会直接断开。
jieba 首次加载字典时会打日志，必须 `jieba.setLogLevel(logging.ERROR)` 压掉。

这个坑很隐蔽：报错不会说「你打了日志」，而是「连接断开」或者干脆没反应。

### 配置

`claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "notes": {
      "command": "<venv 里的 python.exe 绝对路径>",
      "args": ["<项目目录>/notes_mcp.py"]
    }
  }
}
```

路径里的反斜杠要写成 `\\`（JSON 转义），或者直接用正斜杠。
改完配置要**完全退出**客户端再重开 —— 只关窗口不算。

## 已知限制

**说清楚比藏着好：**

- **知识库只到第 18 天**。第 19 天之后的笔记没写进索引，问后期内容它会回答「资料里没有相关内容」
- **单机、无并发优化**。SQLite 和 Chroma 都是本地文件，不适合多实例横向扩展
- **摘要会累积变长**，长期使用需要做二级压缩
- **没有用户体系**，只有一层口令，不区分用户，历史记录是全局的
- **「句式撞车」只缓解了一半**：示例代码里的问句（如 `bot.chat("什么是变量？")`）
  仍会跟真实提问产生高相似度。目前靠 BM25 混合检索 + 按笔记聚合弱化了影响，
  但没有从根上解决（例如给代码块降权、或索引时剔掉示例代码）
- **前端是原生 JS**，没有组件化，再复杂就需要上框架

---

## 项目结构

```
.
├── multi_agent.py        # LangGraph 多智能体图（主管+三个专员）
├── rag_agent.py          # 单 Agent 版本（较早的实现，保留作演进对照）
├── search_chroma.py      # 混合检索（向量 + BM25，RRF 融合，按笔记聚合）
├── build_chroma.py       # 建索引：切分笔记 → 转向量 → 写入 Chroma
├── db.py                 # 会话历史存取（SQLite，参数化查询）
├── server.py             # FastAPI 接口 + 访问口令中间件
├── index.html            # 前端（流式解析 + 步骤显示 + 历史抽屉）
├── eval.py               # 评测：16 题覆盖三条路由，量命中率与耗时
├── test_regression.py    # 回归测试：四类问句各跑一次，人工核对结果
├── Dockerfile            # 容器化
└── chroma_db/            # 向量库（已纳入版本管理，3MB）
```

**关于 `rag_agent.py`**：它是多智能体之前的单 Agent 版本（判断和回答都在一张图里）。保留它是因为多智能体实际上是它的演进结果，对比着看能说明「为什么要拆角色」。

**关于 `test_regression.py`**：不是单元测试，是**人工回归清单**——跑四句话覆盖四条路径（闲聊 / 检索 / 计算 / 依赖摘要），每次改完压缩或路由相关的代码都跑一遍。之所以没写成 pytest，是因为要确认的是「模型行为」而不是「函数返回值」，断言不好写；但这个习惯本身救过我好几次。

**关于 `eval.py`**：`test_regression.py` 是「人眼看」，`eval.py` 是「机器算」。
前者覆盖路径（有没有走对分支），后者量化效果（命中率、耗时）。
两者都不能替代对方 —— 只看命中率，发现不了「答案对但路由错了」；只看回归清单，
说不清「改完到底好了多少」。

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
