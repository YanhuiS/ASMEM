# ASMEM 核心设计与实现

> 代码：`src/asmem/`
> 文档：[core.md](core.md) · [add.md](add.md) · [search.md](search.md)

---

## 1. 系统定位

ASMEM 是一个**长期记忆检索后端**，不是端到端问答模型：

- `POST /add`：把对话、长文档、表格、三元组、agent 轨迹归一化成同一种记录，落成
  `memory.jsonl`。**不摘要、不抽 atom、不改写事实**。文本/代码赛道不调用生成式模型；
  多模态赛道仅在图片本身没有 caption/伴随文本时，可选调用一次 vision caption（见 3.5）。
- `POST /search`：先做完全确定性的多通道粗筛，再由 verifier 决定两条路：
  **通过 → 直接用粗筛证据 answer；不通过 → postagent loop 补证据**。
  它不生成最终答案。

三条边界：

1. **Add 只做格式层。** 原始记录始终可 grep、可引用；多模态图片只派生 caption 与
   `[image: <id>]` 引用，不派生事实。
2. **粗筛不调用 LLM。** 同一份 memory、同一个 query 永远得到同一批候选；dense 可选。
3. **postagent 只做证据，不做答案。** 它留下的 `evidence.jsonl` 就是下游 answer
   看到的全部证据。

---

## 2. 端到端流水线

```text
messages / document / table / triple / trace
  │
  ├─ Add（纯 CPU）
  │    适配 → token 切分 → 统一记录 → 稳定 id → 状态标注
  ▼
memory.jsonl + state.jsonl
  │
  ├─ 粗筛（无 LLM，可缓存）
  │    tokenize / phrase / time / route / intent
  │    → BM25 / Dense / Exact / Time / State / Session / Option
  │    → 加权 RRF → 来源配额 → 近重复 → 邻居扩展
  ▼
inputs/candidates.jsonl + inputs/coarse.json
  │
  ├─ 证据种子
  │    粗筛候选按字符预算写进 evidence.jsonl（origin=coarse, seeded=true）
  ▼
evidence.jsonl
  │
  ├─ verifier（1 次模型调用 + 确定性检查）
  │    ├─ answerable=true  → 直接 answer
  │    └─ answerable=false → postagent loop
  │
  ├─ postagent loop（最多 3 次模型调用）
  │    search(重写 query) → take_note / auto-note → finish
  │    notes 追加到 evidence.jsonl；粗筛 seed 不可删除
  ▼
evidence.jsonl
  │
  ├─ 后处理
  │    source 校验 → parent/chunk/window 解析 → 强制邻接扩展
  │    → 去重 → answer 字符预算裁剪
  ▼
answer-visible evidence
```

---

## 3. 核心设计

### 3.1 Add：确定性格式层

常见做法是在写入时让 LLM 摘要、抽 atom、改写事实。ASMEM 不做：

- 摘要会丢掉 UPDATE、否定、条件和时间锚点；
- atom 化会把指代从上下文中剥离；
- 改写后无法再 grep 原话，也无法把证据追溯到原始消息。

Add 只做四件事：**恢复结构、按 token 切分、分配稳定 id、正则识别显式状态**。
详见 [add.md](add.md)。

关键不变量：

- `text` 是原文（或原文的结构化渲染），不是摘要；
- `id = s{session_index}_{seq:05d}` 稳定且可引用；
- 切分只影响检索单元，`parent_id / part / parts` 保留原始单元；
- `chunk_version` 进入 embedding 身份，换切分策略不会混用旧向量。

### 3.2 粗筛：多通道加权 RRF

粗筛解决“召回和多样性”，不是“精确回答”。七个通道各自给出 ranked list：
`bm25 / dense / exact / time / state / session / option`。

融合用加权 RRF：`weight / (rrf_k + rank)`，不直接相加不同量纲的分数。
随后做来源配额、近重复合并、邻居扩展。详见 [search.md](search.md)。

粗筛是确定性的，**可以单独运行并缓存**：

```bash
python scripts/run_bench.py --config configs/bench_qwen3_gate_loop.yaml \
    --coarse-only --run-name qwen3_coarse_cache --overwrite
# 之后 experiment.reuse_coarse_from: qwen3_coarse_cache
```

### 3.3 Verifier：只判断“具体值在不在”

`gate_verify` 让模型说出 asked slot、写一个短答案、quote 支持文本；不要求完整 claim、
不要求 exact quote、不要求列表完整。

之后做确定性检查：quote 是否在 evidence、答案形状（日期/数字/名字/duration）、
问题专名是否出现在 cited record、first-person 是否来自 user、answer 的日期/数字是否
在 quote 里。单值/排序类题可再做一次 answer↔quote 一致性检查
（`gate_check_consistency`）。

两个出口：

- 通过 → 直接 answer；
- 不通过 → 把 missing slots 交给 postagent loop。

### 3.4 Postagent：有界证据的有界 loop

每轮都是全新的有界 prompt，不累积聊天历史：

```text
QUESTION
MISSING SLOTS
CURRENT EVIDENCE（coarse + notes，top 16）
LAST SEARCH RESULTS（top 5，1-based）
QUERIES ALREADY RUN
→ 一个 JSON action
```

动作：`search` / `take_note` / `finish`。

- `search` 用重写 query 在**已构建的粗筛 index** 上检索；排除已见 id；支持日期过滤；
- `take_note` 把 last search 的指定结果按 raw record 追加到 `evidence.jsonl`；
- 每次 search 自动保存 top `loop_auto_note` 条，避免搜到但没 note；
- loop 结束没有 note 时，自动保存 last search top 2；
- notes 与粗筛 seed 一起走 `postprocess_evidence`/`fit_evidence`，按 source 和
  normalized text 合并去重；
- 最多 `loop_max_steps: 3`；加 verifier 1 次，仍低于 6。

**多跳/聚合的简单增强。** 总额、计数、多来源题最常见的失败是少一跳“事件 → 城市/日期 → 收据/预订”：

- loop 的 rewrite 规则第 7 条：总额/计数题先从 evidence 读出事件的城市和日期，再搜
  receipts / invoices / reservations / bookings / confirmation emails。
- `loop_extra_step_intents: [aggregate, aggregation, multi_hop]` 只给这些 intent 多一轮
  loop（普通题仍是 3 轮）。
- 确定性 bridge：如果 evidence 里有 `[city]` / `[location]` 标签，loop 结束后直接跑一次
  `<city> <month> <year> receipt invoice reservation confirmation`，自动保存
  top 6。ATM-Bench 的 BMVC 2024 住宿题因此能召回两封 gold 酒店确认邮件
  （655.59 + 144.15 GBP）。这个 query 是通用的收据/预订桥，不绑定“酒店”。
- 这套增强只依赖文本和已有的 location 标签，不增加新的模型或外部服务。

### 3.5 多模态：caption 检索，不做 image embedding

多模态赛道只在**报文形态**上和文本赛道不同：`messages[].content` 与 `query` 是有序的
`ContentPart[]`。ASMEM 不训练也不调用图像 encoder，检索仍然只认文本：

```text
ContentPart[]（text + inline base64 image）
  │
  ├─ Add：图片解码 → <user>/images/<sha256>.<ext>（内容寻址）
  │       caption = 显式 caption / 同条 text / 可选 vision caption（按图 hash 缓存）
  │       memory.text = 原文 + "[caption] ..." + "[image: <short id>]"
  │       meta.media = {id, sha256, mime, bytes, path}
  ▼
文本 memory（BM25 / dense 文本向量 / exact / time / state 全部照常）
  │
  ├─ Search：query 里的图片同样先 caption，再进入粗筛
  │       gate 最多看 1 张 query 图；memory 图不进 postagent
  ▼
build_search_items：text ContentPart 在前，命中的 memory 图按预算追加
```

关键约束：

- **base64 永不进入 memory / evidence / prompt。** 单图最大 10 MiB，整条 Add 30 MiB；
  memory 只保留短 id 和 caption（默认 ≤300 字符）。
- **没有 image embedding。** caption 是唯一可检索的图片表示；dense 编码的也是 caption。
- **postagent 只在必要时看多模态。** 文本 query 完全不附图；多模态 query 只在 gate 和
  evidence loop 第一轮附 1 张查询图。memory 图不进入内部 agent prompt，而是随 Search
  响应返回给下游 answer。
- **响应预算。** Search 返回的图片同样受 10 MiB/图、30 MiB/响应和
  `max_response_images` 限制；放不下的图直接跳过，不截断响应。
- caption 端点失败不影响 Add：降级为同条 text / `[image: <id>]`，`/add` 仍返回
  `success=true`。
- **响应格式按赛道，不按单次 query 形态。** 竞赛契约规定多模态赛道的
  `data[].content` 必须是有序 `ContentPart[]`，即使这道题只有文本。旧实现只在
  `query` 本身是数组时才返回 parts，文本 query 会返回字符串，属于多模态赛道的
  契约不匹配。`multimodal.response_parts: always`（api_mm.yaml）让多模态部署始终
  返回 parts；文本/代码赛道保持 `auto`（字符串）。
- **查询图也内容寻址。** Search 的 query 图会写入同一个 `ImageStore`，caption 走
  按 sha256 的缓存。同一张图在多道题/多个用户之间只存一份物理文件，也只会调用
  一次 vision caption。

#### 3.5.0 多模态存储去重

- **图片**：新写入只落一份到 `<store>/shared/images/<sha[:2]>/<sha>.<ext>`，不再
  在 `<user>/images/` 建副本或 hardlink。`meta.media.path` 仍是相对路径，
  `ImageStore.resolve()` 先查用户目录（兼容旧 store），再查 shared 目录。因此
  同一张图跨用户、跨 Add/Search 只有一个物理文件、一个路径。
- **caption**：`<store>/shared/images/captions/<sha>.txt` 是唯一副本；旧 store 的
  用户目录 caption 仍可读（shared 优先），查询图复用同一缓存。
- **memory 元数据**：`meta.media` 只重复很小的 `{id, sha256, mime, bytes, path}`，
  不重复图片字节。
- **runs 快照**：`runs/<id>/inputs/memory.jsonl` 仍是 hardlink，不复制语料。
- **旧 store 迁移**：`python scripts/dedupe_store_images.py --root data/api_store_mm --confirm`
  把用户目录里的重复图片入口移到/合并到 `shared/images`；默认 dry-run，会校验
  SHA-256。若 6006 还是旧进程（只 resolve 用户目录），先重启再运行，否则旧进程会
  暂时看不到已经移到 shared 的图片。

#### 3.5.0.1 原始请求日志（`api/requests`）

平台契约问题靠 summary 无法定位，因此 API 把**原始请求体**按内容哈希保存一次：

```text
<user>/requests/
  add/<sha[:2]>/<sha>.json          # request_id / session_id / messages
  search/<sha[:2]>/<sha>.json       # query / options / top_k
  search/responses/<req sha>/...    # 实际返回的 data[]
  invalid/<sha[:2]>/<sha>.json      # 422 的原始报文（契约排查）
```

- 文件名是**外部化后 payload** 的 SHA-256；同一请求重试只写一次。
- 内联 `data:image/...;base64,...` 被替换成 `asmem-image:<sha256>`，图片字节只存在于
  内容寻址的 ImageStore；日志里**没有 base64**。
- 响应按 `(request sha, response sha)` 写一次；`request_journal_responses: false`
  可关闭。
- 写日志失败不会让 `/add` 或 `/search` 失败；`/health.last_error` 只记录真正影响
  服务的问题。
- 日志是审计产物，不参与检索；30 天保留要求由部署方按需清理。

#### 3.5.1 Mem-Gallery：带 image id 的多 session 对话记忆

Mem-Gallery 的图片本身带人工 caption 和稳定 id（`D1:IMG_003`），因此不需要 vision
caption。`scripts/build_bench_datasets.py::convert_mem_gallery` 把它转成三类可检索记录：

```text
user turn       [round D1:3] <用户问题>
assistant turn  [round D1:3] <回复> [image: D1:IMG_003]
image record    [image: D1:IMG_003]  [caption] <人工 caption>  [round D1:3]
```

每个 image 是**独立检索单元**，BM25 / dense / exact / time 都照常工作。带 query image
的 QA 把数据集自带的 `image_caption` 拼进 query，等价于 `query_content` 对 inline
image 生成的 caption。答案侧按题型使用确定性路径，而不是让 8B 模型自由发挥：

- **VS / `image_id` 题**：让模型只选 evidence 编号（不写 id），再映射回 image id；
  题面给日期时先按日期过滤候选，题面给数量时截断到该数量；没有 query image 且
  没给数量时，优先保留候选最多的 session，避免把全库相似图都选进来。
- **CD 冲突题**：claim-vs-memory 题先问 memory 是 `affirm / deny / not_mentioned`，
  `deny` 即冲突；"Is there a contradiction between X and Y" 这类内部一致性问题走另一条
  判断。`Has X ever ...?` 走存在性检查。
- **TR 时间题**：`When ...?` 只在 `[resolved: ...]` 候选上做一次小选择，避免把 session
  date 当事件日期；`before or after` 题先定位两条记录，再按 `[round Dx:y]` 比较顺序。
- **VR / TTL / KR 文本题**：如果答案是纯 image id（模型把 id 当答案），用 query caption
  + evidence 文本重答一次，强制输出名字 / 地点 / 数量等文本。
- **AR 拒答题**：不按题型强制拒答，仍由普通 answer prompt 判断；按 label 直接输出
  `Not mentioned.` 会把 AR 分数虚高。

**Answer 侧真正看原图。** Qwen3-8B 是纯文本模型，只能验证 caption 路线；4o-mini 是
多模态模型，`configs/bench_mem_gallery_4omini.yaml` 因此把 answer 阶段的输入扩成：

```text
question + evidence 文本（含 [image: id] 标记）
  + query image（high detail）
  + 最终 evidence 里引用到的 memory images（low detail，≤ max_images）
```

图片按 `[attached image: <id>]` 标注后插在 user message 里，和 evidence 的
`[image: <id>]` 一一对应；resize 到长边 512、磁盘缓存，避免每道题重复编码。搜索/verifier
仍是 caption 文本，这样图片只增加 answer 的视觉证据，不改变粗筛的确定性。

`configs/bench_mem_gallery_4omini_caption.yaml` 把 `answer.media.enabled` 关掉，作为
caption-only 对照；A/B 的差值就是“answer 能不能看原图”的贡献。

`Judge` 对 Mem-Gallery 使用 `mem_gallery_score`：image id 题按集合 F1，AR 看是否输出
`Not mentioned`，CD yes/no 看极性；其余开放题仍用 LLM judge。`by_answer_type` 会分别
报告 VS / TTL / VR / CD / AR / TR / FR / MR / KR。

#### 3.5.2 ATM-Bench：邮件 + 照片 + 视频的长期个人记忆

ATM-Bench 是一个人的 ~4 年个人数据：6,742 封邮件、3,759 张照片、533 个视频、
1,013 道 evidence-grounded QA。数据自带图片/视频的 caption、OCR、时间和地点，
所以每个来源都落成一条带稳定 source id 的文本记录：

```text
[email: email202411160004] [summary] ... [detail] ... [time] ...
[image: 20220703_210745]   [caption] ... [tags] ... [location] ... [time] ...
[video: 20220502_172850]   [caption] ... [duration] ... [time] ...
```

转换器按自然月做 add（43 个月），所有 add 完成后才跑 1,013 个 search；benchmark
pipeline 检测到 memory 静态，会把同一用户的 search 并发跑（`search_concurrency`）。
answer 侧和 Mem-Gallery 共用 `answer.media`：`open_end` / `list_recall` 会附上
命中 evidence 里的照片，`number` 题不附图（多为邮件/价格查询）。

ATM-Bench 是 evidence-grounded 的，所以 `scripts/analyze_atm_bench.py` 除了 answer
score，还报告 gold `evidence_ids` 是否出现在 answer-visible evidence 里
（hit-any / full recall / first-hit rank）；这能区分“检索没找到”和“找到了但答错”。

### 3.6 后处理：answer 看到的才是最终证据

`postprocess_evidence`：

- 拒绝不存在的 source id（`["query"]` 除外）；
- 合并重复内容与 source；
- 按记录类型解析 chunk / parent / local window；
- 强制附加同来源邻居，标记 `[context <id>]`；
- 把相对日期解析成绝对日期（`[resolved: ...]`）；
- 按 answer 字符预算裁剪。

---

### 3.6 超时、队列与降级

搜索不会因为一个慢模型调用把请求拖成空结果：

- **硬超时覆盖排队时间。** `search_sem` 的等待在 `asyncio.wait_for` 之内；拿不到 slot
  超过 `search_queue_timeout` 直接走 fallback，而不是排到整个硬超时。
- **模型调用前先落 coarse fallback。** `run_search` 在 gate/loop 之前把 top coarse
  证据写到 API 给的临时文件；硬超时/异常时 API 读取该文件返回，不再固定返回 `data: []`。
- **文件为空时 BM25-only 兜底。** `quick_coarse_fallback` 关闭 dense，只用
  BM25/exact/time/state 生成短候选，不调用生成式或 embedding 模型。
- **dense 超时不再清空候选。** 粗筛 dense 超时会再用 BM25-only 跑一次
  （`_bm25_only_config`），避免 `evidence=0`。
- **embedding 失败可选升级为致命错误。** 默认 `retrieval.dense.fatal_on_error: false`，
  dense 端点故障只记 `coarse.warnings` 并降级为 BM25。设为 `true` 后，embedding
  调用失败（例如百炼欠费返回 `403 AccessDenied.Unpurchased`）会抛 `EmbeddingError`，
  穿过 coarse 的 BM25 兜底：`/search` 返回 503，随后进程发送 SIGTERM 优雅退出
  （兜底 30s 未退出则 `os._exit(1)`），不再静默返回低质量结果。
- **只有带 `request_id` 的请求才在客户端断开后继续运行**；无 `request_id` 的请求被取消，
  立即释放 `search_sem`，避免僵尸任务占满并发。
- **进程级 endpoint breaker**：`llm_global_breaker_failures > 0` 时，端点连续失败会打开
  全局熔断，新请求快速进入 coarse fallback，而不是各自重试。
- **远程端点显式绕过失效代理**：部署环境（例如 AutoDL）常导出 `http_proxy` /
  `https_proxy`，但代理本身可能已经不可用。httpx 默认 `trust_env=true`，会先连接
  代理，失败后报 `ConnectError: All connection attempts failed`——看起来像模型
  端点挂了，实际请求根本没出机器。API 配置用 `trust_env: false` 让 generator、
  captioner、embedding 全部直连；`Captioner` 的 call config 也显式继承该设置。
- **上下文保护**：`model_input_token_cap` 在每次模型调用前做 token 预检；
  端点返回 context overflow 时，`ModelRunner` 会把 prompt 缩到 1/2、1/4 再试一次
  （`context_overflow_retries`），仍失败才降级。

## 4. API

`/add` 和 `/search` 与 benchmark 使用同一套实现：

```text
/add    -> api.store.UserStore.append -> schema.normalize_messages
/search -> search.runner.run_question -> api.server.build_search_items
```

每个 `/add` / `/search` 的原始 JSON 还会按内容哈希写一份到 `<user>/requests/`
（图片只写 `asmem-image:<sha256>`，不写 base64），响应写在 request 旁；写入失败
不影响协议响应。这样平台报契约错误时，可以直接看到它实际发送的 `content` 形态和
我们实际返回的 `data[]`。

`configs/api.yaml` 和 `configs/asmem.yaml` 都是核心配置：

```yaml
postagent_mode: gate_loop
loop_max_steps: 3
loop_top_k: 5
loop_auto_note: 2
loop_note_chars: 1200
loop_evidence_chars: 6000
loop_item_chars: 400
loop_force_intents: []
loop_accept_answer_types: []
loop_extra_step_intents: [aggregate, aggregation, multi_hop]
gate_check_consistency: true
pre_audit: false
# Retrieval (api.yaml uses the local Qwen3-Embedding-4B on :30001)
retrieval:
  coarse:
    candidates: 200
  dense:
    mode: "on"
    model: qwen3-embedding-4b
    base_url: http://127.0.0.1:30001/v1
    dim: 2560
    fatal_on_error: false     # true: embedding 失败即终止服务，不降级为 BM25
max_evidence: 96
max_evidence_chars: 40000
multimodal:
  caption: true
  caption_query: true
  max_image_bytes: 10485760
  max_request_bytes: 31457280
  max_response_bytes: 31457280
  max_response_images: 8
  response_parts: always      # multimodal track: always ContentPart[]
  postagent_images: auto      # auto / never / always
# Raw request audit (content-addressed, images externalized)
request_journal: true
request_journal_responses: true
# Remote endpoints: ignore a possibly-dead http(s)_proxy in the environment.
trust_env: false
# Timeout / queue / fallback
search_queue_timeout: 60
fallback_enabled: true
fallback_timeout: 20
fallback_concurrency: 8
fallback_evidence_limit: 12
llm_global_breaker_failures: 8
llm_global_breaker_cooldown: 30
# Context protection
model_input_token_cap: 24000
model_output_reserve_tokens: 4096
context_overflow_retries: 1
verifier_max_output_tokens: 2048
postagent_max_output_tokens: 4096
```

API 只改 generator/embedding endpoint、`concurrency`、`store_dir` 等部署项；
检索路径与 benchmark 完全一致。`/search` 在模型/端点失败时返回已经写入的粗筛证据，
不返回 5xx。

---

## 5. 结果

全量 2000：

| run | score | judge | rubric |
|---|---:|---:|---:|
| `qwen3_gate_loop_v1` | **0.7170** | 0.7788 | 0.6565 |
| `4omini_gate_loop_v1` | **0.7106** | 0.7740 | 0.6478 |
| `qwen3_b1`（粗筛） | 0.6946 | 0.7490 | 0.6507 |
| `qwen3_b2`（旧 loop） | 0.7185 | 0.7790 | 0.6586 |

214 条验证子集（locomo 富集 + 三个数据集各 50）：

| 方案 | weighted | avg search calls |
|---|---:|---:|
| b1 粗筛 | 0.7061 | 0 |
| b2 loop | 0.7373 | 7.28 |
| fixed pipeline（对比） | 0.7524 | 5.29 |
| **core：verifier + evidence loop** | **0.7637** | **3.12** |

详见 [search.md](search.md)。

### 5.1 Mem-Gallery 400 题验证子集

同一份 memory、同一套粗筛，只比较 answer 侧的多模态路由（20 个 scenario 各取前 20 题）：

| 方案 | score | VS | TTL | VR | CD | TR | AR | FR | MR | KR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 直接 answer（无题型路由） | 0.7239 | 0.543 | 0.619 | 0.556 | 0.360 | 0.686 | 0.962 | 0.875 | 0.931 | 0.750 |
| 题型路由（含已撤回的 AR 强制拒答） | 0.8052 | 0.804 | 0.714 | 0.556 | 0.960 | 0.800 | 0.755 | 0.875 | 0.948 | 0.792 |

主要收益来自 determinism：VS 让模型只选 evidence 编号（不再自由写 id），CD 用
affirm/deny 判定，TR 用 `[resolved]` 候选和 round id 排序。AR 的 0.755 是当时试行的
"按题型强制 Not mentioned" 造成的，已撤回；最终代码恢复普通 answer 路径。

### 5.2 4o-mini + 原图：caption-only vs 多模态 answer

同一份 400 题子集、同一套 4o-mini 搜索/判分，只改 answer 是否看原图：

| 方案 | score | TTL | VS | VR | KR | answer prompt tokens | answer 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| caption-only | 0.7628 | 0.762 | 0.819 | 0.667 | 0.833 | 4.9k | 2.85 |
| 带图 low detail，3 张 | 0.7789 | 0.786 | **0.825** | 0.667 | 0.750 | 13.4k | 5.07 |
| 带图 high detail，5 张 | 约 0.766 | **0.833** | 0.799 | **0.722** | **0.792** | 30.6k | 4.39 |
| **最终：按题型取图 + counting 路径 + AR quote-check** | **0.7995** | **0.881** | 0.814 | **0.750** | **0.792** | 18.7k | 3.88 |

结论与局限：

- **图片的净收益不大（+1.6 分）但方向明确**：TTL 识别类题收益最大；VS 在 low detail
  下略升，high detail 反而因为候选图更多而选错。
- **失败归因不是召回**：20 个图片题失败中 15 个是 `model_or_vision`（gold 图已在
  evidence、且在图片预算内，模型仍选错），3 个 `image_budget`，2 个 `retrieval_miss`。
- **Counting 题需要单独路径**：VR 的 89 个失败里 66 个是 “How many …”。把整段 evidence
  交给模型数会错，改成给一份紧凑的 `id + date + caption + dialogue` 列表、只让它返回
  整数后，TTL/VR 的计数题明显回升（400 题上 VR 0.667 → 0.750，TTL 0.762 → 0.881）。
- 所以最终配置按题型取图：**VS low/3 张，TTL/VR/KR high/5 张**，文本题（FR/MR/TR/CD/AR）
  不附图。这个分型策略来自上表的 400 题消融。
- 成本：带图把 answer prompt 从 4.9k 抬到 13–31k tokens；high detail 的 token 约为
  low 的 3 倍，收益集中在 TTL/VR/KR。最终按题型混合后均值 18.7k、3.88s/题。
- 4o-mini 的 AR 会把“相关事实”当答案，因此 4o-mini 配置启用 `ar_quote_check`：
  先让模型判断 memory 是否**显式**回答，并校验 quote 确实出现在 evidence 里。

### 5.3 全量 1711（4o-mini，最终配置）

| type | n | score |
|---|---:|---:|
| VS | 306 | 0.7719 |
| TTL | 337 | 0.7834 |
| VR | 174 | 0.5678 |
| FR | 219 | 0.7443 |
| MR | 206 | 0.8301 |
| TR | 123 | 0.7642 |
| CD | 81 | 0.7778 |
| AR | 184 | 0.7391 |
| KR | 81 | 0.6111 |
| **overall** | **1711** | **0.7455** |

全量比 400 题子集难（尤其 VR/KR/FR），所以绝对分低于 5.2 的 0.7995。相对上一版
全量（0.7430），counting 路径把 VR 从 0.4931 提到 0.5678，总分为 0.7455；其余题型
在 4o-mini 的 run-to-run 波动范围内。

**局限**（按重要性）：

1. **检索不是主瓶颈。** 图片题剩余失败中 13/17 是 `model_or_vision`（gold 图已在
   evidence、也在图片预算内，模型仍选错），只有 2 个是 `retrieval_miss`。
2. **细粒度视觉匹配仍是上限。** 狗品种、相似地标、服装款式在同 session 多张相似图时
   容易选错；high detail 有帮助但不是决定性的。
3. **4o-mini 的文本推理弱于 Qwen3-8B。** FR/MR/AR/KR 的绝对分低于 Qwen3 在同一
   400 题子集上的表现；4o-mini 的 AR 还会把相关事实当答案（已用 quote-check 缓解）。
   如果目标只是分数而不是单一模型，建议 **文本题用 Qwen3、图片题用 4o-mini** 的混合
   answerer；这是下一步最直接的提升点。
4. **Counting 需要专用路径。** 整段 evidence 直接数会错；紧凑 caption 列表 + 只返回
   整数明显更好，但仍会因 evidence 缺图而少计。
5. **成本。** 最终配置 answer prompt 约 22.5k tokens/题（caption-only 4.9k），
   high-detail 图片是主要成本。

### 5.4 ATM-Bench 全量（4o-mini）

1,013 题，4o-mini + Qwen3-Embedding-4B；`open_end`/`list_recall` 附图，`number` 不附图。

| type | n | score | evidence hit-any | evidence full |
|---|---:|---:|---:|---:|
| list_recall | 139 | 0.4229（id 集合 F1） | 0.863 | 0.791 |
| number | 360 | 0.7500 | 0.931 | 0.889 |
| open_end | 514 | 0.6525 | 0.944 | 0.897 |
| **overall** | **1013** | **0.6557** | **0.928** | **0.880** |

- **检索不是主瓶颈。** gold evidence 的 first-hit rank 中位数是 **1**，hit-any 0.928、
  full 0.880；73 个问题（7.2%）gold id 完全不在最终 evidence 里。
- **`list_recall` 的瓶颈是 id 精度，不是召回。** gold 97/139 是单个 id，但同一个
  burst 里相邻时间戳很多；模型倾向返回整串，id F1 被 precision 拖低。试过“单数问题
  只回 1 个”的规则，离线 F1 从 0.423 降到 0.293，已撤回。
- **`number` 的硬点是多跳。** 例：BMVC 2024 → Glasgow 酒店 → 两封确认邮件求和；
  gold 邮件不含 “BMVC”，240/300 条粗筛候选里都没有，loop 的 query rewrite 也没能
  从 BMVC 推出 Glasgow。
- **`open_end` 主要是 answer 侧。** 例：问“红色草莓甜点”，gold 是 strawberry and
  elderflower mousse，模型答 red cheesecake；图像/描述匹配仍会选错相似记忆。
- 基线（旧 judge 口径、无 id 路径）是 0.6407；id 路径 + id 集合评测把它抬到 0.6557。

详见 [search.md](search.md)。

---

## 6. 对比方法与实现说明

- `fixed pipeline`、旧 `loop`、`compile` 只作为对比/消融，代码保留，配置在
  `configs/ablations/`（不进入核心配置）。
- postagent 采用常见的 retrieve → note → finish 循环；本实现增加了粗筛 index 复用、
  auto-note、coarse+notes 合并去重、确定性 verifier、有界 fresh prompt。
