# Search：粗筛缓存 → verifier → answer | evidence loop

> 代码：`src/asmem/search/`、`src/asmem/retrieval/`
> 入口：`POST /search`、`POST /v1/memories/search`、`python -m asmem.search`、`scripts/coarse`

Search 只有两条路：

```text
粗筛（no LLM，可缓存）
  → verifier（1 次模型调用 + 确定性检查）
      ├─ 通过 → 直接用粗筛证据 answer
      └─ 不通过 → postagent loop（有界证据，最多 3 次模型调用）
                     search(重写 query) → take_note / auto-note → finish
  → coarse + notes 合并去重 → answer-facing evidence
```

没有“只挑部分难例”的规则，也没有固定阶段的 select/gaps/verify/finalize。
postagent 只在 verifier 判为“缺证据”时启动；它自己决定重写什么 query、保存哪些记录、
什么时候认为已经能回答。

---

## 1. 输入与输出

| 项 | 内容 |
|---|---|
| 输入 | `memory.jsonl` 快照、`query`、可选 `options`、`retrieval.*` 配置、embedding 缓存目录 |
| 粗筛输出 | `inputs/candidates.jsonl`、`inputs/coarse.json` |
| postagent 输出 | 追加后的 `evidence.jsonl`、`summary.json`、可选 `steps.jsonl` / `agent_history.jsonl` |
| 返回 | 经过 postprocess 的证据列表（`content` + `metadata.source`） |

`evidence.jsonl` 初始是粗筛 seed（`origin=coarse, seeded=true`）。postagent 只向它追加
notes（`claim_type=copy`）或 claim；粗筛 seed 不会被删除。

---

## 2. 粗筛（无 LLM，可单独缓存）

### 2.1 查询分析

`run_coarse` 在打分前做确定性分析：

| 输出 | 规则 | 代码 |
|---|---|---|
| `tokens` | ASCII 小写、停用词、轻词干；CJK 单字 + 二元组 | `tokenize.tokenize` |
| `phrases` | 引号短语、连续大写词、数字、ID | `tokenize.extract_phrases` |
| `time_expr` | 日期、相对时间、`as of/before/after` | `index.parse_time_expr` |
| `route` | `dialogue / multi_hop / summary / aggregation / policy / self_contained / mcq` | `router.route_query` |
| `intent` | `fact / yesno / list / temporal / update / ...` | `router.detect_intent` |

`self_contained` 不生成候选，直接把 query 作为证据；`policy` 抽取 forbidden 片段供
verifier 检查泄漏。

### 2.2 索引

每次 Search 从快照重建（10k 记录毫秒级）：

- **BM25**：纯 Python 倒排，`k1=1.2`、`b=0.75`；
- **Dense（可选）**：OpenAI-compatible `/embeddings`，长记录按 token 窗口切分并 max-pool；
- **State**：`derive_state(records)`，不依赖磁盘 `state.jsonl`；
- **Exact/Time**：查询时扫描。

Embedding 缓存身份 = `embedder.identity + chunk 版本 + 编码窗口`；换模型/换切分/换窗口
会进入新缓存目录，不会混用向量。

### 2.3 七个召回通道

| 通道 | 信号 |
|---|---|
| `bm25` | 词法匹配 |
| `dense` | 语义相似（可选） |
| `exact` | 引号短语、数字、ID、专名 |
| `time` | 记录时间与查询时间表达的距离/方向 |
| `state` | 当前值优先，superseded 降权保留 |
| `session` | 同会话/同来源覆盖，避免多跳只召回一条 |
| `option` | MCQ 选项与记录重合 |

融合用加权 RRF：

```text
score(d) = Σ_channel weight[channel] / (rrf_k + rank_channel(d))
```

默认 `bm25=0.5, dense=0.5, exact=0.3, time=0.2, state=0.2, session=0.2, option=0.2`，
`rrf_k=60`。融合后做来源配额、近重复合并、邻居扩展，写 `candidates.jsonl`。

### 2.4 粗筛缓存

粗筛是确定性的，不需要每次重跑：

```bash
# 一次性：只跑粗筛，保存 candidates/coarse，跳过 postagent/answer/eval
python scripts/run_bench.py --config configs/bench_qwen3_gate_loop.yaml \
    --coarse-only --run-name qwen3_coarse_cache --overwrite
```

之后的 run 配置：

```yaml
experiment:
  reuse_coarse_from: qwen3_coarse_cache
```

复用时会校验 query、retrieval 配置指纹、candidate id 是否仍在当前 memory snapshot；
不匹配就重算。`qwen3_b1` 本身就是一个全量 coarse-only run，也可以直接复用。

---

## 3. Verifier：只判断“具体值在不在”

`src/asmem/search/verifier.py::gate_verify`，prompt 在 `src/asmem/search/prompts.py::GATE`。

### 3.1 模型部分（1 次调用）

模型只做三件事：

1. 说出 exact asked slot（who / when / where / which item / how many / yes-no / preference）；
2. 用记录写一个短答案；
3. quote 支持文本，并说明它是否属于**同一个 entity/event**。

它**不要求**完整 claim、不要求 exact quote、不要求列表完整；paraphrase、同义词、
relative date 的 resolved 值都可以接受。

### 3.2 确定性检查（不耗模型调用）

- quote 必须真的出现在 evidence（token overlap ≥ 0.5）；
- `when` 要有日期，`how long` 要有 duration，`how many` 要有数字，`who/where` 要有名字；
- `which/what kind` 的 quote 必须含有问题之外的 content token；
- 问题里的专名必须出现在 cited record 的 speaker/content；
- first-person 问题不能只用 assistant plan 回答；
- answer 引入的日期/数字必须出现在 quote 里；answer 复用问题谓词时 quote 也要支持它
  （或显式 yes/no）。

### 3.3 可选一致性检查

`gate_check_consistency: true` 时，对单值/排序类 intent
（`fact / temporal / duration / yesno / update / conflict / compare / order / multi_hop / aggregate`）
再做一次 answer↔quote 检查；rubric/长文档跳过。它用来抓“quote 属于另一个事件”的 false accept，
仍然属于 verifier，不是第三条路。

### 3.4 两个出口

- `answerable=true` → 直接用粗筛证据进入 answer；
- `answerable=false` → 把 `missing` slots 交给 postagent loop。

---

## 4. Postagent：有界证据的有界 loop

`src/asmem/search/evidence_loop.py`。每轮都是全新的有界 prompt，不累积聊天历史：

```text
QUESTION
MISSING SLOTS（verifier 给出）
CURRENT EVIDENCE（coarse + notes，top 16）
LAST SEARCH RESULTS（top 5，1-based index）
QUERIES ALREADY RUN
→ 一个 JSON action
```

### 4.1 动作

| action | 行为 |
|---|---|
| `search` | 用重写 query 在**已构建的粗筛 index** 上检索；排除已见 id；支持 `date_from/date_to`；返回 top-k |
| `take_note` | 把 last search 的指定 1-based 结果按 raw record 追加到 `evidence.jsonl` |
| `finish` | 模型认为当前证据已经能回答 |

### 4.2 Query 重写规则

- 先命名缺失 slot；
- 用 distinctive anchor：专名、数字、引号短语，或刚找到的值；
- 多跳用 A→B→C：找到 A 后，用 A 的值搜 B；
- 时间题用记录自己的相对短语（`last Friday` / `last month`），不自己算日历；
- 有绝对日期时用 `date_from/date_to`；
- 不重复 query；没有新结果就换 anchor 或 finish。

### 4.3 合并与去重

- 每次 search 自动保存 top `loop_auto_note` 条，避免“搜到了但没 note”；
- loop 结束一条 note 都没有时，自动保存 last search top 2；
- notes 以 `claim_type=copy` 写入，和粗筛 seed 一起走
  `postprocess_evidence` / `fit_evidence`，按 source 和 normalized text 合并去重；
- 粗筛 seed 永远保留；notes 只追加。

### 4.4 多模态 query 与 postagent

Search 的 `query` 是多模态 `ContentPart[]` 时：

1. `multimodal.query_content` 把 query 文本与图片 caption 合成检索文本；图片本身不进入
   BM25/dense，也不做 image embedding；
2. 最多保留 1 张合法 data URI（`max_query_images`），只用于 **gate** 和 **evidence loop
   第一轮** 的模型调用；文本 query 永远不附图；
3. memory 里的图片**不进入** postagent prompt —— postagent 只看 caption。命中的 memory
   图由 `api.server.build_search_items` 在响应里返回，受 10 MiB/图、30 MiB/响应和
   `max_response_images` 限制；
4. `agent_history.jsonl` 持久化前会把图片 part 替换成 `[image]`，避免把 base64 写进运行
   记录。

相关配置：

```yaml
multimodal:
  caption_query: true        # query 图也生成 caption
  max_query_images: 1
  postagent_images: auto     # auto: gate + loop 第一轮；never / always
  return_images: true
  max_response_images: 8
```
### 4.5 超时、队列与 fallback

- 硬超时从**获取 `search_sem` 之前**开始计时；`search_queue_timeout` 内拿不到 slot 就
  走 fallback，避免排队把请求拖到 600s。
- `run_search` 在第一个 gate/loop 模型调用前，把 top coarse 证据写入 API 提供的
  fallback 文件；硬超时/异常时 API 返回这份证据，而不是 `data: []`。
- fallback 文件为空（例如 coarse 也超时）时，`quick_coarse_fallback` 关闭 dense，跑一次
  BM25-only 粗筛，不调用任何模型。
- 只有带 `request_id` 的请求在客户端断开后继续运行；无 `request_id` 的请求会被取消，
  释放 `search_sem`。
- `llm_global_breaker_failures` 控制进程级 endpoint 熔断；打开后新搜索快速降级。
- `model_input_token_cap` 做 prompt 预检；`context_overflow_retries` 在 400
  context overflow 时把 prompt 缩小后重试。

### 4.6 上下文与预算

- 每轮只渲染 top 16 证据 + top 5 搜索结果；
- 单条 note `loop_note_chars` 上限；
- `llm.fit_messages` 做 token 预检；
- `loop_max_steps: 3`；加 verifier 1 次（+ 单值题一致性检查 1 次），仍低于 6 次；
- search 复用 `RetrievalIndex`，不重建索引、不重嵌文档。

---

## 5. 后处理与 answer-facing evidence

`postprocess_evidence` 负责：

1. source 校验：只接受 memory 里存在的 id，或 `["query"]`；
2. 合并重复内容和 source；
3. 按 route/记录类型解析 chunk / parent / window；
4. 邻居扩展：每条 anchor 带同来源前后记录，标记 `[context <id>]`；
5. 相对日期解析：`[resolved: ...]`；
6. 按 answer 字符预算裁剪，保留 seed 第一条、agent notes、agent claim。

answer 看到的是：

```text
[1] (s1_00042; speaker=Caroline; observed_at=2023-05-08)
[resolved: last week = 2023-05-01..2023-05-07]
...
[context s1_00043] ...
```

---

## 6. API 与配置

### 6.1 HTTP API

`/add` 和 `/search` 与 benchmark 使用同一套实现：

```text
/add    -> api.store.UserStore.append -> schema.normalize_messages
/search -> search.runner.run_question -> api.server.build_search_items
```

`configs/api.yaml` 与 `configs/asmem.yaml` 都设置：

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
gate_check_consistency: true
pre_audit: false
```

API 只需要改 generator / embedding endpoint、`concurrency`、`store_dir` 等部署项；
检索路径和 benchmark 完全一致。`/search` 在模型/端点失败时返回已经写入的粗筛证据，
不返回 5xx。

### 6.2 配置速查

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
gate_check_consistency: true
pre_audit: false
verifier_max_evidence: 24
verifier_max_chars: 6000
verifier_max_context_tokens: 24000
max_evidence: 64
max_evidence_chars: 40000
final_evidence_chars: 40000
final_item_chars: 4000
final_claim_cap: 12
final_agent_cap: 16
# Timeout / queue / fallback
search_queue_timeout: 60
fallback_enabled: true
fallback_timeout: 20
fallback_evidence_limit: 12
llm_global_breaker_failures: 8
# Context protection
model_input_token_cap: 24000
model_output_reserve_tokens: 4096
context_overflow_retries: 1
verifier_max_output_tokens: 2048
postagent_max_output_tokens: 4096
auto_capture_hits: true
auto_capture_limit: 2
auto_capture_total: 6
```

粗筛：

```yaml
retrieval:
  mode: hybrid
  coarse:
    candidates: 80
    session_cap_ratio: 0.3
    rrf_k: 60
    dedup_jaccard: 0.9
  channels: {bm25: 0.5, dense: 0.5, exact: 0.3, time: 0.2, state: 0.2, session: 0.2, option: 0.2}
  gate:
    evidence_limit: 12
```

---

## 7. 结果

全量 2000：

| run | score | judge | rubric | status |
|---|---:|---:|---:|---|
| `qwen3_gate_loop_v1` | **0.7170** | 0.7788 | 0.6565 | 1997 ok, 2 answer_failed, 1 eval_failed |
| `4omini_gate_loop_v1` | **0.7106** | 0.7740 | 0.6478 | 1999 ok, 1 answer_failed |
| `qwen3_b1`（粗筛） | 0.6946 | 0.7490 | 0.6507 | — |
| `qwen3_b2`（旧 loop） | 0.7185 | 0.7790 | 0.6586 | — |

214 条验证子集（locomo 富集 + 三个数据集各 50；粗筛复用 `qwen3_b1`）：

| 方案 | locomo | personamem | beam | clbench | weighted | avg search calls |
|---|---:|---:|---:|---:|---:|---:|
| b1 粗筛 | 0.7500 | 0.8200 | 0.5147 | 0.7281 | 0.7061 | 0 |
| b2 loop | 0.8438 | 0.7600 | 0.5721 | 0.7434 | 0.7373 | 7.28 |
| fixed pipeline（对比） | 0.8750 | 0.8200 | 0.5351 | 0.7449 | 0.7524 | 5.29 |
| **core：verifier + evidence loop** | **0.8594** | **0.8600** | **0.5661** | **0.7421** | **0.7637** | **3.12** |

core 在 213 条共同评分上比粗筛 +5.76、比 b2 +2.65、比 fixed pipeline +1.13；
search-stage 平均 3.12 次调用 / 2.22M tokens（b2 7.28 / 9.66M）。

> 214 子集的 locomo 被特殊案例富集，不能直接外推全量；全量结果以上表为准。

---

## 8. 对比方法与实现说明

- `fixed pipeline`、旧 `loop`、`compile` 只作为对比/消融保留在代码和
  `configs/ablations/`（不进入核心配置）。
- postagent 采用常见的 retrieve → note → finish 循环；本实现增加了粗筛 index 复用、
  auto-note、coarse+notes 合并去重、确定性 verifier、有界 fresh prompt。
