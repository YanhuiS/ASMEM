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
- **只有带 `request_id` 的请求才在客户端断开后继续运行**；无 `request_id` 的请求被取消，
  立即释放 `search_sem`，避免僵尸任务占满并发。
- **进程级 endpoint breaker**：`llm_global_breaker_failures > 0` 时，端点连续失败会打开
  全局熔断，新请求快速进入 coarse fallback，而不是各自重试。
- **上下文保护**：`model_input_token_cap` 在每次模型调用前做 token 预检；
  端点返回 context overflow 时，`ModelRunner` 会把 prompt 缩到 1/2、1/4 再试一次
  （`context_overflow_retries`），仍失败才降级。

## 4. API

`/add` 和 `/search` 与 benchmark 使用同一套实现：

```text
/add    -> api.store.UserStore.append -> schema.normalize_messages
/search -> search.runner.run_question -> api.server.build_search_items
```

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
gate_check_consistency: true
pre_audit: false
multimodal:
  caption: true
  caption_query: true
  max_image_bytes: 10485760
  max_request_bytes: 31457280
  max_response_bytes: 31457280
  max_response_images: 8
  postagent_images: auto      # auto / never / always
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

---

## 6. 对比方法与实现说明

- `fixed pipeline`、旧 `loop`、`compile` 只作为对比/消融，代码保留，配置在
  `configs/ablations/`（不进入核心配置）。
- postagent 采用常见的 retrieve → note → finish 循环；本实现增加了粗筛 index 复用、
  auto-note、coarse+notes 合并去重、确定性 verifier、有界 fresh prompt。
