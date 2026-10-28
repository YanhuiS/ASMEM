# Add：确定性格式层

> 代码：`src/asmem/adapters.py`、`src/asmem/schema.py`、`src/asmem/state.py`、`src/asmem/api/store.py`
> 入口：`POST /add`、`python -m asmem.add`

Add 是整条流水线里**唯一不调用生成式模型**的写入阶段。它只做四件事：

1. 从平台报文里恢复真实结构（说话人、时间、文档、表格、三元组、轨迹）；
2. 按 token 切分检索单元，保留原始单元的 `parent_id / part / parts`；
3. 分配稳定 id `s{session_index}_{seq:05d}`；
4. 用正则识别显式的三元组和 `UPDATE:`，派生 `state.jsonl`。

它**不摘要、不抽 atom、不改写事实、不翻译、不润色**。原因是检索系统需要的是可 grep、可引用、可追溯的原文；任何生成式改写都会让 UPDATE、否定、条件和时间锚点变得不可验证。

---

## 1. 职责边界

| Add 做 | Add 不做 |
|---|---|
| 恢复说话人/时间/文档/表格/轨迹形态 | 不调用生成式模型 |
| 按 token 切分，写 `parent_id/part/parts` | 不做摘要、事实抽取、atom 化 |
| 分配稳定 id | 不改写、翻译、润色正文 |
| 正则识别显式三元组和 `UPDATE:` | 不猜测自然语言中的版本关系 |
| 写 `memory.jsonl`、派生 `state.jsonl`、维护 WAL | 不在 Add 阶段建 BM25/dense 索引 |

Embedding 是**可选编码器**，不是 Add 的必需步骤。`index_on_add=false` 时，Add 完全不碰 embedding；第一次 `/search` 才构建索引并填缓存。

---

## 2. 输入形态与适配层

平台报文常见三类问题：

1. **说话人丢失**：`role` 恒为 `user/assistant`，真名在正文里（`Caroline: ...`）。
2. **平台标签污染正文**：`[beam_100k][session_1][D1:21][text]` 进入正文后，grep `session_1` 会命中全库。
3. **结构被压平**：表格、三元组、agent 轨迹被塞进一段转义 JSON，grep 和聚合都做不了。

`adapters.adapt_messages` 先剥掉可识别的平台标签，再按首条正文判定整批形态：

| family | 触发 | 产出 |
|---|---|---|
| `triples` | 首条以 `Corpus:` 开头 | `subject \| relation \| object`，`role=corpus` |
| `trajectory` | 首条以 `Streaming ... trajectory:` 开头 | 展开 `trajectory[]`，每条加 `[dataset task index N]` |
| `qa-feedback` | 首条以 `Streaming online QA feedback:` 开头 | 展开成多行可读记录 |
| `document` | 首条以 `document:` 开头 | `role=document` |
| `facts` | 首条以 `source:` 开头 | `role=corpus` |
| `narration` | 正文/标签明确是旁白、舞台指示 | `role=narration` |
| `dialogue` | 能拆出至少一个说话人 | 逐条恢复说话人 |
| `plain` | 以上都不是 | 用协议 `role`，或 `unknown` |

### 2.1 说话人恢复

`split_speaker` 把 `"Christina Baker: I'm trying..."` 拆成 `("Christina Baker", "I'm trying...")`。判据是保守的：

- 前缀超过 4 个词或 40 字符 → 不当说话人；
- 纯数字（`1: ...`）是列表编号；
- `GENERIC_SPEAKERS`（`user/assistant/system`）认得出，但不是真名；
- `NON_SPEAKER_PREFIXES` 是否定表：内容类型（`corpus/document/question`）、场景指示（`close up/stage`）、语气词（`wait/well/no`）；
- **本批次重复出现的前缀**更可信（`Gina:` 出现 20 次 → 是说话人；`Wait:` 只出现一次 → 更可能是正文）。

判错的代价不对称：漏认只是 `role` 退化，误认会从正文里删掉前缀。

### 2.2 三元组与流式记录

- 三元组保留三段原文：`Latvia | official language | Latvian`。不模板动词化，因为 relation 是数据集自己的词（`country of citizenship` / `official language`）。
- `Corpus: Update {...}` 渲染成 `UPDATE: s | r | o (this replaces the prior value ...)`，供 `annotate_state` 识别。
- `Q-id`（`subject_id/relation_id/object_id`）丢弃：只对数据集自己有意义的 id 留在正文里只会污染 grep。
- agent 轨迹展开成逐条消息，并给每条加 `[ALF task index 86]` 这类字面串，否则 query 里的 `task index 86` grep 不到。
- QA feedback 展开成多行，`predicted_answer` 标记为 `answer given at the time (not authoritative)`。

### 2.3 适配层不做的事

- 不改写事实、不摘要、不翻译；
- 不把图片变成理解结果，只保留引用；
- 不猜测自然语言中的更新关系；只把显式 `UPDATE:` 变成状态标记。

---

## 3. 统一记录 schema

`memory.jsonl` 一行一个 JSON：

```json
{"id":"s1_00042","text":"Caroline went to a support group on 7 May 2023.",
 "kind":"message","time":"2023-05-08T13:56:00Z","role":"Caroline",
 "source":"session_1","meta":{}}
```

| 字段 | 语义 |
|---|---|
| `id` | 稳定 id，证据的唯一引用目标 |
| `text` | 可 grep 的正文（消息、段落、表格行、三元组） |
| `kind` | `message / passage / row / triple / record` |
| `time` | **观测时间**（ISO-8601 UTC）；未知为空串 |
| `role` | 说话人/作者；无则为空 |
| `source` | 逻辑来源（会话、文件、任务 id） |
| `meta` | 结构化附加字段 |

`meta` 的关键字段：

| 字段 | 语义 |
|---|---|
| `family / message / unit` | 适配器恢复出的来源信息 |
| `parent_id / part / parts` | 切分前的原始单元 / 当前分片 / 总分片 |
| `parent_chars / parent_kind` | 原始内容长度 / 原始类型 |
| `token_start / token_end / tokens` | 分片在原 token 序列中的区间/长度 |
| `char_start / char_end` | 分片在原文中的字符区间 |
| `chunk_version` | 切分策略版本，参与 embedding 身份 |
| `subject / relation / object` | 三元组结构化字段 |
| `state_key` | `subject\|relation`，状态索引键 |
| `status` | `active / superseded / deleted` |
| `mutation` | `none / replace / delete` |
| `valid_from / valid_to` | 状态生效/失效日期 |

**`time` 是观测时间，不是事件时间。** 一条消息在 2023-05-08 说“我去年搬到了巴黎”，`time` 是 2023-05-08，事件时间是 2022。检索层用 `time` 作为相对时间的锚点，而不是把它当成事件日期。

---

## 4. Token 切分

切分发生在 Add，是**检索单元**的切分，不是事实抽取。长度用 **token** 而不是字符衡量；token 与 BM25 看到的是同一套单位（ASCII 词/数字、单个 CJK 字符、单个标点），因此英文、中文、表格和代码的长度控制一致。

| 类型 | token 预算 | 说明 |
|---|---:|---|
| `message` | 320 | 一条消息通常就是自然检索单元；超长才切 |
| `passage` | 320 | 长文档 |
| `record` | 320 | 任务轨迹/杂项记录 |
| `row / triple` | 不切 | 结构化行和三元组保持原子 |

切分规则：

1. 以 token 窗口为单位，窗口最多 `max_tokens`；
2. 窗口末端优先吸附到**段落边界**，其次句子边界，再次分句/换行边界；只在前 25% 范围内回退，避免块过短；
3. 边界只落在 token 之间，**不会切断 ASCII 单词**；CJK 按单字计数；
4. 下一窗口从上一窗口末端前 `overlap_tokens`（默认 40）个 token 开始，边界事实从两侧都能命中；
5. 所有分片共享 `parent_id`，并写 `part / parts`、`token_start / token_end / tokens`、`char_start / char_end`；
6. `chunk_version` 记录策略版本，并进入 embedding 身份。

`chunk_version` 的格式：

```text
v{version}:{message_tokens}:{passage_tokens}:{record_tokens}:{overlap_tokens}:{min_tokens}
# 例：v3:320:320:320:40:20
```

换模型、换 chunk 策略、换编码窗口会得到新的 embedding 缓存目录，不会混用旧向量。

**切分是检索单元，不是事实单元。** 证据可以引用分片 id，但 `content` 必须自足；如果分片本身缺少上下文，由邻接扩展或 parent 回读补齐。

---

## 5. 稳定 ID

```text
id = s{session_index}_{seq:05d}
```

- `session_index` 在用户内单调分配；
- `seq` 在会话内单调递增；
- 同一 `request_id` 的重试不会产生新 id；
- 切分产生的每个分片占一个 `seq`。

`parent_id` 的形状：

- 一条原始消息被切成多个 passage 分片时，所有分片共享 `p{session_index}_{message_index:05d}`；
- 表格行/三元组各自是独立 parent：`p..._u{unit_index:03d}`；
- `part / parts` 只在共享 parent 的多分片场景下大于 1。

---

## 6. 状态标注与 `state.jsonl`

Add 只做确定性正则识别，不做自然语言推理：

- `kind=triple` 或正文以 `UPDATE:` 开头且包含 `|` 时，解析 `subject | relation | object`；
- 写入 `meta.subject / relation / object / state_key`；
- `UPDATE:` 标记 `mutation=replace`，默认 `status=active`；
- 无法确定的关系保持普通记录，由检索阶段判断。

`state.jsonl` 是 `memory.jsonl` 的**派生状态索引**，按记录顺序确定性推导：

```json
{"state_key":"quarterback|sport","value":"Muay Thai","source":"s2_00001",
 "valid_from":"2027-06-29","valid_to":"","status":"active","conflict":false,
 "supersedes":["s1_00001"],"superseded_by":[]}
```

规则：

- `mutation=replace`：旧值标 `superseded`，写 `valid_to` 和 `superseded_by`；新值写 `supersedes`；
- `mutation=delete`：旧值和新墓碑都标 `deleted`；
- 同一 `state_key` 出现多个不同的 active 值：相关条目 `conflict=true`；
- `memory.jsonl` 保持追加写，不回写旧行；`state.jsonl` 是当前值/历史值的权威视图；
- Search 索引在缺少 `state.jsonl` 时用同一个 `derive_state` 重建，因此两者不会漂移。

`state` 通道使用 `state.jsonl`：active 值优先，superseded 值降权但保留，供“之前是什么”类问题使用。

---

## 7. 写入路径：WAL、幂等与恢复

`UserStore.append` 的流程：

1. 取 `user_id` 的进程内锁；
2. 读取 `meta.json`，先恢复未完成的 `pending`；
3. `request_id` 已在 `applied` → 直接返回，不重复写；若同一 `request_id` 的 payload hash 不同 → `RequestConflictError`；
4. 分配 session index / seq；
5. 归一化 + 切分；
6. 写 `meta.pending[request_id]`；
7. 追加 `memory.jsonl`，推进计数，移入 `applied`；
8. 再写一次 `meta.json`。

`meta.json` 是 WAL：`pending` 里保存了这次请求的完整 records 和下一个 seq。进程在 7/8 之间崩溃时，下次启动 `_recover` 会：

- 如果 `memory.jsonl` 已经包含这些 id → 只推进计数；
- 否则重新追加；
- 最后清空 `pending`。

`_settle` 是幂等的：同一批 records 重复落盘不会产生重复 id。

### 7.1 并发

- 同一用户的 add 串行化（进程内锁）；
- 不同用户互不影响；
- 写入是追加 + 原子替换 `meta.json`，不会出现半条记录。

### 7.2 输出物

```text
data/asmem/<user>/
  memory.jsonl        # 追加写，统一记录
  state.jsonl         # 派生状态索引
  meta.json           # sessions / applied / pending / counts
  index/              # BM25/dense/state 的懒加载产物（可选）
  cache/              # embedding 缓存
  runs/               # search workspace
```

---

## 8. 已知限制

- 说话人恢复是启发式；平台报文如果没有稳定前缀，会退化为 `role=unknown`。
- 状态识别只覆盖显式 `UPDATE:` / 三元组；自然语言里的“其实我更喜欢……”不会被自动标成 replace，需要检索阶段判断。
- 超长表格行/三元组保持原子；如果单行本身超过上下文预算，需要调用方先切分。
- Add 不建索引；第一次检索会有一次索引构建成本（BM25 是毫秒级，dense 取决于 embedding 服务）。
