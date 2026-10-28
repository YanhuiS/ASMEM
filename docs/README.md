# ASMEM 文档

ASMEM 是一个长期记忆检索后端：

- `POST /add`：确定性归一化、切分、编号，落成 `memory.jsonl`；不调用生成式模型。
- `POST /search`：无 LLM 的多通道粗筛 → verifier → 两条路：
  **通过直接 answer；不通过进入有界证据补充 loop**。

文档只保留三篇：

| 文档 | 内容 |
|---|---|
| [core.md](core.md) | 系统定位、端到端流水线、核心设计、数据模型不变量、结果 |
| [add.md](add.md) | 适配层、统一记录、token 切分、稳定 id、状态索引、WAL/幂等/恢复 |
| [search.md](search.md) | 粗筛缓存、verifier、evidence loop、后处理、API/配置、结果与消融 |

历史设计稿在 `docs/archive/`，不作为当前实现说明。

## 一屏流水线

```text
输入 messages / document / table / triple / trace
  │
  ├─ Add（纯 CPU）
  ▼
memory.jsonl + state.jsonl
  │
  ├─ 粗筛（无 LLM，可缓存）
  │    BM25 / Dense / Exact / Time / State / Session / Option
  │      → 加权 RRF → 来源配额 → 近重复 → 邻居扩展
  ▼
inputs/candidates.jsonl + inputs/coarse.json
  │
  ├─ 证据种子
  ▼
evidence.jsonl
  │
  ├─ verifier（1 次模型调用 + 确定性检查）
  │    ├─ 通过 → answer
  │    └─ 不通过 → evidence loop（≤3 次模型调用）
  │                   search(重写 query) → take_note / auto-note → finish
  ▼
coarse + notes 合并去重 → answer-visible evidence
```

## 常用入口

```bash
# 写入
python -m asmem.add --input messages.jsonl --user alice --session session_1

# 检索（核心配置）
python -m asmem.search --memory data/asmem/alice/memory.jsonl \
    --query "Where did Alice move?" --config configs/asmem.yaml

# 粗筛 CLI
export PATH="$PWD/scripts:$PATH"
coarse --query "Where did Alice move?" --top 8

# 粗筛缓存（一次）
python scripts/run_bench.py --config configs/bench_qwen3_gate_loop.yaml \
    --coarse-only --run-name qwen3_coarse_cache --overwrite

# 核心方法 benchmark
python scripts/run_bench.py --config configs/bench_qwen3_gate_loop.yaml \
    --run-name qwen3_gate_loop_v1 --overwrite

python scripts/compare_runs.py --baseline data/bench_runs/qwen3_b1 \
    --candidate data/bench_runs/qwen3_gate_loop_v1
```

对比方法（fixed pipeline / 旧 loop / compile）的配置在 `configs/ablations/`，
只用于消融，不进入核心配置。
