"""评测平台语料的**形态适配层**：把千奇百怪的 ``/add`` 报文归一化成三层记忆能吃的消息。

**为什么需要这一层。** 平台上的语料来自很多个数据集，它们各自把"说话人是谁""时间是什么"
"这是对话还是文档"编码在**报文的不同位置**：

- 有的是 ``{"role": "user", "content": "[beam_100k][session_1][D1:21][text] Christina Baker: ..."}``
  —— 真名在 content 里，``role`` 字段只是"这条是用户侧发的"；
- 有的是两个真人对话（``Caroline:`` / ``Melanie:``），``role`` 全是 ``user``；
- 有的是剧本（``JUDGE:`` / ``FOREMAN:`` / ``#8:``），有的是舞台指示（``CLOSE UP - #8``）；
- 有的是**文档**（``document: Document: xxx.pdf`` 分页）；
- 有的是**知识图谱三元组**（``Corpus: {"subject":..., "relation":..., "object":...}``）；
- 有的是整段 **JSON 记录**（流式 QA 反馈、agent 轨迹），字段全被 ``\\n`` 转义糊在一行里。

**不处理会怎样**（实测到的两个具体故障）：

1. **说话人丢失**：``msgmem.role`` 写成 ``user``/``assistant``，于是 atommem 里全是
   ``"user is trying to implement city autocomplete"`` —— 而这条语料真正的说话人是
   ``Christina Baker``。查询问的是 ``"What fields would Caroline be likely to pursue?"``，
   而 ``Caroline`` 这个名字**一条原子记忆里都没有**，grep 直接落空。
2. **平台标签污染正文**：``[beam_100k][session_1][D1:21][text]`` 进了 ``memory`` 正文，
   于是 grep ``"session_1"`` 命中全库、文档名和正文之间隔着一层噪声，summem 也要花力气
   把它当内容读。

这一层只做**无损的重新编码**：把平台编码的信息搬到该在的字段上（``role`` / ``time``），
把平台自己的记账标签摘掉；**不改写事实、不摘要、不翻译**。原本就是对话的语料经此一层，
取到的正文与原来逐字相同。

用法（只被 ``api.store`` 调用，不依赖任何步骤模块）::

    batch = adapt_messages(messages)          # messages = [{"role","content","timestamp"}]
    batch.family                              # "tagged-dialogue" / "document" / "triples" / ...
    batch.messages                            # [AdaptedMessage(role, content, timestamp, kind)]
    batch.speakers                            # ["Christina Baker"] / ["Caroline", "Melanie"] / []

**只在这里发生一次**：离线 ``add`` 读的是 ``data/*.json``（``dataset.py``），报文里本来
就带着真名与时间，不走这一层。两条入口各自归一化，下游（msgmem / summem / atommem /
search）看到的永远是同一种形状。
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

__all__ = [
    "AdaptedMessage",
    "AdaptedBatch",
    "adapt_messages",
    "split_lead_tags",
    "split_speaker",
    "render_triple",
    "looks_like_json",
    "GENERIC_SPEAKERS",
    "NON_SPEAKER_PREFIXES",
]

# ---------------------------------------------------------------------------
# 一、平台记账标签：``[beam_100k][session_1][D1:21][text]``
# ---------------------------------------------------------------------------
#
# **不能无脑剥所有前导 ``[...]``** —— 正文里本来就有方括号（``[part 1/2] Background`` 是
# clbench 语料的正文开头）。所以只剥**认得出是平台记账**的那几种：会话号、``D1:1`` 式的
# 数据集内 id、没有空格的 slug（数据集名）、以及少数的类型词与 ``Time: ...``。

_SESSION_TAG_RE = re.compile(r"^session_\d+$")
_DATASET_ID_TAG_RE = re.compile(r"^[Dd]\d+:\d+$")
_KIND_TAGS = frozenset({
    "text", "dialogue", "narration", "document", "table", "image", "file",
    "audio", "video", "code", "qa",
})
#: 数据集 slug：``clbench_16_32k`` / ``locomo_refined`` / ``mquake_remastered``。
#: **必须没有空格**，否则 ``[part 1/2]`` 这类正文会被误剥。
_SLUG_TAG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{2,40}$")
_TIME_TAG_RE = re.compile(r"^Time:\s*(.*)$", re.IGNORECASE)

_LEAD_TAG_RE = re.compile(r"^\[([^\[\]]{1,80})\]\s*")


def _is_platform_tag(tag: str, position: int) -> bool:
    """这个前导 ``[tag]`` 是平台记账，还是正文自带的方括号？"""
    if _SESSION_TAG_RE.match(tag) or _DATASET_ID_TAG_RE.match(tag):
        return True
    if _TIME_TAG_RE.match(tag):
        return True
    if tag.lower() in _KIND_TAGS:
        return True
    # slug 只认**最前面那一个**（``[locomo_refined][session_1]...``）——再往后的 slug 型
    # 方括号更可能是正文（例如 ``[part 1/2]`` 不合 slug 规则，故这条只是多加一道保险）。
    return position == 0 and bool(_SLUG_TAG_RE.match(tag))


def split_lead_tags(text: str) -> tuple[list[str], str]:
    """剥掉开头的**平台记账标签**，返回 ``(剥下来的标签, 剩下的正文)``。

    只剥连续的一段；遇到第一个不是平台记账的 ``[...]`` 就停手，把它连同后面全部正文
    原样留下。
    """
    tags: list[str] = []
    rest = text or ""
    while True:
        match = _LEAD_TAG_RE.match(rest)
        if not match:
            break
        tag = match.group(1)
        if not _is_platform_tag(tag, len(tags)):
            break
        tags.append(tag)
        rest = rest[match.end():]
    return tags, rest


def tag_time(tags: Sequence[str]) -> str:
    """从记账标签里取 ``Time: ...`` 的值（没有则空串）。"""
    for tag in tags:
        match = _TIME_TAG_RE.match(tag)
        if match:
            return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# 二、说话人前缀：``Christina Baker: ...`` / ``user: ...`` / ``JUDGE: ...``
# ---------------------------------------------------------------------------

#: 协议 role 的通用写法 —— 认得出，但**不是真名**。
GENERIC_SPEAKERS = {
    "user": "user", "human": "user", "me": "user", "speaker1": "user", "speaker_1": "user",
    "assistant": "assistant", "ai": "assistant", "bot": "assistant", "system": "system",
    "speaker2": "assistant", "speaker_2": "assistant", "model": "assistant",
    "tool": "tool", "function": "tool",
}

#: **不是说话人**的前缀词。它们确实长得像 ``Xxx: ...``，但冒号后面是内容类型或场景描写。
#: 判错的代价不对称：把说话人误判成内容（漏认）只是 role 退化成协议 role；把内容误判成
#: 说话人则会**从正文里删掉这段前缀**。所以这张表宁可长一点。
NON_SPEAKER_PREFIXES = frozenset({
    # 内容类型
    "corpus", "document", "doc", "source", "sources", "context", "input", "output",
    "question", "answer", "options", "option", "passage", "text", "content", "file",
    "record", "task", "goal", "instruction", "instructions", "prompt", "response",
    "summary", "note", "notes", "example", "examples", "format", "output format",
    "schema", "available actions", "actions", "result", "results", "state", "meta",
    # Explicit mutation markers are data, not speakers.
    "update", "delete", "replace", "supersede",
    "title", "subject", "object", "relation", "key", "value", "definition",
    # 剧本的场景/镜头指示（12 Angry Men 那套语料里大量出现）
    "narration", "stage", "stage direction", "scene", "cut to", "fade in", "fade out",
    "close up", "medium shot", "wide shot", "long shot", "dissolve to", "interior",
    "exterior", "int", "ext", "voice over", "voiceover", "o.s.", "v.o.", "camera",
    # 对话里的语气词/连接词（会被误当成人名的那些）
    "wait", "well", "note that", "yes", "yeah", "no", "nope", "ok", "okay", "oh", "hmm",
    "thanks", "thank you", "sorry", "hi", "hello", "hey", "sure", "right", "so", "but",
    "and", "also", "actually", "please", "maybe", "look", "listen", "wow", "great",
    "good", "fine", "true", "false", "however", "therefore", "finally", "first",
    "second", "third", "then", "now", "here", "there", "this", "that", "it", "he",
    "she", "they", "we", "you", "i",
})

#: 说话人前缀最多几个词（``Christina Baker``=2、``Juror #2``=2、``Dr. Stockmann``=2）。
_MAX_SPEAKER_WORDS = 4
_MAX_SPEAKER_CHARS = 40

#: ``Name: rest`` 里的 ``rest`` 必须另起一段 —— 要求冒号后有空白，顺便躲开 ``https://``。
_SPEAKER_RE = re.compile(r"^(?P<name>[^:\n\[\]]{1,40}?)\s*:\s+(?P<rest>.*)$", re.DOTALL)

#: 流水线里出现的"不是对话"的记录前缀（自带 schema，见 §四）。
_STREAMING_PREFIXES = (
    "streaming online qa feedback:",
    "streaming intra-transfer train trajectory:",
    "streaming inter-transfer train trajectory:",
)


def split_speaker(body: str, *, seen: Iterable[str] | None = None) -> tuple[str | None, str]:
    """把 ``"Christina Baker: I'm trying to..."`` 拆成 ``("Christina Baker", "I'm trying to...")``。

    拆不出来（没有冒号、前缀是内容类型/语气词、或长得不像名字）时返回 ``(None, body)`` ——
    **正文一字不动**。

    ``seen`` 是本批次里**已经出现过**的说话人前缀集合：一个前缀如果重复出现，它几乎必然是
    说话人（``Gina:`` 出现 20 次）而不是碰巧的正文冒号（``Wait:`` 出现一次）。这是把
    "该不该信这个前缀"从词表判断变成**语料内证据**判断。
    """
    match = _SPEAKER_RE.match(body or "")
    if not match:
        return None, body
    raw = match.group("name").strip()
    rest = match.group("rest")
    if not raw:
        return None, body

    lowered = raw.lower()
    if lowered in GENERIC_SPEAKERS:
        return GENERIC_SPEAKERS[lowered], rest
    # **只做整词匹配**。``NON_SPEAKER_PREFIXES`` 里有 ``and``/``he``/``no`` 这类短词，
    # 一旦改成"前缀匹配"，``Andrew:`` 会被 ``and`` 吃掉、``Henry:`` 会被 ``he`` 吃掉 ——
    # 真说话人被静默拒掉，而这正是这一层要修的毛病。多词短语（``close up``）同理要整体相等。
    if lowered in NON_SPEAKER_PREFIXES:
        return None, body
    if lowered.startswith(_STREAMING_PREFIXES):
        return None, body
    if len(raw) > _MAX_SPEAKER_CHARS or len(raw.split()) > _MAX_SPEAKER_WORDS:
        return None, body
    # 编号角色（``#8`` / ``Juror #2``）合法；纯数字（``1: ...``）不是名字，是列表编号。
    if raw.isdigit():
        return None, body

    confirmed = seen is not None and lowered in seen
    if not confirmed and not re.match(r"^[A-Z#]", raw):
        return None, body
    return raw, rest


# ---------------------------------------------------------------------------
# 三、JSON 载荷：三元组 / 流式记录
# ---------------------------------------------------------------------------

def looks_like_json(text: str) -> bool:
    """整段（或剥掉一行说明后的剩余）是不是一个 JSON 对象/数组。"""
    stripped = (text or "").strip()
    return bool(stripped) and stripped[0] in "{[" and stripped[-1] in "}]"


def _parse_json_payload(body: str) -> Any | None:
    """从 ``"<一句说明>\\n{...}"`` 里取出 JSON 对象；取不到返回 ``None``。

    容错：说明行可能很长（``Streaming ...:``），也可能被截断（``[part 1/2]`` 那种续写）。
    """
    text = (body or "").strip()
    if not text:
        return None
    if looks_like_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    for opener in ("{", "["):
        start = text.find(opener)
        if start < 0:
            continue
        candidate = text[start:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 被 max_tokens 截断的尾巴：取到最后一个闭合括号再试一次。
            closer = "}" if opener == "{" else "]"
            end = candidate.rfind(closer)
            if end > 0:
                try:
                    return json.loads(candidate[: end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def render_triple(payload: Any, *, update: bool = False) -> str:
    """知识图谱三元组 → 一行自足的陈述。

    源：``{"subject": "Latvia", "relation": "official language", "object": "Latvian", ...}``
    渲染：``Latvia | official language | Latvian``

    为什么是这个形状而不是一句英文：relation 是数据集自己的词（``country of citizenship`` /
    ``official language``），任何"改写成自然语言"的模板都会把它动词化错（``is the country of
    citizenship of`` vs ``is a citizen of``）。**逐字保留三段**既不会改写错，又能被 grep 到
    （query 里就是 ``"official language"`` 这三个词），下游 LLM 也读得懂 ``A | rel | B``。

    ``Q-id``（``subject_id``/``relation_id``/``object_id``）**丢掉**：它们只对数据集自己
    有意义，留在正文里只会污染 grep 与 atom。
    """
    if not isinstance(payload, dict):
        return str(payload)
    subject = str(payload.get("subject") or "").strip()
    relation = str(payload.get("relation") or "").strip()
    obj = str(payload.get("object") or "").strip()
    if not (subject or relation or obj):
        return json.dumps(payload, ensure_ascii=False)
    line = f"{subject} | {relation} | {obj}"
    if update:
        return f"UPDATE: {line} (this replaces the prior value for this subject and relation)"
    return line


def _render_json_lines(payload: Any) -> str:
    """把一个 JSON 对象渲染成 ``key: value`` 的多行文本（供流式记录用）。"""
    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False)
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 四、形态识别 + 逐条归一化
# ---------------------------------------------------------------------------

#: 语料形态。日志与统计用它，预处理的分支也用它。
FAMILIES = (
    "dialogue",        # 逐条对话，说话人在正文前缀里（真名或 user/assistant）
    "narration",       # 剧本旁白 / 舞台指示（``NARRATION: ...``）
    "document",        # 文档分页（``document: Document: xxx.pdf``）
    "triples",         # 知识图谱三元组（``Corpus: {...}``）
    "facts",           # 整段陈述句（``source: ...``）
    "qa-feedback",     # 流式问答反馈（JSON）
    "trajectory",      # agent 轨迹（JSON，内含 messages[]）
    "plain",           # 上面都不是：正文无前缀，role 只能用协议给的
)

# ``FAMILIES`` 是**穷举**：``adapt_messages`` 只会返回上面这八个之一，调用方（日志、统计脚本）
# 可以放心按它建表。多一个形态就要在这里加一行。


@dataclass
class AdaptedMessage:
    """归一化后的一条消息（``store`` 直接拿它建 msg 记录）。

    ``kind`` 是它所属的形态（``FAMILIES`` 之一），只作诊断用 —— 落盘到 msgmem 的是
    ``role`` / ``content`` / ``timestamp`` 三样。
    """

    role: str
    content: str
    timestamp: int | None
    kind: str = "plain"


@dataclass
class AdaptedBatch:
    """一个 ``/add`` 报文的归一化结果。

    ``family`` 是 ``FAMILIES`` 之一；``speakers`` 是本批次里认出来的**真名**说话人
    （按首次出现排序）；全是 ``user``/``assistant`` 时为空。
    """

    family: str
    messages: list[AdaptedMessage] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)


def _prefix_counts(contents: Sequence[str]) -> dict[str, int]:
    """统计每个候选说话人前缀在本批次出现几次（用于确认）。"""
    counts: dict[str, int] = {}
    for text in contents:
        _, body = split_lead_tags(text)
        match = _SPEAKER_RE.match(body)
        if not match:
            continue
        name = match.group("name").strip().lower()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _parse_tag_time(value: str) -> int | None:
    """``"Sep 04, 2025, 09:05:32"`` → Unix 毫秒。``"Unknown"`` 等解析不了的一律 ``None``。"""
    text = (value or "").strip()
    if not text or text.lower() in {"unknown", "n/a", "none", "-"}:
        return None
    for fmt in ("%b %d, %Y, %H:%M:%S", "%b %d, %Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%b %d, %Y"):
        try:
            moment = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return int(moment.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return None


def _adapt_triples(messages: Sequence[dict[str, Any]], tags_per: list[list[str]]) -> list[AdaptedMessage]:
    """``Corpus: {...}`` → ``Subject | relation | object``，role 固定 ``corpus``。"""
    out: list[AdaptedMessage] = []
    for message, tags in zip(messages, tags_per):
        _, body = split_lead_tags(str(message.get("content") or ""))
        update = body.lower().startswith("corpus: update")
        payload = _parse_json_payload(body)
        if payload is None:
            out.append(AdaptedMessage("corpus", body, message.get("timestamp"), "triples"))
            continue
        out.append(AdaptedMessage(
            "corpus", render_triple(payload, update=update),
            message.get("timestamp"), "triples",
        ))
    return out


def _adapt_trajectory(messages: Sequence[dict[str, Any]]) -> list[AdaptedMessage]:
    """``Streaming intra-transfer train trajectory: {...}`` → 逐条展开成对话。

    查询长这样：``"ALF task index 86"`` —— 它在问**第 86 条轨迹**里发生了什么。所以每条
    展开出来的消息都带上 ``ALF task index 86`` 这个字面串，agent 才 grep 得到；光把
    ``"task_index": 86`` 闷在 JSON 里，``grep "task index 86"`` 是搜不到的。

    展开而不是整段塞进一条：轨迹本身是 ``messages[]`` 序列，一条条落成 msg 记录之后，
    search agent 才能像读对话一样 ``read`` 它、按 id 引用它。
    """
    out: list[AdaptedMessage] = []
    for message in messages:
        timestamp = message.get("timestamp")
        _, body = split_lead_tags(str(message.get("content") or ""))
        payload = _parse_json_payload(body)
        if not isinstance(payload, dict):
            out.append(AdaptedMessage("user", body, timestamp, "trajectory"))
            continue
        dataset = str(payload.get("dataset") or "trajectory")
        index = payload.get("task_index")
        header = f"{dataset} task index {index}" if index is not None else dataset
        inner = payload.get("trajectory")
        if not isinstance(inner, list) or not inner:
            out.append(AdaptedMessage(
                "trajectory", f"{header}\n{_render_json_lines(payload)}", timestamp, "trajectory",
            ))
            continue
        for step in inner:
            if isinstance(step, dict):
                role = str(step.get("role") or "unknown")
                text = step.get("content")
                text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
            else:
                role, text = "unknown", str(step)
            out.append(AdaptedMessage(
                role, f"[{header}] {text}", timestamp, "trajectory",
            ))
    return out


def _adapt_qa_feedback(messages: Sequence[dict[str, Any]]) -> list[AdaptedMessage]:
    """``Streaming online QA feedback: {...}`` → 可读的多行记录，role 固定 ``feedback``。

    转义过的 JSON（``\\n`` 全糊在一行）**没法 grep**，也没法当记忆读 —— 展开成真换行之后
    ``question`` 与选项各自成行。``predicted_answer`` 原样保留但**标明是"此前给出的答案"**：
    它是平台流里的一条记录（不是我们编的），但下游不该把它当权威。
    """
    out: list[AdaptedMessage] = []
    for message in messages:
        timestamp = message.get("timestamp")
        _, body = split_lead_tags(str(message.get("content") or ""))
        payload = _parse_json_payload(body)
        if not isinstance(payload, dict):
            out.append(AdaptedMessage("feedback", body, timestamp, "qa-feedback"))
            continue
        lines = ["Streaming online QA feedback"]
        if payload.get("qa_id"):
            lines.append(f"qa_id: {payload['qa_id']}")
        if payload.get("question"):
            lines.append(f"question: {payload['question']}")
        if payload.get("predicted_answer") is not None:
            lines.append(
                f"answer given at the time (not authoritative): {payload['predicted_answer']}"
            )
        if payload.get("score") is not None:
            lines.append(f"score: {payload['score']}")
        if payload.get("method"):
            lines.append(f"method: {payload['method']}")
        for key, value in payload.items():
            if key in {"qa_id", "question", "predicted_answer", "score", "method",
                       "schema_version"}:
                continue
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            lines.append(f"{key}: {rendered}")
        out.append(AdaptedMessage("feedback", "\n".join(lines), timestamp, "qa-feedback"))
    return out


#: 各形态自带的前缀 → 该条消息的 role。这些不是说话人，是**记录类型**。
_RECORD_ROLES = (
    ("narration:", "narration"),
    ("document:", "document"),
    ("source:", "corpus"),
)
#: 舞台指示/镜头说明的前缀词（``STAGE:`` / ``CLOSE UP:`` / ``NOTE:``）—— 冒号后面是场景描写，
#: 不是某人说的话。role 记成 ``narration`` 比记成 ``user`` 诚实：原子抽取若把它当"用户说的"，
#: 会产出 ``"user closed the door"`` 这种把舞台指示拟人化的事实。
_SCENE_PREFIXES = frozenset({
    "stage", "stage direction", "scene", "close up", "medium shot", "wide shot",
    "long shot", "dissolve to", "cut to", "fade in", "fade out", "interior",
    "exterior", "camera", "voice over", "voiceover", "note", "note that",
})
#: 整批形态 → 该类记录的统一 role（``dialogue``/``plain`` 不在此表里，它们靠逐条拆前缀）。
_FAMILY_ROLES = {"facts": "corpus", "document": "document"}


def _consume_record_prefix(body: str) -> tuple[str, str]:
    """剥掉正文开头的"记录类型"前缀，返回 ``(role, 剩余正文)``；不是这类就返回 ``("", body)``。

    只剥一层，且只在冒号后有内容时剥 —— 与 ``split_speaker`` 同一套判据（不猜、无损失）。
    调用顺序上它**排在识别说话人之后**：``NARRATION: FOREMAN: 35 years old`` 里
    ``NARRATION`` 是类型、``FOREMAN`` 才是说话人，先剥类型再看剩下的有没有说话人。
    """
    for prefix, role in _RECORD_ROLES:
        if body.lower().startswith(prefix) and len(body) > len(prefix):
            return role, body[len(prefix):].lstrip()
    match = _SPEAKER_RE.match(body)
    if match and match.group("name").strip().lower() in _SCENE_PREFIXES:
        rest = match.group("rest").lstrip()
        if rest:
            return "narration", rest
    return "", body


def _adapt_dialogue_like(
    messages: Sequence[dict[str, Any]],
    tags_per: list[list[str]],
    counts: dict[str, int],
    *,
    family: str,
) -> tuple[list[AdaptedMessage], list[str]]:
    """``dialogue`` / ``narration`` / ``document`` / ``facts`` / ``plain`` 共用的拆前缀逻辑。

    每条消息的 role 按**从具体到笼统**的顺序取：

    1. 正文里的说话人前缀（``Christina Baker:`` / ``JUDGE:`` / ``user:``）—— 最具体；
    2. 记录类型前缀（``NARRATION:`` → ``narration``、``document:`` → ``document``、
       ``source:`` → ``corpus``）；
    3. 整批形态给的统一 role（``facts``/``document`` 两类）；
    4. 协议 role（``user``/``assistant``）—— 前三者都没有时，它至少说明"这是人机对话的哪一侧"。

    第 1、2 步之间是"先剥类型标记、再拆说话人"（同一个前缀里可能连着一层类型 + 一层说话人，
    见 ``_consume_record_prefix``）。
    """
    out: list[AdaptedMessage] = []
    speakers: list[str] = []
    family_role = _FAMILY_ROLES.get(family, "")
    for message, tags in zip(messages, tags_per):
        protocol_role = str(message.get("role") or "")
        timestamp = message.get("timestamp")
        tag = tag_time(tags)
        if tag:
            parsed = _parse_tag_time(tag)
            if parsed is not None:
                timestamp = parsed
        _, body = split_lead_tags(str(message.get("content") or ""))
        body = body.strip()

        record_role, stripped = _consume_record_prefix(body)
        speaker, rest = split_speaker(stripped, seen=counts)
        if speaker is not None:
            role, content = speaker, (rest or stripped)
            if role not in GENERIC_SPEAKERS.values() and role not in speakers:
                speakers.append(role)
        elif record_role:
            role, content = record_role, stripped
        elif family_role:
            role, content = family_role, body
        else:
            role, content = (protocol_role or "unknown"), body
        out.append(AdaptedMessage(role, content or body, timestamp, family))
    return out, speakers


def _classify(bodies: Sequence[str], tags_per: list[list[str]]) -> str:
    """整批的形态。**看正文前缀，不看协议 role、也不看数据集名。**

    结构性的那几类（三元组 / 流式记录 / 文档 / 整段陈述）由**首条**消息判定 —— 它们整批
    同构。剩下的一律是"对话/剧本"，此时再用**整批**的说话人证据：只要有一条能拆出说话人
    就算 ``dialogue``（剧本的第一条常是演员表 NOTE，没有说话人，但后面全是 ``JUDGE:``）。
    一条都拆不出来的（中文医患对话、纯陈述）落到 ``plain``。
    """
    first = (bodies[0] if bodies else "").lstrip().lower()
    for prefix in _STREAMING_PREFIXES:
        if first.startswith(prefix):
            return "qa-feedback" if "qa feedback" in prefix else "trajectory"
    if first.startswith("corpus:"):
        return "triples"
    if first.startswith("document:"):
        return "document"
    if first.startswith("source:"):
        return "facts"
    if _is_narration_body(bodies[0] if bodies else "", tags_per[0] if tags_per else []):
        return "narration"

    counts = collections.Counter()
    for body in bodies:
        match = _SPEAKER_RE.match(body.lstrip())
        if match:
            counts[match.group("name").strip().lower()] += 1
    for body in bodies:
        speaker, _ = split_speaker(body.lstrip(), seen=counts)
        if speaker is not None:
            return "dialogue"
    return "plain"


def _is_narration_body(body: str, tags: Sequence[str]) -> bool:
    """正文（或记账标签）明说这是旁白/舞台指示吗。"""
    if (body or "").lstrip().lower().startswith("narration:"):
        return True
    return any(tag.lower() == "narration" for tag in tags)


def adapt_messages(messages: Sequence[dict[str, Any]]) -> AdaptedBatch:
    """把一个 ``/add`` 的 messages 归一化成三层记忆能吃的形状。

    ``messages`` 里的 ``content`` 必须**已经是字符串**（多模态由 ``api.models.normalize_content``
    先降级，见 ``server.add``）。返回的 ``AdaptedMessage`` 里：

    - ``role``：真名（``Christina Baker`` / ``Caroline``）优先，其次剧情角色（``JUDGE``），
      再次协议 role（``user``/``assistant``），最后 ``unknown``；
    - ``content``：摘掉平台记账标签与说话人前缀之后的**正文**；
    - ``timestamp``：协议给的毫秒时间戳；协议没给时用 ``[Time: ...]`` 标签里的时间补。

    **怎么判形态**：看**首条消息的正文**（而不是看协议 role、也不看数据集名）。协议 role
    在这些数据集上几乎恒为 ``user``，数据集名是平台内部的 key（换个名字就失效）；只有正文
    的前缀是语料自己长出来的，稳定可判。
    """
    if not messages:
        return AdaptedBatch(family="plain", messages=[], speakers=[])

    split = [split_lead_tags(str(m.get("content") or "")) for m in messages]
    tags_per = [tags for tags, _ in split]
    bodies = [body for _, body in split]
    family = _classify(bodies, tags_per)

    if family == "triples":
        return AdaptedBatch(family=family, messages=_adapt_triples(messages, tags_per), speakers=[])
    if family == "trajectory":
        return AdaptedBatch(family=family, messages=_adapt_trajectory(messages), speakers=[])
    if family == "qa-feedback":
        return AdaptedBatch(family=family, messages=_adapt_qa_feedback(messages), speakers=[])

    counts = _prefix_counts([str(m.get("content") or "") for m in messages])
    adapted, speakers = _adapt_dialogue_like(messages, tags_per, counts, family=family)
    return AdaptedBatch(family=family, messages=adapted, speakers=speakers)
