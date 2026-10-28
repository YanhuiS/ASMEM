"""挑战赛协议的 pydantic 模型 + 多模态 ``content`` 的归一化。

协议里 ``content``（以及 Search 的 ``query``）有两种形态，由赛道决定：

- **文本 / 代码赛道**：一个字符串。
- **多模态赛道**：有序的 ``ContentPart[]`` 数组。

底层语料是**文本** JSONL（search agent 用 grep 检索、用 ``date -d`` 做时间算术），所以
多模态在这里被**降级成文本**：文本 part 原样保留，图片 part 变成 ``[image: <ref>]`` 这样的
可检索、可溯源的文本标记。内联 base64 只按 MIME/大小记一个摘要，**绝不写进正文**。

完整的多模态分支（图片落盘、caption 生成与返回）在 :mod:`asmem.multimodal`；本模块的
``normalize_content`` 是无 store 场景下的纯文本兜底。归一化不做任何语义改写。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """Add 里的一条消息。

    ``timestamp`` 是可选、Unix **毫秒**（协议规定）；分段不改变其值。缺省则该消息 ``time`` 为空串。
    """

    role: str
    content: Any                       # str | list[ContentPart]；在 store 里经 normalize_content 归一
    timestamp: int | None = None


class AddRequest(BaseModel):
    """``POST /add`` 的请求体。``request_id`` 是幂等键（重试时保持不变）。

    空 ``messages`` / 空 id 属于协议错误，直接 422；单条消息内容为空仍在归一化
    阶段被丢弃，不影响同批其他消息。
    """

    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class SearchRequest(BaseModel):
    """``POST /search`` 的请求体。

    ``options`` 只在选择题（含 Streaming）时发送，**不含金标答案**；本实现保持"基准原题"语义，
    不把 options 喂给检索 —— 它们只是原样随请求到达。

    ``request_id`` 是可选幂等键（协议不要求）。带上它时，同一个请求可以在断连后重试：
    已完成的运行直接返回缓存证据，中断的运行从上次写好的证据继续。
    """

    query: Any                         # str | list[ContentPart]
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=0)
    request_id: str | None = None


# ---------------------------------------------------------------------------
# 响应
# ---------------------------------------------------------------------------

class AddResponse(BaseModel):
    """``POST /add`` 的响应：三个 id **原样回显**。"""

    success: bool
    request_id: str
    user_id: str
    session_id: str


class MemoryItem(BaseModel):
    """Search 结果里的一条记忆。

    ``content`` 总会非空：文本/代码赛道是字符串；多模态赛道是有序
    ``ContentPart[]``（本系统只产出文本 part）。``score`` 单调不增，按返回顺序
    表示相关性；``created_at`` 是来源记录的观测时间。
    """

    id: str
    content: Any
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    """``POST /search`` 的响应。**无结果时 ``data`` 是 ``[]``，绝不缺失。**"""

    data: list[MemoryItem]


# ---------------------------------------------------------------------------
# 多模态 content 归一化
# ---------------------------------------------------------------------------

def _part_to_text(part: Any) -> str:
    """把一个 ContentPart 变成文本。

    容错几种常见形态（不同赛道/bedrock 风格写法不一）：

    - ``"some text"``                      —— 已经是字符串
    - ``{"type": "text", "text": "..."}``
    - ``{"type": "image_url", "image_url": {"url": "..."}}``
    - ``{"type": "image", "url"/"source"/"data": "..."}``
    - ``{"image_url": "..."}`` / ``{"url": "..."}`` 等缺 type 的形态

    图片一律降级成 ``[image: <ref>]`` —— 保留引用便于审计与人工核对，但不假装理解它。
    内联 Base64 **绝不进入正文**：只写 ``[image: <mime>, ~<bytes> bytes]``，否则一条图片
    就能把 memory、BM25 和 prompt 撑爆（多模态赛道的 base64 单图可达 10 MiB）。
    """
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return str(part)

    ptype = str(part.get("type") or "").lower()

    # 显式文本 part
    if ptype in ("text", "input_text", "output_text") or "text" in part:
        text = part.get("text")
        if isinstance(text, str):
            return text

    # 图片 part：抽出可读的引用（URL / 文件名 / base64 说明）
    ref = ""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        ref = str(image_url.get("url") or "")
    elif isinstance(image_url, str):
        ref = image_url
    for key in ("url", "file_name", "filename", "path", "name"):
        if not ref and isinstance(part.get(key), str):
            ref = str(part[key])
    if not ref:
        source = part.get("source")
        if isinstance(source, dict):
            ref = str(source.get("url") or source.get("data") or "")
        elif isinstance(source, str):
            ref = source
    if not ref and isinstance(part.get("data"), str):
        ref = f"<{len(part['data'])} bytes>"   # base64：不给全文（会污染语料），只记规模
    if ptype.startswith("image") or ref:
        return _image_ref_text(ref)
    return ""


def _image_ref_text(ref: str) -> str:
    """Short, audit-safe reference; inline base64 is summarized, never copied."""
    ref = str(ref or "").strip()
    if not ref:
        return "[image]"
    if ref.startswith("data:"):
        header, _, payload = ref.partition(",")
        mime = header[5:].split(";", 1)[0].strip() if header.startswith("data:") else ""
        size = (len(payload) * 3) // 4 if payload else 0
        suffix = f", ~{size} bytes" if size else ""
        return f"[image: {mime or 'inline'}{suffix}]"
    return f"[image: {ref}]"


def normalize_content(content: Any) -> str:
    """把 ``str`` 或有序 ``ContentPart[]`` 归一化成一个字符串。

    数组按**原顺序**逐 part 转文本，丢弃空 part，用换行连接 —— 顺序是协议明确要求的，
    所以不做排序、不合并同类项。空数组 / ``None`` 返回空串。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(text for text in (_part_to_text(p) for p in content) if text)
    return str(content)
