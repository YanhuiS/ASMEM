"""统一的日志：级别过滤 + 终端输出 + 文件保存（add/search/answer/eval 共用）。

设计目标（按重要性）：

1. **能看到真实数据流**。``debug`` 级别要把每一步的实际内容打出来 —— 召回 query、
   召回了哪些记忆、完整的 prompt 消息、模型的原始输出、动作解析结果、每轮前后的库状态。
   这些是排查"为什么模型回 NOOP / 为什么删错"的唯一依据。
2. **级别可配**。``configs/search.yaml`` 的 ``log.level`` 控制终端与文件的最低级别；
   命令行 ``--log-level`` 可临时覆盖，方便不改配置就看清一轮。
3. **可保存**。``log.output`` 指定目录时，日志同时写入
   ``{output}/{level}_{ts}.log``，跑完还能回看；为空则只打终端。
4. **并发安全**。多个 speaker 目录并行时日志会交错，所以每行都带 ``[目录名]`` 前缀
   （``bind`` 出的 logger），并且写文件加锁。

级别：``debug`` < ``info`` < ``warn`` < ``error``。``silent`` 关掉全部输出。
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "silent": 100}
DEFAULT_LEVEL = "info"

from .io import PROJECT_ROOT


def parse_level(value: Any) -> str:
    """把配置里的级别规整成小写合法值；非法值退回 info。"""
    text = str(value or "").strip().lower()
    return text if text in LEVELS else DEFAULT_LEVEL


class Logger:
    """一个简单的分级 logger：终端 + 可选文件，线程安全，支持 ``bind`` 加前缀。"""

    def __init__(
        self,
        level: str = DEFAULT_LEVEL,
        path: Path | None = None,
        prefix: str = "",
        _lock: threading.Lock | None = None,
        _stream: Any = None,
        _handle: Any = None,
    ) -> None:
        self.level = parse_level(level)
        self.threshold = LEVELS[self.level]
        self.prefix = prefix
        self.path = path
        self._lock = _lock or threading.Lock()
        self._stream = _stream if _stream is not None else sys.stdout
        self._handle = _handle
        self.counts: dict[str, int] = {name: 0 for name in LEVELS}

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        prefix: str = "asmem",
    ) -> "Logger":
        """按 ``config['log']`` 建 logger。

        ``log.output`` 语义：
        - 未设置 / 空 → 只打终端，不落盘
        - 目录路径 → 写入 ``{该目录}/{prefix}_{level}_{ts}.log``
        ``prefix`` 用来区分是哪个步骤的日志（``add`` / ``search`` / ``answer`` / ``eval``）。
        """
        section = config.get("log") or {}
        level = parse_level(section.get("level", DEFAULT_LEVEL))
        path: Path | None = None
        target = str(section.get("output") or "").strip()
        if target:
            directory = Path(target)
            if not directory.is_absolute():
                directory = PROJECT_ROOT / directory
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = directory / f"{prefix}_{level}_{stamp}.log"
        handle = path.open("w", encoding="utf-8") if path else None
        logger = cls(level=level, path=path, _handle=handle)
        if path:
            logger.info(f"日志落盘 {path}（级别 {level}）")
        return logger

    def bind(self, prefix: str) -> "Logger":
        """派生出带前缀的 logger，共享级别/文件/锁。并行跑多目录时用它区分来源。"""
        child = Logger(
            level=self.level,
            prefix=prefix,
            _lock=self._lock,
            _stream=self._stream,
            _handle=self._handle,
        )
        return child

    # -- 输出 ---------------------------------------------------------------
    def _emit(self, name: str, message: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        if LEVELS[name] < self.threshold:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        head = f"[{stamp}] {name.upper():5s}"
        if self.prefix:
            head += f" [{self.prefix}]"
        line = f"{head} {message}"
        with self._lock:
            if self._stream is not None:
                print(line, file=self._stream, flush=True)
            if self._handle is not None:
                self._handle.write(line + "\n")
                self._handle.flush()

    def debug(self, message: str) -> None:
        self._emit("debug", message)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warn(self, message: str) -> None:
        self._emit("warn", message)

    def error(self, message: str) -> None:
        self._emit("error", message)

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.counts.items() if v]
        return "日志 " + " ".join(parts) if parts else "日志 无输出"

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None


def block(title: str, body: str, limit: int = 8000) -> str:
    """把一段多行内容包装成便于阅读/截断的日志块。"""
    body = body if len(body) <= limit else body[:limit] + f"\n... (截断，共 {len(body)} 字符)"
    return f"----- {title} -----\n{body}\n----- /{title} -----"
