"""Deterministic relative-time resolution shared by retrieval and verification.

A record's ``time``/``observed_at`` is the anchor for expressions such as
"yesterday", "last week", or "last year".  Resolving them once, in one place,
keeps the evidence annotation, the verifier input, and claim grounding
consistent.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_REL_DAY = re.compile(r"\b(yesterday|today|tomorrow|last night)\b", re.IGNORECASE)
_REL_WEEK = re.compile(r"\b(last week|this week|next week)\b", re.IGNORECASE)
_REL_MONTH = re.compile(r"\b(last month|this month|next month)\b", re.IGNORECASE)
_REL_YEAR = re.compile(r"\b(last year|this year|next year)\b", re.IGNORECASE)
_REL_AGO = re.compile(
    r"\b(?:(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|fifteen|twenty|thirty)\s+)?"
    r"(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE)
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30,
}
_REL_WEEKDAY = re.compile(
    r"\b(last|this|next)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE)
_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def parse_observed(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_relative_dates(text: str, observed_at: str) -> dict[str, str]:
    """Resolve common relative expressions against the observation time."""
    moment = parse_observed(observed_at)
    if moment is None:
        return {}
    out: dict[str, str] = {}
    body = text or ""
    for match in _REL_DAY.finditer(body):
        phrase = match.group(1).lower()
        if phrase in {"yesterday", "last night"}:
            day = moment - timedelta(days=1)
        elif phrase == "tomorrow":
            day = moment + timedelta(days=1)
        else:
            day = moment
        out[phrase] = day.strftime("%Y-%m-%d")
    for match in _REL_WEEK.finditer(body):
        phrase = match.group(1).lower()
        monday = (moment - timedelta(days=moment.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        if phrase == "last week":
            start = monday - timedelta(days=7)
        elif phrase == "next week":
            start = monday + timedelta(days=7)
        else:
            start = monday
        out[phrase] = f"{start:%Y-%m-%d}..{(start + timedelta(days=6)):%Y-%m-%d}"
    for match in _REL_MONTH.finditer(body):
        phrase = match.group(1).lower()
        year, month = moment.year, moment.month
        if phrase == "last month":
            year, month = (year - 1, 12) if month == 1 else (year, month - 1)
        elif phrase == "next month":
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        out[phrase] = f"{year:04d}-{month:02d}"
    for match in _REL_YEAR.finditer(body):
        phrase = match.group(1).lower()
        year = moment.year + (-1 if phrase == "last year" else 1 if phrase == "next year" else 0)
        out[phrase] = f"{year:04d}"
    for match in _REL_AGO.finditer(body):
        # "a few years ago" / "several years ago" are deliberately vague; the
        # optional count group would otherwise read them as one year and
        # produce a confident but wrong absolute date.
        prefix = body[max(0, match.start() - 16):match.start()].lower()
        if re.search(r"(?:a\s+few|few|several|a\s+couple\s+of|couple\s+of)\s*$",
                     prefix):
            continue
        raw_count = (match.group("count") or "1").lower()
        count = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS.get(
            raw_count, 1)
        unit = match.group("unit").lower()
        days = count * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        day = moment - timedelta(days=days)
        phrase = match.group(0).lower()
        if unit == "year":
            out[phrase] = f"{day.year:04d}"
        elif unit == "month":
            out[phrase] = f"{day.year:04d}-{day.month:02d}"
        else:
            out[phrase] = f"{day:%Y-%m-%d}"
    # "last Friday" is extremely common in conversational memory ("I passed
    # the interview last Friday").  It was previously unresolved, so the model
    # had to do calendar arithmetic itself -- and occasionally invented a date
    # (e.g. 2023-10-18 instead of the Friday before 2023-10-22).
    for match in _REL_WEEKDAY.finditer(body):
        modifier = match.group(1).lower()
        weekday = match.group(2).lower()
        target = _WEEKDAY_INDEX[weekday]
        if modifier == "last":
            # "last Friday" on a Friday means the previous Friday, not today.
            days = (moment.weekday() - target) % 7 or 7
            day = moment - timedelta(days=days)
        elif modifier == "this":
            # "this Friday" is the Friday in the current Monday-Sunday week,
            # even when that day is still ahead (Wednesday -> Friday).
            monday = moment - timedelta(days=moment.weekday())
            day = monday + timedelta(days=target)
        else:  # next
            day = moment + timedelta(
                days=(target - moment.weekday()) % 7 or 7)
        out[f"{modifier} {weekday}"] = f"{day:%Y-%m-%d}"
    return out
