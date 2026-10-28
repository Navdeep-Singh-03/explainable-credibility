"""Handling of untrusted retrieved content.

Everything that comes back from a provider (titles, excerpts, ratings, URLs, dates) is DATA.
This module reduces it to inert, bounded plain text. The module never executes it, never follows
links found inside it, and never feeds it to anything that interprets instructions.
"""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")  # zero-width + bidi overrides
_TAG = re.compile(r"<[^>]{0,2000}>")
_WS = re.compile(r"\s+")
_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|igshid|ref|ref_src)$", re.I)


def clean_text(value: object, max_chars: Optional[int] = None) -> str:
    """Return inert plain text: no markup, control characters, bidi tricks; whitespace collapsed; bounded."""
    if not isinstance(value, str):
        return ""
    v = html.unescape(value)
    v = _TAG.sub(" ", v)
    v = unicodedata.normalize("NFC", v)
    v = _CTRL.sub(" ", v)
    v = _INVISIBLE.sub("", v)
    v = _WS.sub(" ", v).strip()
    if max_chars is not None and len(v) > max_chars:
        v = v[: max(1, max_chars - 1)].rstrip() + "\u2026"
    return v


def safe_url(value: object, allowed_schemes: Iterable[str] = ("http", "https"), max_len: int = 2048) -> Optional[str]:
    """Return the URL unchanged if it is a plain http(s) URL, else None. Never repairs or invents URLs."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or len(v) > max_len or re.search(r"[\s\x00-\x1f\x7f]", v):
        return None
    try:
        p = urlsplit(v)
        _ = p.port  # raises ValueError on a malformed port
    except ValueError:
        return None
    if p.scheme.lower() not in {s.lower() for s in allowed_schemes} or not p.hostname:
        return None
    if p.username or p.password:  # credentials in URLs are a phishing / leakage pattern
        return None
    return v


def hostname(url: str) -> str:
    try:
        h = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def canonical_url(url: str) -> str:
    """Key for de-duplication only (never shown as the evidence URL)."""
    p = urlsplit(url)
    q = urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k)])
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), hostname(url), path, q, ""))


_FRAC = re.compile(r"(\.\d{6})\d+")


def parse_date(value: object, now: Optional[datetime] = None) -> Optional[str]:
    """Parse an RFC-3339 / ISO date into 'YYYY-MM-DDTHH:MM:SSZ'. Unparseable or future dates give None."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = _FRAC.sub(r"\1", value.strip())
    if v.endswith(("Z", "z")):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    now = now or datetime.now(timezone.utc)
    if dt > now + timedelta(days=1):  # a future publication date is a data error, not information
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_to_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def iso_now(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
