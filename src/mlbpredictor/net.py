"""Tiny HTTP helper with retries and an on-disk text cache.

Outbound HTTPS in this environment goes through a pre-configured agent proxy whose
CA is already in the system trust store, so ``requests`` works with default TLS
verification (and honours ``REQUESTS_CA_BUNDLE`` if set). We keep a simple file
cache under ``data/mlb/retrosheet`` so the pipeline runs offline after first fetch.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .paths import RETRO_DIR

_DEFAULT_TIMEOUT = 30


def fetch_text(url: str, *, timeout: int = _DEFAULT_TIMEOUT, retries: int = 4) -> str | None:
    """GET ``url`` and return text, or ``None`` on a 404 / persistent failure."""
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:      # network / TLS / 5xx
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    print(f"  [net] giving up on {url}: {last_exc}")
    return None


def cached_text(url: str, cache_name: str, *, refresh: bool = False,
                timeout: int = _DEFAULT_TIMEOUT) -> str | None:
    """Return ``url`` text, caching the raw bytes under ``RETRO_DIR/cache_name``."""
    path = RETRO_DIR / cache_name
    if path.exists() and not refresh:
        return path.read_text()
    text = fetch_text(url, timeout=timeout)
    if text is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text
