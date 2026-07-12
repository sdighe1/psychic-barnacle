"""Player identity bridge, backed by the Chadwick Bureau register.

The offline model is keyed by **Retrosheet** ids (from event files / game logs); the
live ``statsapi.mlb.com`` feed is keyed by **MLBAM** ids and full names. The register
(``people-*.csv``) carries ``key_retro``, ``key_mlbam`` and names, letting us join the
two. It also carries ``birth_year`` for the projection age curve.

The register is sharded into ``people-0.csv`` ... ``people-9.csv``; we download and
concatenate them once, cache the slim subset we need, and expose lookups.
"""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd

from .config import load_config
from .net import fetch_text
from .paths import CACHE_DIR

_COLS = ["key_retro", "key_mlbam", "name_last", "name_first", "birth_year"]


def load_register(refresh: bool = False) -> pd.DataFrame:
    """Return the slim register: retro_id, mlbam_id, name, birth_year. Cached."""
    cache = CACHE_DIR / "register.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    cfg = load_config()["data"]
    base = cfg["register_base"]
    # The register is sharded by the first hex digit of each person's UUID:
    # people-0.csv ... people-9.csv, people-a.csv ... people-f.csv (16 shards).
    n = int(cfg["register_shards"])
    suffixes = "0123456789abcdef"[:n] if n <= 16 else "0123456789abcdef"
    frames = []
    for suffix in suffixes:
        text = fetch_text(f"{base}/people-{suffix}.csv")
        if not text:
            continue
        df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in _COLS,
                         dtype=str, low_memory=False)
        frames.append(df)
    if not frames:
        raise RuntimeError("Could not download the Chadwick register (people-*.csv).")

    reg = pd.concat(frames, ignore_index=True)
    reg = reg.dropna(subset=["key_retro"])
    reg = reg[reg["key_retro"].str.len() > 0]
    reg["name"] = (reg["name_first"].fillna("") + " " + reg["name_last"].fillna("")).str.strip()
    reg["birth_year"] = pd.to_numeric(reg["birth_year"], errors="coerce")
    reg = reg[["key_retro", "key_mlbam", "name", "birth_year"]].rename(
        columns={"key_retro": "retro_id", "key_mlbam": "mlbam_id"})
    reg = reg.drop_duplicates(subset=["retro_id"])
    reg.to_parquet(cache, index=False)
    return reg


@lru_cache(maxsize=1)
def _maps() -> tuple[dict, dict, dict, dict]:
    reg = load_register()
    retro_to_name = dict(zip(reg["retro_id"], reg["name"]))
    retro_to_birth = dict(zip(reg["retro_id"], reg["birth_year"]))
    has_mlbam = reg.dropna(subset=["mlbam_id"])
    has_mlbam = has_mlbam[has_mlbam["mlbam_id"].str.len() > 0]
    mlbam_to_retro = dict(zip(has_mlbam["mlbam_id"], has_mlbam["retro_id"]))
    name_to_retro = dict(zip(reg["name"].str.lower(), reg["retro_id"]))
    return retro_to_name, retro_to_birth, mlbam_to_retro, name_to_retro


def name_for(retro_id: str) -> str:
    return _maps()[0].get(retro_id, retro_id)


def birth_year_for(retro_id: str) -> float | None:
    y = _maps()[1].get(retro_id)
    return None if y is None or pd.isna(y) else float(y)


def retro_for_mlbam(mlbam_id: str | int) -> str | None:
    """Map a live statsapi MLBAM id to the offline Retrosheet id."""
    return _maps()[2].get(str(mlbam_id))


def retro_for_name(name: str) -> str | None:
    """Fallback: map a display name to a Retrosheet id (case-insensitive)."""
    return _maps()[3].get((name or "").strip().lower())
