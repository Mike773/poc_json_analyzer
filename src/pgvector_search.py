"""Postgres + pgvector — кеш семантического поиска метрик (ТЗ §2.2, §9.2).

Слой опциональный: активируется когда уникальных метрик в направлении достаточно
много (config.postgres_min_metrics_per_direction), либо вручную через
populate_search_cache().

Структура — см. db/init/01_schema.sql.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from openai import OpenAI


DEFAULT_DSN = os.environ.get(
    "POC_PG_DSN",
    "postgresql://poc:poc@localhost:5434/poc_metrics",
)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def connect(dsn: str = DEFAULT_DSN) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=False)


def ping(dsn: str = DEFAULT_DSN) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tokens / hashing
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _name_hash(name: str) -> str:
    return hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()


def _tokens(name: str) -> List[str]:
    """Извлекаем токены имени, приводим все к нижнему регистру (case-insensitive matching)."""
    return [t.lower() for t in _TOKEN_RE.findall(name)]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


_openai_client: Optional[OpenAI] = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY не задан — нужен для embeddings (text-embedding-3-small)"
            )
        _openai_client = OpenAI()
    return _openai_client


def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _openai().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Populate cache from SQLite catalog
# ---------------------------------------------------------------------------


def _format_vector(v: List[float]) -> str:
    """pgvector принимает строку '[v1,v2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def populate_search_cache(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    direction_key: Optional[str] = None,
) -> Dict[str, int]:
    """Идемпотентно: грузит новые/изменённые метрики (по name_hash) из metric_catalog → metric_search_cache.

    direction_key: если None — берётся department у любого сотрудника в данной сессии.
    """
    # direction_key
    if direction_key is None:
        row = sqlite_conn.execute(
            "SELECT department FROM dim_employee WHERE department IS NOT NULL LIMIT 1"
        ).fetchone()
        direction_key = row["department"] if row else "default"

    metrics = list(
        sqlite_conn.execute("SELECT metric_id, name FROM metric_catalog").fetchall()
    )

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT metric_id, name_hash FROM metric_search_cache WHERE direction_key = %s",
            (direction_key,),
        )
        existing = {r[0]: r[1] for r in cur.fetchall()}

    to_process: List[Tuple[int, str, str, List[str]]] = []
    for m in metrics:
        mid = m["metric_id"]
        name = m["name"]
        nh = _name_hash(name)
        if existing.get(mid) == nh:
            continue
        to_process.append((mid, name, nh, _tokens(name)))

    stats = {"total": len(metrics), "unchanged": len(metrics) - len(to_process), "upserted": 0}
    if not to_process:
        return stats

    vectors = embed_batch([item[1] for item in to_process])
    assert len(vectors) == len(to_process)

    with pg_conn.cursor() as cur:
        for (mid, name, nh, toks), vec in zip(to_process, vectors):
            cur.execute(
                """INSERT INTO metric_search_cache
                     (direction_key, metric_id, name, name_hash, abbreviations, vector, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s::vector, now())
                   ON CONFLICT (direction_key, metric_id) DO UPDATE
                     SET name = EXCLUDED.name,
                         name_hash = EXCLUDED.name_hash,
                         abbreviations = EXCLUDED.abbreviations,
                         vector = EXCLUDED.vector,
                         updated_at = now()""",
                (direction_key, mid, name, nh, toks, _format_vector(vec)),
            )
            stats["upserted"] += 1
    pg_conn.commit()
    return stats


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_metrics_pgvector(
    pg_conn: psycopg.Connection,
    query: str,
    direction_key: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Гибридный поиск: точное совпадение токенов (через GIN на abbreviations)
    + cosine similarity по vector. Результаты с exact-match идут первыми."""
    q = query.strip()
    if not q:
        return []

    tokens = _tokens(q)
    query_vec = embed_batch([q])[0]
    qvec_str = _format_vector(query_vec)

    with pg_conn.cursor() as cur:
        # Exact: пересечение токенов запроса с abbreviations
        exact_rows: List[Dict[str, Any]] = []
        if tokens:
            cur.execute(
                """SELECT metric_id, name, abbreviations,
                          (vector <=> %s::vector) AS cosine_dist
                   FROM metric_search_cache
                   WHERE direction_key = %s
                     AND abbreviations && %s::text[]
                   ORDER BY cosine_dist ASC
                   LIMIT %s""",
                (qvec_str, direction_key, tokens, top_k),
            )
            for r in cur.fetchall():
                exact_rows.append(
                    {
                        "metric_id": r[0],
                        "name": r[1],
                        "abbreviations": r[2],
                        "cosine_distance": float(r[3]),
                        "match": "exact_token",
                    }
                )

        # Vector: top_k по cosine, исключая exact-матчи
        exact_ids = {r["metric_id"] for r in exact_rows}
        cur.execute(
            """SELECT metric_id, name, abbreviations, (vector <=> %s::vector) AS cosine_dist
               FROM metric_search_cache
               WHERE direction_key = %s
                 AND NOT (metric_id = ANY(%s))
               ORDER BY cosine_dist ASC
               LIMIT %s""",
            (qvec_str, direction_key, list(exact_ids) or [0], top_k),
        )
        vector_rows = [
            {
                "metric_id": r[0],
                "name": r[1],
                "abbreviations": r[2],
                "cosine_distance": float(r[3]),
                "match": "vector",
            }
            for r in cur.fetchall()
        ]

    return (exact_rows + vector_rows)[:top_k]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def get_direction_key(sqlite_conn: sqlite3.Connection) -> str:
    row = sqlite_conn.execute(
        "SELECT department FROM dim_employee WHERE department IS NOT NULL LIMIT 1"
    ).fetchone()
    return row["department"] if row else "default"
