"""Postgres + pgvector — кеш семантического поиска метрик (ТЗ §2.2, §9.2).

Слой опциональный: активируется когда уникальных метрик в направлении достаточно
много (config.postgres_min_metrics_per_direction), либо вручную через
populate_search_cache().

Эмбеддинги считаются через LangChain (src/providers.py): провайдер openai или
gigachat. Размерность вектора зависит от провайдера, поэтому DDL таблицы
metric_search_cache держим в коде (ensure_schema), а не в db/init/01_schema.sql.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from langchain_core.embeddings import Embeddings

from .providers import embedding_dim, make_embeddings


DEFAULT_DSN = os.environ.get(
    "POC_PG_DSN",
    "postgresql://poc:poc@localhost:5434/poc_metrics",
)


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
# Embeddings (через LangChain — провайдер из src/providers.py)
# ---------------------------------------------------------------------------


_embeddings: Optional[Embeddings] = None


def _emb() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = make_embeddings()
    return _embeddings


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Эмбеддинги для набора документов (имён метрик)."""
    if not texts:
        return []
    return _emb().embed_documents(list(texts))


def embed_query(text: str) -> List[float]:
    """Эмбеддинг одного поискового запроса."""
    return _emb().embed_query(text)


# ---------------------------------------------------------------------------
# Populate cache from SQLite catalog
# ---------------------------------------------------------------------------


def _format_vector(v: List[float]) -> str:
    """pgvector принимает строку '[v1,v2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _existing_vector_dim(cur: psycopg.Cursor) -> Optional[int]:
    """Размерность колонки vector в metric_search_cache; None — если таблицы нет."""
    cur.execute("SELECT to_regclass('metric_search_cache')")
    if cur.fetchone()[0] is None:
        return None
    # Для типа vector pgvector хранит размерность напрямую в atttypmod.
    cur.execute(
        """SELECT atttypmod FROM pg_attribute
           WHERE attrelid = 'metric_search_cache'::regclass AND attname = 'vector'"""
    )
    row = cur.fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return None
    return int(row[0])


def ensure_schema(pg_conn: psycopg.Connection) -> None:
    """Создаёт metric_search_cache с размерностью вектора под активный провайдер.

    Если таблица уже есть, но с другой размерностью (сменили провайдер эмбеддингов) —
    пересоздаёт её: это кеш, он полностью восстанавливается populate_search_cache().
    """
    dim = embedding_dim()
    with pg_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        existing = _existing_vector_dim(cur)
        if existing is not None and existing != dim:
            print(
                f"metric_search_cache: размерность вектора {existing} ≠ {dim} "
                f"(сменился провайдер эмбеддингов) — пересоздаю кеш.",
                file=sys.stderr,
            )
            cur.execute("DROP TABLE metric_search_cache")
            existing = None
        if existing is None:
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS metric_search_cache (
                       direction_key text NOT NULL,
                       metric_id     bigint NOT NULL,
                       name          text NOT NULL,
                       name_hash     text NOT NULL,
                       abbreviations text[] DEFAULT '{{}}',
                       vector        vector({dim}),
                       created_at    timestamptz NOT NULL DEFAULT now(),
                       updated_at    timestamptz NOT NULL DEFAULT now(),
                       PRIMARY KEY (direction_key, metric_id)
                   )"""
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metric_search_cache_vector "
                "ON metric_search_cache USING hnsw (vector vector_cosine_ops)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metric_search_cache_abbr "
                "ON metric_search_cache USING gin (abbreviations)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metric_search_cache_direction "
                "ON metric_search_cache (direction_key)"
            )
    pg_conn.commit()


def populate_search_cache(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    direction_key: Optional[str] = None,
) -> Dict[str, int]:
    """Идемпотентно: грузит новые/изменённые метрики (по name_hash) из metric_catalog → metric_search_cache.

    direction_key: если None — берётся department у любого сотрудника в данной сессии.
    """
    ensure_schema(pg_conn)

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
    query_vec = embed_query(q)
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
