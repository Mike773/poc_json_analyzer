"""Загружает metric_catalog (SQLite) в metric_search_cache (Postgres + pgvector).

Идемпотентно: повторный запуск пропускает неизменённые метрики (по name_hash).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import open_session_db
from src.ingest import load_json
from src.pgvector_search import connect, populate_search_cache, get_direction_key, ping


JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "call_center_metrics.json")
JSON_PATH = os.path.normpath(JSON_PATH)


def main() -> None:
    if not ping():
        print("Postgres недоступен на DSN по умолчанию.", file=sys.stderr)
        print("Проверьте: docker compose up -d", file=sys.stderr)
        sys.exit(1)

    print("Загружаю JSON в SQLite-сессию…")
    sqlite_conn = open_session_db()
    load_json(sqlite_conn, JSON_PATH)
    direction_key = get_direction_key(sqlite_conn)
    print(f"direction_key: {direction_key!r}")

    metrics_count = sqlite_conn.execute("SELECT COUNT(*) FROM metric_catalog").fetchone()[0]
    print(f"Метрик в каталоге: {metrics_count}")

    print("Заливаю эмбеддинги в Postgres…")
    pg_conn = connect()
    try:
        stats = populate_search_cache(sqlite_conn, pg_conn, direction_key=direction_key)
    finally:
        pg_conn.close()

    print(f"Итог: total={stats['total']}, unchanged={stats['unchanged']}, upserted={stats['upserted']}")


if __name__ == "__main__":
    main()
