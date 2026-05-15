"""Интерактивный чат с агентом-аналитиком.

Запуск:
  export OPENAI_API_KEY=sk-...
  python3 scripts/chat.py

Опционально:
  python3 scripts/chat.py --verbose      # показывать вызовы тулов
  python3 scripts/chat.py --question "..." # one-shot, без интерактива
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from src.schema import open_session_db
from src.ingest import load_json
from src.peers import build_peer_groups
from src.dynamics import build_metric_dynamics
from src.severity import compute_severity
from src.llm import Agent


JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "call_center_metrics.json")
JSON_PATH = os.path.normpath(JSON_PATH)


def _fmt(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _step(label: str, fn):
    t = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t
    print(f"  [{_fmt(dt):>7}] {label}", flush=True)
    return result


def init_session():
    print("Инициализация сессии:", flush=True)
    t_total = time.perf_counter()
    conn = _step("открытие SQLite in-memory", open_session_db)
    _step(f"загрузка JSON из {os.path.basename(JSON_PATH)}", lambda: load_json(conn, JSON_PATH))

    n_emp = conn.execute("SELECT COUNT(*) FROM dim_employee").fetchone()[0]
    n_metrics = conn.execute("SELECT COUNT(*) FROM metric_catalog").fetchone()[0]
    n_facts = conn.execute("SELECT COUNT(*) FROM fact_metric").fetchone()[0]
    print(f"  [    -  ] загружено: сотрудников={n_emp}, метрик={n_metrics}, фактов={n_facts}", flush=True)

    _step("peer-группы", lambda: build_peer_groups(conn))
    _step("динамика (тренды, аномалии, peer-агрегаты)", lambda: build_metric_dynamics(conn))
    _step("severity (per-cell + рoll-up)", lambda: compute_severity(conn))

    dt_total = time.perf_counter() - t_total
    print(f"  всего init: {_fmt(dt_total)}\n", flush=True)
    return conn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="печатать вызовы тулов")
    parser.add_argument("--question", "-q", type=str, help="одиночный вопрос (без интерактива)")
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "openai"),
        choices=["openai", "gigachat"],
        help="провайдер LLM (по умолчанию из LLM_PROVIDER или openai)",
    )
    parser.add_argument(
        "--effort",
        default="none",
        choices=["none", "low", "medium", "high"],
        help="reasoning_effort для reasoning-моделей OpenAI; 'none' — не передавать (GigaChat игнорирует)",
    )
    parser.add_argument(
        "--model", default=None, help="имя модели (по умолчанию — дефолтная модель провайдера)"
    )
    mem_group = parser.add_mutually_exclusive_group()
    mem_group.add_argument(
        "--memory",
        dest="enable_memory",
        action="store_true",
        default=True,
        help="включить память диалога (default)",
    )
    mem_group.add_argument(
        "--no-memory",
        dest="enable_memory",
        action="store_false",
        help="отключить память: каждый вопрос обрабатывается изолированно",
    )
    args = parser.parse_args()

    conn = init_session()
    effort = None if args.effort == "none" else args.effort
    agent = Agent(
        conn,
        model=args.model,
        provider=args.provider,
        reasoning_effort=effort,
        verbose=args.verbose,
        enable_memory=args.enable_memory,
    )
    print(
        f"Готово. Провайдер: {agent.provider}, модель: {agent.model}, "
        f"reasoning_effort: {effort or '—'}, память: {'on' if args.enable_memory else 'off'}.\n"
    )

    if args.question:
        answer = agent.ask(args.question)
        print(answer)
        return

    print("Введите вопрос (пустая строка или Ctrl+D — выход):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            answer = agent.ask(q)
        except Exception as e:
            print(f"[ошибка] {e}")
            continue
        print()
        print(answer)


if __name__ == "__main__":
    main()
