"""Запуск insight engine + LLM-нарратор → связный отчёт.

Запуск:
  export OPENAI_API_KEY=sk-...
  python3 scripts/narrate.py
  python3 scripts/narrate.py --employee 10005   # только по одному
  python3 scripts/narrate.py --top 5            # топ-5 инсайтов на сотрудника
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
from src.insights import run_all_detectors
from src.llm import Narrator


JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "call_center_metrics.json")
JSON_PATH = os.path.normpath(JSON_PATH)


def _fmt(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _step(label, fn):
    t = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t
    print(f"  [{_fmt(dt):>7}] {label}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee", help="ограничить одним сотрудником (employee_id)")
    parser.add_argument("--top", type=int, default=8, help="макс. инсайтов на сотрудника")
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
        help="reasoning_effort для reasoning-моделей OpenAI; 'none' — не передавать",
    )
    parser.add_argument(
        "--model", default=None, help="имя модели (по умолчанию — дефолтная модель провайдера)"
    )
    args = parser.parse_args()
    effort = None if args.effort == "none" else args.effort

    print("Инициализация сессии:", flush=True)
    t0 = time.perf_counter()
    conn = _step("открытие SQLite in-memory", open_session_db)
    _step("загрузка JSON", lambda: load_json(conn, JSON_PATH))
    _step("peer-группы", lambda: build_peer_groups(conn))
    _step("динамика", lambda: build_metric_dynamics(conn))
    _step("severity", lambda: compute_severity(conn))
    insights = _step("insight engine (детекторы + дедуп)", lambda: run_all_detectors(conn))
    print(f"  всего init: {_fmt(time.perf_counter() - t0)}\n", flush=True)

    if args.employee:
        insights = [i for i in insights if i.employee_id == args.employee]

    narrator = Narrator(
        conn, model=args.model, provider=args.provider, reasoning_effort=effort
    )
    print(
        f"Инсайтов: {len(insights)}. Запрос к {narrator.provider}/{narrator.model}…",
        flush=True,
    )
    t_llm = time.perf_counter()
    report = narrator.narrate(insights, top_per_employee=args.top, verbose=True)
    print(f"  итог narrator: {_fmt(time.perf_counter() - t_llm)}\n", flush=True)

    print(report)


if __name__ == "__main__":
    main()
