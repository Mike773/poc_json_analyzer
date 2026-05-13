"""Демонстрация POC: загрузка JSON, расчёт динамики/severity, прогон ключевых тулов и insight engine."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict

# чтобы можно было запускать как `python3 scripts/demo.py` из корня проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import open_session_db
from src.ingest import load_json
from src.peers import build_peer_groups, get_peers
from src.dynamics import build_metric_dynamics
from src.severity import compute_severity
from src.tools import (
    get_employee_profile,
    list_elements,
    expand_metric,
    expand_by_element,
    rank_elements_for_employee,
    rank_metrics_for_employee,
    compare_employees_overview,
    compare_to_group,
    search_metrics,
    get_metric_timeseries,
)
from src.insights import run_all_detectors, group_for_narrator


JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "call_center_metrics.json")
JSON_PATH = os.path.normpath(JSON_PATH)


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    section("1. Init session")
    conn = open_session_db()
    load_json(conn, JSON_PATH)
    build_peer_groups(conn)
    build_metric_dynamics(conn)
    compute_severity(conn)

    counts = {
        "employees": conn.execute("SELECT COUNT(*) FROM dim_employee").fetchone()[0],
        "metrics": conn.execute("SELECT COUNT(*) FROM metric_catalog").fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM metric_edge").fetchone()[0],
        "facts": conn.execute("SELECT COUNT(*) FROM fact_metric").fetchone()[0],
        "peer_pairs": conn.execute("SELECT COUNT(*) FROM peer_groups").fetchone()[0],
        "dynamics_cells": conn.execute("SELECT COUNT(*) FROM metric_dynamics").fetchone()[0],
    }
    print(jdump(counts))

    section("2. Peer groups (sanity check: manager has empty peers)")
    print(f"Manager 10001 peers: {get_peers(conn, '10001')}")
    print(f"Operator 10005 peers: {get_peers(conn, '10005')}")

    section("3. get_employee_profile(10005) — L1 со severity")
    print(jdump(get_employee_profile(conn, "10005")))

    section("4. Сценарий: «Какой продукт самый плохой у оператора 10005?»")
    print("→ rank_elements_for_employee('10005', top_n=3)")
    print(jdump(rank_elements_for_employee(conn, "10005", top_n=3)))

    section("5. Сценарий: «По какой метрике 10005 хуже всего?»")
    print("→ rank_metrics_for_employee('10005', top_n=5)")
    print(jdump(rank_metrics_for_employee(conn, "10005", top_n=5)))

    section("6. Drill: AHT по элементам у 10005")
    out = expand_by_element(conn, 7842156, "10005")
    out["elements"] = out["elements"][:5]
    print(jdump(out))

    section("7. compare_to_group: AHT Продукт 4 у 10005")
    print(jdump(compare_to_group(conn, "10005", 7842156, "Продукт 4")))

    section("8. compare_employees_overview(10003, 10005)")
    print(jdump(compare_employees_overview(conn, "10003", "10005", top_n=3)))

    section("9. Insight engine — раскладка")
    insights = run_all_detectors(conn)
    print(f"Всего инсайтов (после дедупа): {len(insights)}")
    c = Counter(i.type for i in insights)
    print("По типам:")
    for t, n in c.most_common():
        print(f"  {t}: {n}")
    print("\nТоп-5 по severity:")
    for i in sorted(insights, key=lambda x: x.severity, reverse=True)[:5]:
        print(f"  [{i.type}] emp={i.employee_id} | {i.metric_name} | element={i.element} | sev={i.severity:.3f}")
        if "also" in i.evidence:
            for a in i.evidence["also"]:
                print(f"        also: {a['type']} sev={a['severity']:.3f}")

    section("10. Группировка инсайтов для нарратора: оператор 10005")
    g = group_for_narrator(insights, conn)
    if "10005" in g:
        for root_mid, items in g["10005"].items():
            root_name = conn.execute(
                "SELECT name FROM metric_catalog WHERE metric_id = ?", (root_mid,)
            ).fetchone()["name"]
            print(f"  Корень: {root_name} (id={root_mid}) — {len(items)} инсайтов")
            for it in items[:3]:
                print(f"    {it['type']} | {it['metric_name']} | element={it['element']} | sev={it['severity']:.3f}")


if __name__ == "__main__":
    main()
