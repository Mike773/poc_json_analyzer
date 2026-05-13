from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional


DIRECTION_MAP = {"прямая": "direct", "обратная": "inverse"}


def _direction_role(post: str) -> str:
    p = (post or "").lower()
    if "руковод" in p or "директор" in p or "начальник" in p:
        return "manager"
    return "employee"


def _walk(
    conn: sqlite3.Connection,
    metrics: List[Dict[str, Any]],
    parent_id: Optional[int],
    level: int,
    owner_id: str,
) -> None:
    catalog_seen: Dict[int, Dict[str, Any]] = {}
    for m in metrics:
        _walk_node(conn, m, parent_id, level, owner_id, catalog_seen)


def _walk_node(
    conn: sqlite3.Connection,
    m: Dict[str, Any],
    parent_id: Optional[int],
    level: int,
    owner_id: str,
    catalog_seen: Dict[int, Dict[str, Any]],
) -> None:
    metric_id = m["id"]
    name = m["metric_name"]
    description = m.get("metric_description")
    direction = DIRECTION_MAP.get(m["metric_type"], m["metric_type"])

    # upsert catalog (берём минимальный level)
    cur = conn.execute(
        "SELECT level FROM metric_catalog WHERE metric_id = ?", (metric_id,)
    )
    row = cur.fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO metric_catalog
               (metric_id, name, description, direction, has_plan, has_benchmark,
                has_element_breakdown, element_kind, level)
               VALUES (?, ?, ?, ?, 0, 0, 0, NULL, ?)""",
            (metric_id, name, description, direction, level),
        )
    else:
        if level < row["level"]:
            conn.execute(
                "UPDATE metric_catalog SET level = ? WHERE metric_id = ?",
                (level, metric_id),
            )

    # upsert edge (parent → metric_id) — игнорируем дубликаты
    if parent_id is not None:
        conn.execute(
            """INSERT OR IGNORE INTO metric_edge (parent_metric_id, child_metric_id, weight)
               VALUES (?, ?, ?)""",
            (parent_id, metric_id, m.get("influent_percent")),
        )

    # insert fact
    conn.execute(
        """INSERT OR REPLACE INTO fact_metric
           (employee_id, metric_id, snapshot_date, element, fact, plan, benchmark, calc_period)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            owner_id,
            metric_id,
            m["date"],
            m.get("element"),
            m.get("fact"),
            m.get("plan"),
            m.get("benchmark"),
            m.get("calc_period"),
        ),
    )

    for child in m.get("child_metrics", []) or []:
        _walk_node(conn, child, metric_id, level + 1, owner_id, catalog_seen)


def _infer_element_kind(values: List[str]) -> Optional[str]:
    """Простая эвристика: общий префикс из первого слова, если совпадает у всех значений."""
    if not values:
        return None
    first_words = []
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        first_words.append(v.split()[0])
    if not first_words:
        return None
    common = first_words[0]
    for fw in first_words[1:]:
        if fw != common:
            return "элемент"
    return common.lower()


def load_json(conn: sqlite3.Connection, path: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # boss
    boss = data["boss"]
    conn.execute(
        """INSERT INTO dim_employee (employee_id, fio, post, department, role)
           VALUES (?, ?, ?, ?, ?)""",
        (
            str(boss["tabnum"]),
            boss.get("fio"),
            boss.get("post"),
            boss.get("depart"),
            "manager",
        ),
    )
    _walk(conn, boss.get("metrics", []), parent_id=None, level=1, owner_id=str(boss["tabnum"]))

    # employees
    for emp in data.get("employees", []):
        role = _direction_role(emp.get("post", ""))
        conn.execute(
            """INSERT INTO dim_employee (employee_id, fio, post, department, role)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(emp["tabnum"]),
                emp.get("fio"),
                emp.get("post"),
                emp.get("depart"),
                role,
            ),
        )
        _walk(conn, emp.get("metrics", []), parent_id=None, level=1, owner_id=str(emp["tabnum"]))

    _finalize_catalog_flags(conn)


def _finalize_catalog_flags(conn: sqlite3.Connection) -> None:
    """После загрузки фактов вычисляем has_plan / has_benchmark / has_element_breakdown / element_kind."""
    cur = conn.execute("SELECT metric_id FROM metric_catalog")
    metric_ids = [r["metric_id"] for r in cur.fetchall()]
    for mid in metric_ids:
        has_plan = conn.execute(
            "SELECT 1 FROM fact_metric WHERE metric_id = ? AND plan IS NOT NULL LIMIT 1",
            (mid,),
        ).fetchone() is not None
        has_bench = conn.execute(
            "SELECT 1 FROM fact_metric WHERE metric_id = ? AND benchmark IS NOT NULL LIMIT 1",
            (mid,),
        ).fetchone() is not None
        has_elem = conn.execute(
            "SELECT 1 FROM fact_metric WHERE metric_id = ? AND element IS NOT NULL LIMIT 1",
            (mid,),
        ).fetchone() is not None

        element_kind = None
        if has_elem:
            vals = [
                r["element"]
                for r in conn.execute(
                    "SELECT DISTINCT element FROM fact_metric WHERE metric_id = ? AND element IS NOT NULL",
                    (mid,),
                ).fetchall()
            ]
            element_kind = _infer_element_kind(vals)

        conn.execute(
            """UPDATE metric_catalog
               SET has_plan = ?, has_benchmark = ?, has_element_breakdown = ?, element_kind = ?
               WHERE metric_id = ?""",
            (int(has_plan), int(has_bench), int(has_elem), element_kind, mid),
        )
    conn.commit()


def get_root_metrics(conn: sqlite3.Connection) -> List[int]:
    """Корневые метрики — те, для которых нет входящего ребра."""
    cur = conn.execute(
        """SELECT c.metric_id FROM metric_catalog c
           LEFT JOIN metric_edge e ON e.child_metric_id = c.metric_id
           WHERE e.child_metric_id IS NULL"""
    )
    return [r["metric_id"] for r in cur.fetchall()]
