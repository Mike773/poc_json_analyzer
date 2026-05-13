from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import direction_sign


def _dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def get_target_date(conn: sqlite3.Connection) -> str:
    r = conn.execute("SELECT MAX(target_date) FROM metric_dynamics").fetchone()
    return r[0]


def get_employee_profile(
    conn: sqlite3.Connection, employee_id: str, target_date: Optional[str] = None
) -> Dict[str, Any]:
    """Возвращает атрибуты сотрудника + L1-метрики со severity_total (значения per-метрика, не per-element)."""
    if target_date is None:
        target_date = get_target_date(conn)
    emp = _dict(
        conn.execute(
            "SELECT employee_id, fio, post, department, role FROM dim_employee WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()
    )
    if emp is None:
        return {"error": f"employee {employee_id} not found"}

    # L1 metrics
    l1 = list(
        conn.execute(
            """SELECT c.metric_id, c.name, c.direction, c.has_plan, c.has_element_breakdown, c.element_kind
               FROM metric_catalog c
               LEFT JOIN metric_edge e ON e.child_metric_id = c.metric_id
               WHERE e.child_metric_id IS NULL
               ORDER BY c.metric_id"""
        ).fetchall()
    )
    metrics_out = []
    for m in l1:
        # одна агрегированная severity_total на (emp, metric_id) — берём первую попавшуюся строку
        sev = conn.execute(
            """SELECT severity_total, rollup_quality FROM metric_dynamics
               WHERE employee_id = ? AND metric_id = ? AND target_date = ?
               LIMIT 1""",
            (employee_id, m["metric_id"], target_date),
        ).fetchone()
        metrics_out.append(
            {
                "metric_id": m["metric_id"],
                "name": m["name"],
                "direction": m["direction"],
                "has_plan": bool(m["has_plan"]),
                "has_element_breakdown": bool(m["has_element_breakdown"]),
                "element_kind": m["element_kind"],
                "severity_total": sev["severity_total"] if sev else None,
                "rollup_quality": sev["rollup_quality"] if sev else None,
            }
        )

    return {**emp, "target_date": target_date, "l1_metrics": metrics_out}


def list_elements(
    conn: sqlite3.Connection, metric_id: Optional[int] = None
) -> Dict[str, Any]:
    if metric_id is not None:
        cur = conn.execute(
            """SELECT DISTINCT element FROM fact_metric
               WHERE metric_id = ? AND element IS NOT NULL ORDER BY element""",
            (metric_id,),
        )
        elements = [r["element"] for r in cur.fetchall()]
        kind = conn.execute(
            "SELECT element_kind FROM metric_catalog WHERE metric_id = ?", (metric_id,)
        ).fetchone()
        return {"metric_id": metric_id, "elements": elements, "element_kind": kind["element_kind"] if kind else None}
    # глобально
    by_metric = {}
    for r in conn.execute(
        """SELECT metric_id, element_kind FROM metric_catalog WHERE has_element_breakdown = 1"""
    ):
        by_metric[r["metric_id"]] = r["element_kind"]
    return {"metrics_with_breakdown": by_metric}


def expand_metric(
    conn: sqlite3.Connection,
    metric_id: int,
    employee_id: str,
    element: Optional[str] = None,
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    if target_date is None:
        target_date = get_target_date(conn)

    children = list(
        conn.execute(
            """SELECT e.child_metric_id AS metric_id, e.weight, c.name, c.direction,
                      c.has_plan, c.has_element_breakdown, c.element_kind
               FROM metric_edge e
               JOIN metric_catalog c ON c.metric_id = e.child_metric_id
               WHERE e.parent_metric_id = ?
               ORDER BY e.weight DESC NULLS LAST, e.child_metric_id""",
            (metric_id,),
        ).fetchall()
    )

    out = []
    for ch in children:
        if element is not None:
            row = conn.execute(
                """SELECT d.severity_self, d.severity_total, d.rollup_quality,
                          f.fact, f.plan, f.benchmark
                   FROM metric_dynamics d
                   LEFT JOIN fact_metric f
                     ON f.employee_id = d.employee_id AND f.metric_id = d.metric_id
                        AND f.snapshot_date = d.target_date
                        AND COALESCE(f.element, '') = COALESCE(d.element, '')
                   WHERE d.employee_id = ? AND d.metric_id = ? AND d.element = ?
                     AND d.target_date = ?""",
                (employee_id, ch["metric_id"], element, target_date),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT severity_total, rollup_quality
                   FROM metric_dynamics
                   WHERE employee_id = ? AND metric_id = ? AND target_date = ?
                   LIMIT 1""",
                (employee_id, ch["metric_id"], target_date),
            ).fetchone()

        entry = {
            "metric_id": ch["metric_id"],
            "name": ch["name"],
            "direction": ch["direction"],
            "weight": ch["weight"],
            "has_element_breakdown": bool(ch["has_element_breakdown"]),
            "element_kind": ch["element_kind"],
        }
        if row is not None:
            entry.update(_dict(row) or {})
        out.append(entry)

    return {
        "parent_metric_id": metric_id,
        "element": element,
        "target_date": target_date,
        "children": out,
    }


def expand_by_element(
    conn: sqlite3.Connection,
    metric_id: int,
    employee_id: str,
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    if target_date is None:
        target_date = get_target_date(conn)

    cat = _dict(
        conn.execute(
            "SELECT name, direction, has_plan, has_element_breakdown, element_kind FROM metric_catalog WHERE metric_id = ?",
            (metric_id,),
        ).fetchone()
    )
    rows = list(
        conn.execute(
            """SELECT d.element, d.severity_self, d.deviation_plan_pct, d.deviation_benchmark_pct,
                      d.peer_z_score, d.peer_group_quality,
                      f.fact, f.plan, f.benchmark
               FROM metric_dynamics d
               LEFT JOIN fact_metric f
                 ON f.employee_id = d.employee_id AND f.metric_id = d.metric_id
                    AND f.snapshot_date = d.target_date
                    AND COALESCE(f.element, '') = COALESCE(d.element, '')
               WHERE d.employee_id = ? AND d.metric_id = ? AND d.target_date = ?
               ORDER BY d.severity_self DESC""",
            (employee_id, metric_id, target_date),
        ).fetchall()
    )
    return {
        "metric_id": metric_id,
        "metric": cat,
        "target_date": target_date,
        "elements": [_dict(r) for r in rows],
    }


def rank_elements_for_employee(
    conn: sqlite3.Connection,
    employee_id: str,
    target_date: Optional[str] = None,
    top_n: int = 5,
    scope: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Сводит severity_self по всем метрикам в разрезе элементов: топ-N самых проблемных элементов."""
    if target_date is None:
        target_date = get_target_date(conn)

    where = "d.employee_id = ? AND d.target_date = ? AND d.element IS NOT NULL"
    params: List[Any] = [employee_id, target_date]
    if scope:
        ph = ",".join(["?"] * len(scope))
        where += f" AND d.metric_id IN ({ph})"
        params.extend(scope)

    rows = list(
        conn.execute(
            f"""SELECT d.element, d.metric_id, c.name AS metric_name, d.severity_self
                FROM metric_dynamics d
                JOIN metric_catalog c ON c.metric_id = d.metric_id
                WHERE {where}""",
            params,
        ).fetchall()
    )

    by_elem: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"sum": 0.0, "max": 0.0, "metrics": []})
    for r in rows:
        sev = r["severity_self"] or 0.0
        e = r["element"]
        by_elem[e]["sum"] += sev
        if sev > by_elem[e]["max"]:
            by_elem[e]["max"] = sev
        by_elem[e]["metrics"].append(
            {"metric_id": r["metric_id"], "name": r["metric_name"], "severity": sev}
        )

    ranked = sorted(by_elem.items(), key=lambda kv: kv[1]["sum"], reverse=True)[:top_n]
    out = []
    for elem, agg in ranked:
        top_metrics = sorted(agg["metrics"], key=lambda m: m["severity"], reverse=True)[:5]
        out.append(
            {
                "element": elem,
                "aggregate_severity": agg["sum"],
                "max_severity": agg["max"],
                "top_metrics": top_metrics,
            }
        )
    return {"employee_id": employee_id, "target_date": target_date, "ranking": out}


def rank_metrics_for_employee(
    conn: sqlite3.Connection,
    employee_id: str,
    target_date: Optional[str] = None,
    top_n: int = 5,
    level: Optional[int] = None,
) -> Dict[str, Any]:
    if target_date is None:
        target_date = get_target_date(conn)
    # выбираем уникальные (metric_id) с их severity_total + summed severity_self по элементам
    where = "d.employee_id = ? AND d.target_date = ?"
    params: List[Any] = [employee_id, target_date]
    if level is not None:
        where += " AND c.level = ?"
        params.append(level)
    rows = list(
        conn.execute(
            f"""SELECT d.metric_id, c.name, c.level,
                       MAX(d.severity_total) AS severity_total,
                       MAX(d.severity_self) AS max_self,
                       SUM(d.severity_self) AS sum_self,
                       MAX(d.rollup_quality) AS rollup_quality
                FROM metric_dynamics d
                JOIN metric_catalog c ON c.metric_id = d.metric_id
                WHERE {where}
                GROUP BY d.metric_id
                ORDER BY severity_total DESC, max_self DESC""",
            params,
        ).fetchall()
    )
    return {
        "employee_id": employee_id,
        "target_date": target_date,
        "ranking": [_dict(r) for r in rows[:top_n]],
    }


def compare_employees_overview(
    conn: sqlite3.Connection,
    emp_a: str,
    emp_b: str,
    target_date: Optional[str] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Сравнивает двух сотрудников по всем метрикам, возвращает top-N максимальных расхождений (по нормированному dev)."""
    if target_date is None:
        target_date = get_target_date(conn)
    rows = list(
        conn.execute(
            """SELECT da.metric_id, da.element, c.name, c.direction,
                      fa.fact AS fact_a, fb.fact AS fact_b,
                      fa.benchmark AS bench_a, fb.benchmark AS bench_b
               FROM fact_metric fa
               JOIN fact_metric fb
                 ON fb.metric_id = fa.metric_id
                 AND fb.snapshot_date = fa.snapshot_date
                 AND COALESCE(fb.element, '') = COALESCE(fa.element, '')
               JOIN metric_catalog c ON c.metric_id = fa.metric_id
               JOIN metric_dynamics da
                 ON da.employee_id = fa.employee_id AND da.metric_id = fa.metric_id
                 AND COALESCE(da.element, '') = COALESCE(fa.element, '')
                 AND da.target_date = fa.snapshot_date
               WHERE fa.employee_id = ? AND fb.employee_id = ?
                 AND fa.snapshot_date = ?
                 AND fa.fact IS NOT NULL AND fb.fact IS NOT NULL""",
            (emp_a, emp_b, target_date),
        ).fetchall()
    )
    diffs = []
    for r in rows:
        # знак: положительное = A хуже B
        sign = direction_sign(r["direction"])
        raw = (r["fact_a"] - r["fact_b"]) * sign
        # нормируем относительно benchmark_a (если есть) или среднего по двум
        base = r["bench_a"] if r["bench_a"] not in (None, 0) else (r["fact_a"] + r["fact_b"]) / 2
        if base in (None, 0):
            continue
        norm_diff = raw / abs(base)
        diffs.append(
            {
                "metric_id": r["metric_id"],
                "metric_name": r["name"],
                "element": r["element"],
                "fact_a": r["fact_a"],
                "fact_b": r["fact_b"],
                "diff_pct": norm_diff,
                "direction": r["direction"],
            }
        )
    diffs.sort(key=lambda d: abs(d["diff_pct"]), reverse=True)
    return {
        "emp_a": emp_a,
        "emp_b": emp_b,
        "target_date": target_date,
        "top_diffs": diffs[:top_n],
    }


def compare_to_group(
    conn: sqlite3.Connection,
    employee_id: str,
    metric_id: int,
    element: Optional[str] = None,
    target_date: Optional[str] = None,
    departments: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Сравнение сотрудника с группой.

    Если departments/roles НЕ заданы — используется зашитая peer-группа (same post + same department),
    предрасчитанная в metric_dynamics.

    Если заданы — на лету собирает peer-группу из fact_metric по фильтру (employees из указанных
    департаментов и/или ролей, исключая самого сотрудника) и вычисляет mean/median/std/percentile/
    rank/z-score с учётом direction.
    """
    if target_date is None:
        target_date = get_target_date(conn)

    custom_filter = bool(departments or roles)

    # Метаданные метрики (нужны для direction в кастомном пути и для базовой инфы)
    cat = conn.execute(
        "SELECT name, direction FROM metric_catalog WHERE metric_id = ?", (metric_id,)
    ).fetchone()
    direction = cat["direction"] if cat else "direct"

    if not custom_filter:
        # Кешированный путь — берём из metric_dynamics
        if element is None:
            d = conn.execute(
                """SELECT * FROM metric_dynamics
                   WHERE employee_id = ? AND metric_id = ? AND element IS NULL AND target_date = ?""",
                (employee_id, metric_id, target_date),
            ).fetchone()
        else:
            d = conn.execute(
                """SELECT * FROM metric_dynamics
                   WHERE employee_id = ? AND metric_id = ? AND element = ? AND target_date = ?""",
                (employee_id, metric_id, element, target_date),
            ).fetchone()
        if d is None:
            return {"error": "no dynamics row found"}
        return {
            "employee_id": employee_id,
            "metric_id": metric_id,
            "element": element,
            "target_date": target_date,
            "group_filter": {"mode": "default_peer_group", "scope": "same post + same department"},
            "deviation_benchmark_pct": d["deviation_benchmark_pct"],
            "deviation_plan_pct": d["deviation_plan_pct"],
            "peer_group_size": d["peer_group_size"],
            "peer_group_quality": d["peer_group_quality"],
            "peer_mean": d["peer_mean"],
            "peer_median": d["peer_median"],
            "peer_z_score": d["peer_z_score"],
            "peer_percentile": d["peer_percentile"],
            "peer_rank": d["peer_rank"],
            "small_group_warning": d["peer_group_quality"] == "small",
            "missing_group_warning": d["peer_group_quality"] == "none",
            "benchmark_unavailable": bool(d["benchmark_unavailable"]),
            "benchmark_peer_disagreement": bool(d["benchmark_peer_disagreement"]),
        }

    # ──── Кастомный путь: ad-hoc группа по departments/roles ────
    own_clauses = ["employee_id = ?", "metric_id = ?", "snapshot_date = ?"]
    own_params: List[Any] = [employee_id, metric_id, target_date]
    if element is None:
        own_clauses.append("element IS NULL")
    else:
        own_clauses.append("element = ?")
        own_params.append(element)
    own_row = conn.execute(
        "SELECT fact, plan, benchmark FROM fact_metric WHERE " + " AND ".join(own_clauses),
        own_params,
    ).fetchone()
    if own_row is None:
        return {"error": "no fact for this employee/metric/element/date"}
    own_fact = own_row["fact"]

    # Peer facts по фильтру
    peer_clauses = [
        "f.metric_id = ?",
        "f.snapshot_date = ?",
        "f.fact IS NOT NULL",
        "f.employee_id != ?",
    ]
    peer_params: List[Any] = [metric_id, target_date, employee_id]
    if element is None:
        peer_clauses.append("f.element IS NULL")
    else:
        peer_clauses.append("f.element = ?")
        peer_params.append(element)
    if departments:
        ph = ",".join("?" * len(departments))
        peer_clauses.append(f"e.department IN ({ph})")
        peer_params.extend(departments)
    if roles:
        ph = ",".join("?" * len(roles))
        peer_clauses.append(f"e.role IN ({ph})")
        peer_params.extend(roles)

    peer_facts = [
        float(r["fact"])
        for r in conn.execute(
            f"""SELECT f.fact FROM fact_metric f
                JOIN dim_employee e ON e.employee_id = f.employee_id
                WHERE {' AND '.join(peer_clauses)}""",
            peer_params,
        ).fetchall()
    ]
    size = len(peer_facts)

    out: Dict[str, Any] = {
        "employee_id": employee_id,
        "metric_id": metric_id,
        "element": element,
        "target_date": target_date,
        "group_filter": {
            "mode": "custom",
            "departments": departments,
            "roles": roles,
        },
        "own_fact": own_fact,
        "deviation_plan_pct": _signed_pct_local(own_fact, own_row["plan"], direction),
        "deviation_benchmark_pct": _signed_pct_local(own_fact, own_row["benchmark"], direction),
        "peer_group_size": size,
        "peer_mean": None,
        "peer_median": None,
        "peer_std": None,
        "peer_min": None,
        "peer_max": None,
        "peer_z_score": None,
        "peer_percentile": None,
        "peer_rank": None,
        "peer_group_quality": "none",
        "small_group_warning": False,
        "missing_group_warning": True,
        "benchmark_unavailable": own_row["benchmark"] is None,
        "benchmark_peer_disagreement": False,
    }
    if size == 0:
        return out

    import numpy as np

    arr = np.array(peer_facts, dtype=float)
    out["peer_mean"] = float(arr.mean())
    out["peer_median"] = float(np.median(arr))
    out["peer_std"] = float(arr.std(ddof=0))
    out["peer_min"] = float(arr.min())
    out["peer_max"] = float(arr.max())
    out["missing_group_warning"] = False
    out["peer_group_quality"] = "ok" if size >= 5 else ("small" if size >= 3 else "small")
    out["small_group_warning"] = size < 5

    if size >= 3 and own_fact is not None:
        combined = np.append(arr, own_fact)
        if direction == "direct":
            out["peer_percentile"] = float((combined < own_fact).sum()) / len(combined) * 100.0
            order = np.argsort(-combined)
        else:
            out["peer_percentile"] = float((combined > own_fact).sum()) / len(combined) * 100.0
            order = np.argsort(combined)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order)) + 1
        out["peer_rank"] = int(ranks[len(arr)])

    if size >= 5 and own_fact is not None and out["peer_std"] and out["peer_std"] > 0:
        raw_z = (own_fact - out["peer_mean"]) / out["peer_std"]
        out["peer_z_score"] = raw_z * (-1 if direction == "direct" else 1)

    return out


def _signed_pct_local(fact: Optional[float], target: Optional[float], direction: str) -> Optional[float]:
    if fact is None or target is None or target == 0:
        return None
    from .config import direction_sign

    raw = (fact - target) / abs(target)
    return raw * direction_sign(direction)


def rank_employees_by_metric(
    conn: sqlite3.Connection,
    metric_id: int,
    element: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    target_date: Optional[str] = None,
    agg: str = "mean",
    top_n: int = 3,
    bottom_n: int = 0,
    roles: Optional[List[str]] = None,
    departments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ранжирование сотрудников по метрике за период или одну дату.

    Поддерживает агрегацию по интервалу [date_from, date_to] (включительно).
    Если интервал не задан — используется target_date (или последняя дата).

    "Лучшие" определяются с учётом direction:
      - direct: max value лучше
      - inverse: min value лучше

    agg ∈ {'mean', 'last', 'min', 'max', 'sum'}
    roles: фильтр по dim_employee.role — например ['employee'] чтобы исключить руководителя.
    """
    cat = conn.execute(
        "SELECT name, direction, has_element_breakdown FROM metric_catalog WHERE metric_id = ?",
        (metric_id,),
    ).fetchone()
    if cat is None:
        return {"error": f"unknown metric_id {metric_id}"}
    direction = cat["direction"]

    # Дата/интервал
    where = ["f.metric_id = ?", "f.fact IS NOT NULL"]
    params: List[Any] = [metric_id]
    if date_from or date_to:
        if date_from:
            where.append("f.snapshot_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("f.snapshot_date <= ?")
            params.append(date_to)
    else:
        if target_date is None:
            target_date = get_target_date(conn)
        where.append("f.snapshot_date = ?")
        params.append(target_date)

    if element is not None:
        where.append("f.element = ?")
        params.append(element)
    elif not cat["has_element_breakdown"]:
        where.append("f.element IS NULL")
    # иначе агрегируем по всем элементам (редкий кейс, но допустим)

    if roles:
        ph = ",".join("?" * len(roles))
        where.append(f"e.role IN ({ph})")
        params.extend(roles)

    if departments:
        ph = ",".join("?" * len(departments))
        where.append(f"e.department IN ({ph})")
        params.extend(departments)

    agg = (agg or "mean").lower()
    if agg == "mean":
        agg_expr = "AVG(f.fact)"
    elif agg == "min":
        agg_expr = "MIN(f.fact)"
    elif agg == "max":
        agg_expr = "MAX(f.fact)"
    elif agg == "sum":
        agg_expr = "SUM(f.fact)"
    elif agg == "last":
        # последнее значение в окне (через коррелированный подзапрос)
        agg_expr = (
            "(SELECT f2.fact FROM fact_metric f2 "
            " WHERE f2.employee_id = f.employee_id AND f2.metric_id = f.metric_id "
            "       AND COALESCE(f2.element,'') = COALESCE(f.element,'') "
            "       AND f2.snapshot_date <= MAX(f.snapshot_date) "
            "       AND f2.fact IS NOT NULL "
            " ORDER BY f2.snapshot_date DESC LIMIT 1)"
        )
    else:
        return {"error": f"unknown agg: {agg}; allowed: mean|min|max|sum|last"}

    sql = f"""
        SELECT f.employee_id, e.fio, e.role, e.department,
               {agg_expr} AS value,
               COUNT(*) AS points_used,
               MIN(f.snapshot_date) AS first_date,
               MAX(f.snapshot_date) AS last_date
        FROM fact_metric f
        JOIN dim_employee e ON e.employee_id = f.employee_id
        WHERE {' AND '.join(where)}
        GROUP BY f.employee_id, e.fio, e.role, e.department
        ORDER BY value {'DESC' if direction == 'direct' else 'ASC'}
    """
    rows = list(conn.execute(sql, params).fetchall())
    ranked = [_dict(r) for r in rows]

    # Список использованных дат
    sql_dates = f"""SELECT DISTINCT f.snapshot_date FROM fact_metric f
                    JOIN dim_employee e ON e.employee_id = f.employee_id
                    WHERE {' AND '.join(where)}
                    ORDER BY f.snapshot_date"""
    snapshot_dates = [r["snapshot_date"] for r in conn.execute(sql_dates, params).fetchall()]

    return {
        "metric_id": metric_id,
        "metric_name": cat["name"],
        "direction": direction,
        "element": element,
        "agg": agg,
        "date_from": date_from,
        "date_to": date_to,
        "target_date": target_date,
        "snapshot_dates_used": snapshot_dates,
        "top": ranked[:top_n] if top_n else [],
        "bottom": list(reversed(ranked[-bottom_n:])) if bottom_n else [],
        "all": ranked,
    }


def rank_departments_by_metric(
    conn: sqlite3.Connection,
    metric_id: int,
    element: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    target_date: Optional[str] = None,
    agg: str = "mean",
    top_n: int = 5,
    bottom_n: int = 0,
    roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ранжирование ДЕПАРТАМЕНТОВ по метрике.

    Сначала агрегирует значения внутри каждого департамента (mean / min / max / sum по всем точкам
    всех сотрудников департамента), затем сортирует с учётом direction.
    """
    cat = conn.execute(
        "SELECT name, direction, has_element_breakdown FROM metric_catalog WHERE metric_id = ?",
        (metric_id,),
    ).fetchone()
    if cat is None:
        return {"error": f"unknown metric_id {metric_id}"}
    direction = cat["direction"]

    where = ["f.metric_id = ?", "f.fact IS NOT NULL"]
    params: List[Any] = [metric_id]
    if date_from or date_to:
        if date_from:
            where.append("f.snapshot_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("f.snapshot_date <= ?")
            params.append(date_to)
    else:
        if target_date is None:
            target_date = get_target_date(conn)
        where.append("f.snapshot_date = ?")
        params.append(target_date)

    if element is not None:
        where.append("f.element = ?")
        params.append(element)
    elif not cat["has_element_breakdown"]:
        where.append("f.element IS NULL")

    if roles:
        ph = ",".join("?" * len(roles))
        where.append(f"e.role IN ({ph})")
        params.extend(roles)

    agg = (agg or "mean").lower()
    if agg == "mean":
        agg_expr = "AVG(f.fact)"
    elif agg == "min":
        agg_expr = "MIN(f.fact)"
    elif agg == "max":
        agg_expr = "MAX(f.fact)"
    elif agg == "sum":
        agg_expr = "SUM(f.fact)"
    else:
        return {"error": f"unknown agg: {agg}; allowed: mean|min|max|sum"}

    sql = f"""
        SELECT e.department,
               {agg_expr} AS value,
               COUNT(DISTINCT f.employee_id) AS employees_count,
               COUNT(*) AS points_used,
               MIN(f.snapshot_date) AS first_date,
               MAX(f.snapshot_date) AS last_date
        FROM fact_metric f
        JOIN dim_employee e ON e.employee_id = f.employee_id
        WHERE {' AND '.join(where)}
        GROUP BY e.department
        HAVING value IS NOT NULL
        ORDER BY value {'DESC' if direction == 'direct' else 'ASC'}
    """
    rows = list(conn.execute(sql, params).fetchall())
    ranked = [_dict(r) for r in rows]

    return {
        "metric_id": metric_id,
        "metric_name": cat["name"],
        "direction": direction,
        "element": element,
        "agg": agg,
        "date_from": date_from,
        "date_to": date_to,
        "target_date": target_date,
        "departments_count": len(ranked),
        "top": ranked[:top_n] if top_n else [],
        "bottom": list(reversed(ranked[-bottom_n:])) if bottom_n else [],
        "all": ranked,
    }


def list_employees(
    conn: sqlite3.Connection,
    role: Optional[str] = None,
    department: Optional[str] = None,
    departments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Список сотрудников с фильтрами по role/department/departments."""
    clauses: List[str] = []
    params: List[Any] = []
    if role:
        clauses.append("role = ?")
        params.append(role)
    if department:
        clauses.append("department = ?")
        params.append(department)
    if departments:
        ph = ",".join("?" * len(departments))
        clauses.append(f"department IN ({ph})")
        params.extend(departments)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = list(
        conn.execute(
            f"""SELECT employee_id, fio, post, department, role
                FROM dim_employee{where_sql}
                ORDER BY department, role DESC, employee_id""",
            params,
        ).fetchall()
    )
    return {
        "filter": {"role": role, "department": department, "departments": departments},
        "count": len(rows),
        "employees": [_dict(r) for r in rows],
    }


def get_metrics_matrix(
    conn: sqlite3.Connection,
    metric_ids: List[int],
    employee_ids: Optional[List[str]] = None,
    elements: Optional[List[str]] = None,
    target_date: Optional[str] = None,
    departments: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Батч-выборка: крест-произведение (metric × employee × element) за один target_date.

    Способы задать сотрудников (хотя бы один):
      - employee_ids: явный список табельных номеров
      - departments: фильтр по департаментам (все сотрудники из них)
      - roles: фильтр по ролям ('employee' / 'manager')
    Фильтры комбинируются (AND).

    elements: None/пустой массив → все элементы метрики; иначе — фильтр по значениям.
    """
    if not metric_ids:
        return {"error": "metric_ids обязателен и непуст", "cells": []}
    if not (employee_ids or departments or roles):
        return {"error": "нужен хотя бы один из: employee_ids / departments / roles", "cells": []}
    if target_date is None:
        target_date = get_target_date(conn)

    where = [
        f"f.metric_id IN ({','.join('?' * len(metric_ids))})",
        "f.snapshot_date = ?",
    ]
    params: List[Any] = list(metric_ids) + [target_date]

    if employee_ids:
        where.append(f"f.employee_id IN ({','.join('?' * len(employee_ids))})")
        params.extend(employee_ids)
    if departments:
        where.append(f"e.department IN ({','.join('?' * len(departments))})")
        params.extend(departments)
    if roles:
        where.append(f"e.role IN ({','.join('?' * len(roles))})")
        params.extend(roles)
    if elements:
        where.append(
            f"(f.element IS NULL OR f.element IN ({','.join('?' * len(elements))}))"
        )
        params.extend(elements)

    sql = f"""
        SELECT f.employee_id, e.fio, e.department, e.role,
               f.metric_id, c.name AS metric_name, c.direction,
               f.element, f.fact, f.plan, f.benchmark,
               d.deviation_plan_pct, d.deviation_benchmark_pct,
               d.peer_z_score, d.peer_group_quality, d.severity_self
        FROM fact_metric f
        JOIN dim_employee e ON e.employee_id = f.employee_id
        JOIN metric_catalog c ON c.metric_id = f.metric_id
        LEFT JOIN metric_dynamics d
          ON d.employee_id = f.employee_id
         AND d.metric_id = f.metric_id
         AND COALESCE(d.element, '') = COALESCE(f.element, '')
         AND d.target_date = f.snapshot_date
        WHERE {' AND '.join(where)}
        ORDER BY f.metric_id, e.department, f.employee_id, f.element
    """
    rows = list(conn.execute(sql, params).fetchall())
    return {
        "target_date": target_date,
        "metric_ids": metric_ids,
        "employee_ids": employee_ids,
        "departments": departments,
        "roles": roles,
        "elements_filter": elements,
        "cells_count": len(rows),
        "cells": [_dict(r) for r in rows],
    }


def get_metric_timeseries(
    conn: sqlite3.Connection,
    metric_id: int,
    employee_id: str,
    element: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Временной ряд значений метрики у сотрудника.

    Семантика element:
      - element=<значение>     → фильтр по этому элементу;
      - element=None для метрик БЕЗ has_element_breakdown → фильтр element IS NULL;
      - element=None для метрик С has_element_breakdown → ВСЕ элементы (с колонкой element в каждой точке).
    """
    has_breakdown = conn.execute(
        "SELECT has_element_breakdown FROM metric_catalog WHERE metric_id = ?",
        (metric_id,),
    ).fetchone()
    has_breakdown = bool(has_breakdown["has_element_breakdown"]) if has_breakdown else False

    clauses = ["employee_id = ?", "metric_id = ?"]
    params: List[Any] = [employee_id, metric_id]

    if element is not None:
        clauses.append("element = ?")
        params.append(element)
    elif not has_breakdown:
        clauses.append("element IS NULL")
    # иначе (element=None и has_breakdown=true) — без фильтра по element, отдаём все

    if date_from:
        clauses.append("snapshot_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("snapshot_date <= ?")
        params.append(date_to)

    sql = (
        "SELECT snapshot_date, element, fact, plan, benchmark FROM fact_metric WHERE "
        + " AND ".join(clauses)
        + " ORDER BY element, snapshot_date"
    )
    rows = list(conn.execute(sql, params).fetchall())
    return {
        "metric_id": metric_id,
        "employee_id": employee_id,
        "element_filter": element,
        "has_element_breakdown": has_breakdown,
        "points_count": len(rows),
        "points": [_dict(r) for r in rows],
    }


def search_metrics(conn: sqlite3.Connection, query: str) -> List[Dict[str, Any]]:
    """Гибридный поиск.

    1) Сначала пробуем pgvector (если доступен) — точное совпадение токенов + cosine similarity.
    2) Иначе — in-memory токен-match по имени/описанию (fallback).
    """
    try:
        from . import pgvector_search

        if pgvector_search.ping():
            direction_key = pgvector_search.get_direction_key(conn)
            pg_conn = pgvector_search.connect()
            try:
                results = pgvector_search.search_metrics_pgvector(
                    pg_conn, query, direction_key=direction_key, top_k=10
                )
            finally:
                pg_conn.close()

            # обогащаем недостающими полями каталога из SQLite
            out: List[Dict[str, Any]] = []
            for r in results:
                cat = _dict(
                    conn.execute(
                        """SELECT metric_id, name, description, direction,
                                  has_plan, has_element_breakdown, element_kind, level
                           FROM metric_catalog WHERE metric_id = ?""",
                        (r["metric_id"],),
                    ).fetchone()
                )
                if cat is None:
                    continue
                cat["cosine_distance"] = r["cosine_distance"]
                cat["match"] = r["match"]
                out.append(cat)
            return out
    except Exception:
        # пробрасываемся на in-memory, если pgvector/embeddings недоступны
        pass

    # Fallback: in-memory
    q = query.lower().strip()
    tokens = [t for t in q.replace(",", " ").split() if t]
    rows = list(
        conn.execute(
            "SELECT metric_id, name, description, direction, has_plan, has_element_breakdown, element_kind, level FROM metric_catalog"
        ).fetchall()
    )
    scored = []
    for r in rows:
        name = (r["name"] or "").lower()
        desc = (r["description"] or "").lower()
        score = 0
        for t in tokens:
            if t in name:
                score += 3
            if t in desc:
                score += 1
        if q in name:
            score += 5
        if score > 0:
            scored.append((score, _dict(r)))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:10]]
