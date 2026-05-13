from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Tuple

from .config import Config, DEFAULT_CONFIG, norm
from .ingest import get_root_metrics


def _cell_signals(
    row: sqlite3.Row, direction: str, has_plan: bool, cfg: Config
) -> Tuple[float, float, float, float, float]:
    """Возвращает (plan_signal, benchmark_signal, peer_signal, trend_signal, anomaly_signal).

    NULL-safe: отсутствующий показатель даёт сигнал 0.
    """
    # Plan signal
    plan_signal = 0.0
    if has_plan and row["deviation_plan_pct"] is not None:
        dev = row["deviation_plan_pct"]
        if abs(dev) > cfg.plan_miss_threshold:
            plan_signal = norm(dev, cfg.norm_scale_pct)

    # Benchmark signal
    benchmark_signal = 0.0
    if row["deviation_benchmark_pct"] is not None:
        dev = row["deviation_benchmark_pct"]
        if abs(dev) > cfg.benchmark_threshold:
            benchmark_signal = norm(dev, cfg.norm_scale_pct)

    # Peer signal (only when computed quality is 'ok')
    peer_signal = 0.0
    if row["peer_z_score"] is not None and row["peer_group_quality"] == "ok":
        peer_signal = norm(row["peer_z_score"], cfg.norm_scale_z)

    # Trend signal: max нормированных |slope| по окнам с p < threshold и against direction
    trend_signal = 0.0
    windows = [
        (row["trend_slope_short"], row["trend_pvalue_short"], row["window_short_days"]),
        (row["trend_slope_medium"], row["trend_pvalue_medium"], row["window_medium_days"]),
        (row["trend_slope_long"], row["trend_pvalue_long"], row["window_long_days"]),
    ]
    for slope, pv, window_days in windows:
        if slope is None or pv is None or window_days is None:
            continue
        if pv > cfg.trend_pvalue_threshold:
            continue
        # against direction:
        # direct → плохо когда slope < 0 (значение падает)
        # inverse → плохо когда slope > 0 (значение растёт)
        bad = (direction == "direct" and slope < 0) or (direction == "inverse" and slope > 0)
        if not bad:
            continue
        # нормируем относительно среднего значения серии
        # mean берём из peer_mean как proxy: дёшево и доступно; если нет — пропускаем
        mean = row["peer_mean"]
        if mean is None or mean == 0:
            continue
        rel_change = abs(slope) * window_days / abs(mean)
        sig = norm(rel_change, cfg.norm_scale_slope_pct_per_window)
        if sig > trend_signal:
            trend_signal = sig

    # Anomaly signal
    anomaly_signal = 0.0
    if row["anomaly_score"] is not None and abs(row["anomaly_score"]) > cfg.anomaly_z_threshold:
        anomaly_signal = norm(row["anomaly_score"], cfg.norm_scale_z)

    return plan_signal, benchmark_signal, peer_signal, trend_signal, anomaly_signal


def _children_rollup(
    children: List[Tuple[int, Optional[float], float]], cfg: Config
) -> Tuple[float, str]:
    """children: list of (metric_id, weight | None, severity_total).

    Возвращает (children_severity, rollup_quality).
    """
    if not children:
        return 0.0, "no_children"

    weights_known = [w for _, w, _ in children if w is not None]
    if len(weights_known) == len(children) and sum(weights_known) > 0:
        norm_factor = sum(weights_known)
        s = sum(sev * (w / norm_factor) for _, w, sev in children)
        return s, "weighted"

    if weights_known and len(weights_known) < len(children):
        # часть весов известна, часть нет
        sum_known = sum(weights_known)
        weighted_part = 0.0
        if sum_known > 0:
            weighted_part = sum(
                sev * (w / sum_known) for _, w, sev in children if w is not None
            )
        unknown_part = max((sev for _, w, sev in children if w is None), default=0.0)
        return max(weighted_part, unknown_part), "partial_weights"

    # все None
    return max(sev for _, _, sev in children), "equal_weights"


def compute_severity(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> None:
    """Вычисляет severity по ячейкам, агрегирует по элементам, потом катит вверх по дереву.

    Запись severity_total идёт в metric_dynamics. Для агрегата по элементам в строке с element IS NULL —
    если её ещё нет, создаём «сводную» запись (target_date = тот же).
    """
    # 1) Per-cell severity
    rows = list(
        conn.execute(
            """SELECT d.*, c.direction, c.has_plan
               FROM metric_dynamics d
               JOIN metric_catalog c ON c.metric_id = d.metric_id"""
        ).fetchall()
    )
    target_date = rows[0]["target_date"] if rows else None
    if target_date is None:
        return

    cell_severity: Dict[Tuple[str, int, Optional[str]], Dict[str, float]] = {}
    for r in rows:
        key = (r["employee_id"], r["metric_id"], r["element"])
        plan_s, bench_s, peer_s, trend_s, anom_s = _cell_signals(
            r, r["direction"], bool(r["has_plan"]), cfg
        )
        severity_static = max(plan_s, bench_s, peer_s)
        severity_dynamic = max(trend_s, anom_s)
        severity_self = (
            cfg.severity_weight_static * severity_static
            + cfg.severity_weight_dynamic * severity_dynamic
        )
        cell_severity[key] = {
            "static": severity_static,
            "dynamic": severity_dynamic,
            "self": severity_self,
        }

    # 2) Element aggregation per (employee, metric)
    # element_aggregator: 'max' (default) — берём максимум по элементам.
    severity_by_metric: Dict[Tuple[str, int], float] = {}  # severity_total post-elements
    for (emp, mid, _elem), s in cell_severity.items():
        key = (emp, mid)
        cur = severity_by_metric.get(key, 0.0)
        if s["self"] > cur:
            severity_by_metric[key] = s["self"]

    # 3) Дерево: для каждого сотрудника считаем roll-up
    employees = [r["employee_id"] for r in conn.execute("SELECT employee_id FROM dim_employee")]
    metric_ids = [r["metric_id"] for r in conn.execute("SELECT metric_id FROM metric_catalog")]
    edges = list(conn.execute("SELECT parent_metric_id, child_metric_id, weight FROM metric_edge"))
    children_map: Dict[int, List[Tuple[int, Optional[float]]]] = {}
    for e in edges:
        children_map.setdefault(e["parent_metric_id"], []).append(
            (e["child_metric_id"], e["weight"])
        )

    # Topo order: считаем «снизу вверх». В простом случае hierarchy compact — можем рекурсивно с мемо.
    severity_total: Dict[Tuple[str, int], Tuple[float, str]] = {}
    rollup_quality_map: Dict[Tuple[str, int], str] = {}

    def total_for(emp: str, mid: int) -> Tuple[float, str]:
        if (emp, mid) in severity_total:
            return severity_total[(emp, mid)]
        self_sev = severity_by_metric.get((emp, mid), 0.0)
        children = children_map.get(mid, [])
        if not children:
            severity_total[(emp, mid)] = (self_sev, "leaf")
            return severity_total[(emp, mid)]
        child_triples = []
        for child_mid, w in children:
            child_sev, _ = total_for(emp, child_mid)
            child_triples.append((child_mid, w, child_sev))
        c_sev, q = _children_rollup(child_triples, cfg)
        total = max(self_sev, c_sev)
        severity_total[(emp, mid)] = (total, q)
        return severity_total[(emp, mid)]

    for emp in employees:
        for mid in metric_ids:
            total_for(emp, mid)

    # 4) Запись обратно. Для ячеек проставляем severity_self и severity_total из их metric_id-уровня.
    # severity_total in metric_dynamics: total всего поддерева для (emp, metric_id); для элементной строки —
    # дублируем (т.к. element-aggregator = max).
    for (emp, mid, elem), s in cell_severity.items():
        total, quality = severity_total.get((emp, mid), (s["self"], "leaf"))
        conn.execute(
            """UPDATE metric_dynamics SET
                 severity_static = ?, severity_dynamic = ?,
                 severity_self = ?, severity_total = ?, rollup_quality = ?
               WHERE employee_id = ? AND metric_id = ?
                 AND COALESCE(element, '') = COALESCE(?, '')
                 AND target_date = ?""",
            (
                s["static"], s["dynamic"], s["self"], total, quality,
                emp, mid, elem, target_date,
            ),
        )
    conn.commit()
