from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .config import Config, DEFAULT_CONFIG


@dataclass
class Insight:
    type: str
    employee_id: str
    metric_id: int
    metric_name: str
    element: Optional[str]
    severity: float
    evidence: Dict[str, Any] = field(default_factory=dict)


def _against_direction(slope: float, direction: str) -> bool:
    return (direction == "direct" and slope < 0) or (direction == "inverse" and slope > 0)


def _load_catalog(conn: sqlite3.Connection) -> Dict[int, sqlite3.Row]:
    return {r["metric_id"]: r for r in conn.execute("SELECT * FROM metric_catalog").fetchall()}


def detect_plan_miss(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> List[Insight]:
    cat = _load_catalog(conn)
    out: List[Insight] = []
    for r in conn.execute(
        "SELECT * FROM metric_dynamics WHERE deviation_plan_pct IS NOT NULL"
    ):
        if not cat[r["metric_id"]]["has_plan"]:
            continue
        dev = r["deviation_plan_pct"]
        if abs(dev) > cfg.plan_miss_threshold:
            out.append(
                Insight(
                    type="plan_miss",
                    employee_id=r["employee_id"],
                    metric_id=r["metric_id"],
                    metric_name=cat[r["metric_id"]]["name"],
                    element=r["element"],
                    severity=min(1.0, abs(dev) / cfg.norm_scale_pct),
                    evidence={"deviation_plan_pct": dev},
                )
            )
    return out


def detect_benchmark_gap(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> List[Insight]:
    cat = _load_catalog(conn)
    out: List[Insight] = []
    for r in conn.execute(
        "SELECT * FROM metric_dynamics WHERE deviation_benchmark_pct IS NOT NULL"
    ):
        dev = r["deviation_benchmark_pct"]
        if abs(dev) > cfg.benchmark_gap_threshold:
            out.append(
                Insight(
                    type="benchmark_gap",
                    employee_id=r["employee_id"],
                    metric_id=r["metric_id"],
                    metric_name=cat[r["metric_id"]]["name"],
                    element=r["element"],
                    severity=min(1.0, abs(dev) / cfg.norm_scale_pct),
                    evidence={"deviation_benchmark_pct": dev},
                )
            )
    return out


def detect_trends(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> List[Insight]:
    cat = _load_catalog(conn)
    out: List[Insight] = []
    for r in conn.execute("SELECT * FROM metric_dynamics"):
        direction = cat[r["metric_id"]]["direction"]
        windows = [
            ("short", r["trend_slope_short"], r["trend_pvalue_short"], r["window_short_days"]),
            ("medium", r["trend_slope_medium"], r["trend_pvalue_medium"], r["window_medium_days"]),
            ("long", r["trend_slope_long"], r["trend_pvalue_long"], r["window_long_days"]),
        ]
        sig_against = []
        for w_name, slope, pv, w_days in windows:
            if slope is None or pv is None or w_days is None:
                continue
            if pv > cfg.trend_pvalue_threshold:
                continue
            if not _against_direction(slope, direction):
                continue
            sig_against.append((w_name, slope, pv, w_days))

        if not sig_against:
            continue

        # Trend emerging — только short в списке
        if {w[0] for w in sig_against} == {"short"}:
            t = "trend_emerging"
        elif any(w[0] in ("medium", "long") for w in sig_against):
            t = "trend_systemic"
        else:
            t = "trend_emerging"  # резерв

        # severity по самой агрессивной точке
        mean = r["peer_mean"] if r["peer_mean"] not in (None, 0) else None
        best = 0.0
        for _, slope, _, w_days in sig_against:
            if mean is None:
                continue
            rel = abs(slope) * w_days / abs(mean)
            sev = min(1.0, rel / cfg.norm_scale_slope_pct_per_window)
            if sev > best:
                best = sev

        if best == 0:
            continue

        out.append(
            Insight(
                type=t,
                employee_id=r["employee_id"],
                metric_id=r["metric_id"],
                metric_name=cat[r["metric_id"]]["name"],
                element=r["element"],
                severity=best,
                evidence={"windows": [w[0] for w in sig_against], "slopes": [w[1] for w in sig_against]},
            )
        )
    return out


def detect_anomaly(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> List[Insight]:
    cat = _load_catalog(conn)
    out: List[Insight] = []
    for r in conn.execute(
        "SELECT * FROM metric_dynamics WHERE anomaly_score IS NOT NULL"
    ):
        z = r["anomaly_score"]
        if abs(z) <= cfg.anomaly_z_threshold:
            continue
        out.append(
            Insight(
                type="anomaly",
                employee_id=r["employee_id"],
                metric_id=r["metric_id"],
                metric_name=cat[r["metric_id"]]["name"],
                element=r["element"],
                severity=min(1.0, abs(z) / cfg.norm_scale_z),
                evidence={"anomaly_score": z},
            )
        )
    return out


def detect_peer_outlier(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> List[Insight]:
    cat = _load_catalog(conn)
    out: List[Insight] = []
    for r in conn.execute(
        "SELECT * FROM metric_dynamics WHERE peer_group_quality = 'ok'"
    ):
        z = r["peer_z_score"]
        p = r["peer_percentile"]
        triggered = False
        if z is not None and abs(z) > cfg.peer_outlier_z_threshold:
            triggered = True
        elif p is not None and (p < cfg.peer_outlier_percentile_low or p > cfg.peer_outlier_percentile_high):
            triggered = True
        if not triggered:
            continue
        sev = 0.0
        if z is not None:
            sev = min(1.0, abs(z) / cfg.norm_scale_z)
        out.append(
            Insight(
                type="peer_outlier",
                employee_id=r["employee_id"],
                metric_id=r["metric_id"],
                metric_name=cat[r["metric_id"]]["name"],
                element=r["element"],
                severity=sev,
                evidence={"peer_z": z, "peer_percentile": p},
            )
        )
    return out


def detect_element_concentration(
    conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG
) -> List[Insight]:
    """Для метрик с has_element_breakdown — если топ-1 элемент даёт > threshold от суммы severity_self."""
    cat = _load_catalog(conn)
    # группируем по (employee, metric)
    by_key: Dict[tuple, List[sqlite3.Row]] = defaultdict(list)
    for r in conn.execute(
        """SELECT d.* FROM metric_dynamics d
           JOIN metric_catalog c ON c.metric_id = d.metric_id
           WHERE c.has_element_breakdown = 1 AND d.element IS NOT NULL"""
    ):
        by_key[(r["employee_id"], r["metric_id"])].append(r)

    out: List[Insight] = []
    for (emp, mid), rows in by_key.items():
        rows_sorted = sorted(rows, key=lambda r: r["severity_self"] or 0, reverse=True)
        total = sum((r["severity_self"] or 0) for r in rows_sorted)
        if total <= 0:
            continue
        top = rows_sorted[0]
        top_sev = top["severity_self"] or 0
        concentration = top_sev / total
        if concentration < cfg.element_concentration_threshold:
            continue
        out.append(
            Insight(
                type="element_concentration",
                employee_id=emp,
                metric_id=mid,
                metric_name=cat[mid]["name"],
                element=top["element"],
                severity=concentration,
                evidence={
                    "concentration_ratio": concentration,
                    "top_element_severity": top_sev,
                    "total_severity_across_elements": total,
                },
            )
        )
    return out


def deduplicate(insights: List[Insight]) -> List[Insight]:
    """В рамках (employee, metric_id, element) оставляем самый «сильный» сигнал как primary,
    остальные складываем в evidence['also'].

    Если element различается — это разные сущности. Element concentration не дедуплицируется
    с обычными детекторами на том же листе — он подсвечивает map проблемы, а не корневую причину.
    """
    grouped: Dict[tuple, List[Insight]] = defaultdict(list)
    for ins in insights:
        if ins.type == "element_concentration":
            # отдельная категория — он остаётся как есть
            grouped[(ins.employee_id, ins.metric_id, ins.element, ins.type)].append(ins)
        else:
            grouped[(ins.employee_id, ins.metric_id, ins.element, "primary")].append(ins)

    out: List[Insight] = []
    for key, items in grouped.items():
        if len(items) == 1:
            out.append(items[0])
            continue
        items.sort(key=lambda i: i.severity, reverse=True)
        primary = items[0]
        also = [{"type": i.type, "severity": i.severity, "evidence": i.evidence} for i in items[1:]]
        primary.evidence = dict(primary.evidence)
        primary.evidence["also"] = also
        out.append(primary)
    return out


def run_all_detectors(conn: sqlite3.Connection, cfg: Config = DEFAULT_CONFIG) -> List[Insight]:
    raw: List[Insight] = []
    raw.extend(detect_plan_miss(conn, cfg))
    raw.extend(detect_benchmark_gap(conn, cfg))
    raw.extend(detect_trends(conn, cfg))
    raw.extend(detect_anomaly(conn, cfg))
    raw.extend(detect_peer_outlier(conn, cfg))
    raw.extend(detect_element_concentration(conn, cfg))
    return deduplicate(raw)


def group_for_narrator(insights: List[Insight], conn: sqlite3.Connection) -> Dict[str, Any]:
    """Группирует инсайты по сотруднику и корню иерархии (для рендера)."""
    # Построим map metric_id → root_metric_id
    edges = list(conn.execute("SELECT parent_metric_id, child_metric_id FROM metric_edge"))
    parent_of = {e["child_metric_id"]: e["parent_metric_id"] for e in edges}

    def root_of(mid: int) -> int:
        cur = mid
        while cur in parent_of:
            cur = parent_of[cur]
        return cur

    grouped: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ins in sorted(insights, key=lambda i: i.severity, reverse=True):
        root = root_of(ins.metric_id)
        d = asdict(ins)
        grouped[ins.employee_id][root].append(d)

    return {emp: dict(roots) for emp, roots in grouped.items()}
