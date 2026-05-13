from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .config import Config, DEFAULT_CONFIG, direction_sign


def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _signed_pct(fact: Optional[float], target: Optional[float], direction: str) -> Optional[float]:
    """(fact - target) / |target| с инверсией для direct (где меньше — хуже).

    Возвращаемое значение: положительное = хуже целевого, отрицательное = лучше.
    """
    if fact is None or target is None or target == 0:
        return None
    raw = (fact - target) / abs(target)
    return raw * direction_sign(direction)


def get_target_date(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(snapshot_date) AS d FROM fact_metric").fetchone()
    return row["d"]


def _enumerate_cells(conn: sqlite3.Connection) -> List[Tuple[str, int, Optional[str]]]:
    """Перечисляем уникальные (employee_id, metric_id, element) ячейки."""
    cur = conn.execute(
        """SELECT DISTINCT employee_id, metric_id, element
           FROM fact_metric
           ORDER BY employee_id, metric_id, element"""
    )
    return [(r["employee_id"], r["metric_id"], r["element"]) for r in cur.fetchall()]


def _series(
    conn: sqlite3.Connection, employee_id: str, metric_id: int, element: Optional[str]
) -> List[Tuple[date, float]]:
    if element is None:
        cur = conn.execute(
            """SELECT snapshot_date, fact FROM fact_metric
               WHERE employee_id = ? AND metric_id = ? AND element IS NULL
                 AND fact IS NOT NULL
               ORDER BY snapshot_date""",
            (employee_id, metric_id),
        )
    else:
        cur = conn.execute(
            """SELECT snapshot_date, fact FROM fact_metric
               WHERE employee_id = ? AND metric_id = ? AND element = ?
                 AND fact IS NOT NULL
               ORDER BY snapshot_date""",
            (employee_id, metric_id, element),
        )
    return [(_to_date(r["snapshot_date"]), float(r["fact"])) for r in cur.fetchall()]


def _resolve_period_days(
    points: List[Tuple[date, float]], conn: sqlite3.Connection, metric_id: int, cfg: Config
) -> Tuple[float, bool]:
    """Возвращает (period_days, period_regular).

    Первичный источник — calc_period; fallback — медиана шагов между точками.
    """
    calc_period = conn.execute(
        "SELECT calc_period FROM fact_metric WHERE metric_id = ? AND calc_period IS NOT NULL LIMIT 1",
        (metric_id,),
    ).fetchone()
    cp = calc_period["calc_period"] if calc_period else None

    if cp and cp in cfg.calc_period_to_days:
        period_days = float(cfg.calc_period_to_days[cp])
    else:
        if len(points) >= 2:
            diffs = [(points[i + 1][0] - points[i][0]).days for i in range(len(points) - 1)]
            period_days = float(np.median(diffs))
        else:
            period_days = 1.0

    # Регулярность
    if len(points) >= 3:
        diffs = [(points[i + 1][0] - points[i][0]).days for i in range(len(points) - 1)]
        mad = float(np.median(np.abs(np.array(diffs) - period_days)))
        variance = mad / period_days if period_days > 0 else 1.0
        regular = variance < cfg.regularity_threshold
    else:
        regular = True
    return period_days, regular


def _trend_in_window(
    points: List[Tuple[date, float]], target: date, window_days: float, cfg: Config
) -> Tuple[Optional[float], Optional[float], int]:
    """linregress(value ~ days_since_start) на точках в окне. Возвращает (slope, pvalue, n)."""
    cutoff = target.toordinal() - window_days
    in_w = [(d, v) for d, v in points if cutoff <= d.toordinal() <= target.toordinal()]
    n = len(in_w)
    if n < cfg.min_points_for_trend:
        return None, None, n
    days = np.array([d.toordinal() for d, _ in in_w], dtype=float)
    days = days - days.min()
    vals = np.array([v for _, v in in_w], dtype=float)
    if np.all(vals == vals[0]):
        return 0.0, 1.0, n
    res = stats.linregress(days, vals)
    return float(res.slope), float(res.pvalue), n


def _anomaly_score(
    points: List[Tuple[date, float]],
    target: date,
    period_days: float,
    period_regular: bool,
    window_long: float,
    cfg: Config,
) -> Optional[float]:
    if period_regular:
        cutoff = target.toordinal() - window_long
        sample = [v for d, v in points if cutoff <= d.toordinal() <= target.toordinal()]
    else:
        # «регулярные» точки — те, у которых шаг от соседа близок к period_days
        if len(points) < 2:
            sample = [v for _, v in points]
        else:
            sample = []
            tol = cfg.regular_step_tolerance
            for i, (d, v) in enumerate(points):
                if i == 0:
                    sample.append(v)
                    continue
                step = (d - points[i - 1][0]).days
                if abs(step - period_days) <= tol * period_days:
                    sample.append(v)
    if len(sample) < cfg.min_points_for_anomaly:
        return None
    arr = np.array(sample, dtype=float)
    std = float(arr.std(ddof=0))
    if std <= 0:
        return None
    mean = float(arr.mean())
    last = arr[-1]
    return (last - mean) / std


def _peer_aggregates(
    conn: sqlite3.Connection,
    employee_id: str,
    metric_id: int,
    element: Optional[str],
    target_date: str,
    direction: str,
    cfg: Config,
) -> Dict[str, Any]:
    peers = [
        r["peer_employee_id"]
        for r in conn.execute(
            "SELECT peer_employee_id FROM peer_groups WHERE employee_id = ?", (employee_id,)
        ).fetchall()
    ]
    if not peers:
        return {
            "peer_group_size": 0,
            "peer_group_quality": "none",
            "peer_mean": None,
            "peer_median": None,
            "peer_std": None,
            "peer_min": None,
            "peer_max": None,
            "peer_p25": None,
            "peer_p75": None,
            "peer_z_score": None,
            "peer_percentile": None,
            "peer_rank": None,
        }

    if element is None:
        clause = "element IS NULL"
        params = (metric_id, target_date) + tuple(peers)
    else:
        clause = "element = ?"
        params = (metric_id, target_date, element) + tuple(peers)
    placeholders = ",".join(["?"] * len(peers))
    sql = f"""SELECT fact FROM fact_metric
              WHERE metric_id = ? AND snapshot_date = ? AND {clause}
                AND fact IS NOT NULL
                AND employee_id IN ({placeholders})"""
    peer_facts = [float(r["fact"]) for r in conn.execute(sql, params).fetchall()]
    size = len(peer_facts)

    if size == 0:
        return {
            "peer_group_size": 0,
            "peer_group_quality": "none",
            "peer_mean": None,
            "peer_median": None,
            "peer_std": None,
            "peer_min": None,
            "peer_max": None,
            "peer_p25": None,
            "peer_p75": None,
            "peer_z_score": None,
            "peer_percentile": None,
            "peer_rank": None,
        }

    arr = np.array(peer_facts, dtype=float)
    quality = "ok" if size >= cfg.peer_group_min_for_zscore else (
        "small" if size >= cfg.peer_group_min_for_percentile else "small"
    )

    # employee own value
    own = conn.execute(
        f"""SELECT fact FROM fact_metric
            WHERE employee_id = ? AND metric_id = ? AND snapshot_date = ?
              AND {('element IS NULL' if element is None else 'element = ?')}""",
        (employee_id, metric_id, target_date) if element is None else (employee_id, metric_id, target_date, element),
    ).fetchone()
    own_fact = float(own["fact"]) if own and own["fact"] is not None else None

    # percentile / rank — с учётом direction
    percentile: Optional[float] = None
    rank: Optional[int] = None
    z: Optional[float] = None
    if own_fact is not None and size >= cfg.peer_group_min_for_percentile:
        # все значения вместе с собственным
        combined = np.append(arr, own_fact)
        if direction == "direct":
            # выше = лучше; percentile = доля тех, кто ниже сотрудника
            percentile = float((combined < own_fact).sum()) / len(combined) * 100.0
            order = np.argsort(-combined)  # от большего к меньшему
        else:
            percentile = float((combined > own_fact).sum()) / len(combined) * 100.0
            order = np.argsort(combined)
        # rank: позиция own_fact в combined согласно order
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order)) + 1
        own_idx = len(arr)  # own добавлен в конец
        rank = int(ranks[own_idx])

    if own_fact is not None and size >= cfg.peer_group_min_for_zscore:
        std = float(arr.std(ddof=0))
        if std > 0:
            mean = float(arr.mean())
            # с учётом direction: положительный z = «хуже» peer mean
            raw_z = (own_fact - mean) / std
            z = raw_z * (-1 if direction == "direct" else 1)

    return {
        "peer_group_size": size,
        "peer_group_quality": quality,
        "peer_mean": float(arr.mean()),
        "peer_median": float(np.median(arr)),
        "peer_std": float(arr.std(ddof=0)),
        "peer_min": float(arr.min()),
        "peer_max": float(arr.max()),
        "peer_p25": float(np.percentile(arr, 25)),
        "peer_p75": float(np.percentile(arr, 75)),
        "peer_z_score": z,
        "peer_percentile": percentile,
        "peer_rank": rank,
    }


def build_metric_dynamics(
    conn: sqlite3.Connection, target_date: Optional[str] = None, cfg: Config = DEFAULT_CONFIG
) -> None:
    if target_date is None:
        target_date = get_target_date(conn)
    target_d = _to_date(target_date)

    # Карта направлений
    direction_by_metric = {
        r["metric_id"]: r["direction"]
        for r in conn.execute("SELECT metric_id, direction FROM metric_catalog").fetchall()
    }

    for employee_id, metric_id, element in _enumerate_cells(conn):
        points = _series(conn, employee_id, metric_id, element)
        if len(points) < 2:
            continue

        direction = direction_by_metric[metric_id]
        period_days, period_regular = _resolve_period_days(points, conn, metric_id, cfg)

        w_short = period_days * cfg.window_multipliers[0]
        w_med = period_days * cfg.window_multipliers[1]
        w_long = period_days * cfg.window_multipliers[2]

        slope_s, pv_s, n_s = _trend_in_window(points, target_d, w_short, cfg)
        slope_m, pv_m, n_m = _trend_in_window(points, target_d, w_med, cfg)
        slope_l, pv_l, n_l = _trend_in_window(points, target_d, w_long, cfg)

        anomaly = _anomaly_score(points, target_d, period_days, period_regular, w_long, cfg)

        # отклонения на target_date
        if element is None:
            row = conn.execute(
                """SELECT fact, plan, benchmark FROM fact_metric
                   WHERE employee_id = ? AND metric_id = ? AND snapshot_date = ? AND element IS NULL""",
                (employee_id, metric_id, target_date),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT fact, plan, benchmark FROM fact_metric
                   WHERE employee_id = ? AND metric_id = ? AND snapshot_date = ? AND element = ?""",
                (employee_id, metric_id, target_date, element),
            ).fetchone()
        dev_plan = None
        dev_bench = None
        if row is not None:
            dev_plan = _signed_pct(row["fact"], row["plan"], direction)
            dev_bench = _signed_pct(row["fact"], row["benchmark"], direction)

        peer = _peer_aggregates(conn, employee_id, metric_id, element, target_date, direction, cfg)

        benchmark_unavailable = 1 if dev_bench is None else 0

        # disagreement: benchmark говорит «норма», computed peer — нет
        disagreement = 0
        if dev_bench is not None and peer["peer_z_score"] is not None:
            bench_alarm = abs(dev_bench) > cfg.benchmark_threshold
            peer_alarm = abs(peer["peer_z_score"]) > cfg.peer_outlier_z_threshold
            if bench_alarm != peer_alarm:
                disagreement = 1

        conn.execute(
            """INSERT OR REPLACE INTO metric_dynamics (
                employee_id, metric_id, element, target_date,
                period_days, period_regular,
                points_total, points_in_short, points_in_medium, points_in_long,
                window_short_days, window_medium_days, window_long_days,
                trend_slope_short, trend_slope_medium, trend_slope_long,
                trend_pvalue_short, trend_pvalue_medium, trend_pvalue_long,
                anomaly_score,
                deviation_plan_pct, deviation_benchmark_pct,
                peer_group_size, peer_mean, peer_median, peer_std,
                peer_min, peer_max, peer_p25, peer_p75,
                peer_z_score, peer_percentile, peer_rank,
                peer_group_quality, benchmark_unavailable, benchmark_peer_disagreement
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                employee_id, metric_id, element, target_date,
                period_days, int(period_regular),
                len(points), n_s, n_m, n_l,
                w_short, w_med, w_long,
                slope_s, slope_m, slope_l,
                pv_s, pv_m, pv_l,
                anomaly,
                dev_plan, dev_bench,
                peer["peer_group_size"], peer["peer_mean"], peer["peer_median"], peer["peer_std"],
                peer["peer_min"], peer["peer_max"], peer["peer_p25"], peer["peer_p75"],
                peer["peer_z_score"], peer["peer_percentile"], peer["peer_rank"],
                peer["peer_group_quality"], benchmark_unavailable, disagreement,
            ),
        )
    conn.commit()
