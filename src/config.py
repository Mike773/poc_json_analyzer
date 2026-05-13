from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Config:
    regularity_threshold: float = 0.3
    regular_step_tolerance: float = 0.5
    window_multipliers: tuple = (3, 6, 12)
    calc_period_to_days: Dict[str, int] = field(
        default_factory=lambda: {"неделя": 7, "месяц": 30, "квартал": 90}
    )

    min_points_for_anomaly: int = 4
    min_points_for_trend: int = 4
    trend_pvalue_threshold: float = 0.1
    anomaly_z_threshold: float = 2.0

    plan_miss_threshold: float = 0.05
    benchmark_threshold: float = 0.02
    benchmark_gap_threshold: float = 0.05

    peer_group_min_for_percentile: int = 3
    peer_group_min_for_zscore: int = 5
    peer_outlier_z_threshold: float = 1.5
    peer_outlier_percentile_low: float = 10.0
    peer_outlier_percentile_high: float = 90.0

    severity_weight_static: float = 0.5
    severity_weight_dynamic: float = 0.5

    norm_scale_pct: float = 0.20
    norm_scale_z: float = 3.0
    norm_scale_slope_pct_per_window: float = 0.30

    element_aggregator: str = "max"  # 'max' | 'weighted_mean'
    trend_divergence_majority: float = 0.6
    element_concentration_threshold: float = 0.4

    # Postgres-слой отключён, если уникальных метрик в направлении ≤ этого порога
    postgres_min_metrics_per_direction: int = 50


DEFAULT_CONFIG = Config()


def direction_sign(direction: str) -> int:
    """Знак для нормирования отклонений: direct → выше плохо имеет минус; inverse — наоборот.

    Конвенция: возвращает множитель такой, что положительное (fact - target) * sign(direction)
    означает «хуже плана/бенчмарка».
    """
    if direction == "direct":
        return -1  # для прямой метрики fact < target = хуже → (fact-target) отрицательно, домножая на -1 получаем "плохо"
    elif direction == "inverse":
        return 1  # для обратной метрики fact > target = хуже → положительное "плохо"
    raise ValueError(f"unknown direction: {direction!r}")


def norm(value: float, scale: float) -> float:
    """Клипаем |value|/scale в [0, 1]."""
    if value is None:
        return 0.0
    v = abs(value) / scale
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
