from __future__ import annotations

import sqlite3


SCHEMA_SQL = """
CREATE TABLE dim_employee (
  employee_id TEXT PRIMARY KEY,
  fio TEXT,
  post TEXT,
  department TEXT,
  role TEXT NOT NULL CHECK (role IN ('manager', 'employee'))
);

CREATE TABLE metric_catalog (
  metric_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  direction TEXT NOT NULL CHECK (direction IN ('direct', 'inverse')),
  has_plan INTEGER NOT NULL DEFAULT 0,
  has_benchmark INTEGER NOT NULL DEFAULT 0,
  has_element_breakdown INTEGER NOT NULL DEFAULT 0,
  element_kind TEXT,
  level INTEGER NOT NULL
);

-- Корень не имеет ребра; у всех остальных метрик ровно одно ребро (parent, self).
CREATE TABLE metric_edge (
  parent_metric_id INTEGER NOT NULL,
  child_metric_id INTEGER NOT NULL,
  weight REAL,  -- nullable: influent_percent может отсутствовать
  PRIMARY KEY (parent_metric_id, child_metric_id)
);
CREATE INDEX idx_edge_child ON metric_edge(child_metric_id);

CREATE TABLE fact_metric (
  employee_id TEXT NOT NULL,
  metric_id INTEGER NOT NULL,
  snapshot_date TEXT NOT NULL,
  element TEXT,  -- nullable
  fact REAL,
  plan REAL,
  benchmark REAL,
  calc_period TEXT
);
CREATE UNIQUE INDEX idx_fact_pk
  ON fact_metric(employee_id, metric_id, snapshot_date, COALESCE(element, ''));
CREATE INDEX idx_fact_metric_emp ON fact_metric(metric_id, employee_id);
CREATE INDEX idx_fact_metric_elem ON fact_metric(metric_id, element);
CREATE INDEX idx_fact_emp_date ON fact_metric(employee_id, snapshot_date);

CREATE TABLE peer_groups (
  employee_id TEXT NOT NULL,
  peer_employee_id TEXT NOT NULL,
  PRIMARY KEY (employee_id, peer_employee_id)
);

CREATE TABLE metric_dynamics (
  employee_id TEXT NOT NULL,
  metric_id INTEGER NOT NULL,
  element TEXT,
  target_date TEXT NOT NULL,
  period_days REAL,
  period_regular INTEGER,
  points_total INTEGER,
  points_in_short INTEGER,
  points_in_medium INTEGER,
  points_in_long INTEGER,
  window_short_days REAL,
  window_medium_days REAL,
  window_long_days REAL,
  trend_slope_short REAL, trend_slope_medium REAL, trend_slope_long REAL,
  trend_pvalue_short REAL, trend_pvalue_medium REAL, trend_pvalue_long REAL,
  anomaly_score REAL,
  deviation_plan_pct REAL,
  deviation_benchmark_pct REAL,
  peer_group_size INTEGER,
  peer_mean REAL, peer_median REAL, peer_std REAL,
  peer_min REAL, peer_max REAL, peer_p25 REAL, peer_p75 REAL,
  peer_z_score REAL, peer_percentile REAL, peer_rank INTEGER,
  peer_group_quality TEXT,
  benchmark_unavailable INTEGER,
  benchmark_peer_disagreement INTEGER,
  severity_static REAL,
  severity_dynamic REAL,
  severity_self REAL,
  severity_total REAL,
  rollup_quality TEXT
);
CREATE UNIQUE INDEX idx_dyn_pk
  ON metric_dynamics(employee_id, metric_id, COALESCE(element, ''), target_date);
CREATE INDEX idx_dyn_metric_elem ON metric_dynamics(metric_id, element);
CREATE INDEX idx_dyn_emp ON metric_dynamics(employee_id);
"""


def open_session_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn
