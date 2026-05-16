"""Детальный e2e-тест function-calling: проверяет, что LLM-провайдер умеет вызывать
ВСЕ 14 аналитических тулов агента.

Что делает:
  1. Инициализирует сессию (SQLite in-memory + полный пайплайн), один раз.
  2. Считывает из БД реальные id сотрудников, имена метрик, департаменты — подставляет
     их в вопросы (ничего не хардкодится).
  3. Перехватывает src.llm.execute_tool — логирует каждый вызов тула.
  4. Прогоняет агента по 15 вопросам, каждый нацелен на конкретный тул.
  5. Сверяет числа из результата тула с независимым запросом к SQLite (уровень A)
     и проверяет чистоту текста ответа (уровень B).
  6. Печатает матрицу покрытия 14 тулов и итоговый вердикт.

Запуск (конфигурация берётся из .env):
  python3 scripts/test_tools_gigachat.py
  python3 scripts/test_tools_gigachat.py --provider gigachat --model GigaChat-2-Max
  python3 scripts/test_tools_gigachat.py --quiet     # без трейса вызовов тулов

Код возврата: 0 — все критичные проверки прошли; 1 — есть провалы.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from src.schema import open_session_db
from src.ingest import load_json
from src.peers import build_peer_groups
from src.dynamics import build_metric_dynamics
from src.severity import compute_severity
from src.providers import resolve_model
import src.llm as llm_mod
from src.llm import Agent, TOOL_SCHEMAS


JSON_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "call_center_metrics.json")
)

ALL_TOOLS = [t["function"]["name"] for t in TOOL_SCHEMAS]


# ───────────────────────────────────────────────────────────────────────────
# Перехват вызовов тулов
# ───────────────────────────────────────────────────────────────────────────

_ORIG_EXECUTE_TOOL = llm_mod.execute_tool
_TOOL_LOG: list = []


def _patched_execute_tool(name, args, conn):
    t0 = time.perf_counter()
    exc = None
    try:
        result = _ORIG_EXECUTE_TOOL(name, args, conn)
    except Exception as e:  # noqa: BLE001
        result = {"error": repr(e)}
        exc = e
    _TOOL_LOG.append(
        {
            "tool": name,
            "args": dict(args or {}),
            "result": result,
            "exception": exc,
            "dt": time.perf_counter() - t0,
        }
    )
    return result


llm_mod.execute_tool = _patched_execute_tool


# ───────────────────────────────────────────────────────────────────────────
# Утилиты сверки
# ───────────────────────────────────────────────────────────────────────────

# Внутренние поля/имена, которых не должно быть в ответе пользователю.
FORBIDDEN = [
    "severity",
    "peer_z",
    "peer_percentile",
    "peer_rank",
    "peer_mean",
    "deviation_plan_pct",
    "deviation_benchmark_pct",
    "rollup_quality",
    "snapshot_date",
    "aggregate_severity",
    "metric_id",
] + ALL_TOOLS


def _close(a, b, tol: float = 0.01) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if abs(a - b) <= 1e-6:
        return True
    return abs(a - b) / max(abs(a), abs(b), 1e-9) <= tol


def _numbers(text: str) -> list:
    out = []
    for m in re.findall(r"\d+(?:[.,]\d+)?", text or ""):
        try:
            out.append(float(m.replace(",", ".")))
        except ValueError:
            pass
    return out


def _num_present(text: str, value, tol: float = 0.03) -> bool:
    if value is None:
        return True
    return any(_close(n, value, tol) for n in _numbers(text))


def _find_call(calls: list, tool: str):
    for c in calls:
        if c["tool"] == tool:
            return c
    return None


# ───────────────────────────────────────────────────────────────────────────
# Независимые запросы к SQLite для сверки (уровень A)
# ───────────────────────────────────────────────────────────────────────────

_AGG = {"mean": "AVG", "min": "MIN", "max": "MAX", "sum": "SUM"}


def _metric_meta(conn, metric_id):
    return conn.execute(
        "SELECT direction, has_element_breakdown FROM metric_catalog WHERE metric_id = ?",
        (metric_id,),
    ).fetchone()


def _indep_rank_employees(conn, metric_id, element, dates, agg, roles, departments):
    """Победитель ранжирования сотрудников по метрике — независимым SQL."""
    meta = _metric_meta(conn, metric_id)
    if meta is None:
        return None
    agge = _AGG.get((agg or "mean").lower())
    if agge is None:  # 'last' и пр. — точную сверку пропускаем
        return None
    where = ["f.metric_id = ?", "f.fact IS NOT NULL"]
    params: list = [metric_id]
    if dates:
        where.append(f"f.snapshot_date IN ({','.join('?' * len(dates))})")
        params += list(dates)
    if element is not None:
        where.append("f.element = ?")
        params.append(element)
    elif not meta["has_element_breakdown"]:
        where.append("f.element IS NULL")
    if roles:
        where.append(f"e.role IN ({','.join('?' * len(roles))})")
        params += list(roles)
    if departments:
        where.append(f"e.department IN ({','.join('?' * len(departments))})")
        params += list(departments)
    order = "DESC" if meta["direction"] == "direct" else "ASC"
    row = conn.execute(
        f"""SELECT f.employee_id, {agge}(f.fact) AS v
            FROM fact_metric f JOIN dim_employee e ON e.employee_id = f.employee_id
            WHERE {' AND '.join(where)}
            GROUP BY f.employee_id
            ORDER BY v {order} LIMIT 1""",
        params,
    ).fetchone()
    return (row["employee_id"], row["v"]) if row else None


def _indep_rank_departments(conn, metric_id, element, date_from, date_to, target_date, agg, roles):
    meta = _metric_meta(conn, metric_id)
    if meta is None:
        return None
    agge = _AGG.get((agg or "mean").lower())
    if agge is None:
        return None
    where = ["f.metric_id = ?", "f.fact IS NOT NULL"]
    params: list = [metric_id]
    # rank_departments_by_metric: интервал, иначе одна target_date
    if date_from or date_to:
        if date_from:
            where.append("f.snapshot_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("f.snapshot_date <= ?")
            params.append(date_to)
    elif target_date:
        where.append("f.snapshot_date = ?")
        params.append(target_date)
    if element is not None:
        where.append("f.element = ?")
        params.append(element)
    elif not meta["has_element_breakdown"]:
        where.append("f.element IS NULL")
    if roles:
        where.append(f"e.role IN ({','.join('?' * len(roles))})")
        params += list(roles)
    order = "DESC" if meta["direction"] == "direct" else "ASC"
    row = conn.execute(
        f"""SELECT e.department, {agge}(f.fact) AS v
            FROM fact_metric f JOIN dim_employee e ON e.employee_id = f.employee_id
            WHERE {' AND '.join(where)}
            GROUP BY e.department
            ORDER BY v {order} LIMIT 1""",
        params,
    ).fetchone()
    return (row["department"], row["v"]) if row else None


def _indep_timeseries_count(conn, metric_id, employee_id, element_filter, date_from, date_to):
    meta = _metric_meta(conn, metric_id)
    has_brk = bool(meta["has_element_breakdown"]) if meta else False
    where = ["employee_id = ?", "metric_id = ?"]
    params: list = [employee_id, metric_id]
    if element_filter is not None:
        where.append("element = ?")
        params.append(element_filter)
    elif not has_brk:
        where.append("element IS NULL")
    if date_from:
        where.append("snapshot_date >= ?")
        params.append(date_from)
    if date_to:
        where.append("snapshot_date <= ?")
        params.append(date_to)
    return conn.execute(
        f"SELECT COUNT(*) AS c FROM fact_metric WHERE {' AND '.join(where)}", params
    ).fetchone()["c"]


def _indep_matrix_count(conn, metric_ids, employee_ids, departments, roles, elements, target_date):
    where = [
        f"f.metric_id IN ({','.join('?' * len(metric_ids))})",
        "f.snapshot_date = ?",
    ]
    params: list = list(metric_ids) + [target_date]
    if employee_ids:
        where.append(f"f.employee_id IN ({','.join('?' * len(employee_ids))})")
        params += list(employee_ids)
    if departments:
        where.append(f"e.department IN ({','.join('?' * len(departments))})")
        params += list(departments)
    if roles:
        where.append(f"e.role IN ({','.join('?' * len(roles))})")
        params += list(roles)
    if elements:
        where.append(f"(f.element IS NULL OR f.element IN ({','.join('?' * len(elements))}))")
        params += list(elements)
    return conn.execute(
        f"""SELECT COUNT(*) AS c
            FROM fact_metric f JOIN dim_employee e ON e.employee_id = f.employee_id
            WHERE {' AND '.join(where)}""",
        params,
    ).fetchone()["c"]


def _indep_peer_mean(conn, employee_id, metric_id, element, target_date):
    """Среднее по дефолтной peer-группе сотрудника."""
    peers = [
        r["peer_employee_id"]
        for r in conn.execute(
            "SELECT peer_employee_id FROM peer_groups WHERE employee_id = ?", (employee_id,)
        ).fetchall()
    ]
    if not peers:
        return None
    where = [
        f"employee_id IN ({','.join('?' * len(peers))})",
        "metric_id = ?",
        "snapshot_date = ?",
        "fact IS NOT NULL",
    ]
    params: list = list(peers) + [metric_id, target_date]
    if element is None:
        where.append("element IS NULL")
    else:
        where.append("element = ?")
        params.append(element)
    row = conn.execute(
        f"SELECT AVG(fact) AS m, COUNT(*) AS n FROM fact_metric WHERE {' AND '.join(where)}",
        params,
    ).fetchone()
    return (row["m"], row["n"])


# ───────────────────────────────────────────────────────────────────────────
# Проверки результата (возвращают список (label, ok, detail, fatal))
# ───────────────────────────────────────────────────────────────────────────


def check_clean(answer: str) -> list:
    res = []
    ok_nonempty = bool(answer) and len(answer.strip()) >= 15
    res.append(("ответ непустой", ok_nonempty, f"длина={len(answer or '')}", True))
    low = (answer or "").lower()
    hits = sorted({t for t in FORBIDDEN if t.lower() in low})
    res.append(
        ("без внутренних полей/имён тулов", not hits, f"найдено: {hits}" if hits else "чисто", True)
    )
    has_cyr = bool(re.search(r"[а-яё]", low))
    res.append(("ответ на русском", has_cyr, "ок" if has_cyr else "нет кириллицы", False))
    return res


def check_profile(conn, calls, answer, fx):
    c = _find_call(calls, "get_employee_profile")
    if c is None:
        return [("Level A: get_employee_profile вызван", False, "тул не вызван", True)]
    r = c["result"]
    if not isinstance(r, dict) or "error" in r:
        return [("Level A: результат тула", False, f"{r}", True)]
    emp = conn.execute(
        "SELECT fio, post, department FROM dim_employee WHERE employee_id = ?",
        (fx["emp1"],),
    ).fetchone()
    ok = r.get("fio") == emp["fio"] and r.get("department") == emp["department"]
    out = [
        (
            "Level A: профиль == dim_employee",
            ok,
            f"тул: {r.get('fio')}/{r.get('department')} | БД: {emp['fio']}/{emp['department']}",
            True,
        )
    ]
    surname = (emp["fio"] or "").split()[0] if emp["fio"] else ""
    if surname:
        out.append(
            ("Level B: ФИО в ответе", surname.lower() in (answer or "").lower(),
             f"ищем '{surname}'", False)
        )
    return out


def check_rank_employees(conn, calls, answer, fx):
    c = _find_call(calls, "rank_employees_by_metric")
    if c is None:
        return [("Level A: rank_employees_by_metric вызван", False, "тул не вызван", True)]
    r = c["result"]
    if not isinstance(r, dict) or "error" in r:
        return [("Level A: результат тула", False, f"{r}", True)]
    top = r.get("top") or r.get("all") or []
    if not top:
        return [("Level A: непустой ранкинг", False, "ранкинг пуст", True)]
    winner = top[0]
    indep = _indep_rank_employees(
        conn, r["metric_id"], r.get("element"), r.get("snapshot_dates_used"),
        r.get("agg"), c["args"].get("roles"), c["args"].get("departments"),
    )
    out = []
    if indep is None:
        out.append(("Level A: сверка победителя", True, "agg не из {mean,min,max,sum} — пропуск", False))
    else:
        ok = indep[0] == winner.get("employee_id") and _close(indep[1], winner.get("value"))
        out.append(
            ("Level A: победитель == SQL", ok,
             f"тул: {winner.get('employee_id')}={winner.get('value')} | "
             f"SQL: {indep[0]}={indep[1]}", True)
        )
        out.append(
            ("Level B: значение лидера в ответе", _num_present(answer, winner.get("value")),
             f"значение={winner.get('value')}", False)
        )
    return out


def check_rank_departments(conn, calls, answer, fx):
    c = _find_call(calls, "rank_departments_by_metric")
    if c is None:
        return [("Level A: rank_departments_by_metric вызван", False, "тул не вызван", True)]
    r = c["result"]
    if not isinstance(r, dict) or "error" in r:
        return [("Level A: результат тула", False, f"{r}", True)]
    top = r.get("top") or r.get("all") or []
    if not top:
        return [("Level A: непустой ранкинг", False, "ранкинг пуст", True)]
    winner = top[0]
    indep = _indep_rank_departments(
        conn, r["metric_id"], r.get("element"),
        r.get("date_from"), r.get("date_to"), r.get("target_date"),
        r.get("agg"), c["args"].get("roles"),
    )
    out = []
    if indep is None:
        out.append(("Level A: сверка лидера", True, "agg не из {mean,min,max,sum} — пропуск", False))
    else:
        ok = indep[0] == winner.get("department") and _close(indep[1], winner.get("value"))
        out.append(
            ("Level A: лидер-департамент == SQL", ok,
             f"тул: {winner.get('department')}={winner.get('value')} | "
             f"SQL: {indep[0]}={indep[1]}", True)
        )
        out.append(
            ("Level B: департамент в ответе",
             str(winner.get("department", "")).lower() in (answer or "").lower(),
             f"департамент={winner.get('department')}", False)
        )
    return out


def check_timeseries(conn, calls, answer, fx):
    c = _find_call(calls, "get_metric_timeseries")
    if c is None:
        return [("Level A: get_metric_timeseries вызван", False, "тул не вызван", True)]
    r = c["result"]
    if not isinstance(r, dict) or "error" in r:
        return [("Level A: результат тула", False, f"{r}", True)]
    indep = _indep_timeseries_count(
        conn, r["metric_id"], r["employee_id"], r.get("element_filter"),
        c["args"].get("date_from"), c["args"].get("date_to"),
    )
    ok = indep == r.get("points_count")
    return [
        ("Level A: число точек == SQL COUNT", ok,
         f"тул={r.get('points_count')} | SQL={indep}", True)
    ]


def check_matrix(conn, calls, answer, fx):
    c = _find_call(calls, "get_metrics_matrix")
    if c is None:
        return [("Level A: get_metrics_matrix вызван", False, "тул не вызван", True)]
    r = c["result"]
    if not isinstance(r, dict) or "error" in r:
        return [("Level A: результат тула", False, f"{r}", True)]
    indep = _indep_matrix_count(
        conn, r.get("metric_ids") or [], r.get("employee_ids"), r.get("departments"),
        r.get("roles"), r.get("elements_filter"), r.get("target_date"),
    )
    ok = indep == r.get("cells_count")
    return [
        ("Level A: число ячеек == SQL COUNT", ok,
         f"тул={r.get('cells_count')} | SQL={indep}", True)
    ]


def check_compare_to_group(conn, calls, answer, fx):
    c = _find_call(calls, "compare_to_group")
    if c is None:
        return [("Level A: compare_to_group вызван", False, "тул не вызван", True)]
    r = c["result"]
    if not isinstance(r, dict) or "error" in r:
        return [("Level A: результат тула", False, f"{r}", True)]
    mode = (r.get("group_filter") or {}).get("mode")
    if mode == "custom":
        # кастомная группа — сверяем own_fact с fact_metric
        elem = r.get("element")
        where = ["employee_id = ?", "metric_id = ?", "snapshot_date = ?"]
        params: list = [r["employee_id"], r["metric_id"], r["target_date"]]
        if elem is None:
            where.append("element IS NULL")
        else:
            where.append("element = ?")
            params.append(elem)
        row = conn.execute(
            f"SELECT fact FROM fact_metric WHERE {' AND '.join(where)}", params
        ).fetchone()
        ok = row is not None and _close(row["fact"], r.get("own_fact"))
        return [
            ("Level A: own_fact == fact_metric", ok,
             f"тул={r.get('own_fact')} | SQL={row['fact'] if row else None}", True)
        ]
    # дефолтная peer-группа — сверяем peer_mean
    indep = _indep_peer_mean(conn, r["employee_id"], r["metric_id"], r.get("element"), r["target_date"])
    if indep is None or r.get("peer_mean") is None:
        return [
            ("Level A: peer_mean", True,
             f"peer-группа пуста/нет данных (тул peer_mean={r.get('peer_mean')}) — пропуск", False)
        ]
    ok = _close(indep[0], r.get("peer_mean"))
    return [
        ("Level A: peer_mean == среднее peer-группы", ok,
         f"тул={r.get('peer_mean')} | SQL={indep[0]} (n={indep[1]})", True)
    ]


# ───────────────────────────────────────────────────────────────────────────
# Инициализация и фикстуры
# ───────────────────────────────────────────────────────────────────────────


def init_session():
    print("Инициализация сессии…", flush=True)
    conn = open_session_db()
    load_json(conn, JSON_PATH)
    build_peer_groups(conn)
    build_metric_dynamics(conn)
    compute_severity(conn)
    n_emp = conn.execute("SELECT COUNT(*) FROM dim_employee").fetchone()[0]
    n_metrics = conn.execute("SELECT COUNT(*) FROM metric_catalog").fetchone()[0]
    n_facts = conn.execute("SELECT COUNT(*) FROM fact_metric").fetchone()[0]
    print(f"  загружено: сотрудников={n_emp}, метрик={n_metrics}, фактов={n_facts}\n", flush=True)
    return conn


def pick_fixtures(conn) -> dict:
    """Подбирает реальные значения из БД для подстановки в вопросы."""
    employees = [
        r["employee_id"]
        for r in conn.execute(
            "SELECT employee_id FROM dim_employee WHERE role = 'employee' ORDER BY employee_id"
        ).fetchall()
    ]
    if len(employees) < 3:
        raise SystemExit("Нужно минимум 3 сотрудника роли 'employee' для теста")

    # Все метрики + сколько у каждой строк fact_metric
    fact_counts = {
        r["metric_id"]: r["n"]
        for r in conn.execute(
            "SELECT metric_id, COUNT(*) AS n FROM fact_metric GROUP BY metric_id"
        ).fetchall()
    }
    all_metrics = list(
        conn.execute(
            """SELECT metric_id, name, level, has_element_breakdown
               FROM metric_catalog ORDER BY level, metric_id"""
        ).fetchall()
    )
    child_ids = {
        r["child_metric_id"]
        for r in conn.execute("SELECT child_metric_id FROM metric_edge").fetchall()
    }
    have_children = {
        r["parent_metric_id"]
        for r in conn.execute("SELECT DISTINCT parent_metric_id FROM metric_edge").fetchall()
    }
    l1_ids = {m["metric_id"] for m in all_metrics if m["metric_id"] not in child_ids}

    def _has_facts(m):
        return fact_counts.get(m["metric_id"], 0) > 0

    # l1_main: L1-метрика с фактами, без element-разреза, с детьми (для expand_metric)
    l1_pref = [
        m for m in all_metrics
        if m["metric_id"] in l1_ids and _has_facts(m)
        and not m["has_element_breakdown"] and m["metric_id"] in have_children
    ]
    l1_any = [m for m in all_metrics if m["metric_id"] in l1_ids and _has_facts(m)]
    l1_main = (l1_pref or l1_any or all_metrics)[0]

    # l2: другая метрика с фактами, без element-разреза (для get_metrics_matrix)
    l2_pref = [
        m for m in all_metrics
        if m["metric_id"] != l1_main["metric_id"] and _has_facts(m)
        and not m["has_element_breakdown"]
    ]
    l2_any = [
        m for m in all_metrics
        if m["metric_id"] != l1_main["metric_id"] and _has_facts(m)
    ]
    l1_second = (l2_pref or l2_any or [l1_main])[0]

    # метрика с element-разрезом + один её element
    brk = conn.execute(
        """SELECT metric_id, name, element_kind FROM metric_catalog
           WHERE has_element_breakdown = 1 ORDER BY metric_id LIMIT 1"""
    ).fetchone()
    brk_elem = None
    if brk is not None:
        er = conn.execute(
            """SELECT DISTINCT element FROM fact_metric
               WHERE metric_id = ? AND element IS NOT NULL ORDER BY element LIMIT 1""",
            (brk["metric_id"],),
        ).fetchone()
        brk_elem = er["element"] if er else None

    dept = conn.execute(
        """SELECT department FROM dim_employee
           GROUP BY department ORDER BY COUNT(*) DESC LIMIT 1"""
    ).fetchone()["department"]

    dates = [
        r["snapshot_date"]
        for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM fact_metric ORDER BY snapshot_date"
        ).fetchall()
    ]

    return {
        "emp1": employees[0],
        "emp2": employees[1],
        "emp3": employees[2],
        "l1_name": l1_main["name"],
        "l1_id": l1_main["metric_id"],
        "l2_name": l1_second["name"],
        "brk_name": brk["name"] if brk else l1_main["name"],
        "brk_kind": (brk["element_kind"] if brk and brk["element_kind"] else "элементам"),
        "brk_elem": brk_elem,
        "dept": dept,
        "date_from": dates[0],
        "date_to": dates[-1],
    }


def build_cases(fx: dict) -> list:
    return [
        {
            "tool": "get_employee_profile",
            "question": f"Дай общий обзор по сотруднику {fx['emp1']}: кто это и какие у него метрики верхнего уровня?",
            "check": check_profile,
        },
        {
            "tool": "list_elements",
            "question": "По каким продуктам или каналам вообще разбиваются метрики? Дай сводку по разрезам.",
            "check": None,
        },
        {
            "tool": "expand_metric",
            "question": (
                f"Раскрой метрику «{fx['l1_name']}» сотрудника {fx['emp1']} на один уровень вниз по дереву: "
                f"перечисли её дочерние метрики и их вклад."
            ),
            "check": None,
        },
        {
            "tool": "expand_by_element",
            "question": f"Разложи метрику «{fx['brk_name']}» сотрудника {fx['emp1']} по всем {fx['brk_kind']}.",
            "check": None,
        },
        {
            "tool": "rank_elements_for_employee",
            "question": f"Какой продукт или канал самый проблемный у сотрудника {fx['emp1']}?",
            "check": None,
        },
        {
            "tool": "rank_metrics_for_employee",
            "question": f"По какой метрике у сотрудника {fx['emp1']} хуже всего обстоят дела?",
            "check": None,
        },
        {
            "tool": "compare_employees_overview",
            "question": f"Сравни сотрудников {fx['emp1']} и {fx['emp2']}: где между ними самые большие расхождения?",
            "check": None,
        },
        {
            "tool": "compare_to_group",
            "question": (
                f"Сравни сотрудника {fx['emp1']} с его peer-группой коллег по метрике «{fx['l1_name']}»: "
                f"насколько его значение отклоняется от среднего по группе — выше или ниже?"
            ),
            "check": check_compare_to_group,
        },
        {
            "tool": "rank_employees_by_metric",
            "question": f"Кто из операторов лучший по метрике «{fx['l1_name']}»?",
            "check": check_rank_employees,
        },
        {
            "tool": "rank_departments_by_metric",
            "question": f"Какой департамент сильнее всех по метрике «{fx['l1_name']}»?",
            "check": check_rank_departments,
        },
        {
            "tool": "list_employees",
            "question": f"Кто входит в департамент «{fx['dept']}»? Перечисли сотрудников.",
            "check": None,
        },
        {
            "tool": "get_metrics_matrix",
            "question": (
                f"Сведи значения метрик «{fx['l1_name']}» и «{fx['l2_name']}» сразу по сотрудникам "
                f"{fx['emp1']}, {fx['emp2']} и {fx['emp3']} в одну таблицу."
            ),
            "check": check_matrix,
        },
        {
            "tool": "get_metric_timeseries",
            "question": f"Покажи динамику метрики «{fx['l1_name']}» у сотрудника {fx['emp1']} по датам.",
            "check": check_timeseries,
        },
        {
            "tool": "search_metrics",
            "question": "Найди метрики, связанные с переводами звонков.",
            "check": None,
        },
        {
            "tool": "rank_employees_by_metric",
            "question": (
                f"Кто был лучшим оператором по метрике «{fx['l1_name']}» "
                f"за период с {fx['date_from']} по {fx['date_to']}?"
            ),
            "check": check_rank_employees,
        },
    ]


# ───────────────────────────────────────────────────────────────────────────
# Прогон
# ───────────────────────────────────────────────────────────────────────────


def is_schema_error(exc: BaseException) -> bool:
    s = (repr(exc) + " " + str(exc)).lower()
    return any(k in s for k in ("validation", "pydantic", "schema", "is not an allowed value"))


def run(provider: str, model: str, quiet: bool) -> int:
    conn = init_session()
    fx = pick_fixtures(conn)
    print("Фикстуры из БД:")
    for k, v in fx.items():
        print(f"  {k:12}= {v}")
    print()

    agent = Agent(
        conn,
        model=model,
        provider=provider,
        verbose=not quiet,
        enable_memory=False,
    )
    print(f"Агент готов: provider={agent.provider}, model={agent.model}\n")

    cases = build_cases(fx)
    results = []  # по кейсу: dict

    for idx, case in enumerate(cases, 1):
        print("═" * 78)
        print(f"  Кейс {idx}/{len(cases)} — целевой тул: {case['tool']}")
        print(f"  Вопрос: {case['question']}")
        print("═" * 78, flush=True)

        start = len(_TOOL_LOG)
        answer, ask_exc = "", None
        t0 = time.perf_counter()
        try:
            answer = agent.ask(case["question"])
        except Exception as e:  # noqa: BLE001
            ask_exc = e
            traceback.print_exc()
        dt = time.perf_counter() - t0
        calls = _TOOL_LOG[start:]

        if not quiet and answer:
            print(f"\n  ОТВЕТ АГЕНТА:\n  {answer}\n", flush=True)

        # Проверки
        checks = []
        if ask_exc is not None:
            checks.append(
                ("вызов агента без исключений", False,
                 f"{'SCHEMA ' if is_schema_error(ask_exc) else ''}{ask_exc!r}", True)
            )
        else:
            checks.extend(check_clean(answer))
        for c in calls:
            if c["exception"] is not None:
                checks.append(
                    (f"execute_tool({c['tool']}) без исключений", False, repr(c["exception"]), True)
                )
            elif isinstance(c["result"], dict) and "error" in c["result"]:
                checks.append(
                    (f"{c['tool']} без error-результата", False, str(c["result"]["error"]), True)
                )
        if case["check"] is not None and ask_exc is None:
            try:
                checks.extend(case["check"](conn, calls, answer, fx))
            except Exception as e:  # noqa: BLE001
                checks.append(("выполнение сверки", False, f"исключение в чекере: {e!r}", True))

        called = [c["tool"] for c in calls]
        results.append(
            {
                "idx": idx,
                "target": case["tool"],
                "question": case["question"],
                "answer": answer,
                "ask_exc": ask_exc,
                "calls": calls,
                "called": called,
                "checks": checks,
                "dt": dt,
            }
        )

        target_hit = case["tool"] in called
        print(f"  Тулы вызваны ({len(calls)}): {called or '—'}")
        print(f"  Целевой тул {case['tool']}: {'ВЫЗВАН' if target_hit else 'НЕ вызван'}")
        for label, ok, detail, fatal in checks:
            mark = "OK  " if ok else ("FAIL" if fatal else "WARN")
            print(f"    [{mark}] {label}: {detail}")
        print(f"  ⏱ {dt:.1f}s", flush=True)
        time.sleep(1.0)  # лёгкая пауза против rate-limit

    return report(results)


# ───────────────────────────────────────────────────────────────────────────
# Отчёт
# ───────────────────────────────────────────────────────────────────────────


def report(results: list) -> int:
    print("\n" + "█" * 78)
    print("  ИТОГОВЫЙ ОТЧЁТ")
    print("█" * 78 + "\n")

    called_by_tool: dict = {t: [] for t in ALL_TOOLS}
    for r in results:
        for t in set(r["called"]):
            if t in called_by_tool:
                called_by_tool[t].append(r["idx"])

    print("Матрица покрытия тулов (14):")
    uncovered = []
    for t in ALL_TOOLS:
        cases_hit = called_by_tool[t]
        if cases_hit:
            print(f"  [✓] {t:30} ← кейсы {cases_hit}")
        else:
            print(f"  [✗] {t:30} ← НЕ ВЫЗВАН")
            uncovered.append(t)

    schema_errors = [r for r in results if r["ask_exc"] is not None and is_schema_error(r["ask_exc"])]
    other_ask_errors = [
        r for r in results if r["ask_exc"] is not None and not is_schema_error(r["ask_exc"])
    ]
    tool_exceptions = [
        (r["idx"], c["tool"], c["exception"])
        for r in results
        for c in r["calls"]
        if c["exception"] is not None
    ]
    error_results = [
        (r["idx"], c["tool"], c["result"].get("error"))
        for r in results
        for c in r["calls"]
        if isinstance(c["result"], dict) and "error" in c["result"] and c["exception"] is None
    ]
    fatal_checks = [
        (r["idx"], label, detail)
        for r in results
        for (label, ok, detail, fatal) in r["checks"]
        if fatal and not ok
    ]
    warn_checks = [
        (r["idx"], label, detail)
        for r in results
        for (label, ok, detail, fatal) in r["checks"]
        if not fatal and not ok
    ]

    print("\nПроблемы:")
    if uncovered:
        print(f"  • Непокрытые тулы: {uncovered}")
    if schema_errors:
        for r in schema_errors:
            print(f"  • SCHEMA-ошибка GigaChat, кейс {r['idx']}: {r['ask_exc']!r}")
    if other_ask_errors:
        for r in other_ask_errors:
            print(f"  • Ошибка вызова агента, кейс {r['idx']}: {r['ask_exc']!r}")
    if tool_exceptions:
        for idx, tool, exc in tool_exceptions:
            print(f"  • Исключение в execute_tool, кейс {idx}, тул {tool}: {exc!r}")
    if error_results:
        for idx, tool, err in error_results:
            print(f"  • Тул вернул error, кейс {idx}, тул {tool}: {err}")
    if fatal_checks:
        for idx, label, detail in fatal_checks:
            print(f"  • Провал проверки, кейс {idx}: {label} — {detail}")
    if not any([uncovered, schema_errors, other_ask_errors, tool_exceptions, error_results, fatal_checks]):
        print("  нет критичных проблем")

    if warn_checks:
        print("\nПредупреждения (не критично):")
        for idx, label, detail in warn_checks:
            print(f"  • кейс {idx}: {label} — {detail}")

    ok = not any(
        [uncovered, schema_errors, other_ask_errors, tool_exceptions, error_results, fatal_checks]
    )
    covered = sum(1 for t in ALL_TOOLS if called_by_tool[t])
    print(f"\nПокрытие: {covered}/{len(ALL_TOOLS)} тулов.")
    print("ВЕРДИКТ:", "ВСЕ КРИТИЧНЫЕ ПРОВЕРКИ ПРОЙДЕНЫ" if ok else "ЕСТЬ ПРОВАЛЫ — см. выше")
    print("█" * 78)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="e2e-тест function-calling по всем 14 тулам")
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "gigachat"),
        choices=["openai", "gigachat"],
    )
    parser.add_argument("--model", default=None, help="имя модели; по умолчанию — из .env / дефолт")
    parser.add_argument("--quiet", action="store_true", help="не печатать трейс вызовов тулов")
    args = parser.parse_args()

    model = resolve_model(args.model, args.provider)
    print(f"Тест function-calling: provider={args.provider}, model={model}\n")
    sys.exit(run(args.provider, model, args.quiet))


if __name__ == "__main__":
    main()
