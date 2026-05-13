"""LLM-слой: агент-оркестратор и нарратор инсайтов через OpenAI o4-mini."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from . import tools as analytical_tools
from .humanize import Humanizer
from .insights import Insight


DEFAULT_MODEL = "o4-mini"


def _jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY не задан. Установите: export OPENAI_API_KEY=sk-..."
        )


# ---------------------------------------------------------------------------
# Schemas: описание тулов для function calling
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_profile",
            "description": "Возвращает атрибуты сотрудника (ФИО, должность) и список L1-метрик с агрегированным severity_total. Используй для начального обзора.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "Табельный номер сотрудника"},
                    "target_date": {"type": ["string", "null"], "description": "Дата снимка YYYY-MM-DD; по умолчанию — последняя"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_elements",
            "description": "Возвращает уникальные значения element для метрики (или для всех с has_element_breakdown). Используй когда пользователь спрашивает 'по каким продуктам/каналам/регионам разрезана метрика'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {"type": ["integer", "null"], "description": "ID метрики; null — общая сводка по всем метрикам с breakdown"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_metric",
            "description": "Раскрывает детей метрики в дереве (например, AHT → ring/ACW/AUX/ACD/HOLD). С element фиксированным — значения детей для этого элемента; без — агрегированный severity по элементам.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {"type": "integer", "description": "ID родительской метрики"},
                    "employee_id": {"type": "string"},
                    "element": {"type": ["string", "null"], "description": "Конкретный элемент или null"},
                    "target_date": {"type": ["string", "null"]},
                },
                "required": ["metric_id", "employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_by_element",
            "description": "Раскладывает метрику по всем её элементам (продуктам/каналам). Возвращает fact/plan/benchmark/severity для каждого элемента. Сортировано по severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {"type": "integer"},
                    "employee_id": {"type": "string"},
                    "target_date": {"type": ["string", "null"]},
                },
                "required": ["metric_id", "employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_elements_for_employee",
            "description": "Топ-N самых проблемных элементов (продуктов/каналов) для сотрудника, агрегированных по всем метрикам. Используй для запросов 'какой продукт самый плохой'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "target_date": {"type": ["string", "null"]},
                    "top_n": {"type": "integer", "default": 5},
                    "scope": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "Список metric_id для ограничения"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_metrics_for_employee",
            "description": "Топ-N самых проблемных метрик для сотрудника по severity_total. Используй для запросов 'по какой метрике хуже всего'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "target_date": {"type": ["string", "null"]},
                    "top_n": {"type": "integer", "default": 5},
                    "level": {"type": ["integer", "null"], "description": "Ограничить уровнем дерева"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_employees_overview",
            "description": "Сравнивает двух сотрудников по всем метрикам, возвращает топ-N максимальных нормированных расхождений с учётом direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emp_a": {"type": "string"},
                    "emp_b": {"type": "string"},
                    "target_date": {"type": ["string", "null"]},
                    "top_n": {"type": "integer", "default": 5},
                },
                "required": ["emp_a", "emp_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_group",
            "description": "Сравнивает значение сотрудника с группой коллег. По умолчанию (без фильтров) — peer-группа same post + same department, предрассчитанная при init. Если задать departments или roles — на лету собирает кастомную группу по фильтру (например, 'все операторы', 'операторы Секторов 1 и 2'). Возвращает benchmark-сигнал, peer-агрегаты (mean/median/z/percentile/rank) и флаги.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "metric_id": {"type": "integer"},
                    "element": {"type": ["string", "null"]},
                    "target_date": {"type": ["string", "null"]},
                    "departments": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Фильтр группы по департаментам. null = peer-группа по умолчанию.",
                    },
                    "roles": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "enum": ["employee", "manager"]},
                        "description": "Фильтр группы по ролям.",
                    },
                },
                "required": ["employee_id", "metric_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_employees_by_metric",
            "description": "Ранжирует сотрудников по одной метрике. Поддерживает фильтр по ролям и департаментам, агрегацию за интервал date_from/date_to. 'Лучшие' определяются автоматически с учётом direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {"type": "integer"},
                    "element": {"type": ["string", "null"], "description": "Для метрик с breakdown — обязателен; иначе null"},
                    "date_from": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD, включительно"},
                    "date_to": {"type": ["string", "null"], "description": "ISO, включительно. Если оба null — берётся одна дата (target_date)"},
                    "target_date": {"type": ["string", "null"], "description": "Используется только если date_from/date_to не заданы"},
                    "agg": {
                        "type": "string",
                        "enum": ["mean", "min", "max", "sum", "last"],
                        "description": "Агрегация в окне; default 'mean'",
                    },
                    "top_n": {"type": "integer", "default": 3},
                    "bottom_n": {"type": "integer", "default": 0},
                    "roles": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "enum": ["employee", "manager"]},
                    },
                    "departments": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Фильтр по департаментам",
                    },
                },
                "required": ["metric_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_departments_by_metric",
            "description": "Ранжирует ДЕПАРТАМЕНТЫ по метрике (агрегация значений сотрудников внутри каждого департамента, потом сортировка). Используй для запросов 'какой департамент лучше/хуже по X', 'сравни отделы по AHT'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {"type": "integer"},
                    "element": {"type": ["string", "null"]},
                    "date_from": {"type": ["string", "null"]},
                    "date_to": {"type": ["string", "null"]},
                    "target_date": {"type": ["string", "null"]},
                    "agg": {"type": "string", "enum": ["mean", "min", "max", "sum"], "default": "mean"},
                    "top_n": {"type": "integer", "default": 5},
                    "bottom_n": {"type": "integer", "default": 0},
                    "roles": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "enum": ["employee", "manager"]},
                    },
                },
                "required": ["metric_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_employees",
            "description": "Список сотрудников с фильтрами по role и/или department. Используй когда нужно узнать состав департамента или собрать employee_ids для последующего батч-запроса.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": ["string", "null"], "enum": ["employee", "manager", None]},
                    "department": {"type": ["string", "null"], "description": "Точное название департамента"},
                    "departments": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Несколько департаментов (OR)",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics_matrix",
            "description": "БАТЧ-выборка значений: крест-произведение metric_ids × (employee_ids | departments | roles) × elements за одну дату. Возвращает плоский список ячеек с fact/plan/benchmark/deviation/severity. Используй ВМЕСТО циклических per-employee вызовов. Сотрудников можно задать ИЛИ списком employee_ids, ИЛИ фильтром по departments / roles (или комбинацией) — нужен хотя бы один из трёх.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Список ID метрик. Можно одну.",
                    },
                    "employee_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Конкретные табельные номера. Если null — используется фильтр по departments/roles.",
                    },
                    "departments": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Все сотрудники из этих департаментов",
                    },
                    "roles": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "enum": ["employee", "manager"]},
                        "description": "Фильтр по роли",
                    },
                    "elements": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Фильтр по элементам (продуктам/каналам). null или пустой = все элементы.",
                    },
                    "target_date": {"type": ["string", "null"]},
                },
                "required": ["metric_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_timeseries",
            "description": "Возвращает временной ряд значений метрики (fact/plan/benchmark по датам). Для метрик с has_element_breakdown=true: если element НЕ задан — вернутся ВСЕ элементы (с колонкой element в каждой точке); задан — фильтр по этому элементу. Для метрик без breakdown element следует оставить null.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {"type": "integer"},
                    "employee_id": {"type": "string"},
                    "element": {"type": ["string", "null"], "description": "Конкретный элемент или null для всех (см. описание тула)"},
                    "date_from": {"type": ["string", "null"], "description": "ISO дата YYYY-MM-DD"},
                    "date_to": {"type": ["string", "null"]},
                },
                "required": ["metric_id", "employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_metrics",
            "description": "Поиск метрик по имени или описанию. Возвращает топ совпадений с metric_id, именем, флагами.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def execute_tool(name: str, args: Dict[str, Any], conn: sqlite3.Connection) -> Any:
    """Маршрутизирует вызов тула к analytical_tools."""
    if name == "get_employee_profile":
        return analytical_tools.get_employee_profile(conn, args["employee_id"], args.get("target_date"))
    if name == "list_elements":
        return analytical_tools.list_elements(conn, args.get("metric_id"))
    if name == "expand_metric":
        return analytical_tools.expand_metric(
            conn, args["metric_id"], args["employee_id"], args.get("element"), args.get("target_date")
        )
    if name == "expand_by_element":
        return analytical_tools.expand_by_element(
            conn, args["metric_id"], args["employee_id"], args.get("target_date")
        )
    if name == "rank_elements_for_employee":
        return analytical_tools.rank_elements_for_employee(
            conn, args["employee_id"], args.get("target_date"), args.get("top_n", 5), args.get("scope")
        )
    if name == "rank_metrics_for_employee":
        return analytical_tools.rank_metrics_for_employee(
            conn, args["employee_id"], args.get("target_date"), args.get("top_n", 5), args.get("level")
        )
    if name == "compare_employees_overview":
        return analytical_tools.compare_employees_overview(
            conn, args["emp_a"], args["emp_b"], args.get("target_date"), args.get("top_n", 5)
        )
    if name == "compare_to_group":
        return analytical_tools.compare_to_group(
            conn,
            args["employee_id"],
            args["metric_id"],
            args.get("element"),
            args.get("target_date"),
            args.get("departments"),
            args.get("roles"),
        )
    if name == "get_metrics_matrix":
        return analytical_tools.get_metrics_matrix(
            conn,
            args["metric_ids"],
            args.get("employee_ids"),
            args.get("elements"),
            args.get("target_date"),
            args.get("departments"),
            args.get("roles"),
        )
    if name == "rank_employees_by_metric":
        return analytical_tools.rank_employees_by_metric(
            conn,
            args["metric_id"],
            args.get("element"),
            args.get("date_from"),
            args.get("date_to"),
            args.get("target_date"),
            args.get("agg", "mean"),
            args.get("top_n", 3),
            args.get("bottom_n", 0),
            args.get("roles"),
            args.get("departments"),
        )
    if name == "rank_departments_by_metric":
        return analytical_tools.rank_departments_by_metric(
            conn,
            args["metric_id"],
            args.get("element"),
            args.get("date_from"),
            args.get("date_to"),
            args.get("target_date"),
            args.get("agg", "mean"),
            args.get("top_n", 5),
            args.get("bottom_n", 0),
            args.get("roles"),
        )
    if name == "list_employees":
        return analytical_tools.list_employees(
            conn,
            args.get("role"),
            args.get("department"),
            args.get("departments"),
        )
    if name == "get_metric_timeseries":
        return analytical_tools.get_metric_timeseries(
            conn, args["metric_id"], args["employee_id"], args.get("element"), args.get("date_from"), args.get("date_to")
        )
    if name == "search_metrics":
        return analytical_tools.search_metrics(conn, args["query"])
    return {"error": f"unknown tool: {name}"}


# ---------------------------------------------------------------------------
# System prompt: каталог + сотрудники, БЕЗ значений и severity (по ТЗ)
# ---------------------------------------------------------------------------


def build_system_prompt(conn: sqlite3.Connection) -> str:
    # L1-метрики
    l1_rows = list(
        conn.execute(
            """SELECT c.metric_id, c.name, c.description, c.direction, c.has_plan,
                      c.has_element_breakdown, c.element_kind
               FROM metric_catalog c
               LEFT JOIN metric_edge e ON e.child_metric_id = c.metric_id
               WHERE e.child_metric_id IS NULL
               ORDER BY c.metric_id"""
        ).fetchall()
    )
    l1_lines = []
    for r in l1_rows:
        flags = []
        if r["has_plan"]:
            flags.append("has_plan")
        if r["has_element_breakdown"]:
            flags.append(f"breakdown={r['element_kind']}")
        l1_lines.append(
            f"  - id={r['metric_id']}, name={r['name']!r}, direction={r['direction']}, {', '.join(flags) or '—'} | {r['description']}"
        )

    # Все метрики (свёрнутый список — агент может через search_metrics доставать L2/L3)
    all_metrics = list(
        conn.execute(
            "SELECT metric_id, name, direction, level FROM metric_catalog ORDER BY level, metric_id"
        ).fetchall()
    )
    all_metrics_lines = "\n".join(
        f"  L{r['level']} id={r['metric_id']} {r['name']!r} ({r['direction']})" for r in all_metrics
    )

    # Сотрудники + департаменты
    emp_rows = list(
        conn.execute(
            "SELECT employee_id, fio, post, department, role FROM dim_employee ORDER BY department, role DESC, employee_id"
        ).fetchall()
    )
    emp_lines = "\n".join(
        f"  - {r['employee_id']}: {r['fio']} [dept={r['department']}, {r['post']}, {r['role']}]" for r in emp_rows
    )

    dept_rows = list(
        conn.execute(
            """SELECT department,
                      COUNT(*) AS n_total,
                      SUM(CASE WHEN role = 'manager' THEN 1 ELSE 0 END) AS n_mgr,
                      SUM(CASE WHEN role = 'employee' THEN 1 ELSE 0 END) AS n_emp
               FROM dim_employee
               GROUP BY department
               ORDER BY department"""
        ).fetchall()
    )
    dept_lines = "\n".join(
        f"  - {r['department']}: всего {r['n_total']} (managers={r['n_mgr']}, employees={r['n_emp']})"
        for r in dept_rows
    )

    target_date = conn.execute("SELECT MAX(snapshot_date) FROM fact_metric").fetchone()[0]

    # Доступные даты снимков (нужны агенту, чтобы не угадывать несуществующие)
    dates_rows = conn.execute(
        "SELECT DISTINCT snapshot_date FROM fact_metric ORDER BY snapshot_date"
    ).fetchall()
    snapshot_dates = [r["snapshot_date"] for r in dates_rows]
    calc_period = conn.execute(
        "SELECT calc_period FROM fact_metric WHERE calc_period IS NOT NULL LIMIT 1"
    ).fetchone()
    calc_period = calc_period["calc_period"] if calc_period else "—"

    return f"""Ты — аналитический ассистент по метрикам сотрудников колл-центра.

Целевая дата сессии по умолчанию: {target_date}.
Периодичность снимков: {calc_period}. Доступные даты ({len(snapshot_dates)}): {', '.join(snapshot_dates)}.
ВАЖНО: используй ТОЛЬКО даты из этого списка как target_date. Для интервалов («за сентябрь», «последний месяц») задавай date_from/date_to и используй rank_employees_by_metric или get_metric_timeseries с диапазоном.

Департаменты в анализе:
{dept_lines}

Сотрудники в анализе:
{emp_lines}

Каталог корневых (L1) метрик:
{chr(10).join(l1_lines)}

Полный список метрик (для понимания дерева):
{all_metrics_lines}

Правила интерпретации:
- direction=direct → больше значит лучше; direction=inverse → больше значит хуже.
- Если у метрики has_element_breakdown=True, она существует в нескольких разрезах по element_kind (например, по продуктам). «Общего итога» по элементам нет — используй expand_by_element для разреза или rank_elements_for_employee для агрегата по сотруднику.
- deviation_plan_pct и deviation_benchmark_pct: положительное значение = хуже плана/бенчмарка с учётом direction.
- benchmark — внешний peer-сигнал от источника (per-row), варьируется по сотрудникам и датам. Первичный сигнал для сравнения с коллегами.
- peer_z_score — вычисленный по локальной группе коллег того же поста; недоступен у руководителя (он один в своей группе).
- benchmark_peer_disagreement=true означает, что внешний benchmark и computed peer расходятся — упомяни оба угла зрения.
- severity_total ∈ [0,1]; чем выше — тем серьёзнее проблема. rollup_quality показывает, на чём построен роллап:
  - 'weighted' — все веса детей известны;
  - 'partial_weights' — часть весов отсутствует;
  - 'equal_weights' — все веса отсутствуют, агрегат по max.

Стиль ответа пользователю:
- Отвечай на русском. Кратко, фактологически.
- Имена сущностей оставляй как есть: «AHT», «доля переводов», «Продукт 4», «Сектор 1», ФИО.
- Значения сопровождай контекстом: «AHT 350 секунд (план 270, бенчмарк 277)».
- Отклонения формулируй процентом: «превышение плана на 30%», «ниже среднего на 5%».

ВАЖНО — НЕ упоминай в ответе внутренние/системные сущности. Это поля API и метрики качества, они для тебя, не для пользователя:
- severity, severity_self, severity_total, aggregate_severity — НЕ упоминай;
  вместо этого формулируй конкретными отклонениями (план, бенчмарк, сравнение с группой).
- deviation_plan_pct, deviation_benchmark_pct — НЕ упоминай эти имена; используй «отклонение от плана»/«от бенчмарка» в процентах.
- peer_z_score, peer_percentile, peer_rank, peer_group_quality, peer_group_size, peer_mean/median — НЕ упоминай напрямую.
  Переводи естественно: z≈1.4 → «значительно выше среднего по группе»; percentile 10 (для inverse) → «входит в 10% худших»; rank 9/10 → «9-й из 10» или «почти худший».
- rollup_quality, benchmark_unavailable, benchmark_peer_disagreement, *_warning — это служебные флаги.
  Если важно, перескажи смысл: «расхождение между внешним бенчмарком и сравнением с коллегами», но без слова с подчёркиванием.
- metric_id, employee_id, snapshot_date — НЕ выводи как технические идентификаторы.
  Табельный номер допустим в скобках при первом упоминании ФИО: «Михайлов А. (10008)».
- Даты в формате YYYY-MM-DD преобразуй в читаемые: «2025-11-03» → «3 ноября 2025».
- Имена тулов (rank_employees_by_metric, get_metrics_matrix и т.п.) НЕ упоминай вообще.

Эффективное использование тулов (это для тебя, не для пользователя):
- Несколько сотрудников × несколько метрик → один get_metrics_matrix.
- Топ N лучших/худших за период → rank_employees_by_metric с date_from/date_to и agg.
- Ранжирование подразделений → rank_departments_by_metric.
- Список людей в отделе → list_employees.
- Сравнение с произвольной группой (не только peer по умолчанию) → compare_to_group с departments / roles.
"""


# ---------------------------------------------------------------------------
# Agent: один пользовательский запрос → tool-цикл → финальный ответ
# ---------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        conn: sqlite3.Connection,
        model: str = DEFAULT_MODEL,
        reasoning_effort: Optional[str] = None,
        max_iterations: int = 12,
        verbose: bool = False,
        enable_memory: bool = True,
    ):
        """reasoning_effort: 'low' | 'medium' | 'high' | None.
        None — параметр не передаётся в API (работает с не-reasoning моделями).

        enable_memory: если False, каждый ask() работает изолированно (без истории пар user/assistant).
        """
        _ensure_api_key()
        self.conn = conn
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.enable_memory = enable_memory
        self.client = OpenAI()
        self.system_prompt = build_system_prompt(conn)
        # История диалога: только пары user/assistant (без tool_calls / tool results).
        self.history: List[Dict[str, Any]] = []
        self.humanizer = Humanizer(conn)

    def reset_history(self) -> None:
        self.history = []

    def _trace(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _preview(obj: Any, limit: int = 400) -> str:
        s = _jdump(obj)
        if len(s) <= limit:
            return s
        return s[:limit] + f"… ({len(s)} симв.)"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 1.0:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60.0:
            return f"{seconds:.2f}s"
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}m{s:.1f}s"

    def ask(self, question: str) -> str:
        if self.enable_memory:
            messages: List[Dict[str, Any]] = (
                [{"role": "system", "content": self.system_prompt}]
                + self.history
                + [{"role": "user", "content": question}]
            )
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]

        t_total_start = time.perf_counter()
        t_llm_total = 0.0
        t_tools_total = 0.0
        tokens_prompt_total = 0
        tokens_completion_total = 0

        if self.enable_memory:
            n_pairs = len(self.history) // 2
            history_hint = f" | history: {n_pairs} пар(ы)" if n_pairs else ""
        else:
            history_hint = " | memory: OFF"
        self._trace(f"\n┌─ user → {question}{history_hint}")

        for iteration in range(self.max_iterations):
            step = iteration + 1
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

            self._trace(
                f"├─ [#{step}] LLM call → model={self.model}"
                f"{', effort=' + self.reasoning_effort if self.reasoning_effort else ''}"
            )
            t_llm = time.perf_counter()
            response = self.client.chat.completions.create(**kwargs)
            dt_llm = time.perf_counter() - t_llm
            t_llm_total += dt_llm
            msg = response.choices[0].message

            usage = response.usage
            if usage:
                tokens_prompt_total += usage.prompt_tokens
                tokens_completion_total += usage.completion_tokens
                self._trace(
                    f"│  ⏱ {self._fmt_time(dt_llm)} | tokens: prompt={usage.prompt_tokens}, "
                    f"completion={usage.completion_tokens}, total={usage.total_tokens}"
                )
            else:
                self._trace(f"│  ⏱ {self._fmt_time(dt_llm)}")

            # Промежуточный текст ассистента — печатаем только если впереди есть tool-вызовы,
            # иначе финальный текст продублируется (вызывающий код печатает ответ сам).
            if msg.tool_calls and msg.content and msg.content.strip():
                self._trace(f"│  💭 {msg.content.strip()}")

            # Append assistant turn — preserve tool_calls structure
            assistant_dict: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_dict)

            # Done?
            if not msg.tool_calls:
                final_text = msg.content or ""
                # Сохраняем ТОЛЬКО пару user/assistant в историю — без tool_calls и tool results.
                if self.enable_memory:
                    self.history.append({"role": "user", "content": question})
                    self.history.append({"role": "assistant", "content": final_text})
                dt_total = time.perf_counter() - t_total_start
                if self.enable_memory:
                    mem_suffix = f"history → {len(self.history) // 2} пар(ы)"
                else:
                    mem_suffix = "memory OFF (история не сохранена)"
                self._trace(
                    f"└─ финал (step #{step}) | total={self._fmt_time(dt_total)} "
                    f"(llm={self._fmt_time(t_llm_total)}, tools={self._fmt_time(t_tools_total)}) | "
                    f"tokens prompt+completion={tokens_prompt_total}+{tokens_completion_total} | "
                    f"{mem_suffix}\n"
                )
                return final_text

            # Execute tools and append results
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    result = {"error": f"invalid JSON arguments: {e}"}
                    self._trace(f"│  🔧 {self.humanizer.action(name)} ({name})")
                    self._trace(f"│     ⚠ ошибка парсинга аргументов: {e}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": _jdump(result)})
                    continue

                self._trace(f"│  🔧 {self.humanizer.action(name)}  →  {name}")
                for line in self.humanizer.format_args(name, args):
                    self._trace(f"│     {line}")

                t_tool = time.perf_counter()
                try:
                    result = execute_tool(name, args, self.conn)
                except Exception as e:
                    result = {"error": str(e)}
                dt_tool = time.perf_counter() - t_tool
                t_tools_total += dt_tool

                self._trace(f"│  📊 результат  ({self._fmt_time(dt_tool)}):")
                for line in self.humanizer.format_result(name, result):
                    self._trace(f"│     {line}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _jdump(result),
                    }
                )

        return "[превышено максимальное число итераций tool-цикла]"


# ---------------------------------------------------------------------------
# Narrator: insights[] → связный текст
# ---------------------------------------------------------------------------


NARRATOR_SYSTEM = """Ты — аналитик. На вход подаётся структурированный список инсайтов о сотрудниках колл-центра.

Сделай связный отчёт на русском. Правила:
- Группируй по сотруднику (по ФИО), внутри — по проблемным областям.
- Для каждого сотрудника начни с одного-двух предложений общей картины.
- Конкретные значения упоминай аккуратно: fact, plan, deviation_plan_pct, deviation_benchmark_pct.
- Element упоминай по element_kind (например «Продукт 4», а не «элемент»).
- Не более 6 буллетов на сотрудника; самые серьёзные первыми (severity).
- Тип сигнала указывай человеческим языком: plan_miss → «не выполняет план», benchmark_gap → «отстаёт от бенчмарка», trend_systemic → «системный негативный тренд», anomaly → «аномалия», peer_outlier → «выбивается из группы», element_concentration → «проблема сконцентрирована в одном элементе».
- Не повторяй одно и то же дважды (после дедупа также есть поле 'also' внутри evidence — учитывай его при упоминании сопутствующих сигналов).
- Сухо, без эмоциональных оборотов.
"""


class Narrator:
    def __init__(
        self,
        conn: sqlite3.Connection,
        model: str = DEFAULT_MODEL,
        reasoning_effort: Optional[str] = None,
    ):
        """reasoning_effort: None отключает параметр (для не-reasoning моделей)."""
        _ensure_api_key()
        self.conn = conn
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI()
        self.emp_names = {
            r["employee_id"]: f"{r['fio']} ({r['post']})"
            for r in conn.execute("SELECT employee_id, fio, post FROM dim_employee").fetchall()
        }
        self.metric_names = {
            r["metric_id"]: r["name"]
            for r in conn.execute("SELECT metric_id, name FROM metric_catalog").fetchall()
        }

    def narrate(self, insights: List[Insight], top_per_employee: int = 8, verbose: bool = False) -> str:
        # Подготовим компактный JSON: ФИО, имя метрики, тип, severity, evidence
        from collections import defaultdict

        by_emp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for i in sorted(insights, key=lambda x: x.severity, reverse=True):
            by_emp[i.employee_id].append(
                {
                    "type": i.type,
                    "metric": self.metric_names.get(i.metric_id, str(i.metric_id)),
                    "element": i.element,
                    "severity": round(i.severity, 3),
                    "evidence": i.evidence,
                }
            )

        report_input = []
        for emp_id, items in by_emp.items():
            report_input.append(
                {
                    "employee": self.emp_names.get(emp_id, emp_id),
                    "insights": items[:top_per_employee],
                }
            )

        user_msg = (
            "Инсайты:\n```json\n"
            + json.dumps(report_input, ensure_ascii=False, indent=2, default=str)
            + "\n```"
        )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": NARRATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        t_start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t_start
        usage = response.usage
        if verbose:
            tu = (
                f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
                if usage
                else "—"
            )
            print(f"  ⏱ narrator LLM call: {Agent._fmt_time(dt)} | tokens: {tu}", flush=True)
        return response.choices[0].message.content or ""
