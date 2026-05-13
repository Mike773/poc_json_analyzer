"""Человекочитаемые описания tool calls и их результатов для трейса агента."""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional


TOOL_ACTION = {
    "get_employee_profile": "Профиль сотрудника",
    "list_elements": "Перечень элементов метрики",
    "list_employees": "Список сотрудников (с фильтрами)",
    "expand_metric": "Раскрытие метрики по детям дерева",
    "expand_by_element": "Раскрытие метрики по элементам",
    "rank_elements_for_employee": "Топ проблемных элементов сотрудника",
    "rank_metrics_for_employee": "Топ проблемных метрик сотрудника",
    "rank_employees_by_metric": "Ранжирование сотрудников по метрике",
    "rank_departments_by_metric": "Ранжирование департаментов по метрике",
    "compare_employees": "Сравнение двух сотрудников по метрике",
    "compare_employees_overview": "Обзорное сравнение двух сотрудников",
    "compare_to_group": "Сравнение сотрудника с peer-группой",
    "get_metric_timeseries": "Временной ряд метрики",
    "get_metrics_matrix": "Матрица значений (metric × employee)",
    "search_metrics": "Поиск метрик",
}


class Humanizer:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.metric_names = {
            r["metric_id"]: r["name"]
            for r in conn.execute("SELECT metric_id, name FROM metric_catalog").fetchall()
        }
        self.metric_dirs = {
            r["metric_id"]: r["direction"]
            for r in conn.execute("SELECT metric_id, direction FROM metric_catalog").fetchall()
        }
        self.emp_fios = {
            r["employee_id"]: r["fio"]
            for r in conn.execute("SELECT employee_id, fio FROM dim_employee").fetchall()
        }

    # ─── resolvers ──────────────────────────────────────────────────────

    def metric_label(self, mid: Optional[int]) -> str:
        if mid is None:
            return "—"
        name = self.metric_names.get(mid, f"id={mid}")
        direction = self.metric_dirs.get(mid)
        if direction:
            return f"{name} (id={mid}, {direction})"
        return f"{name} (id={mid})"

    def emp_label(self, eid: Optional[str]) -> str:
        if eid is None:
            return "—"
        fio = self.emp_fios.get(eid)
        return f"{fio} ({eid})" if fio else str(eid)

    # ─── action header ──────────────────────────────────────────────────

    def action(self, tool_name: str) -> str:
        return TOOL_ACTION.get(tool_name, tool_name)

    # ─── args formatting ────────────────────────────────────────────────

    def format_args(self, tool_name: str, args: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        # порядок «важных» полей
        order = [
            "metric_id", "metric_ids", "employee_id", "employee_ids",
            "emp_a", "emp_b",
            "element", "elements",
            "department", "departments",
            "date_from", "date_to", "target_date",
            "agg", "top_n", "bottom_n", "roles", "role", "level", "scope",
            "query", "axis",
        ]
        seen = set()
        for k in order:
            if k not in args:
                continue
            seen.add(k)
            v = args[k]
            if v is None:
                continue
            lines.append(self._format_kv(k, v))
        # хвост — всё остальное
        for k, v in args.items():
            if k in seen or v is None:
                continue
            lines.append(self._format_kv(k, v))
        return lines

    def _format_kv(self, k: str, v: Any) -> str:
        if k == "metric_id":
            return f"метрика     : {self.metric_label(v)}"
        if k == "metric_ids":
            return f"метрики     : {', '.join(self.metric_label(m) for m in v)}"
        if k in ("employee_id", "emp_a"):
            return f"сотрудник   : {self.emp_label(v)}"
        if k == "emp_b":
            return f"сотрудник Б : {self.emp_label(v)}"
        if k == "employee_ids":
            return f"сотрудники  : {', '.join(self.emp_label(e) for e in v)}"
        if k == "element":
            return f"элемент     : {v!s}"
        if k == "elements":
            return f"элементы    : {', '.join(map(str, v))}"
        if k == "date_from":
            return f"с           : {v}"
        if k == "date_to":
            return f"по          : {v}"
        if k == "target_date":
            return f"на дату     : {v}"
        if k == "agg":
            return f"агрегация   : {v}"
        if k == "top_n":
            return f"top         : {v}"
        if k == "bottom_n":
            return f"bottom      : {v}"
        if k == "roles":
            return f"роли        : {', '.join(v)}"
        if k == "role":
            return f"роль        : {v}"
        if k == "department":
            return f"департамент : {v}"
        if k == "departments":
            return f"департаменты: {', '.join(v)}"
        if k == "axis":
            return f"ось         : {v}"
        if k == "query":
            return f"запрос      : {v!r}"
        if k == "level":
            return f"уровень     : {v}"
        if k == "scope":
            return f"scope       : {v}"
        return f"{k:<12}: {v}"

    # ─── result formatting ──────────────────────────────────────────────

    def _fallback(self, result: Dict[str, Any]) -> List[str]:
        # Компактный JSON-обзор: верхние ключи
        out = []
        for k, v in result.items():
            if isinstance(v, list):
                out.append(f"{k}: список из {len(v)}")
            elif isinstance(v, dict):
                out.append(f"{k}: dict({len(v)} полей)")
            else:
                s = str(v)
                out.append(f"{k}: {s if len(s) < 80 else s[:80] + '…'}")
        return out

    # — per-tool formatters —

    def _fmt_get_employee_profile(self, r: Dict[str, Any]) -> List[str]:
        out = [
            f"{r.get('fio')} — {r.get('post')} ({r.get('role')})",
            f"target_date: {r.get('target_date')}",
            "L1-метрики:",
        ]
        for m in r.get("l1_metrics", []):
            sev = m.get("severity_total")
            sev_s = f"{sev:.3f}" if isinstance(sev, (int, float)) else "—"
            elem = f", element_kind={m['element_kind']}" if m.get("element_kind") else ""
            out.append(f"  • {m.get('name')} (id={m.get('metric_id')}, {m.get('direction')}{elem}): severity={sev_s}")
        return out

    def _fmt_list_elements(self, r: Dict[str, Any]) -> List[str]:
        if "elements" in r:
            els = r["elements"]
            return [
                f"метрика id={r.get('metric_id')}, element_kind={r.get('element_kind')!r}",
                f"всего элементов: {len(els)}",
                f"  {', '.join(els)}",
            ]
        # global
        bm = r.get("metrics_with_breakdown", {})
        return [f"метрик с breakdown: {len(bm)}"] + [
            f"  • id={mid}: {kind}" for mid, kind in list(bm.items())[:10]
        ]

    def _fmt_expand_metric(self, r: Dict[str, Any]) -> List[str]:
        out = [
            f"родитель: {self.metric_label(r.get('parent_metric_id'))}",
            f"элемент:  {r.get('element') or '—'}",
            f"target:   {r.get('target_date')}",
            f"детей: {len(r.get('children', []))}",
        ]
        for c in r.get("children", []):
            sev = c.get("severity_total")
            sev_s = f"{sev:.3f}" if isinstance(sev, (int, float)) else "—"
            w = c.get("weight")
            extras = []
            if w is not None:
                extras.append(f"weight={w}")
            if "fact" in c:
                extras.append(f"fact={c.get('fact')}")
            if "plan" in c and c.get("plan") is not None:
                extras.append(f"plan={c.get('plan')}")
            if "benchmark" in c and c.get("benchmark") is not None:
                extras.append(f"bench={c.get('benchmark')}")
            extras_s = ", ".join(extras)
            out.append(f"  • {c.get('name')} (id={c.get('metric_id')}, {c.get('direction')}): severity={sev_s}{', ' + extras_s if extras_s else ''}")
        return out

    def _fmt_expand_by_element(self, r: Dict[str, Any]) -> List[str]:
        m = r.get("metric") or {}
        out = [
            f"метрика: {m.get('name')} ({m.get('direction')}, kind={m.get('element_kind')})",
            f"target:  {r.get('target_date')}",
            f"элементов: {len(r.get('elements', []))}, отсортировано по severity_self",
        ]
        for e in r.get("elements", []):
            sev = e.get("severity_self")
            sev_s = f"{sev:.3f}" if isinstance(sev, (int, float)) else "—"
            out.append(
                f"  • {e.get('element')}: fact={e.get('fact')}, plan={e.get('plan')}, "
                f"bench={e.get('benchmark')}, severity={sev_s}"
            )
        return out

    def _fmt_rank_elements_for_employee(self, r: Dict[str, Any]) -> List[str]:
        out = [f"сотрудник {r.get('employee_id')}, target {r.get('target_date')}, ранжировка по severity:"]
        for item in r.get("ranking", []):
            agg = item.get("aggregate_severity")
            agg_s = f"{agg:.2f}" if isinstance(agg, (int, float)) else "—"
            tops = item.get("top_metrics", [])[:3]
            tops_s = "; ".join(f"{t['name']}={t['severity']:.2f}" for t in tops if isinstance(t.get('severity'), (int, float)))
            out.append(f"  • {item.get('element')}: aggregate_severity={agg_s} (топ метрик: {tops_s})")
        return out

    def _fmt_rank_metrics_for_employee(self, r: Dict[str, Any]) -> List[str]:
        out = [f"сотрудник {r.get('employee_id')}, ранжировка по severity_total:"]
        for item in r.get("ranking", []):
            sev = item.get("severity_total")
            sev_s = f"{sev:.3f}" if isinstance(sev, (int, float)) else "—"
            out.append(
                f"  • {item.get('name')} (L{item.get('level')}, id={item.get('metric_id')}): severity_total={sev_s}"
            )
        return out

    def _fmt_rank_departments_by_metric(self, r: Dict[str, Any]) -> List[str]:
        out = [
            f"метрика: {r.get('metric_name')} (id={r.get('metric_id')}, {r.get('direction')})",
            f"элемент: {r.get('element') or '—'}",
            f"агрегация: {r.get('agg')}",
            f"всего департаментов с данными: {r.get('departments_count')}",
        ]
        if r.get("top"):
            out.append("ТОП (лучшие):")
            for x in r["top"]:
                v = x.get("value")
                v_s = f"{v:.2f}" if isinstance(v, (int, float)) else "—"
                out.append(
                    f"  • {x.get('department')}: {v_s} (сотрудников={x.get('employees_count')}, точек={x.get('points_used')})"
                )
        if r.get("bottom"):
            out.append("BOTTOM (худшие):")
            for x in r["bottom"]:
                v = x.get("value")
                v_s = f"{v:.2f}" if isinstance(v, (int, float)) else "—"
                out.append(f"  • {x.get('department')}: {v_s}")
        return out

    def _fmt_list_employees(self, r: Dict[str, Any]) -> List[str]:
        filt = r.get("filter") or {}
        out = [
            f"фильтр: role={filt.get('role') or '—'}, department={filt.get('department') or '—'}, departments={filt.get('departments') or '—'}",
            f"найдено: {r.get('count')}",
        ]
        for e in r.get("employees", [])[:20]:
            out.append(
                f"  • {e.get('employee_id')} {e.get('fio')} [{e.get('department')}, {e.get('post')}, {e.get('role')}]"
            )
        if r.get("count", 0) > 20:
            out.append(f"  … ещё {r['count'] - 20}")
        return out

    def _fmt_rank_employees_by_metric(self, r: Dict[str, Any]) -> List[str]:
        out = [
            f"метрика: {r.get('metric_name')} (id={r.get('metric_id')}, {r.get('direction')})",
            f"элемент: {r.get('element') or '—'}",
            f"агрегация: {r.get('agg')} по {len(r.get('snapshot_dates_used', []))} снимкам "
            f"({', '.join(r.get('snapshot_dates_used', []))})",
        ]
        if r.get("top"):
            out.append("ТОП (лучшие):")
            for x in r["top"]:
                v = x.get("value")
                v_s = f"{v:.2f}" if isinstance(v, (int, float)) else "—"
                out.append(f"  • {x.get('fio')} ({x.get('employee_id')}, {x.get('role')}): {v_s} ({x.get('points_used')} точек)")
        if r.get("bottom"):
            out.append("BOTTOM (худшие):")
            for x in r["bottom"]:
                v = x.get("value")
                v_s = f"{v:.2f}" if isinstance(v, (int, float)) else "—"
                out.append(f"  • {x.get('fio')} ({x.get('employee_id')}): {v_s}")
        return out

    def _fmt_compare_employees_overview(self, r: Dict[str, Any]) -> List[str]:
        out = [
            f"сравнение {self.emp_label(r.get('emp_a'))} vs {self.emp_label(r.get('emp_b'))}",
            f"target: {r.get('target_date')}",
            "топ расхождений (отриц. = A лучше, полож. = A хуже):",
        ]
        for d in r.get("top_diffs", []):
            pct = d.get("diff_pct")
            pct_s = f"{pct*100:+.1f}%" if isinstance(pct, (int, float)) else "—"
            out.append(
                f"  • {d.get('metric_name')} / {d.get('element') or '—'}: "
                f"A={d.get('fact_a')} vs B={d.get('fact_b')} → {pct_s} ({d.get('direction')})"
            )
        return out

    def _fmt_compare_to_group(self, r: Dict[str, Any]) -> List[str]:
        flags = []
        if r.get("small_group_warning"):
            flags.append("small_group")
        if r.get("missing_group_warning"):
            flags.append("missing_group")
        if r.get("benchmark_unavailable"):
            flags.append("benchmark_unavailable")
        if r.get("benchmark_peer_disagreement"):
            flags.append("benchmark_peer_disagreement")
        flags_s = " | ".join(flags) if flags else "—"

        gf = r.get("group_filter") or {}
        if gf.get("mode") == "custom":
            scope_bits = []
            if gf.get("departments"):
                scope_bits.append(f"departments={gf['departments']}")
            if gf.get("roles"):
                scope_bits.append(f"roles={gf['roles']}")
            group_line = f"группа: КАСТОМНАЯ ({', '.join(scope_bits) or '—'})"
        else:
            group_line = f"группа: peer по умолчанию ({gf.get('scope', 'same post + dept')})"

        return [
            f"сотрудник: {self.emp_label(r.get('employee_id'))}",
            f"метрика:   {self.metric_label(r.get('metric_id'))}, элемент={r.get('element') or '—'}",
            f"target:    {r.get('target_date')}",
            group_line,
            f"deviation_plan_pct:      {r.get('deviation_plan_pct')}",
            f"deviation_benchmark_pct: {r.get('deviation_benchmark_pct')}",
            f"peer (qual={r.get('peer_group_quality')}, size={r.get('peer_group_size')}): "
            f"mean={r.get('peer_mean')}, median={r.get('peer_median')}, "
            f"z={r.get('peer_z_score')}, percentile={r.get('peer_percentile')}, rank={r.get('peer_rank')}",
            f"флаги: {flags_s}",
        ]

    def _fmt_get_metric_timeseries(self, r: Dict[str, Any]) -> List[str]:
        pts = r.get("points") or []
        out = [
            f"метрика: {self.metric_label(r.get('metric_id'))}",
            f"сотрудник: {self.emp_label(r.get('employee_id'))}",
            f"element_filter: {r.get('element_filter') or '—'} (has_breakdown={r.get('has_element_breakdown')})",
            f"точек: {r.get('points_count', len(pts))}",
        ]
        # Превью первых 3 и последних 3
        preview = pts[:3] + ([{"...": "..."}] if len(pts) > 6 else []) + (pts[-3:] if len(pts) > 3 else [])
        for p in preview:
            if p.get("...") == "...":
                out.append("  …")
                continue
            elem = f"[{p.get('element')}] " if p.get("element") else ""
            out.append(
                f"  • {p.get('snapshot_date')} {elem}fact={p.get('fact')}, plan={p.get('plan')}, bench={p.get('benchmark')}"
            )
        return out

    def _fmt_get_metrics_matrix(self, r: Dict[str, Any]) -> List[str]:
        cells = r.get("cells") or []
        out = [
            f"target: {r.get('target_date')}",
            f"метрик×сотрудников×элементов = {r.get('cells_count', len(cells))} ячеек",
        ]
        for c in cells[:8]:
            sev = c.get("severity_self")
            sev_s = f"{sev:.3f}" if isinstance(sev, (int, float)) else "—"
            elem = f" [{c.get('element')}]" if c.get("element") else ""
            out.append(
                f"  • {c.get('fio')} | {c.get('metric_name')}{elem}: "
                f"fact={c.get('fact')}, plan={c.get('plan')}, bench={c.get('benchmark')}, severity={sev_s}"
            )
        if len(cells) > 8:
            out.append(f"  … ещё {len(cells) - 8} ячеек")
        return out

    def _fmt_search_metrics(self, r: Any) -> List[str]:
        # search_metrics возвращает list, не dict
        if isinstance(r, list):
            results = r
        else:
            results = r.get("results") or []
        out = [f"найдено: {len(results)}"]
        for item in results:
            extras = []
            if "match" in item:
                extras.append(item["match"])
            if "cosine_distance" in item:
                extras.append(f"cosine={item['cosine_distance']:.3f}")
            extras_s = f" [{', '.join(extras)}]" if extras else ""
            out.append(
                f"  • {item.get('name')} (id={item.get('metric_id')}, L{item.get('level')}){extras_s}"
            )
        return out

    def format_result(self, tool_name: str, result: Any) -> List[str]:  # type: ignore[override]
        # search_metrics возвращает list — обработаем отдельно
        if tool_name == "search_metrics":
            return self._fmt_search_metrics(result)
        if not isinstance(result, dict):
            return [f"(непредвиденный формат: {type(result).__name__})"]
        if "error" in result:
            return [f"⚠ ошибка: {result['error']}"]
        fmt = getattr(self, f"_fmt_{tool_name}", None)
        if fmt is not None:
            try:
                return fmt(result)
            except Exception as e:
                return [f"(ошибка форматирования: {e}; fallback)", *self._fallback(result)]
        return self._fallback(result)
