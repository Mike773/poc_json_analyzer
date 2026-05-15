# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

POC аналитического агента по иерархическим метрикам колл-центра. На вход — один JSON
(`call_center_metrics.json`) с деревом метрик по сотрудникам; на выходе — LLM-агент, отвечающий
на аналитические вопросы, и нарратор, генерирующий отчёт об инсайтах. Код, комментарии и
промпты — на русском; ответы пользователю тоже русскоязычные.

## Commands

Зависимости: `pip install -r requirements.txt` (numpy, scipy, psycopg, langchain-*,
python-dotenv). Python 3.9+.

Конфигурация — через `.env` в корне проекта (скрипты подхватывают его автоматически
через `python-dotenv`; шаблон — `.env.example`). LLM и embeddings идут через LangChain
с двумя провайдерами:

- `LLM_PROVIDER` — `openai` (по умолчанию) или `gigachat`; флаг `--provider` перекрывает.
- `LLM_MODEL` — имя модели; флаг `--model` перекрывает, иначе дефолт провайдера
  (`o4-mini` / `GigaChat-Max`).
- Учётные данные: `OPENAI_API_KEY` (openai) или `GIGACHAT_CREDENTIALS` + `GIGACHAT_SCOPE`
  (gigachat). Резолв модели — `providers.resolve_model()`.

Скрипты запускаются из корня проекта (каждый сам добавляет корень в `sys.path`):

```bash
python3 scripts/demo.py                          # прогон всего пайплайна + тулов, без LLM
python3 scripts/chat.py                           # интерактивный чат с агентом
python3 scripts/chat.py -q "вопрос" --verbose     # one-shot + трейс вызовов тулов
python3 scripts/chat.py --provider gigachat       # переключить провайдера
python3 scripts/narrate.py --employee 10005       # insight engine + LLM-отчёт
```

Опциональный слой pgvector (семантический поиск метрик):

```bash
docker compose up -d                  # Postgres+pgvector на :5434
python3 scripts/populate_pg_cache.py  # залить эмбеддинги метрик в кеш
```

Тестов и линтера в репозитории нет. `scripts/demo.py` служит фактическим smoke-тестом пайплайна.

## Architecture

**Всё эфемерно.** Постоянного хранилища нет: каждый запуск скрипта заново читает
`call_center_metrics.json` в in-memory SQLite (`schema.open_session_db`). Postgres используется
только как опциональный кеш эмбеддингов для поиска, не для фактов.

**Пайплайн инициализации сессии — порядок шагов обязателен** (см. `chat.py:init_session`):

1. `schema.open_session_db()` — in-memory SQLite со звёздной схемой.
2. `ingest.load_json()` — рекурсивно разворачивает дерево `child_metrics` из JSON в плоские
   таблицы: `dim_employee`, `metric_catalog` (дедуплицирован по `metric_id`), `metric_edge`
   (рёбра дерева, по одному входящему на не-корневую метрику), `fact_metric` (значения per-employee).
3. `peers.build_peer_groups()` — peer-группы по точному совпадению (post, department); fallback'а нет.
4. `dynamics.build_metric_dynamics()` — тренды (linregress по 3 окнам), аномалии (z-score),
   отклонения от plan/benchmark, peer-агрегаты → таблица `metric_dynamics`.
5. `severity.compute_severity()` — **читает `metric_dynamics`, поэтому строго после шага 4**;
   считает severity по ячейкам, агрегирует по элементам, катит вверх по дереву через `metric_edge`.

**Слои поверх данных:**

- `tools.py` — чистые аналитические функции-запросы к SQLite (ранжирование, сравнение, drill-down).
  Это «инструменты», которые видит LLM.
- `providers.py` — абстракция провайдера поверх LangChain: `make_chat_model()` /
  `make_embeddings()` отдают `ChatOpenAI`/`OpenAIEmbeddings` или `GigaChat`/`GigaChatEmbeddings`
  в зависимости от `LLM_PROVIDER`. Единственное место, где импортируются `langchain_openai` /
  `langchain_gigachat` (импорты ленивые — нужен только пакет активного провайдера).
- `llm.py` — `Agent` (оркестратор: function-calling цикл через LangChain `bind_tools` →
  `execute_tool` → ответ) и `Narrator` (insights[] → связный текст). `TOOL_SCHEMAS` и
  `execute_tool` должны оставаться синхронными с сигнатурами в `tools.py`. Сообщения — объекты
  `langchain_core.messages` (`SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`).
  `build_system_prompt` встраивает каталог метрик и список сотрудников в системный промпт.
- `insights.py` — детекторы (plan_miss, benchmark_gap, trend, anomaly, peer_outlier,
  element_concentration) → `Insight` → дедуп. Питает `Narrator`.
- `humanize.py` — человекочитаемый рендер вызовов тулов для `--verbose` трейса.
- `pgvector_search.py` — опциональный семантический поиск; `tools.search_metrics` пытается
  использовать его и тихо откатывается на in-memory токен-поиск, если Postgres недоступен.
  Эмбеддинги — через `providers.make_embeddings()`. Размерность вектора зависит от провайдера
  (openai=1536, gigachat=1024), поэтому DDL таблицы `metric_search_cache` — в коде
  (`ensure_schema()`), а не в `01_schema.sql`; при смене провайдера кеш пересоздаётся.

## Conventions to know

- **`direction`**: `direct` → больше = лучше; `inverse` → больше = хуже. Знак отклонений
  нормирован так, что положительное значение всегда = «хуже плана/бенчмарка» (см.
  `config.direction_sign` и `dynamics._signed_pct`).
- **Конфигурация порогов** централизована в `config.Config` (`DEFAULT_CONFIG`) — пороги
  аномалий, трендов, peer-групп, веса severity. Менять поведение детекторов следует там.
- **Внутренние поля скрыты от пользователя.** Системный промпт в `llm.py` явно запрещает
  агенту упоминать `severity*`, `peer_z_score`, `deviation_*_pct`, `metric_id`, имена тулов
  и т.п. — при правках промпта/тулов сохраняй это разделение «внутреннее vs пользователю».
- **`element`** — разрез метрики (продукт/канал). У метрик с `has_element_breakdown` нет
  «общего итога»: запросы должны идти либо по конкретному element, либо через агрегирующие тулы.
- В коде встречаются ссылки на ТЗ (`§2.2`, `§9.2` и т.п.) — это внешний документ, в репозитории его нет.
