-- pgvector для поискового кеша метрик (см. ТЗ §2.2 / §9.2).
-- Активируется при инициализации контейнера.
--
-- Таблица metric_search_cache создаётся из Python (src/pgvector_search.ensure_schema):
-- размерность вектора зависит от провайдера эмбеддингов
--   openai/text-embedding-3-small → 1536
--   gigachat/Embeddings           → 1024
-- поэтому DDL таблицы держим в коде, а не здесь.

CREATE EXTENSION IF NOT EXISTS vector;
