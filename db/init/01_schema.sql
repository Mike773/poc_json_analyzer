-- pgvector + схема для поискового кеша метрик (см. ТЗ §2.2 / §9.2).
-- Активируется при загрузке инициализации контейнера.

CREATE EXTENSION IF NOT EXISTS vector;

-- Размерность 1536 = text-embedding-3-small (OpenAI).
CREATE TABLE IF NOT EXISTS metric_search_cache (
  direction_key text NOT NULL,
  metric_id     bigint NOT NULL,
  name          text NOT NULL,
  name_hash     text NOT NULL,           -- sha256 имени, для diff'а
  abbreviations text[] DEFAULT '{}',     -- токены/аббревиатуры из имени
  vector        vector(1536),
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (direction_key, metric_id)
);

-- HNSW по cosine; ivfflat — альтернатива, для больших объёмов.
-- На < 100 строк индекс не критичен, но создаём заранее.
CREATE INDEX IF NOT EXISTS idx_metric_search_cache_vector
  ON metric_search_cache USING hnsw (vector vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_metric_search_cache_abbr
  ON metric_search_cache USING gin (abbreviations);

CREATE INDEX IF NOT EXISTS idx_metric_search_cache_direction
  ON metric_search_cache (direction_key);
