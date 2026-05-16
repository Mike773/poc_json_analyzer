"""Провайдеры LLM и embeddings поверх LangChain.

Поддерживаются два провайдера:
  - openai   → langchain_openai (ChatOpenAI, OpenAIEmbeddings)
  - gigachat → langchain_gigachat (GigaChat, GigaChatEmbeddings)

Активный провайдер выбирается переменной окружения LLM_PROVIDER
(значение по умолчанию — openai). Имя модели и провайдер можно также
передать явно в make_chat_model() / make_embeddings().

Учётные данные (через окружение):
  - openai:   OPENAI_API_KEY
  - gigachat: GIGACHAT_CREDENTIALS (+ опционально GIGACHAT_SCOPE)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


PROVIDERS = ("openai", "gigachat")
DEFAULT_PROVIDER = "openai"

_DEFAULT_CHAT_MODEL = {
    "openai": "o4-mini",
    "gigachat": "GigaChat-Max",
}

_EMBED_MODEL = {
    "openai": "text-embedding-3-small",
    "gigachat": "Embeddings",
}

# Размерность вектора эмбеддинга по провайдеру — нужна для схемы pgvector.
_EMBED_DIM = {
    "openai": 1536,
    "gigachat": 1024,
}


def get_provider() -> str:
    """Активный провайдер из переменной окружения LLM_PROVIDER (по умолчанию openai)."""
    return _resolve(os.environ.get("LLM_PROVIDER"))


def _resolve(provider: Optional[str]) -> str:
    provider = (provider or DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"неизвестный провайдер {provider!r}; допустимо: {', '.join(PROVIDERS)}"
        )
    return provider


def default_chat_model(provider: Optional[str] = None) -> str:
    """«Чистый» дефолт модели для провайдера (без учёта окружения)."""
    return _DEFAULT_CHAT_MODEL[_resolve(provider)]


def resolve_model(model: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Имя чат-модели: явный аргумент > LLM_MODEL из окружения > дефолт провайдера."""
    if model:
        return model
    env_model = os.environ.get("LLM_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    return default_chat_model(provider)


def embedding_dim(provider: Optional[str] = None) -> int:
    """Размерность вектора эмбеддинга активного (или указанного) провайдера."""
    return _EMBED_DIM[_resolve(provider)]


def make_chat_model(
    model: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> BaseChatModel:
    """Чат-модель LangChain для активного (или указанного) провайдера.

    reasoning_effort применяется только к reasoning-моделям OpenAI;
    для GigaChat параметр игнорируется.
    """
    provider = _resolve(provider)
    model = model or default_chat_model(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Установите: export OPENAI_API_KEY=sk-..."
            )
        kwargs: Dict[str, Any] = {"model": model}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        return ChatOpenAI(**kwargs)

    # gigachat
    from langchain_gigachat import GigaChat

    if not os.environ.get("GIGACHAT_CREDENTIALS"):
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS не задан. Установите ключ авторизации GigaChat: "
            "export GIGACHAT_CREDENTIALS=..."
        )
    return GigaChat(model=model, verify_ssl_certs=False)


def make_embeddings(provider: Optional[str] = None) -> Embeddings:
    """Embeddings-модель LangChain для активного (или указанного) провайдера."""
    provider = _resolve(provider)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY не задан — нужен для embeddings.")
        return OpenAIEmbeddings(model=_EMBED_MODEL["openai"])

    # gigachat
    from langchain_gigachat import GigaChatEmbeddings

    if not os.environ.get("GIGACHAT_CREDENTIALS"):
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS не задан — нужен для embeddings GigaChat."
        )
    return GigaChatEmbeddings(model=_EMBED_MODEL["gigachat"], verify_ssl_certs=False)
