"""
Centralized Configuration & LLM/Embeddings Factory.

Educational Concept:
--------------------
Instead of coupling our RAG pipeline or agents to a specific LLM vendor,
this module implements the **Factory Pattern** on top of Pydantic Settings:
  - `get_Settings()` loads and validates environment variables from `.env`.
  - `get_chat_model()` returns a LangChain `BaseChatModel` (Gemini or OpenAI).
  - `get_embeddings()` returns a LangChain `Embeddings` model and automatically
    tracks the expected vector dimension (768 for Gemini, 1536 for OpenAI).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DOCS_DIR = DATA_DIR / "static_docs"
SQLITE_DB_PATH = DATA_DIR / "parking_dynamic.db"
CONFIRMED_RESERVATIONS_FILE = DATA_DIR / "confirmed_reservations.txt"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    llm_provider: Literal["gemini", "openai"] = Field(
        default="gemini",
        alias="LLM_PROVIDER",
        description="Which LLM/Embedding provider to use ('gemini' or 'openai').",
    )

    # Google Gemini configuration
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_chat_model: str = Field(
        default="gemini-3.1-flash-lite", alias="GEMINI_CHAT_MODEL"
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )

    # OpenAI configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # Pinecone configuration
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        default="parking-knowledge-base", alias="PINECONE_INDEX_NAME"
    )
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    use_local_vector_fallback: bool = Field(
        default=False, alias="USE_LOCAL_VECTOR_FALLBACK"
    )

    # Security keys for Stage 2 (Admin API) and Stage 3 (MCP Server)
    admin_api_key: str = Field(
        default="secret-admin-approval-key-2026", alias="ADMIN_API_KEY"
    )
    mcp_server_api_key: str = Field(
        default="secret-mcp-writer-key-2026", alias="MCP_SERVER_API_KEY"
    )
    admin_api_base_url: str = Field(
        default="http://127.0.0.1:8000", alias="ADMIN_API_BASE_URL"
    )

    @property
    def embedding_dimension(self) -> int:
        """
        Return the exact vector dimension produced by the active embedding model.
        - Google `models/gemini-embedding-001`: 3072 dimensions (or auto-detected)
        - OpenAI `text-embedding-3-small`: 1536 dimensions
        """
        if self.llm_provider == "gemini":
            return 768
        return 1536


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()


def get_chat_model(
    temperature: float = 0.0,
    settings: Settings | None = None,
) -> BaseChatModel:
    """
    Factory function that returns a configured LangChain Chat Model.
    We default to temperature=0.0 for deterministic RAG answers and slot extraction.
    """
    cfg = settings or get_settings()

    if cfg.llm_provider == "gemini":
        if not cfg.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing! Get a free key at https://aistudio.google.com/apikey "
                "or set LLM_PROVIDER=openai in your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.gemini_chat_model,
            google_api_key=cfg.google_api_key,
            temperature=temperature,
            timeout=8.0,
            max_retries=0,
        )

    if cfg.llm_provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing in your .env file.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.openai_chat_model,
            api_key=cfg.openai_api_key,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {cfg.llm_provider}")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """
    Factory function that returns the configured LangChain Embeddings model.
    """
    cfg = settings or get_settings()

    if cfg.llm_provider == "gemini":
        if not cfg.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing! Needed for Google gemini-embedding-001."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=cfg.gemini_embedding_model,
            google_api_key=cfg.google_api_key,
        )

    if cfg.llm_provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing in your .env file.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {cfg.llm_provider}")


def extract_llm_text(content: object) -> str:
    """
    Normalize LangChain `AIMessage.content` into a clean string.
    Handles both plain strings and Gemini 3.x structured content blocks
    (`[{"type": "text", "text": "..."}]`).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(p for p in parts if p).strip()
    return str(content)

