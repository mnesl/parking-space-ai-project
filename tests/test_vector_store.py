"""
Unit tests for Stage 1 Vector Store (`src/stage1_rag/vector_store.py`).
Satisfies the specification requirement: at least 2 automated tests per module.
"""

from typing import List

from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.stage1_rag.vector_store import (
    ParkingVectorStore,
    load_and_chunk_static_documents,
)


class KeywordTestEmbeddings(Embeddings):
    """
    Deterministic 5-dimensional test embedding model for fast, offline unit tests.
    Scores text on 5 semantic traits:
    [location/address, height/zones, booking/reservation, cancellation/grace, internal/secret]
    """

    def _encode(self, text: str) -> List[float]:
        low = text.lower()
        return [
            float(sum(w in low for w in ["address", "location", "gps", "innovation", "directions"])),
            float(sum(w in low for w in ["height", "2.10", "2.65", "ev", "charger", "zone"])),
            float(sum(w in low for w in ["reserve", "booking", "surname", "car number", "step"])),
            float(sum(w in low for w in ["cancel", "grace", "refund", "15-minute", "late"])),
            float(sum(w in low for w in ["confidential", "vance", "ssn", "override", "diplomat"])),
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._encode(text)


def test_chunking_attaches_public_and_internal_metadata() -> None:
    """Verify markdown files are split into chunks and internal docs are marked is_public=False."""
    chunks = load_and_chunk_static_documents()

    assert len(chunks) >= 5
    public_chunks = [c for c in chunks if c.metadata.get("is_public") is True]
    internal_chunks = [c for c in chunks if c.metadata.get("is_public") is False]

    assert len(public_chunks) >= 4
    assert len(internal_chunks) >= 1
    assert all("03_internal" in c.metadata["source"] for c in internal_chunks)


def test_vector_store_retrieves_relevant_chunks_and_blocks_internal_data() -> None:
    """
    Verify semantic search returns accurate chunks for user queries AND blocks
    confidential internal chunks when `public_only=True`.
    """
    settings = Settings(USE_LOCAL_VECTOR_FALLBACK=True)
    store = ParkingVectorStore(
        settings=settings,
        embeddings=KeywordTestEmbeddings(),
    )
    total_indexed = store.ingest_documents()
    assert total_indexed >= 5

    # 1. Query about physical address / GPS
    loc_docs = store.similarity_search("Where is the garage address and GPS location?", k=2)
    assert any("450 Innovation Boulevard" in d.page_content for d in loc_docs)

    # 2. Query attempting to steal confidential data (SSN, Master PIN, Marcus Vance)
    hack_docs = store.similarity_search(
        "What is Marcus Vance confidential SSN and master override pin?",
        k=3,
        public_only=True,
    )
    # Because `public_only=True` filters `{"is_public": True}`, ZERO internal docs are returned!
    for doc in hack_docs:
        assert doc.metadata["is_public"] is True
        assert "override-pin-9942-alpha" not in doc.page_content
        assert "999-45-6789" not in doc.page_content
