"""
Stage 1: Pinecone Vector Database Ingestion & Semantic Retriever (`vector_store.py`).

Educational Concepts Covered Here:
----------------------------------
1. **Document Chunking (`RecursiveCharacterTextSplitter`):**
   Why don't we turn an entire 10-page document into a single vector?
   - Because if a document talks about 20 different topics (GPS coordinates, EV chargers,
     refund rules), its single vector becomes a "muddy average" of all 20 topics!
   - By splitting documents into smaller overlapping **Chunks** (`chunk_size=500`,
     `chunk_overlap=80`), each chunk represents ONE clear idea, dramatically boosting
     `Recall@K` and `Precision@K`!

2. **Metadata Tagging & Metadata Filtering (`is_public: True`):**
   Every chunk we store in Pinecone carries a metadata dictionary:
   `{"source": "01_general_info_and_location.md", "is_public": True, "category": "general"}`
   Notice that `03_internal_sensitive_audit.md` is tagged with `"is_public": False`.
   When a public user searches Pinecone, we pass `filter={"is_public": True}` so
   confidential chunks are filtered out at the database level BEFORE they ever reach the LLM!

3. **Cosine Similarity Search:**
   Measures the angle between the user's question vector and each document chunk vector.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import STATIC_DOCS_DIR, Settings, get_embeddings, get_settings


def load_and_chunk_static_documents(
    docs_dir: Path = STATIC_DOCS_DIR,
    chunk_size: int = 550,
    chunk_overlap: int = 80,
) -> List[Document]:
    """
    Read all markdown files in `data/static_docs/`, attach security & source metadata,
    and split them into overlapping semantic chunks for vector indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    all_chunks: List[Document] = []

    for md_file in sorted(docs_dir.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        filename = md_file.name

        # Security classification based on filename / content header
        is_public = "internal_sensitive" not in filename.lower() and (
            "INTERNAL_ONLY" not in raw_text
        )

        if "01_" in filename:
            category = "general_and_location"
        elif "02_" in filename:
            category = "booking_and_policies"
        else:
            category = "internal_audit"

        base_doc = Document(
            page_content=raw_text,
            metadata={
                "source": filename,
                "category": category,
                "is_public": is_public,
            },
        )

        file_chunks = splitter.split_documents([base_doc])
        for idx, chunk in enumerate(file_chunks):
            chunk.metadata["chunk_id"] = f"{filename}#chunk_{idx}"
            all_chunks.append(chunk)

    return all_chunks


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Pure-Python Cosine Similarity formula (for educational clarity & local fallback):
    similarity = (A . B) / (||A|| * ||B||)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ParkingVectorStore:
    """
    Manages document ingestion and semantic search using **Pinecone Serverless**
    (with an automatic in-memory Cosine Similarity fallback when running offline
    unit tests or when `USE_LOCAL_VECTOR_FALLBACK=true`).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings: Optional[Embeddings] = None,
        docs_dir: Path = STATIC_DOCS_DIR,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings
        self.docs_dir = docs_dir

        # Offline / fast unit-test vector memory store
        self._local_chunks: List[Document] = []
        self._local_vectors: List[List[float]] = []
        self._pinecone_vectorstore = None

    @property
    def uses_pinecone_cloud(self) -> bool:
        """Return True if configured to connect to live Pinecone Serverless."""
        return (
            bool(self.settings.pinecone_api_key)
            and not self.settings.use_local_vector_fallback
        )

    def _ensure_embeddings(self) -> Embeddings:
        if self.embeddings is None:
            self.embeddings = get_embeddings(self.settings)
        return self.embeddings

    def ingest_documents(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 80,
    ) -> int:
        """
        Chunk all markdown files in `data/static_docs/` and index them into
        Pinecone Serverless (or the local vector store if running offline).
        Returns the total number of chunks indexed.
        """
        chunks = load_and_chunk_static_documents(
            docs_dir=self.docs_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if self.uses_pinecone_cloud:
            self._ingest_into_pinecone(chunks)
        else:
            self._ingest_into_local_memory(chunks)

        return len(chunks)

    def _ingest_into_pinecone(self, chunks: List[Document]) -> None:
        """
        Connect to Pinecone Serverless, auto-create the index with the exact
        embedding dimension (768 for Gemini, 1536 for OpenAI) if it doesn't exist,
        and upsert all document chunks with deterministic IDs.
        """
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        index_name = self.settings.pinecone_index_name
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )

        emb = self._ensure_embeddings()
        ids = [c.metadata["chunk_id"] for c in chunks]
        self._pinecone_vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=emb,
            index_name=index_name,
            ids=ids,
            pinecone_api_key=self.settings.pinecone_api_key,
        )

    def _ingest_into_local_memory(self, chunks: List[Document]) -> None:
        """Embed chunks into memory using the active Embeddings model."""
        emb = self._ensure_embeddings()
        texts = [c.page_content for c in chunks]
        self._local_chunks = chunks
        self._local_vectors = emb.embed_documents(texts)

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        public_only: bool = True,
    ) -> List[Document]:
        """
        Retrieve the top-K most semantically similar document chunks.
        If `public_only=True` (the default for all user queries), applies a strict
        metadata filter `{"is_public": True}` so internal/confidential documents
        can NEVER be retrieved!
        """
        metadata_filter: Optional[Dict[str, object]] = (
            {"is_public": True} if public_only else None
        )

        if self.uses_pinecone_cloud:
            from langchain_pinecone import PineconeVectorStore

            if self._pinecone_vectorstore is None:
                self._pinecone_vectorstore = PineconeVectorStore(
                    index_name=self.settings.pinecone_index_name,
                    embedding=self._ensure_embeddings(),
                    pinecone_api_key=self.settings.pinecone_api_key,
                )
            return self._pinecone_vectorstore.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter,
            )

        # Local in-memory cosine similarity search with metadata filtering
        if not self._local_chunks:
            self.ingest_documents()

        emb = self._ensure_embeddings()
        query_vec = emb.embed_query(query)

        scored_results: List[tuple[float, Document]] = []
        for chunk, vec in zip(self._local_chunks, self._local_vectors):
            if public_only and not chunk.metadata.get("is_public", False):
                continue
            score = _cosine_similarity(query_vec, vec)
            scored_results.append((score, chunk))

        scored_results.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_results[:k]]
