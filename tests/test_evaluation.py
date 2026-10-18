"""
Unit tests for Stage 1 RAG Evaluation (`src/stage1_rag/evaluation.py`).
Satisfies the specification requirement: at least 2 automated tests per module.
"""

from pathlib import Path

from src.config import Settings
from src.stage1_rag.chatbot_agent import ParkingRAGChatbot
from src.stage1_rag.evaluation import (
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_rag_system,
)
from src.stage1_rag.sql_store import DynamicParkingSQLStore
from src.stage1_rag.vector_store import ParkingVectorStore
from tests.test_vector_store import KeywordTestEmbeddings


def test_recall_and_precision_at_k_math_formulas() -> None:
    """Verify exact mathematical calculation of Recall@K and Precision@K."""
    retrieved = [
        "01_general_info_and_location.md#chunk_0",
        "01_general_info_and_location.md#chunk_1",
        "02_booking_process_and_policies.md#chunk_0",
    ]
    expected = {"01_general_info_and_location.md"}

    # Recall@3: Found the 1 expected source inside Top-3 -> 1 / 1 = 1.0 (100%)
    assert compute_recall_at_k(retrieved, expected, k=3) == 1.0

    # Precision@3: Out of 3 retrieved chunks, 2 belong to the expected source -> 2 / 3 = 0.6667
    assert compute_precision_at_k(retrieved, expected, k=3) == 0.6667


def test_full_stage1_rag_evaluation_generates_report(tmp_path: Path) -> None:
    """Verify end-to-end RAG evaluation achieves high Recall@K, Accuracy, and 100% Guardrail block rate."""
    settings = Settings(USE_LOCAL_VECTOR_FALLBACK=True)
    vec_store = ParkingVectorStore(
        settings=settings,
        embeddings=KeywordTestEmbeddings(),
    )
    vec_store.ingest_documents()
    sql_store = DynamicParkingSQLStore(db_path=tmp_path / "eval_test.db")
    bot = ParkingRAGChatbot(
        vector_store=vec_store,
        sql_store=sql_store,
        settings=settings,
    )

    report_path = tmp_path / "STAGE_1_EVALUATION_REPORT.md"
    report = evaluate_rag_system(bot, k=3, save_report_path=report_path)

    assert report.mean_recall_at_k >= 0.80
    assert report.mean_precision_at_k >= 0.60
    assert report.mean_answer_accuracy >= 0.85
    assert report.guardrail_block_rate == 1.0
    assert report_path.exists()

