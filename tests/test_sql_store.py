"""
Unit tests for Stage 1 Dynamic SQL Store (`src/stage1_rag/sql_store.py`).
Satisfies the specification requirement: at least 2 automated tests per module.
"""

from pathlib import Path

from src.stage1_rag.sql_store import DynamicParkingSQLStore


def test_sql_store_seeds_and_returns_pydantic_snapshot(tmp_path: Path) -> None:
    """Verify the SQL store seeds default zones/hours and returns validated Pydantic models."""
    test_db = tmp_path / "test_parking.db"
    store = DynamicParkingSQLStore(db_path=test_db)

    snapshot = store.get_live_snapshot()

    # Check total available spaces (42 in Zone A + 11 in Zone B + 5 in Zone C = 58)
    assert snapshot.total_available_spaces == 58
    assert len(snapshot.zones) == 3
    assert len(snapshot.operating_hours) == 3

    # Verify `.model_dump()` works on our Pydantic model
    zone_a_dict = snapshot.zones[0].model_dump()
    assert zone_a_dict["zone_code"] == "ZONE_A"
    assert zone_a_dict["hourly_rate_usd"] == 4.50


def test_sql_store_reserve_space_decrements_availability(tmp_path: Path) -> None:
    """Verify reserving a space decrements available_spaces in SQLite."""
    test_db = tmp_path / "test_parking.db"
    store = DynamicParkingSQLStore(db_path=test_db)

    before = store.get_live_snapshot().total_available_spaces
    success = store.reserve_space("ZONE_A")
    after = store.get_live_snapshot().total_available_spaces

    assert success is True
    assert after == before - 1

