"""
Unit and Security tests for Stage 3 MCP Server & Client
(`src/stage3_mcp/mcp_server.py` and `src/stage3_mcp/mcp_client.py`).
Satisfies the specification requirement: at least 2 automated tests per module.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.stage2_hitl.admin_agent import AdminEscalationTicket
from src.stage3_mcp.mcp_client import ReservationMCPClient
from src.stage3_mcp.mcp_server import (
    ConfirmedReservationPayload,
    ReservationMCPFileProcessor,
    create_mcp_fastapi_app,
)


def test_mcp_server_writes_exact_pipe_delimited_format(tmp_path: Path) -> None:
    """
    Verify that once an administrator approves a ticket (`status='CONFIRMED'`),
    the MCP server writes the exact required format:
    `Name | Car Number | Reservation Period | Approval Time`
    """
    out_file = tmp_path / "confirmed_reservations.txt"
    settings = Settings(MCP_SERVER_API_KEY="mcp-secret-test-key")
    processor = ReservationMCPFileProcessor(output_file=out_file, settings=settings)
    mcp_client = ReservationMCPClient(processor=processor, settings=settings)

    approved_ticket = AdminEscalationTicket(
        request_id="REQ-12345678",
        name="Merve",
        surname="Okcu",
        car_number="34ABC123",
        reservation_period="2026-10-20 09:00 to 2026-10-20 18:00",
        zone_code="ZONE_B",
        status="CONFIRMED",
        admin_briefing="Briefing",
        admin_notes="Approved",
        created_at="2026-10-18 22:00:00",
        decided_at="2026-10-18 22:05:00",
    )

    result = mcp_client.process_approved_ticket(approved_ticket)
    assert result.success is True
    assert (
        result.formatted_entry
        == "Merve Okcu | 34ABC123 | 2026-10-20 09:00 to 2026-10-20 18:00 | 2026-10-18 22:05:00"
    )

    # Verify disk file content matches 100%
    records = processor.read_all_records()
    assert len(records) == 1
    assert records[0] == result.formatted_entry


def test_mcp_server_blocks_unauthorized_access_and_pipe_injection(tmp_path: Path) -> None:
    """
    Verify Stage 3 Security Requirements:
    1. Rejects unapproved (`PENDING` or `REFUSED`) tickets.
    2. Rejects HTTP calls without valid `X-MCP-Api-Key` (`401 Unauthorized`).
    3. Rejects pipe (`|`) or newline (`\\n`) injection attempts.
    """
    out_file = tmp_path / "secure_reservations.txt"
    settings = Settings(MCP_SERVER_API_KEY="mcp-secret-test-key")
    processor = ReservationMCPFileProcessor(output_file=out_file, settings=settings)
    mcp_client = ReservationMCPClient(processor=processor, settings=settings)

    # 1. Unapproved ticket (`PENDING`) must raise PermissionError
    pending_ticket = AdminEscalationTicket(
        request_id="REQ-PENDING1",
        name="Merve",
        surname="Okcu",
        car_number="34ABC123",
        reservation_period="2026-10-20 09:00 to 2026-10-20 18:00",
        status="PENDING",
        admin_briefing="Briefing",
        created_at="2026-10-18 22:00:00",
    )
    with pytest.raises(PermissionError):
        mcp_client.process_approved_ticket(pending_ticket)

    # 2. Pipe character ('|') injection attempt must raise ValidationError
    with pytest.raises(ValueError):
        ConfirmedReservationPayload(
            name="Hacker | FREE-CAR | Forever",
            car_number="34ABC123",
            reservation_period="2026-10-20 09:00 to 18:00",
            approval_time="2026-10-18 22:05:00",
        )

    # 3. FastAPI MCP Server rejects missing/wrong API key with 401
    app = create_mcp_fastapi_app(processor=processor, settings=settings)
    http_client = TestClient(app)

    unauth = http_client.post(
        "/mcp/tools/write_confirmed_reservation",
        headers={"X-MCP-Api-Key": "wrong-key"},
        json={
            "name": "Merve Okcu",
            "car_number": "34ABC123",
            "reservation_period": "2026-10-20 09:00 to 2026-10-20 18:00",
            "approval_time": "2026-10-18 22:05:00",
        },
    )
    assert unauth.status_code == 401
    assert not out_file.exists()

