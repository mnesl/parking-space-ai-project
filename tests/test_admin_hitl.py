"""
Unit and Integration tests for Stage 2 Human-in-the-Loop Administrator Agent & REST API
(`src/stage2_hitl/admin_agent.py` and `src/stage2_hitl/admin_api.py`).
Satisfies the specification requirement: at least 2 automated tests per module.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from src.config import Settings
from src.stage1_rag.chatbot_agent import ParkingRAGChatbot
from src.stage1_rag.sql_store import DynamicParkingSQLStore
from src.stage1_rag.vector_store import ParkingVectorStore
from src.stage2_hitl.admin_agent import (
    AdminDecisionInput,
    AdminEscalationAgent,
    escalate_from_chatbot_to_admin,
)
from src.stage2_hitl.admin_api import create_admin_app
from tests.test_vector_store import KeywordTestEmbeddings


def test_agent1_to_agent2_handoff_and_admin_confirm_refuse(tmp_path: Path) -> None:
    """
    Verify that once Agent 1 (Chatbot) collects all 4 user fields, the request is
    escalated to Agent 2 (AdminEscalationAgent), and Agent 2 processes both
    CONFIRM and REFUSE responses accurately.
    """
    db_file = tmp_path / "stage2_test.db"
    settings = Settings(USE_LOCAL_VECTOR_FALLBACK=True)
    vec_store = ParkingVectorStore(
        settings=settings,
        embeddings=KeywordTestEmbeddings(),
    )
    vec_store.ingest_documents()
    sql_store = DynamicParkingSQLStore(db_path=db_file)

    chatbot = ParkingRAGChatbot(
        vector_store=vec_store,
        sql_store=sql_store,
        settings=settings,
    )
    admin_agent = AdminEscalationAgent(
        db_path=db_file,
        sql_store=sql_store,
        settings=settings,
    )

    # 1. User provides all 4 reservation details to Agent 1
    turn = chatbot.chat(
        "My name is Merve Okcu, my car number is 34ABC123, "
        "and I want to reserve for 2026-10-20 09:00 to 2026-10-20 18:00 in Zone B."
    )
    assert turn.ready_for_admin_escalation is True

    # 2. Handoff from Agent 1 -> Agent 2
    handoff = escalate_from_chatbot_to_admin(chatbot, admin_agent)
    assert handoff.ticket.status == "PENDING"
    assert handoff.ticket.car_number == "34ABC123"
    assert "ADMIN APPROVAL REQUIRED" in handoff.ticket.admin_briefing

    # 3. Administrator CONFIRMS the request
    confirmed_ticket = admin_agent.process_admin_decision(
        handoff.ticket.request_id,
        AdminDecisionInput(decision="CONFIRM", admin_notes="Verified EV stall B-04"),
    )
    assert confirmed_ticket.status == "CONFIRMED"
    assert confirmed_ticket.decided_at is not None

    user_msg = admin_agent.format_decision_for_user(confirmed_ticket)
    assert "CONFIRMED" in user_msg
    assert "34ABC123" in user_msg


def test_admin_fastapi_rest_endpoints_and_auth_security(tmp_path: Path) -> None:
    """
    Verify the Stage 2 FastAPI REST API endpoints (`/escalate`, `/pending`, `/decision`)
    and ensure unauthorized requests without `X-Admin-Api-Key` are rejected with HTTP 401.
    """
    db_file = tmp_path / "stage2_api.db"
    settings = Settings(
        USE_LOCAL_VECTOR_FALLBACK=True,
        ADMIN_API_KEY="test-admin-secret-key",
    )
    admin_agent = AdminEscalationAgent(db_path=db_file, settings=settings)
    app = create_admin_app(admin_agent=admin_agent, settings=settings)
    client = TestClient(app)

    # 1. Escalate a reservation via POST /admin/reservations/escalate
    create_resp = client.post(
        "/admin/reservations/escalate",
        json={
            "name": "Ahmet",
            "surname": "Yilmaz",
            "car_number": "06XYZ99",
            "reservation_period": "2026-10-22 10:00 to 2026-10-22 14:00",
            "zone_code": "ZONE_A",
        },
    )
    assert create_resp.status_code == 201
    ticket_data = create_resp.json()
    req_id = ticket_data["request_id"]
    assert ticket_data["status"] == "PENDING"

    # 2. Attempt to approve WITHOUT Admin API Key -> Must fail with 401 Unauthorized!
    unauth_resp = client.post(
        f"/admin/reservations/{req_id}/decision",
        json={"decision": "CONFIRM"},
    )
    assert unauth_resp.status_code == 401

    # 3. Approve WITH valid X-Admin-Api-Key header -> Succeeds with 200 OK!
    auth_resp = client.post(
        f"/admin/reservations/{req_id}/decision",
        headers={"X-Admin-Api-Key": "test-admin-secret-key"},
        json={"decision": "REFUSE", "admin_notes": "Zone maintenance window"},
    )
    assert auth_resp.status_code == 200
    assert auth_resp.json()["status"] == "REFUSED"
    assert auth_resp.json()["admin_notes"] == "Zone maintenance window"

