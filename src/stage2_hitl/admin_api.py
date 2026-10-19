"""
Stage 2: FastAPI REST API for Human-in-the-Loop Administrator Approval (`admin_api.py`).

Specification Requirement:
--------------------------
"Chat bot should be able to send a reservation request to administrator and get
confirm/refuse response from them (e.g. via email server, messenger, rest api)."

Educational Concepts Covered Here:
----------------------------------
1. **Why FastAPI + Pydantic Is the Industry Standard for AI Microservices:**
   Notice how FastAPI uses our exact Pydantic models (`ReservationDraft`,
   `AdminDecisionInput`, `AdminEscalationTicket`) as type annotations on the
   endpoint functions! FastAPI automatically:
   - Validates the incoming JSON body,
   - Rejects malformed requests with HTTP 422,
   - Serializes the returned Pydantic model to JSON, and
   - Builds an interactive Swagger UI dashboard at `http://127.0.0.1:8000/docs`!

2. **API Key Security (`X-Admin-Api-Key` Header):**
   Only an authorized Human Administrator with the valid `ADMIN_API_KEY` can approve
   or refuse reservations (`POST /admin/reservations/{request_id}/decision`).
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status

from src.config import Settings, get_settings
from src.stage1_rag.chatbot_agent import ReservationDraft
from src.stage2_hitl.admin_agent import (
    AdminDecisionInput,
    AdminEscalationAgent,
    AdminEscalationTicket,
)


def create_admin_app(
    admin_agent: Optional[AdminEscalationAgent] = None,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """
    Factory that builds the Stage 2 FastAPI application for Human-in-the-Loop
    Administrator escalation and Confirm/Refuse responses.
    """
    cfg = settings or get_settings()
    agent = admin_agent or AdminEscalationAgent(settings=cfg)

    app = FastAPI(
        title="AeroPark Stage 2 - Human-in-the-Loop Administrator API",
        description=(
            "REST API allowing Agent 1 (Chatbot) to escalate reservation requests "
            "and allowing the Human Administrator to CONFIRM or REFUSE them."
        ),
        version="1.0.0",
    )

    def verify_admin_api_key(
        x_admin_api_key: str = Header(default="", alias="X-Admin-Api-Key"),
    ) -> None:
        """Security dependency verifying the Administrator's API key."""
        if x_admin_api_key != cfg.admin_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-Admin-Api-Key header.",
            )

    @app.post(
        "/admin/reservations/escalate",
        response_model=AdminEscalationTicket,
        status_code=status.HTTP_201_CREATED,
        summary="Escalate a completed user reservation from Agent 1 to Agent 2",
    )
    def escalate_reservation_endpoint(draft: ReservationDraft) -> AdminEscalationTicket:
        try:
            return agent.escalate_reservation(draft)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err

    @app.get(
        "/admin/reservations/pending",
        response_model=list[AdminEscalationTicket],
        dependencies=[Depends(verify_admin_api_key)],
        summary="List all PENDING reservation requests awaiting Admin decision",
    )
    def list_pending_reservations_endpoint() -> list[AdminEscalationTicket]:
        return agent.list_pending_tickets()

    @app.get(
        "/admin/reservations/{request_id}",
        response_model=AdminEscalationTicket,
        summary="Check the status of a specific reservation request",
    )
    def get_reservation_ticket_endpoint(request_id: str) -> AdminEscalationTicket:
        ticket = agent.get_ticket(request_id)
        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reservation ticket '{request_id}' not found.",
            )
        return ticket

    @app.post(
        "/admin/reservations/{request_id}/decision",
        response_model=AdminEscalationTicket,
        dependencies=[Depends(verify_admin_api_key)],
        summary="Submit Administrator CONFIRM or REFUSE decision for a reservation",
    )
    def submit_admin_decision_endpoint(
        request_id: str,
        decision: AdminDecisionInput,
    ) -> AdminEscalationTicket:
        try:
            return agent.process_admin_decision(request_id, decision)
        except KeyError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(err),
            ) from err

    return app


app = create_admin_app()

