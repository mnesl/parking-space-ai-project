"""
Stage 3: MCP Client & LangChain Tool Fallback (`mcp_client.py`).

Specification Requirement:
--------------------------
"In case of MCP server is not possible to implement, use tool/function call for
writing data into file. Once the administrator (second agent) approves the
reservation, the server should write the reservation details to a text file."

Educational Concepts Covered Here:
----------------------------------
1. **How an MCP Client Invokes an MCP Server (`ReservationMCPClient`):**
   When Stage 2's `AdminEscalationTicket` transitions to `status == "CONFIRMED"`,
   our client verifies that the ticket is indeed `CONFIRMED` (never `PENDING` or
   `REFUSED`!), decrements available spaces in SQLite, and invokes the MCP Server
   tool `write_confirmed_reservation` with the secret `MCP_SERVER_API_KEY`.

2. **LangChain `@tool` Decorator (`write_reservation_langchain_tool`):**
   We also expose a standard LangChain `@tool` function wrapper around our MCP
   processor so that both MCP protocol callers and LangChain tool-calling agents
   can invoke the exact same secure recording logic!
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from src.config import CONFIRMED_RESERVATIONS_FILE, Settings, get_settings
from src.stage1_rag.sql_store import DynamicParkingSQLStore
from src.stage2_hitl.admin_agent import AdminEscalationTicket
from src.stage3_mcp.mcp_server import (
    ConfirmedReservationPayload,
    MCPRecordResult,
    ReservationMCPFileProcessor,
)


class ReservationMCPClient:
    """
    Client that connects Stage 2 (`AdminEscalationTicket` with `status == 'CONFIRMED'`)
    to our Stage 3 MCP Server (`ReservationMCPFileProcessor` / FastAPI MCP Server).
    """

    def __init__(
        self,
        processor: Optional[ReservationMCPFileProcessor] = None,
        sql_store: Optional[DynamicParkingSQLStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.processor = processor or ReservationMCPFileProcessor(
            settings=self.settings
        )
        self.sql_store = sql_store

    def process_approved_ticket(
        self,
        ticket: AdminEscalationTicket,
        api_key: Optional[str] = None,
    ) -> MCPRecordResult:
        """
        Verify that the Administrator has CONFIRMED the ticket, write the reservation
        to the flat file via the MCP Server in the format:
            `Name | Car Number | Reservation Period | Approval Time`
        and decrement live parking availability in SQLite if `sql_store` is attached.
        """
        if ticket.status != "CONFIRMED":
            raise PermissionError(
                f"Cannot record unapproved reservation '{ticket.request_id}'. "
                f"Current status is '{ticket.status}', expected 'CONFIRMED'."
            )

        approval_time = ticket.decided_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        payload = ConfirmedReservationPayload(
            name=ticket.full_name,
            car_number=ticket.car_number,
            reservation_period=ticket.reservation_period,
            approval_time=approval_time,
        )

        active_key = api_key if api_key is not None else self.settings.mcp_server_api_key
        result = self.processor.record_confirmed_reservation(
            payload=payload,
            api_key=active_key,
        )

        if self.sql_store is not None:
            self.sql_store.reserve_space(ticket.zone_code)

        return result


@tool
def write_confirmed_reservation_tool(
    name: str,
    car_number: str,
    reservation_period: str,
    approval_time: str,
    output_file: str = str(CONFIRMED_RESERVATIONS_FILE),
) -> str:
    """
    LangChain Tool fallback that writes a confirmed parking reservation to disk
    in the exact format: Name | Car Number | Reservation Period | Approval Time.
    """
    settings = get_settings()
    processor = ReservationMCPFileProcessor(
        output_file=Path(output_file),
        settings=settings,
    )
    payload = ConfirmedReservationPayload(
        name=name,
        car_number=car_number,
        reservation_period=reservation_period,
        approval_time=approval_time,
    )
    res = processor.record_confirmed_reservation(
        payload=payload,
        api_key=settings.mcp_server_api_key,
    )
    return res.formatted_entry

