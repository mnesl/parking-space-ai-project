"""
Stage 3: Secure MCP Server & FastAPI Service for Confirmed Reservations (`mcp_server.py`).

Specification Requirements Covered:
-----------------------------------
1. "Use any open-source MCP server that provides functionality to write data to file.
   Alternatively, develop a simple MCP server using Python + FastAPI to process
   confirmed reservations."
2. "Once the administrator (second agent) approves the reservation, the server should
   write the reservation details to a text file."
3. "File entry format: Name | Car Number | Reservation Period | Approval Time."
4. "Ensure the server is secure and resistant to unauthorized access, while ensuring
   reliable service."

Educational Concepts Covered Here:
----------------------------------
1. **What is the Model Context Protocol (MCP)?**
   MCP (created by Anthropic and adopted across the AI industry) is a universal standard
   (like a "USB-C port for AI applications") that defines how AI Agents (MCP Clients)
   discover and execute external **Tools**, read **Resources**, and use **Prompts**
   hosted on an **MCP Server**.

2. **Exact Flat-File Ledger Format (`Name | Car Number | Reservation Period | Approval Time`):**
   Every approved reservation is appended as a single line to `data/confirmed_reservations.txt`:
   `Merve Okcu | 34ABC123 | 2026-10-20 09:00 to 2026-10-20 18:00 | 2026-10-18 22:30:00`

3. **3 Security & Reliability Defenses Built Into This Server:**
   - **Defense #1 (Token Authentication):** Every tool call / HTTP request requires a
     valid `MCP_SERVER_API_KEY` (`X-MCP-Api-Key` header or tool token).
   - **Defense #2 (Delimiter & Newline Injection Prevention):** Using Pydantic's
     `@field_validator`, any attempt to inject `"|"` or `"\n"` into `name`, `car_number`,
     or `reservation_period` is immediately rejected so an attacker can never forge fake
     columns or extra lines in `confirmed_reservations.txt`!
   - **Defense #3 (Thread-Safe Atomic File Locking):** Uses a `threading.Lock()` +
     explicit `flush()` so concurrent approvals never corrupt the file!
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.config import CONFIRMED_RESERVATIONS_FILE, Settings, get_settings

# Global lock ensuring thread-safe atomic writes to confirmed_reservations.txt
_FILE_WRITE_LOCK = threading.Lock()


# ============================================================================
# 1. PYDANTIC MODELS FOR CONFIRMED RESERVATION RECORDING
# ============================================================================
class ConfirmedReservationPayload(BaseModel):
    """
    Validated payload sent to the MCP Server after the Stage 2 Administrator
    CONFIRMS a reservation.
    """

    name: str = Field(
        min_length=2,
        description="Customer's full name (e.g., 'Merve Okcu')",
    )
    car_number: str = Field(
        min_length=4,
        max_length=15,
        description="Confirmed vehicle license plate (e.g., '34ABC123')",
    )
    reservation_period: str = Field(
        min_length=5,
        description="Approved reservation window (e.g., '2026-10-20 09:00 to 2026-10-20 18:00')",
    )
    approval_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        description="Timestamp when the administrator approved the reservation",
    )

    @field_validator("name", "car_number", "reservation_period", "approval_time")
    @classmethod
    def prevent_delimiter_and_newline_injection(cls, value: str) -> str:
        """
        Security Validator: Block pipe ('|') and newline ('\\n', '\\r') characters
        so nobody can corrupt the pipe-delimited file format!
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace only.")
        if "|" in cleaned or "\n" in cleaned or "\r" in cleaned:
            raise ValueError(
                "Security Violation: Pipe ('|') and newline characters are forbidden "
                "in reservation fields to prevent file injection."
            )
        return cleaned

    def to_file_line(self) -> str:
        """
        Format the record in the EXACT specification format:
        `Name | Car Number | Reservation Period | Approval Time`
        """
        return (
            f"{self.name} | {self.car_number} | "
            f"{self.reservation_period} | {self.approval_time}"
        )


class MCPRecordResult(BaseModel):
    """Structured response returned by the MCP Server after writing to disk."""

    success: bool
    formatted_entry: str = Field(
        description="The exact line written: 'Name | Car Number | Reservation Period | Approval Time'"
    )
    file_path: str
    total_records_in_file: int


# ============================================================================
# 2. CORE SECURE RESERVATION FILE WRITER ENGINE
# ============================================================================
class ReservationMCPFileProcessor:
    """
    Core engine behind both the FastMCP Tool Server and the FastAPI MCP HTTP endpoint.
    Handles authentication, Pydantic validation, and thread-safe file persistence.
    """

    def __init__(
        self,
        output_file: Path | str = CONFIRMED_RESERVATIONS_FILE,
        settings: Optional[Settings] = None,
    ) -> None:
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings or get_settings()

    def verify_token(self, api_key: str) -> None:
        """Verify that the caller possesses the valid `MCP_SERVER_API_KEY`."""
        if not api_key or api_key != self.settings.mcp_server_api_key:
            raise PermissionError(
                "Unauthorized MCP access: Invalid or missing MCP_SERVER_API_KEY."
            )

    def record_confirmed_reservation(
        self,
        payload: ConfirmedReservationPayload,
        api_key: str,
    ) -> MCPRecordResult:
        """
        Authenticate caller, format entry as `Name | Car Number | Reservation Period | Approval Time`,
        and atomically append to `confirmed_reservations.txt`.
        """
        self.verify_token(api_key)
        line = payload.to_file_line()

        with _FILE_WRITE_LOCK:
            with self.output_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

            all_lines = self.read_all_records()

        return MCPRecordResult(
            success=True,
            formatted_entry=line,
            file_path=str(self.output_file),
            total_records_in_file=len(all_lines),
        )

    def read_all_records(self) -> List[str]:
        """Read all recorded reservation lines from the output file."""
        if not self.output_file.exists():
            return []
        content = self.output_file.read_text(encoding="utf-8")
        return [ln.strip() for ln in content.splitlines() if ln.strip()]


# ============================================================================
# 3. OFFICIAL MCP SERVER DEFINITION (`mcp` SDK / FastMCP)
# ============================================================================
def create_fastmcp_server(
    processor: Optional[ReservationMCPFileProcessor] = None,
):
    """
    Create a standard Model Context Protocol (MCP) Server using the official `mcp` SDK.
    Exposes:
      - Tool: `write_confirmed_reservation`
      - Resource: `reservations://confirmed_list`
    """
    from mcp.server.fastmcp import FastMCP

    proc = processor or ReservationMCPFileProcessor()
    mcp_server = FastMCP("AeroPark-Reservation-MCP-Server")

    @mcp_server.tool(
        name="write_confirmed_reservation",
        description=(
            "Write an administrator-approved parking reservation to the official ledger file "
            "in the format: Name | Car Number | Reservation Period | Approval Time"
        ),
    )
    def write_confirmed_reservation_tool(
        name: str,
        car_number: str,
        reservation_period: str,
        approval_time: str,
        api_key: str,
    ) -> dict:
        payload = ConfirmedReservationPayload(
            name=name,
            car_number=car_number,
            reservation_period=reservation_period,
            approval_time=approval_time,
        )
        result = proc.record_confirmed_reservation(payload=payload, api_key=api_key)
        return result.model_dump()

    @mcp_server.resource("reservations://confirmed_list")
    def get_confirmed_reservations_resource() -> str:
        records = proc.read_all_records()
        return "\n".join(records) if records else "No confirmed reservations yet."

    return mcp_server


# ============================================================================
# 4. FASTAPI MCP HTTP TRANSPORT SERVER
# ============================================================================
def create_mcp_fastapi_app(
    processor: Optional[ReservationMCPFileProcessor] = None,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """
    Create the Python + FastAPI MCP Server specified in Stage 3:
    "Alternatively, develop a simple MCP server using Python + FastAPI to process
    confirmed reservations."
    """
    cfg = settings or get_settings()
    proc = processor or ReservationMCPFileProcessor(settings=cfg)

    app = FastAPI(
        title="AeroPark Stage 3 — Secure MCP Reservation Server",
        description=(
            "Processes confirmed reservations from Stage 2 and writes them to storage "
            "in the format: Name | Car Number | Reservation Period | Approval Time."
        ),
        version="1.0.0",
    )

    def verify_mcp_header(
        x_mcp_api_key: str = Header(default="", alias="X-MCP-Api-Key"),
    ) -> str:
        try:
            proc.verify_token(x_mcp_api_key)
            return x_mcp_api_key
        except PermissionError as err:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(err),
            ) from err

    @app.get(
        "/mcp/tools",
        summary="MCP Tool Discovery endpoint listing available server tools",
    )
    def list_mcp_tools() -> dict:
        return {
            "server": "AeroPark-Reservation-MCP-Server",
            "protocol": "Model Context Protocol (MCP) 1.0",
            "tools": [
                {
                    "name": "write_confirmed_reservation",
                    "description": (
                        "Appends an approved reservation to confirmed_reservations.txt "
                        "as: Name | Car Number | Reservation Period | Approval Time"
                    ),
                    "input_schema": ConfirmedReservationPayload.model_json_schema(),
                }
            ],
        }

    @app.post(
        "/mcp/tools/write_confirmed_reservation",
        response_model=MCPRecordResult,
        status_code=status.HTTP_201_CREATED,
        summary="Invoke MCP tool to write a confirmed reservation to file",
    )
    def call_write_reservation_tool(
        payload: ConfirmedReservationPayload,
        api_key: str = Depends(verify_mcp_header),
    ) -> MCPRecordResult:
        try:
            return proc.record_confirmed_reservation(payload=payload, api_key=api_key)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err

    @app.get(
        "/mcp/resources/confirmed_reservations",
        dependencies=[Depends(verify_mcp_header)],
        summary="Read all confirmed reservation lines from storage",
    )
    def read_confirmed_reservations() -> dict:
        records = proc.read_all_records()
        return {"count": len(records), "records": records}

    return app


app = create_mcp_fastapi_app()

