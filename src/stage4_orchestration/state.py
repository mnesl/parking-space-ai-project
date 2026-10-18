"""
Stage 4: LangGraph State Schema (`state.py`).

Educational Concept:
--------------------
In LangGraph, a workflow is a State Machine where every Node receives a shared
dictionary called the **State**, does its work, and returns updates to that State.
Why do we use `TypedDict` with `Annotated[List[str], operator.add]`?
- `TypedDict` defines the exact keys carried from node to node (`user_message`,
  `bot_reply`, `reservation_draft`, `admin_ticket`, `mcp_record`, `workflow_stage`).
- `Annotated[List[str], operator.add]` is called a **Reducer** in LangGraph:
  instead of overwriting `conversation_log` when a node returns a new message,
  LangGraph automatically *appends* (`+`) the new message to the existing list!
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


WorkflowStage = Literal[
    "USER_INTERACTION",
    "BLOCKED_BY_GUARDRAIL",
    "AWAITING_ADMIN_APPROVAL",
    "ADMIN_CONFIRMED",
    "ADMIN_REFUSED",
    "RECORDED_BY_MCP",
]


class ParkingWorkflowState(TypedDict, total=False):
    """
    Shared State dictionary passed across all 3 LangGraph nodes:
      1. `user_interaction_node` (Stage 1 RAG + Guardrails + Slot Collection)
      2. `admin_approval_node` (Stage 2 Human-in-the-Loop Escalation & Decision)
      3. `data_recording_node` (Stage 3 Secure MCP Server File Writer)
    """

    # Current turn input and output
    user_message: str
    bot_reply: str
    retrieved_sources: List[str]
    blocked_by_guardrail: bool

    # Stage 1: ReservationDraft serialized dict (name, surname, car_number, reservation_period, zone_code)
    reservation_draft: Dict[str, Any]
    ready_for_admin_escalation: bool

    # Stage 2: Human-in-the-Loop Admin Ticket & Decision
    admin_ticket: Optional[Dict[str, Any]]
    admin_decision: Optional[Literal["CONFIRM", "REFUSE"]]
    admin_notes: Optional[str]

    # Stage 3: MCP Server File Recording Result
    mcp_record: Optional[Dict[str, Any]]

    # Overall pipeline status & cumulative audit trail (uses operator.add reducer!)
    workflow_stage: WorkflowStage
    audit_trail: Annotated[List[str], operator.add]

