"""
Stage 4: Unified LangGraph Orchestration Pipeline (`graph.py`).

Specification Requirements Covered:
-----------------------------------
1. "Implement orchestration of all components using LangGraph."
2. "Ensure complete integration of all stages:
   - The chatbot (RAG agent) interacts with users.
   - The system escalates reservation requests to the administrator via a
     human-in-the-loop agent (second agent).
   - The MCP server processes data after confirmation."
3. "Implement the workflow logic for the entire pipeline:
   Example graph structure:
   - Node for user interaction (context of RAG and chatbot).
   - Node for administrator approval.
   - Node for data recording."

Educational Concepts Covered Here:
----------------------------------
1. **Nodes, Edges, and Conditional Routing (`add_conditional_edges`):**
   - **Node 1 (`user_interaction_node`):** Runs Stage 1 (`ParkingRAGChatbot`).
     After Node 1 finishes, a **Conditional Edge (`route_after_user_interaction`)**
     inspects the state:
     - If `ready_for_admin_escalation == False` (the user was just asking a question
       about prices/hours, or hasn't finished giving all 4 booking fields yet) ->
       Routes to `END` (waiting for the user's next chat message!).
     - If `ready_for_admin_escalation == True` (all 4 fields collected!) ->
       Routes automatically to **Node 2 (`admin_approval_node`)**!

2. **Node 2 (`admin_approval_node`) & Checkpointer Interrupts (`MemorySaver`):**
   - Escalates the reservation via Stage 2 (`AdminEscalationAgent`).
   - If `admin_decision` is provided (`"CONFIRM"` or `"REFUSE"`), processes the
     decision and routes via **`route_after_admin_approval`**:
     - If `"CONFIRM"` -> Routes to **Node 3 (`data_recording_node`)**!
     - If `"REFUSE"` or still `"PENDING"` -> Routes to `END` and notifies the user.

3. **Node 3 (`data_recording_node`):**
   - Invokes Stage 3 (`ReservationMCPClient.process_approved_ticket()`), which writes
     `Name | Car Number | Reservation Period | Approval Time` to `confirmed_reservations.txt`
     and decrements available spaces in SQLite!
"""

from __future__ import annotations

from typing import Literal, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.stage1_rag.chatbot_agent import ParkingRAGChatbot, ReservationDraft
from src.stage2_hitl.admin_agent import (
    AdminDecisionInput,
    AdminEscalationAgent,
    AdminEscalationTicket,
)
from src.stage3_mcp.mcp_client import ReservationMCPClient
from src.stage4_orchestration.state import ParkingWorkflowState


class ParkingReservationOrchestrator:
    """
    Wraps Stage 1 (`ParkingRAGChatbot`), Stage 2 (`AdminEscalationAgent`), and
    Stage 3 (`ReservationMCPClient`) inside a compiled **LangGraph `StateGraph`**
    with persistent thread checkpointing (`MemorySaver`).
    """

    def __init__(
        self,
        chatbot: Optional[ParkingRAGChatbot] = None,
        admin_agent: Optional[AdminEscalationAgent] = None,
        mcp_client: Optional[ReservationMCPClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chatbot = chatbot or ParkingRAGChatbot(settings=self.settings)
        self.admin_agent = admin_agent or AdminEscalationAgent(
            sql_store=self.chatbot.sql_store,
            settings=self.settings,
        )
        self.mcp_client = mcp_client or ReservationMCPClient(
            sql_store=self.chatbot.sql_store,
            settings=self.settings,
        )
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # ========================================================================
    # NODE 1: User Interaction (RAG + Guardrails + Interactive Slot Filling)
    # ========================================================================
    def user_interaction_node(
        self, state: ParkingWorkflowState
    ) -> ParkingWorkflowState:
        """
        Node 1: Execute Stage 1 RAG & Reservation Slot Collection.
        """
        # Sync any existing reservation_draft from graph state into chatbot
        if state.get("reservation_draft"):
            self.chatbot.reservation_draft = ReservationDraft.model_validate(
                state["reservation_draft"]
            )

        user_msg = state.get("user_message", "")
        turn = self.chatbot.chat(user_msg)

        if turn.blocked_by_guardrail:
            return {
                "bot_reply": turn.reply,
                "retrieved_sources": [],
                "blocked_by_guardrail": True,
                "reservation_draft": turn.reservation_draft.model_dump(),
                "ready_for_admin_escalation": False,
                "workflow_stage": "BLOCKED_BY_GUARDRAIL",
                "audit_trail": ["[Node 1: user_interaction] Blocked by Guardrail"],
            }

        return {
            "bot_reply": turn.reply,
            "retrieved_sources": turn.retrieved_sources,
            "blocked_by_guardrail": False,
            "reservation_draft": turn.reservation_draft.model_dump(),
            "ready_for_admin_escalation": turn.ready_for_admin_escalation,
            "workflow_stage": "USER_INTERACTION",
            "audit_trail": [
                f"[Node 1: user_interaction] Processed query (ready_for_admin={turn.ready_for_admin_escalation})"
            ],
        }

    # ========================================================================
    # NODE 2: Administrator Approval (Stage 2 Human-in-the-Loop Agent)
    # ========================================================================
    def admin_approval_node(
        self, state: ParkingWorkflowState
    ) -> ParkingWorkflowState:
        """
        Node 2: Escalate completed reservation to Agent 2 (Administrator Agent)
        and process the administrator's CONFIRM or REFUSE decision when available.
        """
        draft = ReservationDraft.model_validate(state["reservation_draft"])

        # Create PENDING ticket if not escalated yet in this workflow run
        existing_ticket_dict = state.get("admin_ticket")
        if existing_ticket_dict is None:
            ticket = self.admin_agent.escalate_reservation(draft)
        else:
            ticket = AdminEscalationTicket.model_validate(existing_ticket_dict)

        decision = state.get("admin_decision")
        if decision in ("CONFIRM", "REFUSE"):
            ticket = self.admin_agent.process_admin_decision(
                request_id=ticket.request_id,
                decision_input=AdminDecisionInput(
                    decision=decision,
                    admin_notes=state.get("admin_notes"),
                ),
            )
            user_msg = self.admin_agent.format_decision_for_user(ticket)
            stage = "ADMIN_CONFIRMED" if ticket.status == "CONFIRMED" else "ADMIN_REFUSED"
            return {
                "admin_ticket": ticket.model_dump(),
                "bot_reply": user_msg,
                "workflow_stage": stage,
                "audit_trail": [
                    f"[Node 2: admin_approval] Ticket {ticket.request_id} -> {ticket.status}"
                ],
            }

        # No decision provided yet -> Remains in AWAITING_ADMIN_APPROVAL (HITL pause!)
        pending_msg = (
            f"{state.get('bot_reply', '')}\n\n"
            f"📋 **Admin Ticket Created:** `{ticket.request_id}` (Status: `PENDING`). "
            f"Awaiting Human Administrator CONFIRM / REFUSE decision."
        ).strip()
        return {
            "admin_ticket": ticket.model_dump(),
            "bot_reply": pending_msg,
            "workflow_stage": "AWAITING_ADMIN_APPROVAL",
            "audit_trail": [
                f"[Node 2: admin_approval] Escalated Ticket {ticket.request_id} (PENDING)"
            ],
        }

    # ========================================================================
    # NODE 3: Data Recording via Secure MCP Server (Stage 3)
    # ========================================================================
    def data_recording_node(
        self, state: ParkingWorkflowState
    ) -> ParkingWorkflowState:
        """
        Node 3: Invoke Stage 3 MCP Server to write confirmed reservation to file:
        `Name | Car Number | Reservation Period | Approval Time`
        """
        ticket = AdminEscalationTicket.model_validate(state["admin_ticket"])
        mcp_result = self.mcp_client.process_approved_ticket(ticket)

        final_reply = (
            f"{state.get('bot_reply', '')}\n\n"
            f"✅ **Recorded by MCP Server:** `{mcp_result.formatted_entry}`"
        ).strip()

        return {
            "mcp_record": mcp_result.model_dump(),
            "bot_reply": final_reply,
            "workflow_stage": "RECORDED_BY_MCP",
            "audit_trail": [
                f"[Node 3: data_recording] MCP wrote '{mcp_result.formatted_entry}'"
            ],
        }

    # ========================================================================
    # CONDITIONAL EDGES (ROUTERS)
    # ========================================================================
    @staticmethod
    def route_after_user_interaction(
        state: ParkingWorkflowState,
    ) -> Literal["admin_approval", "__end__"]:
        """
        Route to `admin_approval` ONLY when all 4 reservation fields are collected
        and the request was not blocked by guardrails.
        """
        if state.get("ready_for_admin_escalation") and not state.get(
            "blocked_by_guardrail"
        ):
            return "admin_approval"
        return "__end__"

    @staticmethod
    def route_after_admin_approval(
        state: ParkingWorkflowState,
    ) -> Literal["data_recording", "__end__"]:
        """
        Route to `data_recording` (MCP Server) ONLY when the Human Administrator
        has CONFIRMED the reservation.
        """
        if state.get("workflow_stage") == "ADMIN_CONFIRMED":
            return "data_recording"
        return "__end__"

    def _build_graph(self):
        """Construct and compile the 3-node LangGraph StateGraph."""
        builder = StateGraph(ParkingWorkflowState)

        # Add the 3 specification nodes
        builder.add_node("user_interaction", self.user_interaction_node)
        builder.add_node("admin_approval", self.admin_approval_node)
        builder.add_node("data_recording", self.data_recording_node)

        # Entry edge
        builder.add_edge(START, "user_interaction")

        # Conditional edge 1: User Interaction -> Admin Approval OR End
        builder.add_conditional_edges(
            "user_interaction",
            self.route_after_user_interaction,
            {
                "admin_approval": "admin_approval",
                "__end__": END,
            },
        )

        # Conditional edge 2: Admin Approval -> Data Recording (MCP) OR End
        builder.add_conditional_edges(
            "admin_approval",
            self.route_after_admin_approval,
            {
                "data_recording": "data_recording",
                "__end__": END,
            },
        )

        # Final edge: Data Recording -> End
        builder.add_edge("data_recording", END)

        return builder.compile(checkpointer=self.checkpointer)

    # ========================================================================
    # CONVENIENCE EXECUTION METHODS FOR USERS & ADMINISTRATORS
    # ========================================================================
    def send_user_message(
        self,
        user_message: str,
        thread_id: str = "default-session",
        admin_decision: Optional[Literal["CONFIRM", "REFUSE"]] = None,
        admin_notes: Optional[str] = None,
    ) -> ParkingWorkflowState:
        """
        Send a user message into the LangGraph pipeline for a given conversation `thread_id`.
        """
        config = {"configurable": {"thread_id": thread_id}}
        inputs: ParkingWorkflowState = {
            "user_message": user_message,
            "admin_decision": admin_decision,
            "admin_notes": admin_notes,
        }
        return self.graph.invoke(inputs, config=config)

    def submit_admin_decision_for_thread(
        self,
        thread_id: str,
        decision: Literal["CONFIRM", "REFUSE"],
        admin_notes: Optional[str] = None,
    ) -> ParkingWorkflowState:
        """
        Resume a paused `AWAITING_ADMIN_APPROVAL` thread once the Human Administrator
        submits `CONFIRM` or `REFUSE`, triggering `admin_approval` -> `data_recording`!
        """
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        current_values: ParkingWorkflowState = dict(snapshot.values)  # type: ignore

        if not current_values.get("admin_ticket"):
            raise ValueError(f"No pending admin ticket found for thread '{thread_id}'.")

        updated_state = self.admin_approval_node(
            {
                **current_values,
                "admin_decision": decision,
                "admin_notes": admin_notes,
            }
        )
        merged: ParkingWorkflowState = {**current_values, **updated_state}

        if merged.get("workflow_stage") == "ADMIN_CONFIRMED":
            mcp_state = self.data_recording_node(merged)
            merged = {
                **merged,
                **mcp_state,
                "audit_trail": list(merged.get("audit_trail", []))
                + list(mcp_state.get("audit_trail", [])),
            }

        self.graph.update_state(config, merged)
        return merged

