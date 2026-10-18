"""
Stage 4: Unified Interactive UI & CLI (`src/stage4_orchestration/app_ui.py`).

Provides two ways to demo the full LangGraph pipeline (and capture screenshots
for your PowerPoint presentation!):
  1. **Streamlit Web Dashboard (`streamlit run src/stage4_orchestration/app_ui.py`):**
     - Clean, normal bottom-anchored chat interface (messages scroll above, chat input pinned at the bottom!)
     - Compact, readable typography (Markdown `#` headers in retrieved docs scaled down cleanly)
     - Sidebar / Mode selector for Customer Chat, Human-in-the-Loop Admin Portal, and MCP Server Ledger
  2. **Interactive Terminal CLI (`python -m src.stage4_orchestration.app_ui --cli`)**
"""

from __future__ import annotations

import re
import sys

from src.config import Settings, get_settings
from src.stage1_rag.chatbot_agent import ParkingRAGChatbot
from src.stage1_rag.sql_store import DynamicParkingSQLStore
from src.stage1_rag.vector_store import ParkingVectorStore
from src.stage2_hitl.admin_agent import AdminEscalationAgent
from src.stage3_mcp.mcp_client import ReservationMCPClient
from src.stage3_mcp.mcp_server import ReservationMCPFileProcessor
from src.stage4_orchestration.graph import ParkingReservationOrchestrator
from tests.test_vector_store import KeywordTestEmbeddings


def build_default_orchestrator() -> ParkingReservationOrchestrator:
    """Build a ready-to-run orchestrator (uses live Pinecone/LLM if keys are set, or local fallback)."""
    settings = get_settings()
    has_llm_key = (
        settings.llm_provider == "gemini" and bool(settings.google_api_key)
    ) or (settings.llm_provider == "openai" and bool(settings.openai_api_key))

    if not has_llm_key or not settings.pinecone_api_key:
        settings = Settings(USE_LOCAL_VECTOR_FALLBACK=True)
        vec_store = ParkingVectorStore(
            settings=settings,
            embeddings=KeywordTestEmbeddings(),
        )
    else:
        vec_store = ParkingVectorStore(settings=settings)

    vec_store.ingest_documents()
    sql_store = DynamicParkingSQLStore()
    chatbot = ParkingRAGChatbot(
        vector_store=vec_store,
        sql_store=sql_store,
        settings=settings,
    )
    admin_agent = AdminEscalationAgent(sql_store=sql_store, settings=settings)
    mcp_processor = ReservationMCPFileProcessor(settings=settings)
    mcp_client = ReservationMCPClient(
        processor=mcp_processor,
        sql_store=sql_store,
        settings=settings,
    )
    return ParkingReservationOrchestrator(
        chatbot=chatbot,
        admin_agent=admin_agent,
        mcp_client=mcp_client,
        settings=settings,
    )


def _format_clean_chat_text(raw_text: str) -> str:
    """
    Convert giant markdown `#` and `##` headings from retrieved knowledge-base chunks
    into clean bold labels so chat bubbles look like normal, sleek chat messages!
    """
    cleaned = re.sub(r"^#{1,3}\s+(.+)$", r"**\1**", raw_text, flags=re.MULTILINE)
    return cleaned


def run_cli_demo() -> None:
    """Run an interactive terminal demonstration of all 4 stages."""
    orch = build_default_orchestrator()
    thread_id = "cli-demo-session"
    print("=" * 72)
    print("🚗 AeroPark Smart Garage — Stage 4 LangGraph Interactive CLI")
    print("Type 'exit' to quit, or 'admin' to review pending reservations.")
    print("=" * 72)

    while True:
        user_input = input("\n👤 Customer > ").strip()
        if not user_input or user_input.lower() == "exit":
            break

        if user_input.lower() == "admin":
            pending = orch.admin_agent.list_pending_tickets()
            if not pending:
                print("👩‍💼 [Admin Portal] No PENDING reservation requests.")
                continue
            for t in pending:
                print(f"\n{t.admin_briefing}")
                choice = (
                    input(f"👩‍💼 Admin Decision for {t.request_id} (CONFIRM/REFUSE) > ")
                    .strip()
                    .upper()
                )
                if choice in ("CONFIRM", "REFUSE"):
                    notes = input("👩‍💼 Optional Admin Notes > ").strip() or None
                    state = orch.submit_admin_decision_for_thread(
                        thread_id=thread_id,
                        decision=choice,  # type: ignore
                        admin_notes=notes,
                    )
                    print(f"\n🤖 Bot Notification >\n{state.get('bot_reply')}")
            continue

        state = orch.send_user_message(user_input, thread_id=thread_id)
        print(
            f"\n🤖 AeroPark Bot [{state.get('workflow_stage')}] >\n{state.get('bot_reply')}"
        )


def run_streamlit_app() -> None:
    """
    Render a clean, modern chat interface where:
      - Messages appear in a normal scrollable chat feed at the top
      - The chat input box is pinned at the very bottom of the page
      - Font sizes are compact and clean
    """
    import streamlit as st

    st.set_page_config(
        page_title="AeroPark AI Reservation Chatbot",
        page_icon="🚗",
        layout="centered",
    )

    # Custom CSS to keep chat typography compact, readable, and properly sized
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 4rem !important;
            max-width: 860px !important;
        }
        h1 {
            font-size: 1.35rem !important;
            margin-bottom: 0.25rem !important;
        }
        h2, h3 {
            font-size: 1.05rem !important;
            margin-top: 0.4rem !important;
            margin-bottom: 0.25rem !important;
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            font-size: 0.92rem !important;
            line-height: 1.45 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = build_default_orchestrator()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Welcome to **AeroPark Smart Garage**. Ask me about parking "
                    "availability, hourly prices, or location, or provide your **Name, "
                    "Surname, Car Number, and Reservation Period** to book a space!"
                ),
            }
        ]
        st.session_state.thread_id = "streamlit-session-1"

    orch: ParkingReservationOrchestrator = st.session_state.orchestrator
    pending_tickets = orch.admin_agent.list_pending_tickets()

    # ------------------------------------------------------------------
    # SIDEBAR: Navigation + Live Reservation Slot Tracker + Admin Quick Badge
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🚗 AeroPark Control Panel")
        view_mode = st.radio(
            "Select View:",
            options=[
                "💬 1. Customer Chat (Stage 1)",
                f"👩‍💼 2. Admin Approval Portal ({len(pending_tickets)} Pending)",
                "🔒 3. MCP Ledger & SQLite DB",
            ],
            index=0,
        )

        st.divider()
        st.markdown("**📋 Current Reservation Slots**")
        draft = orch.chatbot.reservation_draft
        st.caption(f"• **Name:** `{draft.name or '—'}`")
        st.caption(f"• **Surname:** `{draft.surname or '—'}`")
        st.caption(f"• **Car Number:** `{draft.car_number or '—'}`")
        st.caption(f"• **Period:** `{draft.reservation_period or '—'}`")
        st.caption(f"• **Zone:** `{draft.zone_code}`")

        if st.button("🔄 Reset Conversation", use_container_width=True):
            orch.chatbot.reset_conversation()
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Conversation reset! How can I help you today?",
                }
            ]
            st.rerun()

    # ------------------------------------------------------------------
    # VIEW 1: NORMAL CUSTOMER CHAT (Messages on Top, Input Box at Bottom!)
    # ------------------------------------------------------------------
    if view_mode.startswith("💬"):
        st.markdown("# 💬 AeroPark Smart Garage — AI Assistant")
        st.caption(
            "Stage 1 Hybrid RAG (Pinecone + SQLite) • 3-Wall Security Guardrails • Interactive Slot Collector"
        )

        # 1. Render all existing chat messages in normal top-to-bottom order
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(_format_clean_chat_text(msg["content"]))

        # 2. Pinned bottom chat input (at the root level so Streamlit anchors it to the bottom!)
        user_prompt = st.chat_input(
            "Type your question or booking details here..."
        )
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            state = orch.send_user_message(
                user_prompt, thread_id=st.session_state.thread_id
            )
            reply = _format_clean_chat_text(state.get("bot_reply", ""))
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.rerun()

    # ------------------------------------------------------------------
    # VIEW 2: HUMAN-IN-THE-LOOP ADMINISTRATOR PORTAL (Stage 2)
    # ------------------------------------------------------------------
    elif view_mode.startswith("👩‍💼"):
        st.markdown("# 👩‍💼 Stage 2: Human-in-the-Loop Administrator Portal")
        st.caption(
            "Review reservation requests escalated by Agent 1 and submit CONFIRM or REFUSE decisions."
        )

        if not pending_tickets:
            st.info("No pending reservation requests awaiting approval right now.")
        for t in pending_tickets:
            with st.container(border=True):
                st.markdown(f"**Ticket `{t.request_id}`** — Created at `{t.created_at}`")
                st.code(t.admin_briefing, language="text")
                notes = st.text_input(
                    f"Optional Admin Notes ({t.request_id})",
                    value="Verified EV stall & license plate",
                    key=f"notes_{t.request_id}",
                )
                col1, col2 = st.columns(2)
                if col1.button(
                    f"✅ CONFIRM ({t.request_id})",
                    key=f"ok_{t.request_id}",
                    use_container_width=True,
                ):
                    res = orch.submit_admin_decision_for_thread(
                        thread_id=st.session_state.thread_id,
                        decision="CONFIRM",
                        admin_notes=notes,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": _format_clean_chat_text(res.get("bot_reply", "")),
                        }
                    )
                    st.success(
                        f"Confirmed `{t.request_id}` & recorded to `confirmed_reservations.txt` via MCP Server!"
                    )
                    st.rerun()
                if col2.button(
                    f"❌ REFUSE ({t.request_id})",
                    key=f"no_{t.request_id}",
                    use_container_width=True,
                ):
                    res = orch.submit_admin_decision_for_thread(
                        thread_id=st.session_state.thread_id,
                        decision="REFUSE",
                        admin_notes=notes,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": _format_clean_chat_text(res.get("bot_reply", "")),
                        }
                    )
                    st.warning(f"Refused reservation `{t.request_id}`.")
                    st.rerun()

    # ------------------------------------------------------------------
    # VIEW 3: MCP SERVER LEDGER & LIVE SQLITE STATE (Stages 3 & 4)
    # ------------------------------------------------------------------
    else:
        st.markdown("# 🔒 Stage 3 & 4: MCP Server Ledger & Live SQLite Database")
        st.markdown("### 📄 `data/confirmed_reservations.txt` (Written by MCP Server)")
        st.caption("Required Format: `Name | Car Number | Reservation Period | Approval Time`")
        records = orch.mcp_client.processor.read_all_records()
        if records:
            st.code("\n".join(records), language="text")
        else:
            st.info("No confirmed reservations recorded yet.")

        st.markdown("### 🗄️ Live SQLite Parking Zones (`parking_dynamic.db`)")
        snapshot = orch.chatbot.sql_store.get_live_snapshot()
        st.dataframe(
            [z.model_dump() for z in snapshot.zones],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli_demo()
    else:
        run_streamlit_app()
