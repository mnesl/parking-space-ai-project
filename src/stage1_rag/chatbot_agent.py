"""
Stage 1, Step 5: Hybrid RAG Chatbot Agent & Interactive Reservation Slot Collector
(`chatbot_agent.py`).

Specification Requirements Covered:
-----------------------------------
1. "Implement the basic architecture of the chatbot using Retrieval-Augmented
   Generation to interact with users."
2. "The chatbot provides information (general information, working hours, prices,
   availability of parking spaces, location)."
3. "The reservation process is based on interactive collection of user data,
   including name, surname, car number, and reservation period."

Educational Concepts Covered Here:
----------------------------------
1. **Interactive Slot Filling with Pydantic (`ReservationDraft`):**
   Remember when we learned about `BaseModel`, `Field`, and `@field_validator`?
   In a real conversation, a user rarely gives all 4 booking details in one message!
   - Turn 1: "I want to reserve a spot for my car 34ABC123."
   - Turn 2: "My name is Merve Okcu."
   - Turn 3: "For 2026-10-20 09:00 to 2026-10-20 18:00."
   Our `ReservationDraft` Pydantic model remembers what slots are already filled,
   validates them (cleaning the license plate and blocking pipe `|` characters!),
   and computes `.missing_fields()` so the chatbot knows *exactly* what follow-up
   question to ask next!

2. **Hybrid RAG Prompt Augmentation:**
   For every user message, our chatbot combines:
   - **Static Context** from `ParkingVectorStore` (Pinecone semantic search, filtered
     with `is_public=True` and scrubbed by `SensitiveDataGuardrail`).
   - **Dynamic Context** from `DynamicParkingSQLStore` (live SQLite spaces, prices,
     and working hours).
"""

from __future__ import annotations

import re
import time
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from src.config import Settings, get_chat_model, get_settings
from src.stage1_rag.guardrails import SensitiveDataGuardrail
from src.stage1_rag.sql_store import DynamicParkingSQLStore
from src.stage1_rag.vector_store import ParkingVectorStore


# ============================================================================
# 1. PYDANTIC MODELS FOR INTERACTIVE RESERVATION SLOT FILLING
# ============================================================================
class ReservationDraft(BaseModel):
    """
    Tracks the 4 required reservation fields across multiple turns of conversation:
      1. `name` (First Name)
      2. `surname` (Last Name)
      3. `car_number` (License Plate)
      4. `reservation_period` (Start and End Date/Time)
    Plus an optional `zone_code` (defaults to 'ZONE_A').
    """

    name: Optional[str] = Field(
        default=None, description="Customer's first name (e.g., 'Merve')"
    )
    surname: Optional[str] = Field(
        default=None, description="Customer's last name / surname (e.g., 'Okcu')"
    )
    car_number: Optional[str] = Field(
        default=None,
        description="Vehicle license plate in uppercase (e.g., '34ABC123')",
    )
    reservation_period: Optional[str] = Field(
        default=None,
        description="Start and end date/time (e.g., '2026-10-20 09:00 to 2026-10-20 18:00')",
    )
    zone_code: str = Field(
        default="ZONE_A",
        description="Preferred parking zone ('ZONE_A', 'ZONE_B', or 'ZONE_C')",
    )

    @field_validator("name", "surname")
    @classmethod
    def clean_names(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if "|" in cleaned or "\n" in cleaned:
            raise ValueError("Invalid characters ('|' or newline) in name/surname.")
        return cleaned

    @field_validator("car_number")
    @classmethod
    def clean_and_validate_plate(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().upper().replace(" ", "")
        if not cleaned:
            return None
        if "|" in cleaned or "\n" in cleaned:
            raise ValueError("Pipe character '|' is forbidden in car_number.")
        if not re.match(r"^[A-Z0-9-]{4,15}$", cleaned):
            raise ValueError(
                "car_number must be 4-15 alphanumeric characters (e.g., 34ABC123)."
            )
        return cleaned

    @field_validator("reservation_period")
    @classmethod
    def clean_period(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if "|" in cleaned or "\n" in cleaned:
            raise ValueError("Pipe character '|' is forbidden in reservation_period.")
        return cleaned

    @property
    def full_name(self) -> str:
        """Return 'Name Surname' combined for Stage 3 file output."""
        parts = [p for p in [self.name, self.surname] if p]
        return " ".join(parts)

    def missing_fields(self) -> List[str]:
        """Return the list of required fields that have not been collected yet."""
        missing: List[str] = []
        if not self.name:
            missing.append("name (First Name)")
        if not self.surname:
            missing.append("surname (Last Name)")
        if not self.car_number:
            missing.append("car_number (License Plate)")
        if not self.reservation_period:
            missing.append("reservation_period (Start & End Date/Time)")
        return missing

    @property
    def is_complete(self) -> bool:
        """Return True when all 4 required user fields are filled and validated."""
        return len(self.missing_fields()) == 0


class ChatbotTurnResult(BaseModel):
    """Structured output returned by the chatbot after processing a single user message."""

    reply: str = Field(description="Sanitized response shown to the user")
    retrieved_sources: List[str] = Field(
        default_factory=list,
        description="List of static chunk IDs retrieved from Pinecone/VectorStore",
    )
    reservation_draft: ReservationDraft = Field(
        description="Current state of collected reservation fields"
    )
    ready_for_admin_escalation: bool = Field(
        default=False,
        description="True when all 4 reservation slots are filled and ready for Stage 2 Admin HITL",
    )
    blocked_by_guardrail: bool = Field(
        default=False,
        description="True if Input Guardrail blocked a prompt injection or PII probe",
    )
    latency_ms: float = Field(
        default=0.0,
        description="End-to-end processing time in milliseconds for performance evaluation",
    )


# ============================================================================
# 2. DETERMINISTIC + LLM SLOT EXTRACTOR HELPER
# ============================================================================
def extract_slots_from_message(
    user_message: str,
    current_draft: ReservationDraft,
) -> ReservationDraft:
    """
    Update `current_draft` with any reservation details found in `user_message`.
    Uses deterministic regex patterns for natural phrasings so slot extraction works
    reliably both with live LLMs and during offline automated testing!
    """
    updated = current_draft.model_dump()

    # 1. Extract Name & Surname (e.g. "My name is Merve Okcu", "I am John Doe", "Name: Jane Smith")
    name_match = re.search(
        r"(?:my\s+name\s+is|i\s+am|i'm|name\s*[:=])\s+([A-Z][a-zA-Z]+)\s+([A-Z][a-zA-Z]+)",
        user_message,
        re.IGNORECASE,
    )
    if name_match:
        updated["name"] = name_match.group(1).strip().title()
        updated["surname"] = name_match.group(2).strip().title()
    else:
        first_only = re.search(
            r"(?:first\s+name\s*[:=]|name\s*[:=])\s*([A-Za-z]+)",
            user_message,
            re.IGNORECASE,
        )
        last_only = re.search(
            r"(?:surname\s*[:=]|last\s+name\s*[:=])\s*([A-Za-z]+)",
            user_message,
            re.IGNORECASE,
        )
        if first_only:
            updated["name"] = first_only.group(1).strip().title()
        if last_only:
            updated["surname"] = last_only.group(1).strip().title()

    # 2. Extract Car Number / License Plate (e.g. "car number 34ABC123", "plate: ABC-1234")
    plate_match = re.search(
        r"(?:car\s+number|license\s+plate|plate|car)\s*(?:is|[:=])?\s*([A-Z0-9]{2,10}(?:-[A-Z0-9]{1,6})?)\b",
        user_message,
        re.IGNORECASE,
    )
    if plate_match:
        candidate = plate_match.group(1).strip().upper()
        # Ensure it contains at least one digit and one letter so normal words aren't matched
        if any(c.isdigit() for c in candidate) and any(c.isalpha() for c in candidate):
            updated["car_number"] = candidate

    # 3. Extract Reservation Period (e.g. "2026-10-20 09:00 to 2026-10-20 18:00" or "period: ...")
    period_match = re.search(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(?:to|-|until)\s+(?:\d{4}-\d{2}-\d{2}\s+)?\d{2}:\d{2})",
        user_message,
        re.IGNORECASE,
    )
    if period_match:
        updated["reservation_period"] = period_match.group(1).strip()
    else:
        explicit_period = re.search(
            r"(?:reservation\s+period|period|for)\s*(?:is|[:=])\s*([0-9A-Za-z\s:,-]{8,45})",
            user_message,
            re.IGNORECASE,
        )
        if explicit_period and any(ch.isdigit() for ch in explicit_period.group(1)):
            updated["reservation_period"] = explicit_period.group(1).strip()

    # 4. Extract Optional Zone Selection
    if re.search(r"\bzone\s*b\b|\bev\s+zone\b", user_message, re.IGNORECASE):
        updated["zone_code"] = "ZONE_B"
    elif re.search(r"\bzone\s*c\b|\bvip\b", user_message, re.IGNORECASE):
        updated["zone_code"] = "ZONE_C"
    elif re.search(r"\bzone\s*a\b", user_message, re.IGNORECASE):
        updated["zone_code"] = "ZONE_A"

    return ReservationDraft.model_validate(updated)


# ============================================================================
# 3. HYBRID RAG CHATBOT AGENT
# ============================================================================
SYSTEM_PROMPT_TEMPLATE = """You are the official AeroPark Smart Garage AI Assistant.
Your job is to:
1. Answer user questions accurately using ONLY the provided Static Context (from Pinecone)
   and Live SQL Database Snapshot (availability, prices, working hours).
2. Help users reserve a parking space by interactively collecting 4 required fields:
   - First Name (`name`)
   - Surname (`surname`)
   - Car Number / License Plate (`car_number`)
   - Reservation Period (`reservation_period`, e.g., '2026-10-20 09:00 to 2026-10-20 18:00')
3. NEVER expose internal staff PII, VIP registries, SSNs, credit cards, or override PINs.

=== STATIC KNOWLEDGE BASE CONTEXT (PINECONE) ===
{static_context}

=== DYNAMIC SQL DATABASE SNAPSHOT (LIVE SPACES, PRICES & HOURS) ===
{dynamic_sql_context}

=== CURRENT RESERVATION DRAFT STATUS ===
- Collected so far: {collected_summary}
- Still missing: {missing_summary}
"""


class ParkingRAGChatbot:
    """
    Stage 1 Hybrid RAG Chatbot combining:
    - Wall #1 Input Guardrail
    - Pinecone Static Vector Search (`k` top chunks, filtered `is_public=True`)
    - Wall #2 Retrieval Context Guardrail
    - Live SQLite Dynamic Snapshot (spaces, prices, hours)
    - Interactive Reservation Slot Collection (`ReservationDraft`)
    - Wall #3 Output Guardrail
    """

    def __init__(
        self,
        vector_store: Optional[ParkingVectorStore] = None,
        sql_store: Optional[DynamicParkingSQLStore] = None,
        guardrail: Optional[SensitiveDataGuardrail] = None,
        llm: Optional[BaseChatModel] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or ParkingVectorStore(settings=self.settings)
        self.sql_store = sql_store or DynamicParkingSQLStore()
        self.guardrail = guardrail or SensitiveDataGuardrail()
        self.llm = llm

        self.chat_history: List[BaseMessage] = []
        self.reservation_draft = ReservationDraft()

    def reset_conversation(self) -> None:
        """Clear conversation history and reset the reservation draft."""
        self.chat_history.clear()
        self.reservation_draft = ReservationDraft()

    def chat(self, user_message: str, top_k: int = 3) -> ChatbotTurnResult:
        """
        Process one conversational turn end-to-end through Guardrails, Hybrid RAG,
        and Reservation Slot Collection.
        """
        start_time = time.perf_counter()

        # ------------------------------------------------------------------
        # STEP 1: Wall #1 — Input Guardrail (Block Prompt Injection / PII Probes)
        # ------------------------------------------------------------------
        input_check = self.guardrail.inspect_user_input(user_message)
        if not input_check.is_safe:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ChatbotTurnResult(
                reply=input_check.blocked_reason or "Request blocked by security guardrail.",
                retrieved_sources=[],
                reservation_draft=self.reservation_draft,
                ready_for_admin_escalation=False,
                blocked_by_guardrail=True,
                latency_ms=round(latency_ms, 2),
            )

        # ------------------------------------------------------------------
        # STEP 2: Interactive Slot Extraction (Update ReservationDraft)
        # ------------------------------------------------------------------
        try:
            self.reservation_draft = extract_slots_from_message(
                user_message, self.reservation_draft
            )
        except ValueError as validation_err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ChatbotTurnResult(
                reply=f"Validation Error in your reservation details: {validation_err}",
                retrieved_sources=[],
                reservation_draft=self.reservation_draft,
                ready_for_admin_escalation=False,
                blocked_by_guardrail=False,
                latency_ms=round(latency_ms, 2),
            )

        # ------------------------------------------------------------------
        # STEP 3: Retrieve Static Context from Pinecone + Wall #2 Guardrail
        # ------------------------------------------------------------------
        raw_docs: List[Document] = self.vector_store.similarity_search(
            query=user_message,
            k=top_k,
            public_only=True,
        )
        safe_docs = self.guardrail.sanitize_retrieved_documents(raw_docs)
        retrieved_sources = [
            str(d.metadata.get("chunk_id", d.metadata.get("source", "unknown")))
            for d in safe_docs
        ]
        static_context_str = "\n\n".join(
            f"[Source: {d.metadata.get('source')}]\n{d.page_content}"
            for d in safe_docs
        )

        # ------------------------------------------------------------------
        # STEP 4: Query Dynamic Context from SQLite (Live Spaces, Prices, Hours)
        # ------------------------------------------------------------------
        live_snapshot = self.sql_store.get_live_snapshot()
        dynamic_sql_str = live_snapshot.to_context_string()

        # ------------------------------------------------------------------
        # STEP 5: Generate Response (via LLM or Deterministic RAG Synthesizer)
        # ------------------------------------------------------------------
        raw_reply = self._generate_response(
            user_message=user_message,
            static_context=static_context_str,
            dynamic_sql_context=dynamic_sql_str,
            live_snapshot=live_snapshot,
            safe_docs=safe_docs,
        )

        # ------------------------------------------------------------------
        # STEP 6: Wall #3 — Output Guardrail (Final PII Scrubbing)
        # ------------------------------------------------------------------
        output_check = self.guardrail.redact_sensitive_text(raw_reply)
        final_reply = output_check.sanitized_text

        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=final_reply))

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return ChatbotTurnResult(
            reply=final_reply,
            retrieved_sources=retrieved_sources,
            reservation_draft=self.reservation_draft,
            ready_for_admin_escalation=self.reservation_draft.is_complete,
            blocked_by_guardrail=False,
            latency_ms=round(latency_ms, 2),
        )

    def _generate_response(
        self,
        user_message: str,
        static_context: str,
        dynamic_sql_context: str,
        live_snapshot,
        safe_docs: List[Document],
    ) -> str:
        """
        Invoke the configured LangChain Chat Model (`get_chat_model()`) if an API key
        is configured, or use the built-in grounded synthesizer when running offline tests.
        """
        missing = self.reservation_draft.missing_fields()
        collected_str = self.reservation_draft.model_dump_json()
        missing_str = ", ".join(missing) if missing else "NONE (All 4 fields collected!)"

        # Try active LLM if injected or if API key is present in .env
        active_llm = self.llm
        if active_llm is None:
            has_key = (
                self.settings.llm_provider == "gemini" and bool(self.settings.google_api_key)
            ) or (
                self.settings.llm_provider == "openai" and bool(self.settings.openai_api_key)
            )
            if has_key:
                active_llm = get_chat_model(temperature=0.0, settings=self.settings)

        if active_llm is not None:
            system_msg = SystemMessage(
                content=SYSTEM_PROMPT_TEMPLATE.format(
                    static_context=static_context,
                    dynamic_sql_context=dynamic_sql_context,
                    collected_summary=collected_str,
                    missing_summary=missing_str,
                )
            )
            messages = [system_msg, *self.chat_history[-6:], HumanMessage(content=user_message)]
            response = active_llm.invoke(messages)
            return str(response.content)

        # Grounded synthesis fallback (used when running tests without external API keys)
        if self.reservation_draft.is_complete:
            return (
                f"Thank you, {self.reservation_draft.full_name}! I have collected all required "
                f"reservation details:\n"
                f"- Name: {self.reservation_draft.full_name}\n"
                f"- Car Number: {self.reservation_draft.car_number}\n"
                f"- Reservation Period: {self.reservation_draft.reservation_period}\n"
                f"- Zone: {self.reservation_draft.zone_code}\n\n"
                f"Your request is now being escalated to the Human Administrator for confirmation."
            )

        low = user_message.lower()
        booking_intent = any(
            w in low
            for w in ["reserve", "book", "name is", "car number", "plate", "period"]
        )
        if booking_intent and len(missing) < 4:
            return (
                f"Got it! I have recorded your details so far. To complete your parking "
                f"reservation, please provide the remaining information: {', '.join(missing)}."
            )

        # Answer informational queries combining Dynamic SQL + Static Pinecone chunks
        parts: List[str] = []
        if any(w in low for w in ["available", "space", "price", "cost", "rate", "hour", "open"]):
            parts.append(dynamic_sql_context)
        if safe_docs:
            joined_chunks = "\n\n".join(d.page_content.strip() for d in safe_docs)
            parts.append("From our AeroPark Knowledge Base:\n" + joined_chunks)
        if booking_intent:
            parts.append(
                f"To book a space, please share: {', '.join(missing)}."
            )
        return "\n\n".join(parts) if parts else dynamic_sql_context
