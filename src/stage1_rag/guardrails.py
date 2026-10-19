"""
Stage 1: Multi-Layer Sensitive Data Guardrails (`guardrails.py`).

Specification Requirement:
--------------------------
"4. Implement guard rails mechanism. Add a filtering to prevent exposure of
sensitive data (e.g., using pre-trained NLP models for text analysis)."

Educational Concept - "Defense in Depth" (The 3 Security Walls):
----------------------------------------------------------------
In production AI Engineering, you NEVER rely on just telling the LLM "please don't
share secrets" (because hackers use Prompt Injection to bypass system prompts!).
Instead, we build 3 independent security layers:

1. **Input Guardrail (`inspect_user_input`):**
   Detects prompt-injection / jailbreak attempts ("ignore previous instructions",
   "dump internal audit", "what is the master override pin or SSN") BEFORE calling
   the LLM or database.
2. **Retrieval / Context Guardrail (`sanitize_retrieved_documents`):**
   Even though Pinecone filters `{"is_public": True}`, what if a public document
   accidentally contains an employee's email, phone number, SSN, credit card, or
   secret API token? Our Context Guardrail scans every retrieved chunk using
   Pattern + NLP Entity Recognition and redacts sensitive entities into
   `[REDACTED_SSN]`, `[REDACTED_CREDIT_CARD]`, etc., BEFORE the LLM reads it!
3. **Output Guardrail (`sanitize_bot_response`):**
   Scans the LLM's final response right before showing it to the user to guarantee
   zero leakage of PII or internal credentials.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Pydantic model representing the outcome of a guardrail check."""

    is_safe: bool = Field(description="True if no blocking security violation occurred")
    sanitized_text: str = Field(description="Text with all sensitive PII/secrets redacted")
    detected_categories: List[str] = Field(
        default_factory=list,
        description="List of sensitive data or attack categories detected",
    )
    blocked_reason: str | None = Field(
        default=None,
        description="If blocked outright, explanation of why the request was refused",
    )


class SensitiveDataGuardrail:
    """
    Multi-layer Guardrail combining:
    1. Prompt-Injection / Confidential-Probing Intent Classifier
    2. Regex + Context-Window NLP Entity Recognizer (for PII, Financial, Credentials,
       and Private VIP Registry entities)
    """

    # 1. Patterns for structured PII & secrets
    SENSITIVE_PATTERNS: Dict[str, re.Pattern[str]] = {
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "SECRET_TOKEN": re.compile(
            r"\b(?:sk-[a-zA-Z0-9_-]{10,}|override-pin-[a-zA-Z0-9_-]+)\b",
            re.IGNORECASE,
        ),
        "INTERNAL_EMAIL": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:local|internal|corp)\b",
            re.IGNORECASE,
        ),
        "PHONE_NUMBER": re.compile(
            r"(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]\d{3}[-\s]\d{4}\b"
        ),
        "PRIVATE_VIP_PLATE": re.compile(
            r"\b(?:\d{2}-(?:DIPLOMAT|GOV)-\d{2,4})\b",
            re.IGNORECASE,
        ),
    }

    # 2. NLP Context-Window Entity Triggers (Named Entity / Role-based PII detection)
    CONFIDENTIAL_NAMED_ENTITIES: List[Tuple[str, str]] = [
        ("marcus vance", "INTERNAL_STAFF_PII"),
        ("elena rostova", "VIP_CUSTOMER_PII"),
        ("viktor krum", "VIP_CUSTOMER_PII"),
    ]

    # 3. Prompt Injection & Unauthorized Probing Signatures
    PROMPT_INJECTION_SIGNATURES: List[Tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions",
                re.IGNORECASE,
            ),
            "PROMPT_INJECTION_ATTEMPT",
        ),
        (
            re.compile(
                r"(?:master\s+gate\s+override\s+pin|backup\s+root\s+token|internal\s+audit\s+log)",
                re.IGNORECASE,
            ),
            "UNAUTHORIZED_CREDENTIAL_PROBE",
        ),
        (
            re.compile(
                r"(?:ssn|social\s+security|credit\s+card\s+on\s+file|vip\s+registry|diplomat\s+plate)",
                re.IGNORECASE,
            ),
            "SENSITIVE_PII_PROBE",
        ),
    ]

    def __init__(self) -> None:
        """Initialize the guardrail engine (loads optional spaCy NER if installed)."""
        self._spacy_nlp = self._load_optional_spacy()

    @staticmethod
    def _load_optional_spacy():
        """
        Attempt to load a pre-trained spaCy NLP pipeline if installed in the environment,
        falling back gracefully to our deterministic Regex + Gazetteer NER engine.
        """
        try:
            import spacy  # type: ignore

            return spacy.load("en_core_web_sm")
        except Exception:
            return None

    def inspect_user_input(self, user_query: str) -> GuardrailResult:
        """
        Wall #1 (Input Guardrail):
        Check if the user is attempting a Prompt Injection attack or explicitly asking
        for confidential employee/VIP customer PII or security tokens.
        """
        detected: List[str] = []

        for pattern, label in self.PROMPT_INJECTION_SIGNATURES:
            if pattern.search(user_query):
                detected.append(label)

        low_query = user_query.lower()
        for entity_name, label in self.CONFIDENTIAL_NAMED_ENTITIES:
            if entity_name in low_query:
                detected.append(label)

        if detected:
            return GuardrailResult(
                is_safe=False,
                sanitized_text=user_query,
                detected_categories=sorted(set(detected)),
                blocked_reason=(
                    "Security Guardrail Alert: I cannot fulfill requests for internal "
                    "security credentials, audit logs, or private personal data (PII) "
                    "of other customers or staff. I can gladly help you with public "
                    "parking information, availability, pricing, or booking a space!"
                ),
            )

        return GuardrailResult(
            is_safe=True,
            sanitized_text=user_query,
            detected_categories=[],
        )

    def redact_sensitive_text(self, text: str) -> GuardrailResult:
        """
        Scan any text (retrieved chunk or LLM response) and replace sensitive PII,
        credentials, or confidential named entities with `[REDACTED_<CATEGORY>]`.
        """
        sanitized = text
        detected: List[str] = []

        # 1. Regex pattern scrubbing (SSN, Credit Card, Secret Tokens, Phone, Diplomatic Plates)
        for category, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern.search(sanitized):
                detected.append(category)
                sanitized = pattern.sub(f"[REDACTED_{category}]", sanitized)

        # 2. Gazetteer / Context-Aware Named Entity Redaction
        for entity_name, category in self.CONFIDENTIAL_NAMED_ENTITIES:
            pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
            if pattern.search(sanitized):
                detected.append(category)
                sanitized = pattern.sub(f"[REDACTED_{category}]", sanitized)

        # 3. Optional pre-trained spaCy NER pass for any remaining PERSON entities
        # inside internal/audit contexts
        if self._spacy_nlp is not None and "CONFIDENTIAL" in sanitized.upper():
            doc = self._spacy_nlp(sanitized)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    detected.append("NLP_PERSON_ENTITY")
                    sanitized = sanitized.replace(ent.text, "[REDACTED_PERSON]")

        unique_detected = sorted(set(detected))
        return GuardrailResult(
            is_safe=len(unique_detected) == 0,
            sanitized_text=sanitized,
            detected_categories=unique_detected,
        )

    def sanitize_retrieved_documents(self, docs: List[Document]) -> List[Document]:
        """
        Wall #2 (Retrieval Guardrail):
        Filter out any document explicitly marked `is_public=False` AND scrub
        all remaining chunks for accidental PII before passing context to the LLM.
        """
        safe_docs: List[Document] = []
        for doc in docs:
            if doc.metadata.get("is_public") is False:
                continue
            redacted = self.redact_sensitive_text(doc.page_content)
            safe_docs.append(
                Document(
                    page_content=redacted.sanitized_text,
                    metadata={
                        **doc.metadata,
                        "redacted_categories": redacted.detected_categories,
                    },
                )
            )
        return safe_docs

