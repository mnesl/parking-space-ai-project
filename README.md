# 🚗 AeroPark Smart Garage — Intelligent Parking Reservation Chatbot (RAG + HITL + MCP + LangGraph)

An enterprise-grade, multi-agent AI system for **Parking Space Information & Reservation** built with **Python, LangChain, LangGraph, Pinecone Serverless, SQLite, FastAPI, and the Model Context Protocol (MCP)**.

---

## 🌟 Key Features Across All 4 Stages

| Stage | Module Path | Key Capabilities Implemented |
| :--- | :--- | :--- |
| **Stage 1: Hybrid RAG, Interactive Chatbot & Guardrails** | [`src/stage1_rag/`](src/stage1_rag) | • **Hybrid Data Split (Bonus):** Static rules/location in **Pinecone Serverless** + Live space availability, prices, and hours in **SQLite**.<br/>• **Interactive Slot Filling:** Multi-turn Pydantic `ReservationDraft` collecting `name`, `surname`, `car_number`, and `reservation_period`.<br/>• **3-Wall Guardrails:** Input prompt-injection blocker + Pinecone `is_public=True` metadata filter + Regex/NLP PII redaction.<br/>• **RAG Evaluation:** Automated `Recall@3` (`90.0%`), `Precision@3` (`93.3%`), Factual Accuracy (`100%`), and `p50`/`p95` Latency benchmark ([Report](docs/STAGE_1_EVALUATION_REPORT.md)). |
| **Stage 2: Human-in-the-Loop (HITL) Admin Agent** | [`src/stage2_hitl/`](src/stage2_hitl) | • **Second LangChain Agent (`AdminEscalationAgent`):** Receives validated reservation handoffs from Agent 1, enriches with live SQLite capacity, and creates `PENDING` tickets.<br/>• **FastAPI Admin Service (`admin_api.py`):** REST endpoints (`/escalate`, `/pending`, `/decision`) protected by `X-Admin-Api-Key`. |
| **Stage 3: Secure MCP Reservation Server** | [`src/stage3_mcp/`](src/stage3_mcp) | • **FastMCP + FastAPI Server (`mcp_server.py`):** Records `CONFIRMED` reservations to `data/confirmed_reservations.txt` in exact format:<br/>`Name | Car Number | Reservation Period | Approval Time`<br/>• **Security & Reliability:** `X-MCP-Api-Key` auth, Pydantic pipe (`|`) & newline (`\n`) delimiter injection protection, and `threading.Lock()` atomic file writes. |
| **Stage 4: LangGraph Orchestration & Load Testing** | [`src/stage4_orchestration/`](src/stage4_orchestration) | • **3-Node LangGraph Pipeline (`graph.py`):** `user_interaction` $\rightarrow$ `admin_approval` $\rightarrow$ `data_recording` with `MemorySaver` checkpointing.<br/>• **Unified UI (`app_ui.py`):** Interactive **Streamlit Dashboard** + Terminal CLI.<br/>• **Load Testing (`scripts/load_test.py`):** Multi-threaded concurrency benchmarks across all 4 stages ([Report](docs/STAGE_4_LOAD_TEST_REPORT.md)). |

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    User(["👤 Customer (CLI / Streamlit UI)"]) <-->|"Chat Input / Output"| Guardrails["🛡️ Stage 1: 3-Wall Guardrails\n(Prompt Injection & PII Filter)"]
    
    subgraph LangGraph["🧠 Stage 4: LangGraph StateGraph Orchestrator"]
        Node1["🤖 Node 1: user_interaction\n(Hybrid RAG + Reservation Slot Collector)"]
        Node2["👩‍💼 Node 2: admin_approval\n(Stage 2: AdminEscalationAgent HITL)"]
        Node3["🔒 Node 3: data_recording\n(Stage 3: MCP Client Node)"]
    end

    Guardrails --> Node1
    Node1 <-->|"Semantic Search (is_public=True)"| Pinecone[("🌲 Pinecone Serverless\n(Static Knowledge Base)")]
    Node1 <-->|"Live Spaces, Prices & Hours"| SQLite[("🗄️ SQLite Database\n(Dynamic Parking State)")]
    
    Node1 -->|"All 4 Slots Validated"| Node2
    Node2 <-->|"REST API (X-Admin-Api-Key)"| AdminPortal["👩‍💼 Human Administrator\n(CONFIRM / REFUSE)"]
    
    Node2 -->|"CONFIRMED"| Node3
    Node2 -->|"REFUSED"| User
    
    Node3 <-->|"MCP Protocol (X-MCP-Api-Key)"| MCPServer["🔒 Stage 3: FastMCP / FastAPI Server"]
    MCPServer -->|"Atomic Locked Write"| FlatFile["📄 data/confirmed_reservations.txt\n(Name | Car # | Period | Approval Time)"]
    Node3 -->|"Decrements Available Spaces"| SQLite
```

---

## 📁 Project Repository Structure

```text
parking-space-ai-project/
├── README.md                        # Complete setup, architecture, and usage documentation
├── requirements.txt                 # Pinned Python dependencies
├── .env.example                     # Environment variable template (Gemini / OpenAI + Pinecone)
├── Dockerfile                       # Container definition for the application
├── docker-compose.yml               # Multi-service orchestration (Streamlit UI + Admin API + MCP Server)
├── .github/workflows/ci.yml         # GitHub Actions CI/CD pipeline (runs pytest + load tests)
├── terraform/                       # Infrastructure as Code (Terraform)
│   └── main.tf                      # Provisions Pinecone Serverless index
├── data/
│   ├── static_docs/                 # Markdown knowledge base for Pinecone ingestion
│   │   ├── 01_general_info_and_location.md
│   │   ├── 02_booking_process_and_policies.md
│   │   └── 03_internal_sensitive_audit.md
│   ├── parking_dynamic.db           # SQLite DB for live spaces, pricing, hours & pending tickets
│   └── confirmed_reservations.txt   # Stage 3 MCP server output file (Name | Car | Period | Time)
├── docs/
│   ├── STAGE_1_EVALUATION_REPORT.md # Recall@3, Precision@3, Accuracy & Latency report
│   ├── STAGE_4_LOAD_TEST_REPORT.md  # Multi-threaded concurrency & throughput report
│   └── AeroPark_AI_Reservation_Chatbot_Presentation.pptx
├── src/
│   ├── __init__.py
│   ├── config.py                    # Settings & LLM/Embeddings Factory (Gemini / OpenAI)
│   ├── stage1_rag/
│   │   ├── __init__.py
│   │   ├── sql_store.py             # Dynamic SQLite database manager (Pydantic validated)
│   │   ├── vector_store.py          # Pinecone Serverless ingestion & similarity search
│   │   ├── guardrails.py            # 3-Wall PII & sensitive data protection (Regex + NLP)
│   │   ├── chatbot_agent.py         # Hybrid RAG + interactive reservation slot collector
│   │   └── evaluation.py            # Recall@K, Precision@K, and latency benchmarks
│   ├── stage2_hitl/
│   │   ├── __init__.py
│   │   ├── admin_agent.py           # Second LangChain agent for Administrator escalation
│   │   └── admin_api.py             # FastAPI REST endpoints for human CONFIRM/REFUSE
│   ├── stage3_mcp/
│   │   ├── __init__.py
│   │   ├── mcp_server.py            # Secure FastMCP / FastAPI reservation writer
│   │   └── mcp_client.py            # MCP client & LangChain @tool fallback
│   └── stage4_orchestration/
│       ├── __init__.py
│       ├── state.py                 # LangGraph TypedDict state schema & reducers
│       ├── graph.py                 # Complete 3-node LangGraph workflow pipeline
│       └── app_ui.py                # Unified Streamlit Dashboard & CLI
├── scripts/
│   ├── load_test.py                 # Stage 4 load & concurrency stress testing suite
│   └── generate_presentation.py     # Automated PowerPoint (.pptx) presentation generator
└── tests/                           # 16 automated unit & integration tests (>= 2 per module)
    ├── test_sql_store.py
    ├── test_vector_store.py
    ├── test_guardrails.py
    ├── test_chatbot_agent.py
    ├── test_evaluation.py
    ├── test_admin_hitl.py
    ├── test_mcp_server.py
    └── test_langgraph_workflow.py
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone & Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env` (Optional — Works Out-of-the-Box with Local Fallback!)
Copy `.env.example` to `.env` and add your **Google AI Studio (`GOOGLE_API_KEY`)** or **OpenAI (`OPENAI_API_KEY`)** and **Pinecone (`PINECONE_API_KEY`)** keys:
```powershell
copy .env.example .env
```

---

## 🎬 Running the Application

### Option A: Launch the Interactive Streamlit Web Dashboard
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\streamlit.exe run src/stage4_orchestration/app_ui.py
```
Opens **`http://localhost:8501`** in your browser:
1. **💬 1. Customer Chat (Stage 1):** Ask questions about location, height limits, EV chargers, live spaces, and prices, then book a space.
2. **👩‍💼 2. Admin Approval Portal (Stage 2):** Review pending tickets and click **✅ CONFIRM** or **❌ REFUSE**.
3. **🔒 3. MCP Ledger & SQLite DB (Stages 3 & 4):** Inspect `data/confirmed_reservations.txt` (`Name | Car Number | Reservation Period | Approval Time`) and watch available spaces decrement in real time!

### Option B: Run the Interactive Terminal CLI
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m src.stage4_orchestration.app_ui --cli
```

### Option C: Launch Stage 2 & Stage 3 FastAPI Swagger UIs (`/docs`)
```powershell
# Stage 2 Admin HITL REST API (http://127.0.0.1:8000/docs)
.\.venv\Scripts\uvicorn.exe src.stage2_hitl.admin_api:app --port 8000

# Stage 3 MCP Server REST API (http://127.0.0.1:8001/docs)
.\.venv\Scripts\uvicorn.exe src.stage3_mcp.mcp_server:app --port 8001
```

---

## 🧪 Running Automated Tests, Evaluations & Load Benchmarks

### 1. Run the Complete Pytest Suite (16 tests across all modules)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 2. Run Stage 4 Concurrency & Load Testing
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe scripts/load_test.py
```
Generates [`docs/STAGE_4_LOAD_TEST_REPORT.md`](docs/STAGE_4_LOAD_TEST_REPORT.md).

### 3. Generate the PowerPoint Presentation (`.pptx`)
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe scripts/generate_presentation.py
```
Generates [`docs/AeroPark_AI_Reservation_Chatbot_Presentation.pptx`](docs/AeroPark_AI_Reservation_Chatbot_Presentation.pptx).
