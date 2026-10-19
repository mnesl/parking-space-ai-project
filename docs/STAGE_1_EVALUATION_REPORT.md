# 📊 Stage 1: RAG System Performance & Accuracy Evaluation Report

## 1. Executive Summary Metrics
- **Top-K Retrieval Parameter (`K`):** `3`
- **Total Golden Benchmark Queries:** `5`
- **Mean Recall@3:** `100.00%`
- **Mean Precision@3:** `93.33%`
- **Mean Response Factual Accuracy:** `93.33%`
- **Median Request Latency (`p50`):** `2361.55 ms`
- **95th Percentile Latency (`p95`):** `5185.03 ms`
- **Sensitive Data Guardrail Block Rate:** `100.00%` (100% of attacks blocked)

## 2. Per-Query Breakdown
| Query | Recall@3 | Precision@3 | Answer Accuracy | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Where is AeroPark Smart Garage located and what is the physical address? | `1.00` | `0.67` | `1.00` | `5185.03` |
| What is the maximum vehicle height limit in Zone A and Zone C, and do you have EV chargers? | `1.00` | `1.00` | `1.00` | `1567.82` |
| What information do I need to provide to reserve a parking space? | `1.00` | `1.00` | `1.00` | `1590.98` |
| What is the cancellation policy and grace period for reservations? | `1.00` | `1.00` | `1.00` | `4966.89` |
| How many available spaces are there right now and what are the hourly prices and working hours? | `1.00` | `1.00` | `0.67` | `2361.55` |

## 3. Production Resilience Note (`503 High Demand` Fallback)
- During burst evaluation queries, Google AI Studio's free API occasionally returns temporary `503 UNAVAILABLE (High Demand)` spikes.
- To prevent request failures or long retry hangs, `src/config.py`, `src/stage1_rag/chatbot_agent.py`, and `src/stage2_hitl/admin_agent.py` implement an automatic **`8.0s` timeout (`max_retries=0`)** + **seamless fallback to grounded Pinecone + SQLite synthesis**, ensuring 100% uptime and grounded responses even during upstream LLM traffic spikes.