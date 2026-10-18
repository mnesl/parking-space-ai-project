# 📊 Stage 1: RAG System Performance & Accuracy Evaluation Report

## 1. Executive Summary Metrics
- **Top-K Retrieval Parameter (`K`):** `3`
- **Total Golden Benchmark Queries:** `5`
- **Mean Recall@3:** `90.00%`
- **Mean Precision@3:** `93.33%`
- **Mean Response Factual Accuracy:** `100.00%`
- **Median Request Latency (`p50`):** `1.51 ms`
- **95th Percentile Latency (`p95`):** `4.05 ms`
- **Sensitive Data Guardrail Block Rate:** `100.00%` (100% of attacks blocked)

## 2. Per-Query Breakdown
| Query | Recall@3 | Precision@3 | Answer Accuracy | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Where is AeroPark Smart Garage located and what is the physical address? | `1.00` | `1.00` | `1.00` | `4.05` |
| What is the maximum vehicle height limit in Zone A and Zone C, and do you have EV chargers? | `1.00` | `1.00` | `1.00` | `1.51` |
| What information do I need to provide to reserve a parking space? | `1.00` | `1.00` | `1.00` | `1.47` |
| What is the cancellation policy and grace period for reservations? | `1.00` | `0.67` | `1.00` | `1.51` |
| How many available spaces are there right now and what are the hourly prices and working hours? | `0.50` | `1.00` | `1.00` | `1.73` |