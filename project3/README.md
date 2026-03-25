
# Project 3: Optimization & Advanced RL Enhancements for Text-to-SQL

##  Live Links

-  **Technical Report:**  
  https://tjhalanigrid.github.io/text2sql_2/

-  **Live Demo:**  
  https://huggingface.co/spaces/tjhalanigrid/text2sql-demo

---

##  Overview

This module extends the base Text-to-SQL system by introducing **system-level optimizations, reinforcement learning improvements, and deployment-focused enhancements**.

The goal is to transform a slow, error-prone baseline into a **scalable, schema-aware, and efficient inference system**.

---

##  Key Results

- ⚡ **5.63× speedup** in SQL execution (1.62s → 0.28s) using parallelization + caching  
-  **Execution Accuracy improved to 44.0%** via soft reward shaping  
-  **~72% schema-valid SQL generation** using constrained decoding  
-  **79.2% errors identified as schema-related (wrong_column)** → guided constraint design  
- ⚡ **~2× latency reduction** with INT8 quantization (3.11s → 1.65s)  
-  **Reduced reward variance** → more stable RL training  

---

##  Key Contributions

### ⚡ Task 1: Parallel SQL Execution Optimization
- Multi-threaded rollout execution using `ThreadPoolExecutor`
- Connection pooling + query caching
- Eliminated database I/O bottleneck

---

###  Task 2: Execution Error Diagnostics & Telemetry
- Error classification system (wrong_column, wrong_table, etc.)
- Identified schema hallucination as dominant failure mode
- Built clause-level error analysis dashboard

---

###  Task 3: Schema-Aware Constrained Decoding
- Logit masking using schema constraint graph
- Prevented invalid table/column generation
- Improved execution correctness over exact match

---

###  Task 4: Soft Reward Shaping
- Introduced dense reward (partial correctness)
- Improved execution accuracy and reduced variance
- Stabilized reinforcement learning training

---

###  Task 5: Quantized Inference Benchmarking
- INT8 dynamic quantization for CPU deployment
- Mixed precision (decoder-only quantization)
- Reduced latency without accuracy loss

---

##  Results Summary

| Metric | Baseline | Optimized |
|--------|---------|----------|
| Execution Speed | 1.62s | **0.28s** |
| Execution Accuracy | ~38% | **44.0%** |
| Schema Validity | Low | **~72%** |
| Inference Latency | 3.1s | **1.65s** |

---

##  Core Implementation Mapping (Tasks 1–5)

### ⚡ Task 01: Parallel SQL Execution
- `execution_reward.py` → Parallel rollout + reward logic  
- `run_sql.py` → SQL execution and timeout handling  
- `scripts/benchmark_parallel_reward.py` → Benchmarking script  

---

###  Task 02: Error Diagnostics
- `sql_validator.py` → SQL validation + error detection  
- `execution_reward.py` → Captures runtime errors  
- `app.py` → Error classification logic  

---

###  Task 03: Constrained Decoding
- `constrained_decoding.py` → Logits masking logic  
- `schema_constraints.py` → Schema rule generation  
- `schema_encoder.py` → Schema representation  

---

###  Task 04: Reward Design
- `execution_reward.py` → Hard reward  
- `execution_reward_soft.py` → Soft reward  
- `train_rl_codet5_reward_soft.py` → Soft reward training  

---

###  Task 05: Quantization
- `quantization_utils.py` → INT8 quantization logic  
- `quantized_text2sql_engine.py` → Quantized inference engine  
- `scripts/quantize_export.py` → Model export  

---


##  Key Improvements Over Base System

- Eliminated SQL execution bottleneck  
- Reduced schema hallucination errors  
- Stabilized RL training with dense rewards  
- Improved execution accuracy  
- Enabled efficient CPU deployment  

---

##  Future Work

- Scale to larger models (CodeT5+, LLaMA, Mistral)  
- Integrate RAG for large schemas  
- Support multi-turn conversational SQL  
- Improve reward shaping strategies  

---

##  Note

This module builds upon the base Text-to-SQL system and focuses on **optimization, robustness, and deployment readiness**.

---

##  Author

**Tanisha Jhalani**  
Machine Learning & Systems Engineering

