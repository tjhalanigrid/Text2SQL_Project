# Project 3: Optimization & Advanced RL Enhancements for Text-to-SQL

## Live Links

Technical Report:
https://tjhalanigrid.github.io/text2sql_2/

Live Demo:
https://huggingface.co/spaces/tjhalanigrid/text2sql-demo

---

## Overview

This module extends the base Text-to-SQL system by introducing system-level optimizations, reinforcement learning improvements, and deployment-focused enhancements.

The goal is to transform a slow, error-prone baseline into a scalable, schema-aware, and efficient inference system suitable for real-world deployment.

---

## Key Results

- Achieved over **5× speedup** in SQL execution using parallelization and caching  
- Improved execution accuracy to **~44%** using soft reward shaping  
- Generated **~70%+ schema-valid SQL** via constrained decoding  
- Identified majority of errors as schema-related → guided constraint design  
- Achieved **~2× latency reduction** using INT8 quantization  
- Reduced reward variance → more stable RL training  

---

## Model Checkpoint Setup

To run this project locally, first clone the project from the main repository and download the trained model checkpoint.

### Step 1: Download Checkpoint

https://drive.google.com/drive/folders/1DrsGSO6sHMuX1h2yozqtzWlLqbIXMLVH?usp=sharing

---

### Step 2: Place in Project Directory


project_root/
│
├── checkpoints/
│ └── <model_files_here>


If the `checkpoints/` folder does not exist, create it manually.

---

### Step 3: Verify Structure


project_root/
│
├── app.py
├── checkpoints/
│ ├── config.json
│ ├── pytorch_model.bin
│ ├── tokenizer.json
│ └── ...


---

### Important Notes

- Do **not rename any files**  
- Ensure all required files are present  
- Incorrect paths will cause model loading failure  

---

## Key Contributions

### Task 1: Parallel SQL Execution Optimization
- Multi-threaded rollout execution using ThreadPoolExecutor  
- Connection pooling and query caching  
- Reduced database I/O bottlenecks  

### Task 2: Execution Error Diagnostics & Telemetry
- Built error classification system (wrong_column, wrong_table, etc.)  
- Identified schema hallucination as a dominant failure mode  
- Developed clause-level error analysis  

### Task 3: Schema-Aware Constrained Decoding
- Logit masking using schema constraint graph  
- Prevented invalid table/column generation  
- Improved execution correctness  

### Task 4: Soft Reward Shaping
- Introduced dense reward for partial correctness  
- Improved execution accuracy and training stability  
- Reduced reward variance  

### Task 5: Quantized Inference Benchmarking
- Applied INT8 dynamic quantization for CPU deployment  
- Used mixed precision (decoder-focused quantization)  
- Reduced inference latency without significant accuracy loss  

---

## Results Summary

| Metric | Baseline | Optimized |
|--------|---------|----------|
| Execution Speed | ~1.6s | ~0.3s |
| Execution Accuracy | ~38% | ~44% |
| Schema Validity | Low | ~70%+ |
| Inference Latency | ~3s | ~1.6s |
 (based on random split and best epoch )
 - for better result please refer report - https://tjhalanigrid.github.io/text2sql_2/
 

---

## Core Implementation Mapping (Tasks 1–5)

### Task 01: Parallel SQL Execution
- execution_reward.py → Parallel rollout and reward logic  
- run_sql.py → SQL execution and timeout handling  
- scripts/benchmark_parallel_reward.py → Benchmarking  

### Task 02: Error Diagnostics
- sql_validator.py → SQL validation and error detection  
- execution_reward.py → Runtime error capture  
- app.py → Error classification logic  

### Task 03: Constrained Decoding
- constrained_decoding.py → Logits masking  
- schema_constraints.py → Schema rules  
- schema_encoder.py → Schema representation  

### Task 04: Reward Design
- execution_reward.py → Hard reward  
- execution_reward_soft.py → Soft reward  
- train_rl_codet5_reward_soft.py → Training  

### Task 05: Quantization
- quantization_utils.py → INT8 quantization  
- quantized_text2sql_engine.py → Inference engine  
- scripts/quantize_export.py → Export  

---

## Key Improvements Over Base System

- Eliminated SQL execution bottlenecks  
- Reduced schema hallucination errors  
- Stabilized RL training with dense rewards  
- Improved execution accuracy  
- Enabled efficient CPU deployment  

---

## Future Work

- Scale to larger models (CodeT5+, LLaMA, Mistral)  
- Integrate RAG for large schemas  
- Support multi-turn conversational SQL  
- Further refine reward shaping strategies  

---

## Author

Tanisha Jhalani  
Machine Learning & Systems Engineering