<!-- # Project 3: Optimization & Advanced RL Enhancements for Text-to-SQL

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
Machine Learning & Systems Engineering -->


# Text-to-SQL using SFT + RLHF (Spider Benchmark)

## Project Overview

This project implements a **cross-domain Text-to-SQL system** trained on the **Spider benchmark dataset**.

The system converts **natural language questions into executable SQL queries** across multiple relational databases.

To improve SQL generation quality, the project explores a **two-stage training pipeline**:

### 1️⃣ Supervised Fine-Tuning (SFT)
Learns SQL syntax and query structure from labeled examples.

### 2️⃣ Reinforcement Learning with Execution Rewards (RLHF / PPO)
Improves logical correctness by rewarding queries that return correct results when executed on the database.

Multiple transformer architectures were evaluated to understand how **different pretraining strategies affect structured SQL generation**.

---

## 🚀 Live Demo

**Gradio Demo:**  
https://huggingface.co/spaces/tjhalanigrid/text2sql-demo

Features:
- Enter **natural language questions**
- Generate **SQL queries automatically**
- Execute queries on **SQLite databases**
- View **result tables**

---

## 📄 Project Reports

**Report 2:**  
https://tjhalanigrid.github.io/Text2SQL_Project/

**Report 3:**  
https://tjhalanigrid.github.io/text2sql_2/

---

## ✨ Features

- Multi-model experimentation (T5, BART, CodeT5)
- Execution-based RL training
- Schema-aware prompting
- Spider-style evaluation pipeline
- Interactive Gradio interface

---

## 📁 Project Structure


Text2SQL_Project/
│
├── src/ # Core logic
├── data/ # Spider dataset & DBs
├── outputs/ # RLHF outputs
├── comparison_plots/ # Graphs
├── docs/ # Reports
├── spider_eval/ # Evaluation scripts
├── checkpoints/ # Models
├── experiments/ # Experiments
├── scripts/ # Utility scripts
├── app.py # Demo app
└── README.md


---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- macOS/Linux (MPS/CUDA/CPU supported)



### ⚡ Quick Start (New Machine)
### 1. Clone repo
```bash
git clone https://github.com/tjhalanigrid/Text2SQL_Project.git
cd Text2SQL_Project
```
### 2. Setup environment
```bash 
python3 -m venv venv
source venv/bin/activate   
pip install --upgrade pip setuptools wheel
pip install -r requirement.txt
```
### 🧠 Task 5 (IMPORTANT - Required before running app)
```bash
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/fp32 --mode fp32 --device cpu
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/int8_dynamic --mode int8_dynamic --device cpu
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/int8_decoder_dynamic --mode int8_decoder_dynamic --device cpu
```

### ▶️ Run App
```bash
export TEXT2SQL_ADAPTER_PATH=checkpoints/best_rlhf_model
python app.py
```

### 📊 Analytics Dashboard
```bash
python comparison_plots/parse_and_plot.py --window 7
open docs/index.html
```

### 🔬 Project 3 Tasks
### Task 1: Parallel Execution
```bash 
python project3/benchmark_parallel_reward.py
```

### Task 2: Error Dashboard
```bash
python scripts/error_dashboard.py
```
### Task 3: Constrained Decoding
```bash
python project3/eval_with_without_constraints.py \
  --adapter_unconstrained checkpoints/best_rlhf_model \
  --adapter_constrained checkpoints/best_rlhf_model_2
```

### Task 4: Reward Comparison
```bash
python project3/eval_task4_rewards.py \
  --adapter_hard checkpoints/best_rlhf_model \
  --adapter_soft checkpoints/best_rlhf_model_2
```
### Task 5: Quantization Benchmark
```bash
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/fp32 --mode fp32 --device cpu
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/int8_dynamic --mode int8_dynamic --device cpu
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/int8_decoder_dynamic --mode int8_decoder_dynamic --device cpu
```
# 📈 Results Summary
Model	SFT Accuracy	RLHF Accuracy
T5-Small	9.0%	8.3%
BART-Base	24.0%	21.23%
CodeT5-Base	41.7%	37.9%
🔮 Future Improvements
Train larger models (CodeT5+, LLaMA)
Improve reward shaping
Better schema linking
Multi-turn SQL support
### (on 500 examples and lora is working)

---
### 👩‍💻 Author

Tanisha Jhalani
Machine Learning & Systems Project



---
