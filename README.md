# Text-to-SQL using SFT + RLHF (Spider Benchmark)

##  Project Overview
This project implements a **cross-domain Text-to-SQL system** trained on the **Spider benchmark dataset**.

The system converts **natural language questions into executable SQL queries** across multiple relational databases.

To improve SQL generation quality, the project explores a **two-stage training pipeline**:

### 1️⃣ Supervised Fine-Tuning (SFT)
Learns SQL syntax and query structure from labeled examples.

### 2️⃣ Reinforcement Learning with Execution Rewards (RLHF / PPO)
Improves logical correctness by rewarding queries that return correct results when executed on the database.

Multiple transformer architectures were evaluated to understand how **different pretraining strategies affect structured SQL generation**.

---

##  Live Demo

Try the interactive Text-to-SQL system here:

 **Gradio Demo**  
https://huggingface.co/spaces/tjhalani/text2sql_final_space

The demo allows users to:

- Enter **natural language questions**
- Generate **SQL queries automatically**
- View the **generated SQL query**
- Execute queries on **SQLite databases**
- View the **result table returned by the query**

---

##  Project Report 2

Full project report with architecture explanation, experiments, and evaluation results:

 https://tjhalanigrid.github.io/Text2SQL_Project/

The report includes:

- System architecture
- Training pipeline (SFT + RLHF)
- Model comparisons
- Execution accuracy evaluation
- Reinforcement learning analysis
- Training curves and visualizations
- Demo screenshots

---

##  Project Report 3

Full technical report covering optimization pipeline, reinforcement learning improvements, and deployment enhancements:

👉 https://tjhalanigrid.github.io/text2sql_2/

The report includes:

- Parallel SQL execution optimization (5.6× speedup)
- Execution error analysis and telemetry dashboard
- Schema-aware constrained decoding
- Hard vs soft reward comparison
- Quantized inference benchmarking (INT8, mixed precision)
- System-level performance improvements and analysis
- End-to-end evaluation and insights

---


##  Features

- **Multi-Model Experimentation**  
  Experiments with **T5-Small, BART-Base, and CodeT5-Base**

- **Execution-Based Reinforcement Learning**  
  SQL queries are executed against SQLite databases and rewarded based on correctness.

- **Schema-Aware Prompting**  
  Database schemas are serialized to improve grounding.

- **Evaluation Pipeline**  
  Includes Spider-style execution accuracy evaluation.

- **Interactive Demo Interface**  
  Natural language → SQL → database results using Gradio.

---

##  Project Structure

```
Text2SQL_Project
│
├── src/                      # Core training, evaluation and inference utilities
│
├── data/                     # Spider dataset and SQLite databases
│
├── outputs/                  # RLHF outputs, logs and training checkpoints
│
├── comparison_plots/         # Training curve visualizations and performance graphs
│
├── docs/                     # HTML report, figures and project documentation
│
├── spider_eval/              # Official Spider evaluation scripts
│
├── checkpoints/              # All models checkpoints 
│
├── experiments/              # Best model checkpoints and experiment configurations
│
├── scripts                   # for evaluation of final model  
| 
├── project3                  # Project3 readme and scripts 
│
├── app.py                    # Gradio interactive demo (Natural Language → SQL)
│
└── README.md                 # Complete project documentation
```

---

##  Repository Files & Folders (Detailed)

- `.vscode/`  
  Local editor settings for VS Code (workspace preferences, optional).

- `checkpoints/`  
  Saved LoRA adapters and model checkpoints used by training/evaluation (`best_rlhf_model`, SFT/RL snapshots).

- `comparison_plots/`  
  Parsed metrics JSON/CSV and generated model comparison plots.

- `data/`  
  Spider data artifacts, evaluation files (`dev.json`, `tables.json`, gold SQL), and selected SQLite databases.

- `docs/`  
  Static project report site files (HTML, CSS, images/figures).

- `experiments/v1_codet5_rlhf/`  
  Experiment-specific RLHF assets/config/results for CodeT5 pipeline.

- `outputs/`  
  Intermediate and epoch-wise output artifacts from training/evaluation runs.

- `scripts/`  
  Utility scripts for evaluation and report/plot generation.

- `spider_eval/`  
  Official Spider evaluation utilities used for exact/execution metrics.

- `src/`  
  Main source code: training, RLHF, inference engine, evaluation, prompt/schema helpers.

- `.gitignore`  
  Git tracking rules for local artifacts, checkpoints, and selected include/exclude paths.

- `.nojekyll`  
  Ensures GitHub Pages serves files without Jekyll processing.

- `README.md`  
  Project documentation (setup, run commands, evaluation, troubleshooting).

- `app.py`  
  Gradio UI entrypoint for Text-to-SQL demo (question -> SQL -> execution results).

- `pred.sql`  
  Generated prediction file produced by evaluation scripts (run artifact; may be regenerated).
- `project3`  
  dedicated folder for project 3 and it's readme file 

- `requirment.txt`  
  Python dependency list used for environment setup.
----
----

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
#### (Task 1 ,2 ,3 ,4 , 5 is integrated with app.py)
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


### Notes for new users

- If adapter files are present in `checkpoints/best_rlhf_model`, UI uses them.
- If not, app automatically falls back to:
  - `experiments/v1_codet5_rlhf/best_rlhf_model`
  - then base model (no LoRA) if adapter is unavailable.
- You can force adapter path with:
  - `export TEXT2SQL_ADAPTER_PATH=experiments/v1_codet5_rlhf/best_rlhf_model`
- Runtime artifacts are written under `outputs/` and `comparison_plots/`.

### Task 5: Quantization Benchmark
```bash
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/fp32 --mode fp32 --device cpu
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/int8_dynamic --mode int8_dynamic --device cpu
python -m scripts.quantize_export --base_model "Salesforce/codet5-base" --out_dir checkpoints/task5/int8_decoder_dynamic --mode int8_decoder_dynamic --device cpu
```
----
----


### Troubleshooting

1. `PermissionError` from `/opt/miniconda3/bin/pip`
   Use venv pip only:
   ```bash
   source venv/bin/activate
   python -m pip install -r requirment.txt
   ```
2. Adapter path / `HFValidationError`
   Set adapter path explicitly:
   ```bash
   export TEXT2SQL_ADAPTER_PATH=checkpoints/best_rlhf_model
   python app.py
   ```
3. `unable to open database file`
   Ensure DB file exists for selected DB (example `chinook_1`):
   ```bash
   ls data/database/chinook_1/chinook_1.sqlite
   ```
4. LoRA not loading (`Can't find weights...adapter_model.bin`)
   Verify adapter weights exist:
   ```bash
   ls checkpoints/best_rlhf_model/adapter_model.bin
   ```

---

##  Core Tasks
## 📌 Project Tasks Breakdown

---

### 🔹 Task 1: Data Preparation

- Prepare Spider JSON and SQLite artifacts needed for training and evaluation  
- Validate schema/database paths and ensure required DBs exist locally  
- Build clean prompt-ready examples for SFT and RLHF pipelines  

---

### 🔹 Task 2: Train and Run Text-to-SQL Inference

- Fine-tune SFT and RLHF models  
- Generate SQL from natural language questions  
- Execute SQL safely and return result tables  

---

### 🔹 Task 3: Run Controlled Model Comparisons

- Compare SFT vs RLHF checkpoints  
- Compare model families (T5, BART, CodeT5)  
- Use consistent evaluation pipelines  

---

### 🔹 Task 4: Evaluate Quality with Benchmark Metrics

- Measure execution accuracy and exact match (Spider evaluation)  
- Track per-model performance  
- Generate aggregate comparison artifacts  

---

### 🔹 Task 5: Presentation Layer

- Launch interactive Gradio demo  
- Regenerate plots and documentation visuals  

---

## 🚀 Advanced Optimization Tasks (Project 3)

---

### 🔹 Task 1: Parallel SQL Execution with Connection Pooling

**Instructions:**
- Use `ThreadPoolExecutor` for parallel SQLite execution (10–20 DBs concurrently)  
- Implement connection pooling to reuse SQLite connections  
- Add query timeout (2s) with safe exception handling  
- Cache query results using hash (query + DB state)  
- Measure execution time: sequential vs parallel (100 rollouts)  
- Profile CPU usage and identify bottlenecks  

**Benefit:**  
🚀 Reduces reward computation bottleneck by **5–10×** and enables larger batch sizes  

---

### 🔹 Task 2: Execution-Guided Reward Attribution & Error Analysis

**Instructions:**
- Capture and classify SQL errors (JOIN, WHERE, NULL handling, etc.)  
- Use attribution methods to identify prompt tokens causing errors  
- Build error dashboard: type → count → examples → fixes  
- Implement reward-based hint mechanism  
- Create adversarial test cases  
- Visualize performance across SQL operations  

**Benefit:**  
📊 Improves interpretability and enables targeted debugging & reward shaping  

---

### 🔹 Task 3: Schema-Aware Constrained Decoding

**Instructions:**
- Parse schema into graph (tables, columns, relationships)  
- Enforce valid SQL generation using constraints  
- Apply grammar-based decoding rules  
- Mask invalid tokens during generation  
- Measure constraint satisfaction rate  
- Compare constrained vs unconstrained decoding  

**Benefit:**  
🧠 Reduces invalid SQL and improves semantic correctness  

---

### 🔹 Task 4: Execution Reward vs Soft Reward Comparison

**Instructions:**
- Implement:
  - Hard reward → exact execution match  
  - Soft reward → partial row match  
- Train models under both reward schemes  
- Compare convergence, variance, accuracy  
- Analyze interpretability and stability  

**Benefit:**  
⚖️ Soft rewards provide denser signals → better training stability  

---

### 🔹 Task 5: Quantized SQL Code Model for Efficient Inference

**Instructions:**
- Quantize models (T5 / CodeT5) to INT8 / INT4  
- Compare performance: fp32 vs int8 vs int4  
- Use mixed precision (encoder fp32, decoder int8)  
- Build lightweight CPU model (<300MB)  
- Benchmark rollout + inference speed  
- Deploy optimized inference pipeline  

**Benefit:**  
⚡ Faster inference + efficient deployment on CPU (Mac-friendly)  

---

##  Key Results Snapshot

## 📈 Results Summary

> *(Evaluated on 500 examples. LoRA applied for best-performing model checkpoints.)*  
> ⚠️ *Note: The reported results correspond to the best-performing model and configuration in this project. Performance may vary across different runs, models, and datasets.*

| Model         | SFT Accuracy | RLHF Accuracy |
|--------------|------------|--------------|
| T5-Small     | 9.0%       | 8.3%         |
| BART-Base    | 24.0%      | 21.23%       |
| CodeT5-Base  | 41.7%      | 37.9%        |

### 🏆 Key Observations

- Best model family in this project: **CodeT5-Base**  
- Best reported SFT execution accuracy: **41.7%**  
- Best reported RLHF execution accuracy: **37.9%**  
- RLHF improved semantic behavior in manual checks, although execution accuracy varied across models  

---

## 🔮 Future Improvements

- Train larger models (CodeT5+, LLaMA)  
- Improve reward shaping strategies  
- Enhance schema linking techniques  
- Support multi-turn conversational SQL  
---

##  Training & Inference

### Train

```bash
python src/train_sft.py
python src/train_sft_codet5.py
python src/train_sft_bart.py
python src/train_rl.py
python src/train_rl_codet5.py
python src/train_rl_bart.py
```

### Inference (example)

```bash
python src/generate_sql.py \
  --question "Show the names of employees who live in Calgary" \
  --db_id chinook_1
```


---




## 🖥️ Gradio UI (Inference)

Main UI file: `app.py`

Launch locally:

```bash
python app.py
```

The UI supports:
- Natural language question input
- Database selection
- Generated SQL display
- Query execution and tabular results

---

##  Plot Dashboard

Generate/rebuild comparison plots:

```bash
python comparison_plots/parse_and_plot.py --window 7
```

Open report/dashboard:

```bash
open docs/index.html
```

---

##  Recommended Production Config

Use this for stable deployment:
- model/adapter: `checkpoints/best_rlhf_model_2`
- base model: `Salesforce/codet5-base`
- decoding mode: deterministic (`do_sample=False`, beam search)
- safety: SQL validation enabled in `src/sql_validator.py`

---

##  Models Evaluated

| Model | Pretraining Type | Parameters |
|------|------------------|-----------|
| CodeT5-Base | Code-pretrained transformer | ~220M |
| T5-Small | General text-to-text transformer | ~60M |
| T5-Base | General text-to-text transformer | ~220M |
| BART-Base | Denoising sequence-to-sequence model | ~139M |

**CodeT5 achieved the best performance** because its code-oriented pretraining improves structured SQL generation.

---

##  Results (Execution Accuracy)

| Model | SFT Accuracy | RLHF Accuracy |
|------|--------------|---------------|
| T5-Small | 9.0% | 8.3% |
| BART-Base | 24.0% | 21.23% |
| CodeT5-Base | **41.7%** | **37.9%** |
(based on random split and best epoch)

Although RLHF slightly reduced execution accuracy in some cases, it improved **semantic alignment and logical reasoning**, confirmed through manual evaluation.

---

##  Training Pipeline

### Stage 1 — Supervised Fine-Tuning (SFT)

Models are trained using **cross-entropy loss** against ground-truth SQL queries.

---

### Stage 2 — Reinforcement Learning (RLHF / PPO)

The model is further optimized using **execution-based rewards**.


---

##  Evaluation

Evaluate execution accuracy using:


---

##  Visualization

Generate training curves and comparisons:


Generated plots include:

- SFT training accuracy curves
- RLHF reward progression
- Cross-model performance comparisons

---



##  Example Query

Example question you can try:

```
Show the names of employees who live in Calgary
```

The system will generate the SQL query and execute it on the database.

---

##  Future Improvements

- Train larger models (**T5-Large, CodeT5+**)
- Improve reward shaping strategies for RLHF
- Better schema linking techniques
- Support multi-turn conversational queries
- Deploy optimized inference pipeline

---

##  Author

**Tanisha Jhalani**   
Machine Learning & Systems Project
