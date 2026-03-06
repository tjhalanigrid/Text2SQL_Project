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
https://huggingface.co/spaces/tjhalanigrid/text2sql-demo

The demo allows users to:

- Enter **natural language questions**
- Generate **SQL queries automatically**
- View the **generated SQL query**
- Execute queries on **SQLite databases**
- View the **result table returned by the query**

---

##  Project Report

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

- `requirment.txt`  
  Python dependency list used for environment setup.

---

##  Installation & Setup

### Prerequisites

- Python 3.10+
- macOS/Linux (MPS/CUDA/CPU supported via PyTorch setup)

### Install dependencies

```bash
python -m pip install -r requirment.txt
```

### Activate venv (recommended)

```bash
source venv/bin/activate
```

## ⚡ Run On Another Computer (Quick Start)

Use these exact commands on a fresh machine.

### 1. Clone and enter project

```bash
git clone https://github.com/tjhalanigrid/Text2SQL_Project.git
cd Text2SQL_Project
```

### 2. Create environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirment.txt
```

### 3. Run UI directly (no retraining required)

```bash
export TEXT2SQL_ADAPTER_PATH=checkpoints/best_rlhf_model
python app.py
```

### 4. Rebuild and open analytics dashboard

```bash
python comparison_plots/parse_and_plot.py --window 7
open docs/index.html
```

### Notes for new users

- If adapter files are present in `checkpoints/best_rlhf_model`, UI uses them.
- If not, app automatically falls back to:
  - `experiments/v1_codet5_rlhf/best_rlhf_model`
  - then base model (no LoRA) if adapter is unavailable.
- You can force adapter path with:
  - `export TEXT2SQL_ADAPTER_PATH=experiments/v1_codet5_rlhf/best_rlhf_model`
- Runtime artifacts are written under `outputs/` and `comparison_plots/`.

### If you already cloned before updates

```bash
cd Text2SQL_Project
git pull
source venv/bin/activate
export TEXT2SQL_ADAPTER_PATH=checkpoints/best_rlhf_model
python app.py
```

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

## ✅ Core Tasks

🔹 **Task 1: Data preparation**
- Prepare Spider JSON and SQLite artifacts needed for training and evaluation.
- Validate schema/database paths and ensure required DBs exist locally.
- Build clean prompt-ready examples for SFT and RLHF pipelines.

🔹 **Task 2: Train and run Text-to-SQL inference**
- Fine-tune SFT models and RLHF models.
- Generate SQL from natural language questions for selected Spider databases.
- Execute generated SQL safely and return result tables.

🔹 **Task 3: Run controlled model comparisons**
- Compare SFT vs RLHF checkpoints.
- Compare model families (T5, BART, CodeT5) using the same evaluation flow.

🔹 **Task 4: Evaluate quality with benchmark metrics**
- Measure execution accuracy and exact match using Spider evaluation.
- Track per-model performance and aggregate comparison artifacts.

🔹 **Task 5: Presentation layer**
- Launch interactive Gradio demo for natural language to SQL.
- Regenerate plots and documentation visuals for reporting.

---

## 📊 Key Results Snapshot

- Best model family in this project: **CodeT5-Base**.
- Best reported SFT execution accuracy: **41.7%**.
- Best reported RLHF execution accuracy: **37.9%**.
- RLHF improved semantic behavior in manual checks while execution accuracy varied by model.

---

## 🏋️ Training & Inference

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

## 📈 Plot Dashboard

Generate/rebuild comparison plots:

```bash
python comparison_plots/parse_and_plot.py --window 7
```

Open report/dashboard:

```bash
open docs/index.html
```

---

## 🎯 Recommended Production Config

Use this for stable deployment:
- model/adapter: `checkpoints/best_rlhf_model`
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
