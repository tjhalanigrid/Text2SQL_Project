# Text-to-SQL using SFT + RLHF (Spider Benchmark)

## 📌 Project Overview
This project implements a **cross-domain Text-to-SQL system** trained on the **Spider benchmark dataset**.

The system converts **natural language questions into executable SQL queries** across multiple relational databases.

To improve SQL generation quality, the project explores a **two-stage training pipeline**:

### 1️⃣ Supervised Fine-Tuning (SFT)
Learns SQL syntax and query structure from labeled examples.

### 2️⃣ Reinforcement Learning with Execution Rewards (RLHF / PPO)
Improves logical correctness by rewarding queries that return correct results when executed on the database.

Multiple transformer architectures were evaluated to understand how **different pretraining strategies affect structured SQL generation**.

---

## 🌐 Live Demo

Try the interactive Text-to-SQL system here:

🔗 **Gradio Demo**  
https://huggingface.co/spaces/tjhalanigrid/text2sql-demo

The demo allows users to:

- Enter **natural language questions**
- Generate **SQL queries automatically**
- View the **generated SQL query**
- Execute queries on **SQLite databases**
- View the **result table returned by the query**

---

## 📄 Project Report

Full project report with architecture explanation, experiments, and evaluation results:

🔗 https://tjhalanigrid.github.io/Text2SQL_Project/

The report includes:

- System architecture
- Training pipeline (SFT + RLHF)
- Model comparisons
- Execution accuracy evaluation
- Reinforcement learning analysis
- Training curves and visualizations
- Demo screenshots

---

## 🚀 Features

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

## 📁 Project Structure

```
text2sql_project
│
├── src/                     # Training, evaluation, inference scripts
│
├── data/                    # Spider dataset and SQLite databases
│
├── outputs/                 # RL outputs and logs
│
├── comparison_plots/        # Training curve visualization
│
├── docs/                    # HTML report and figures
│
├── spider_eval/             # Official Spider evaluation scripts
│
├── app.py                   # Gradio interactive demo
│
├── requirements.txt         # Project dependencies
│
└── README.md                # Project documentation
```

---

## 🧠 Models Evaluated

| Model | Pretraining Type | Parameters |
|------|------------------|-----------|
| CodeT5-Base | Code-pretrained transformer | ~220M |
| T5-Small | General text-to-text transformer | ~60M |
| T5-Base | General text-to-text transformer | ~220M |
| BART-Base | Denoising sequence-to-sequence model | ~139M |

**CodeT5 achieved the best performance** because its code-oriented pretraining improves structured SQL generation.

---

## 📊 Results (Execution Accuracy)

| Model | SFT Accuracy | RLHF Accuracy |
|------|--------------|---------------|
| T5-Small | 9.0% | 8.3% |
| BART-Base | 24.0% | 21.23% |
| CodeT5-Base | **41.7%** | **37.9%** |

Although RLHF slightly reduced execution accuracy in some cases, it improved **semantic alignment and logical reasoning**, confirmed through manual evaluation.

---

## 🧪 Training Pipeline

### Stage 1 — Supervised Fine-Tuning (SFT)

Models are trained using **cross-entropy loss** against ground-truth SQL queries.

Run training:

```bash
python src/train_sft.py
python src/train_sft_codet5.py
python src/train_sft_bart.py
```

---

### Stage 2 — Reinforcement Learning (RLHF / PPO)

The model is further optimized using **execution-based rewards**.

Reward logic:

```
+1 if predicted SQL returns the same result as the ground truth query
Penalty for invalid SQL
```

Training scripts:

```bash
python src/train_rl.py
python src/train_rl_codet5.py
python src/train_rl_bart.py
```

---

## 📈 Evaluation

Evaluate execution accuracy using:

```bash
python src/evaluate_model_codet5.py \
--adapter checkpoints/sft_adapter_codet5 

```

For RLHF models:

```bash
python src/eval_rl_fixed.py \
--adapter checkpoints/best_rlhf_model
```

---

## 📊 Visualization

Generate training curves and comparisons:

```bash
python comparison_plots/parse_and_plot.py --window 7
```

Generated plots include:

- SFT training accuracy curves
- RLHF reward progression
- Cross-model performance comparisons

---



## 📌 Example Query

Example question you can try:

```
Show the names of employees who live in Calgary
```

The system will generate the SQL query and execute it on the database.

---

## 💡 Future Improvements

- Train larger models (**T5-Large, CodeT5+**)
- Improve reward shaping strategies for RLHF
- Better schema linking techniques
- Support multi-turn conversational queries
- Deploy optimized inference pipeline

---

## ✨ Author

**Tanisha Jhalani**   
Machine Learning & Systems Project
