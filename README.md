🚀 Text-to-SQL using SFT + RLHF (Spider Benchmark)
📌 Project Overview

This project implements a cross-domain Text-to-SQL system trained on the Spider benchmark dataset.
The system converts natural language questions into executable SQL queries across multiple relational databases.

To improve SQL generation quality, the project explores a two-stage training pipeline:

1️⃣ Supervised Fine-Tuning (SFT)

Learns SQL syntax and query structure from labeled examples.

2️⃣ Reinforcement Learning with Execution Rewards (RLHF / PPO)

Improves logical correctness by rewarding queries that return correct results when executed on the database.

Multiple transformer architectures were evaluated to analyze how different pretraining strategies affect structured SQL generation.

🌐 Live Demo & Project Report

You can explore the project using the following resources.

🖥 Interactive Text-to-SQL Demo (Gradio Interface)

Try converting natural language → SQL → database results

🔗 Live Demo:
https://huggingface.co/spaces/tjhalanigrid/text2sql-demo

The demo allows users to:

Enter natural language questions

Automatically generate SQL queries

View the generated SQL statement

Execute the query on a SQLite database

See the resulting table returned by the query

📄 Project Report & Experimental Analysis

🔗 Project Report:
https://tjhalanigrid.github.io/Text2SQL_Project/

The report includes:

System architecture

Training pipeline (SFT + RLHF)

Model comparisons

Execution accuracy evaluation

Reinforcement learning analysis

Training curves and visualizations

Demo screenshots

🎯 Objectives

The main objectives of this project are:

Generate accurate and executable SQL queries from natural language questions.

Evaluate the impact of execution-based reinforcement learning.

Compare different transformer architectures for structured query generation.

Build an interactive inference interface for real-time Text-to-SQL conversion.

🚀 Key Features
Multi-Model Experimentation

Three transformer architectures were evaluated:

T5-Small

BART-Base

CodeT5-Base

These models represent different pretraining strategies, allowing comparison of their ability to generate structured SQL.

Execution-Based Reinforcement Learning

Generated SQL queries are executed against SQLite databases to compute reward signals.

Reward mechanism:

+1 reward if predicted SQL returns the same result as the ground truth query

Penalty for invalid or incorrect SQL queries

This encourages the model to produce logically correct and executable SQL.

Schema-Aware Prompting

The database schema is explicitly serialized during input construction.

Example
Tables:
employee(id, name, city)
department(id, department_name)

Question:
Show employees who live in Calgary

This improves schema grounding and reduces hallucinated SQL queries.

Evaluation & Analysis

Model performance is evaluated using Spider-style execution accuracy.

The project includes:

Execution accuracy evaluation

Custom evaluation scripts

Training curve visualization

Cross-model performance comparison

📁 Project Structure
text2sql_project/

├── src/                     # Training, evaluation, inference scripts
│
├── data/                    # Spider dataset and SQLite databases
│
├── checkpoints/             # SFT and RLHF trained model adapters
│
├── outputs/                 # Intermediate RL outputs and logs
│
├── comparison_plots/        # Training curve generation scripts
│
├── docs/                    # HTML report and visualizations
│
├── spider_eval/             # Official Spider evaluation scripts
│
├── app.py                   # Gradio interactive demo interface
│
└── README.md                # Project documentation
🧠 Models Evaluated
Model	Pretraining Type	Parameters
T5-Small	General text-to-text transformer	~60M
BART-Base	Denoising sequence-to-sequence model	~139M
CodeT5-Base	Code-pretrained transformer	~220M

CodeT5 achieved the best performance because its code-oriented pretraining helps generate structured outputs such as SQL queries.

📊 Main Results (Execution Accuracy)
Model	SFT Accuracy	RLHF Accuracy
T5-Small	9.0%	8.3%
BART-Base	24.0%	21.23%
CodeT5-Base	41.7%	37.9%

Although RLHF slightly reduced execution accuracy in some cases, it improved semantic alignment and logical correctness, confirmed through manual inspection of generated queries.

🧪 Training Pipeline
Stage 1 — Supervised Fine-Tuning (SFT)

Models are trained using cross-entropy loss against ground-truth SQL queries.

Run Training
python src/train_sft.py
python src/train_sft_codet5.py
python src/train_sft_bart.py
Stage 2 — Reinforcement Learning (RLHF / PPO)

The model is further optimized using execution-based reward signals.

Reward Logic
+1 if predicted SQL returns same result as ground truth
Penalty for invalid SQL
Training Scripts
python src/train_rl.py
python src/train_rl_codet5.py
python src/train_rl_bart.py
📈 Evaluation

Evaluate execution accuracy using:

python src/evaluate_model_codet5.py \
--adapter checkpoints/sft_adapter_codet5 \
--num_samples 1000

For RLHF-trained models:

python src/eval_rl_fixed.py \
--adapter checkpoints/best_rlhf_model
📊 Visualization

Generate training curves and comparisons:

python comparison_plots/parse_and_plot.py --window 7

Generated plots include:

SFT training accuracy curves

RLHF reward progression

Cross-model performance comparisons

🖥 Running the Demo Locally

Run the interface locally:

python app.py

Then open:

http://127.0.0.1:7860

The interface allows users to enter natural language questions, generate SQL queries, and view database results.

✨ Author

Tanisha Jhalani
B.Tech Mechanical Engineering
Machine Learning & Systems Project

💡 Future Improvements

Possible extensions include:

Training larger models (T5-Large, CodeT5+)

Improved reward shaping strategies for RLHF

Enhanced schema linking mechanisms

Support for multi-turn conversational SQL queries

Deployment with optimized inference pipelines
