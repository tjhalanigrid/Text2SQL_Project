"""
GRADIO DEMO UI
NL → SQL → Result Table
"""

import gradio as gr
import pandas as pd
import re
import time
import os
from src.text2sql_engine import get_engine

# Safe fallbacks for both the adapter path and base model
fallback_adapter = "checkpoints/best_rlhf_model_2"
if not os.path.exists(fallback_adapter):
    fallback_adapter = "checkpoints/sft_adapter_codet5"

adapter_path = os.environ.get("TEXT2SQL_ADAPTER_PATH", fallback_adapter)
base_model_name = os.environ.get("TEXT2SQL_BASE_MODEL", "Salesforce/codet5-base")
use_lora_env = os.environ.get("TEXT2SQL_USE_LORA", "true").strip().lower()
use_lora = use_lora_env not in {"0", "false", "no"}

print("Text2SQL startup config:")
print(f"- TEXT2SQL_ADAPTER_PATH: {adapter_path}")
print(f"- TEXT2SQL_BASE_MODEL: {base_model_name}")
print(f"- TEXT2SQL_USE_LORA: {use_lora}")

engine = get_engine(
    adapter_path=adapter_path,
    base_model_name=base_model_name,
    use_lora=use_lora,
)

# =========================
# SAMPLE QUESTIONS DATA
# =========================
SAMPLES = [
    ("Show 10 distinct employee first names.", "chinook_1"),
    ("Which artist has the most albums?", "chinook_1"),
    ("List all the tracks that belong to the 'Rock' genre.", "chinook_1"),
    ("What are the names of all the cities?", "flight_1"),
    ("Find the flight number and cost of the cheapest flight.", "flight_1"),
    ("List the airlines that fly out of New York.", "flight_1"),
    ("Which campus was opened between 1935 and 1939?", "csu_1"),
    ("Count the number of students in each department.", "college_2"),
    ("List the names of all clubs.", "club_1"),
    ("How many members does each club have?", "club_1"),
    ("Show the names of all cinemas.", "cinema"),
    ("Which cinema has the most screens?", "cinema")
]

SAMPLE_QUESTIONS = [q[0] for q in SAMPLES]

# =========================
# SQL EXPLAINER
# =========================
def explain_sql(sql):
    if not sql:
        return ""
    explanation = "This SQL query retrieves information from the database."
    sql_lower = sql.lower()

    if "join" in sql_lower:
        explanation += "\n• It combines data from multiple tables using JOIN."
    if "where" in sql_lower:
        explanation += "\n• It filters rows using a WHERE condition."
    if "group by" in sql_lower:
        explanation += "\n• It groups results using GROUP BY."
    if "order by" in sql_lower:
        explanation += "\n• It sorts the results using ORDER BY."
    if "limit" in sql_lower:
        explanation += "\n• It limits the number of returned rows."

    return explanation


# =========================
# CORE FUNCTIONS
# =========================
def run_query(method, sample_q, custom_q, db_id):
    
    # 1. Safely determine the question
    question = sample_q if method == "💡 Pick a Sample" else custom_q

    # 2. Validate inputs before hitting the engine
    if not question or str(question).strip() == "":
        return "-- No input provided", pd.DataFrame(columns=["Warning"]), "⚠️ Please enter a question."
    
    if not db_id or str(db_id).strip() == "":
        return "-- No database selected", pd.DataFrame(columns=["Warning"]), "⚠️ Please select a database."

    start_time = time.time()

    # 3. GIANT SAFETY NET to prevent infinite loading spinners
    try:
        result = engine.ask(str(question), str(db_id))
    except Exception as e:
        err_msg = str(e)
        return f"-- ❌ BACKEND CRASH\n-- {err_msg}", pd.DataFrame(columns=["Error Status"]), f"❌ CRITICAL BACKEND CRASH:\n{err_msg}"

    # Extract outputs safely
    final_sql = result.get("sql", "")
    if not isinstance(final_sql, str):
        final_sql = str(final_sql) if final_sql else ""
        
    error_msg = result.get("error", None)
    rows = result.get("rows", [])
    cols = result.get("columns", [])

    end_time = time.time()
    latency = round(end_time - start_time, 3)

    # 🔥 THE FIX 1: Handle block/error display clearly in the UI
    if error_msg:
        # If no SQL was generated (or it was blocked), show it clearly in the SQL code block
        display_sql = final_sql if final_sql.strip() else f"-- ❌ EXECUTION BLOCKED\n-- The input was flagged or failed validation.\n-- Reason: {error_msg}"
        
        # 🔥 THE FIX 2: Explicitly set a column name to avoid Gradio showing 'undefined' headers
        error_df = pd.DataFrame(columns=["Execution Notice"])
        return display_sql, error_df, f"❌ Error Details:\n{error_msg}"

    # 5. Handle Zero Rows gracefully
    if not rows:
        # Safely handle empty columns to prevent 'undefined'
        safe_cols = cols if (cols and len(cols) > 0) else ["Result"]
        df = pd.DataFrame(columns=safe_cols)
        explanation = f"✅ Query executed successfully\n\nRows returned: 0\nExecution Time: {latency} sec\n\n{explain_sql(final_sql)}"
        return final_sql, df, explanation

    # 6. Handle successful execution
    df = pd.DataFrame(rows, columns=cols)
    actual_rows = len(rows)

    explanation = f"✅ Query executed successfully\n\nRows returned: {actual_rows}\nExecution Time: {latency} sec\n\n{explain_sql(final_sql)}"

    limit_match = re.search(r'LIMIT\s+(\d+)', final_sql, re.IGNORECASE)
    if limit_match:
        requested_limit = int(limit_match.group(1))
        if actual_rows < requested_limit:
            explanation += f"\n\nℹ️ Query allowed up to {requested_limit} rows but only {actual_rows} matched."

    return final_sql, df, explanation


def toggle_input_method(method, current_sample):
    if method == "💡 Pick a Sample":
        # Find the DB matching the current sample (fallback to 'chinook_1')
        db = next((db for q, db in SAMPLES if q == current_sample), "chinook_1")
        return (
            gr.update(visible=True),   # Show sample_dropdown
            gr.update(visible=False),  # Hide type_own_warning
            gr.update(visible=False),  # Hide custom_question
            gr.update(value=db, interactive=False) # Lock and reset db_id
        )
    else:
        return (
            gr.update(visible=False),  # Hide sample_dropdown
            gr.update(visible=True),   # Show type_own_warning
            gr.update(visible=True),   # Show custom_question
            gr.update(interactive=True) # Unlock db_id
        )


def load_sample(selected_question):
    if not selected_question:
        return gr.update()
    db = next((db for q, db in SAMPLES if q == selected_question), "chinook_1")
    return gr.update(value=db)


def clear_inputs():
    return (
        gr.update(value="💡 Pick a Sample"),
        gr.update(value=SAMPLE_QUESTIONS[0], visible=True), # sample_dropdown
        gr.update(visible=False),                           # type_own_warning
        gr.update(value="", visible=False),                 # custom_question
        gr.update(value="chinook_1", interactive=False),    # db_id
        "", pd.DataFrame(), ""                              # Outputs (SQL, Table, Explanation)
    )

def update_schema(db_id):
    if not db_id:
        return ""
    try:
        raw_schema = engine.get_schema(db_id)
        html_output = "<div style='max-height: 250px; overflow-y: auto; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9em; line-height: 1.6;'>"
        for line in raw_schema.strip().split('\n'):
            line = line.strip()
            if not line: continue
            match = re.search(r'^([a-zA-Z0-9_]+)\s*\((.*)\)', line)
            if match:
                table_name = match.group(1).upper()
                columns = match.group(2).lower()
                html_output += f"<div style='margin-bottom: 8px;'><strong style='color: #0f172a; font-size: 1.05em; font-weight: 800;'>{table_name}</strong> <span style='color: #64748b;'>( {columns} )</span></div>"
            else:
                html_output += f"<div style='color: #475569;'>{line}</div>"
        html_output += "</div>"
        return html_output
    except Exception as e:
        return f"<div style='color: red;'>Error loading schema: {str(e)}</div>"


# =========================
# UI LAYOUT
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Text-to-SQL RLHF") as demo:

    gr.HTML(
        """
        <div style="text-align: center; background-color: #e0e7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #c7d2fe;">
            <h1 style="color: #3730a3; margin-top: 0; margin-bottom: 10px; font-size: 2.2em;"> Text-to-SQL using RLHF + Execution Reward</h1>
            <p style="color: #4f46e5; font-size: 1.1em; margin: 0;">Convert Natural Language to SQL, strictly validated and safely executed on local SQLite databases.</p>
        </div>
        """
    )

    DBS = sorted([
        "flight_1", "student_assessment", "store_1", "bike_1", "book_2", "chinook_1",
        "academic", "aircraft", "car_1", "cinema", "club_1", "csu_1",
        "college_1", "college_2", "company_1", "company_employee",
        "customer_complaints", "department_store", "employee_hire_evaluation",
        "museum_visit", "products_for_hire", "restaurant_1",
        "school_finance", "shop_membership", "small_bank_1",
         "student_1", "tvshow", "voter_1", "world_1"
    ])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration & Input")

            input_method = gr.Radio(
                choices=["💡 Pick a Sample", "✍️ Type my own"],
                value="💡 Pick a Sample",
                label="How do you want to ask?"
            )

            # --- SAMPLE SECTION ---
            sample_dropdown = gr.Dropdown(
                choices=SAMPLE_QUESTIONS,
                value=SAMPLE_QUESTIONS[0],
                label="Select a Sample Question",
                info="The database will be selected automatically.",
                visible=True
            )

            # --- CUSTOM TYPE WARNING ---
            type_own_warning = gr.Markdown(
                "**⚠️ Please select a Database first, then type your custom question below:**", 
                visible=False
            )

            gr.Markdown("---")

            # --- DATABASE SELECTION (Moved Up) ---
            db_id = gr.Dropdown(
                choices=DBS,
                value="chinook_1",
                label="Select Database",
                interactive=False 
            )

            # --- CUSTOM QUESTION BOX ---
            custom_question = gr.Textbox(
                label="Ask your Custom Question",
                placeholder="Type your own question here...",
                lines=3,
                visible=False
            )

            gr.Markdown("#### 📋 Database Structure")
            gr.HTML("<p style='font-size: 0.85em; color: #64748b; margin-top: -10px; margin-bottom: 5px;'>Use these exact names! Table names are <strong>Dark</strong>, Column names are <span style='color: #94a3b8;'>Light</span>.</p>")
            schema_display = gr.HTML(value=update_schema("chinook_1"))

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                run_btn = gr.Button(" Generate & Run SQL", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Execution Results")
            final_sql = gr.Code(language="sql", label="Final Executed SQL")
            result_table = gr.Dataframe(label="Query Result Table", interactive=False, wrap=True)
            explanation = gr.Textbox(label="AI Explanation + Execution Details", lines=8)

    # =========================
    # EVENT LISTENERS
    # =========================
    
    input_method.change(
        fn=toggle_input_method, 
        inputs=[input_method, sample_dropdown], 
        outputs=[sample_dropdown, type_own_warning, custom_question, db_id]
    )
    
    sample_dropdown.change(fn=load_sample, inputs=[sample_dropdown], outputs=[db_id])
    
    db_id.change(fn=update_schema, inputs=[db_id], outputs=[schema_display])
    
    run_btn.click(
        fn=run_query,
        inputs=[input_method, sample_dropdown, custom_question, db_id],
        outputs=[final_sql, result_table, explanation]
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[input_method, sample_dropdown, type_own_warning, custom_question, db_id, final_sql, result_table, explanation]
    )

if __name__ == "__main__":
    demo.launch()