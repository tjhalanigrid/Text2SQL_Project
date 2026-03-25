
import json
from collections import Counter

# ==============================
# LOAD LOGS
# ==============================
with open("results/error_logs.json") as f:
    logs = json.load(f)

total_errors = len(logs)

# ==============================
# ERROR DISTRIBUTION
# ==============================
error_counts = Counter([e["error_type"] for e in logs])

print("\n" + "="*50)
print("📊 TEXT-to-SQL ERROR DASHBOARD")
print("="*50)

print(f"\n🔢 Total Errors Logged: {total_errors}")

print("\n📊 ERROR DISTRIBUTION:")
print("-"*30)
for k, v in error_counts.items():
    percent = (v / total_errors) * 100
    print(f"{k:<20} : {v:>4} ({percent:.1f}%)")

# ==============================
# TOP ERROR
# ==============================
top_error = error_counts.most_common(1)[0]

print("\n🔥 MOST COMMON ERROR:")
print("-"*30)
print(f"{top_error[0]} ({top_error[1]} times)")

# ==============================
# SQL OPERATION ANALYSIS
# ==============================
join_count = 0
where_count = 0
group_count = 0
order_count = 0

for e in logs:
    sql = e["sql"].lower()

    if "join" in sql:
        join_count += 1
    if "where" in sql:
        where_count += 1
    if "group by" in sql:
        group_count += 1
    if "order by" in sql:
        order_count += 1

print("\n🧠 SQL OPERATION ANALYSIS:")
print("-"*30)
print(f"JOIN used in     : {join_count} queries")
print(f"WHERE used in    : {where_count} queries")
print(f"GROUP BY used in : {group_count} queries")
print(f"ORDER BY used in : {order_count} queries")

# ==============================
# SAMPLE ERRORS
# ==============================
print("\n🧪 SAMPLE ERROR CASES:")
print("-"*50)

for i, e in enumerate(logs[:3], 1):
    print(f"\nCase {i}:")
    print(f"Q   : {e['question']}")
    print(f"SQL : {e['sql']}")
    print(f"Type: {e['error_type']}")

# ==============================
# FINAL INSIGHT
# ==============================
print("\n📌 FINAL INSIGHT:")
print("-"*30)

if top_error[0] == "wrong_column":
    print("⚠️ Model struggles with column selection (schema understanding issue).")

elif top_error[0] == "wrong_table":
    print("⚠️ Model struggles with correct table mapping.")

elif top_error[0] == "syntax_error":
    print("⚠️ Model generates invalid SQL syntax.")

else:
    print("⚠️ Mixed errors — needs general improvement.")

print("\n" + "="*50)
print("✅ DASHBOARD COMPLETE")
print("="*50)

