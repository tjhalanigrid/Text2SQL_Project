import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. EXTRACTED DATA FROM TERMINAL
# ==========================================
# Error Distribution Data
error_types = ['wrong_column', 'wrong_table', 'ambiguous_column', 'other']
error_counts = [61, 11, 4, 1]

# SQL Operation Analysis Data
sql_ops = ['WHERE', 'JOIN', 'ORDER BY', 'GROUP BY']
op_counts = [55, 36, 20, 14]

# ==========================================
# 2. SET UP THE DASHBOARD LAYOUT
# ==========================================
# Use a clean, modern aesthetic
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 3. PLOT 1: ERROR DISTRIBUTION (Horizontal Bar)
# ==========================================
sns.barplot(x=error_counts, y=error_types, ax=ax1, palette="flare")
ax1.set_title('Primary Cause of Failure (Total: 77 Errors)', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('Number of Queries')
ax1.set_ylabel('')

# Add actual numbers next to the bars
for i, v in enumerate(error_counts):
    ax1.text(v + 1.5, i, f"{v}", color='#333333', va='center', fontweight='bold')

# ==========================================
# 4. PLOT 2: SQL OPERATIONS (Vertical Bar)
# ==========================================
sns.barplot(x=sql_ops, y=op_counts, ax=ax2, palette="crest")
ax2.set_title('Clauses Present in Failed Queries', fontsize=14, pad=15, fontweight='bold')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('')

# Add actual numbers on top of the bars
for i, v in enumerate(op_counts):
    ax2.text(i, v + 1, str(v), color='#333333', ha='center', fontweight='bold')

# ==========================================
# 5. RENDER AND SAVE
# ==========================================
plt.suptitle('Text-to-SQL Error Diagnostic Dashboard', fontsize=18, fontweight='heavy', y=1.05)
sns.despine(left=True, bottom=True) # Removes clunky borders
plt.tight_layout()

# Save the plot as a high-res image for your report!
plt.savefig('error_diagnostic_plot.png', dpi=300, bbox_inches='tight')
print("✅ Plot successfully saved as 'error_diagnostic_plot.png'")

# Display the plot
plt.show()