import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. YOUR DATA
# ==========================================
models = ['FP32 (Base)', 'INT8 Dynamic', 'INT8 Decoder-Only']

# Accuracy (multiplied by 100 for percentage)
accuracy = [36.0, 36.0, 38.0]

# Latency metrics
lat_mean = [3.11, 1.65, 1.66]
lat_p50  = [2.94, 1.54, 1.56]
lat_p90  = [4.64, 2.44, 2.48]

# ==========================================
# 2. SET UP THE SIDE-BY-SIDE LAYOUT
# ==========================================
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Model Accuracy (Execution)</b>", 
        "<b>Inference Latency Profile</b>"
    ),
    horizontal_spacing=0.1
)

# ==========================================
# 3. LEFT CHART: ACCURACY
# ==========================================
fig.add_trace(go.Bar(
    x=models, 
    y=accuracy,
    name="Execution Accuracy",
    marker_color=['#94a3b8', '#38bdf8', '#10b981'], # Gray, Blue, Green
    text=[f"{val:.1f}%" for val in accuracy],
    textposition='auto',
    textfont=dict(size=14, color='white', family="Arial Black"),
    showlegend=False
), row=1, col=1)

# ==========================================
# 4. RIGHT CHART: LATENCY PROFILE
# ==========================================
# P50 Latency
fig.add_trace(go.Bar(
    x=models, y=lat_p50,
    name="Median (P50)",
    marker_color="#ece80a" # Light Blue
), row=1, col=2)

# Mean Latency
fig.add_trace(go.Bar(
    x=models, y=lat_mean,
    name="Mean Latency",
    marker_color="#3b4da9" # Standard Blue
), row=1, col=2)

# P90 Latency
fig.add_trace(go.Bar(
    x=models, y=lat_p90,
    name="90th Percentile (P90)",
    marker_color="#d974e2" # Dark Blue
), row=1, col=2)

# ==========================================
# 5. APPLY ULTRA-MODERN STYLING
# ==========================================
fig.update_layout(
    title=dict(
        text="<b>Task 5: FP32 vs. INT8 Quantization Performance</b>",
        font=dict(size=22, color='#1e293b'),
        x=0.5
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    barmode='group',
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.05,
        xanchor="center", x=0.8,
        bgcolor='rgba(255,255,255,0.8)'
    ),
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"),
    margin=dict(t=120, b=60, l=60, r=40)
)

# Style Left Axes
fig.update_yaxes(title_text="<b>Accuracy (%)</b>", range=[0, 45], gridcolor='#f1f5f9', row=1, col=1)
fig.update_xaxes(tickfont=dict(weight='bold'), row=1, col=1)

# Style Right Axes
fig.update_yaxes(title_text="<b>Seconds per Query</b>", gridcolor='#f1f5f9', row=1, col=2)
fig.update_xaxes(tickfont=dict(weight='bold'), row=1, col=2)

# ==========================================
# 6. RENDER AND SAVE
# ==========================================
html_file = "task5_quantization_dashboard.html"
fig.write_html(html_file)
print(f"✅ Interactive Plotly Dashboard saved to: {html_file}")
fig.show()