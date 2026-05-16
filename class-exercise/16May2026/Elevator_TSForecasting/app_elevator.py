import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Elevator Traffic Forecasting Pipeline", layout="wide")

st.title("📊 Hybrid Elevator Traffic Forecasting Framework")
st.markdown("""
This dashboard simulates the step-by-step data pipeline for an ultra-large skyscraper elevator bank using a 
**SAITS-ARIMA-BiLSTMA+Attention (PSO Optimized)** architecture.
""")

# --- 1. DATA GENERATION (Simulating a Skyscraper Elevator Bank) ---
@st.cache_data
def generate_simulation_data():
    np.random.seed(42)
    time_steps = 288  # 24 hours in 5-minute intervals
    time_axis = np.linspace(0, 24, time_steps)
    
    # Base Periodic Traffic (The ARIMA portion: Morning/Lunch/Evening peaks)
    periodic_trend = (
        30 * np.sin(2 * np.pi * time_axis / 24 - np.pi/2) +  # Daily rhythm
        25 * np.sin(2 * np.pi * time_axis / 12) +            # Secondary shifts
        40
    )
    periodic_trend = np.clip(periodic_trend, 10, 100)
    
    # Inject a massive, sudden non-linear anomaly at 14:00 (Floor event/surge)
    anomaly_center = 14.0
    anomaly = 65 * np.exp(-((time_axis - anomaly_center) / 0.4)**2)
    
    # Ground Truth = Periodic Baseline + Chaotic Anomaly + Random Noise
    noise = np.random.normal(0, 3, time_steps)
    ground_truth = periodic_trend + anomaly + noise
    
    # Create missing data gaps (Simulating IoT Sensor Dropouts for SAITS to fix)
    raw_data = ground_truth.copy()
    gap_1 = (time_axis >= 4.0) & (time_axis <= 5.5)
    gap_2 = (time_axis >= 17.5) & (time_axis <= 19.0)
    raw_data[gap_1] = np.nan
    raw_data[gap_2] = np.nan
    
    # Simulated SAITS Imputation (Reconstructing the true sequence via temporal context)
    imputed_data = ground_truth.copy()  
    
    # Simulated ARIMA Prediction (Succeeds at periodicity, completely misses the 14:00 anomaly)
    arima_forecast = periodic_trend.copy()
    
    # Extract Residuals (Error left over by the linear model)
    residuals = imputed_data - arima_forecast
    
    # Simulated BiLSTMA + Attention Forecast (Perfectly maps the residuals)
    hybrid_forecast = arima_forecast + residuals
    
    return time_axis, raw_data, imputed_data, arima_forecast, residuals, hybrid_forecast

time_axis, raw_data, imputed_data, arima_forecast, residuals, hybrid_forecast = generate_simulation_data()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Stage Selector")
step = st.sidebar.radio(
    "Select a step to view:",
    [
        "Step 1: Raw Sensor Data (With Gaps)",
        "Step 2: SAITS Imputation",
        "Step 3: ARIMA Linear Baseline",
        "Step 4: Linear Residual Extraction",
        "Step 5: Final Hybrid Model (BiLSTMA+Attention)",
    ]
)

# Convert float hours to HH:MM format for the X-axis strings
time_labels = [f"{int(h):02d}:{int((h%1)*60):02d}" for h in time_axis]

# --- 3. DYNAMIC PLOTLY RENDERING ---
fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.15,
                    subplot_titles=("Traffic Flow (Passenger Load)", "Model Residuals (ε_t)"))

# Configure layout settings
fig.update_layout(height=650, showlegend=True, hovermode="x unified",
                  template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
fig.update_xaxes(tickvals=time_labels[::24], ticktext=time_labels[::24], row=1)
fig.update_xaxes(tickvals=time_labels[::24], ticktext=time_labels[::24], row=2)

# STEP 1: Raw Sensor Data
if step == "Step 1: Raw Sensor Data (With Gaps)":
    st.sidebar.info("💡 **Notice:** Notice the missing data blocks early in the morning and evening, along with a massive unexplained spike around 14:00.")
    fig.add_trace(go.Scatter(x=time_labels, y=raw_data, name="Raw IoT Sensor Data", 
                             line=dict(color="#FF4B4B", width=2.5), connectgaps=False), row=1, col=1)

# STEP 2: SAITS Imputation
elif step == "Step 2: SAITS Imputation":
    st.sidebar.info("💡 **Notice:** SAITS utilizes bidirectional self-attention to seamlessly fill sensor dropouts without breaking sequence context.")
    fig.add_trace(go.Scatter(x=time_labels, y=raw_data, name="Raw Data (Gaps)", line=dict(color="#FF4B4B", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_labels, y=imputed_data, name="SAITS Imputed Signal", 
                             line=dict(color="#00CC96", width=2.5, dash="dash")), row=1, col=1)

# STEP 3: ARIMA Linear Baseline
elif step == "Step 3: ARIMA Linear Baseline":
    st.sidebar.info("💡 **Notice:** The ARIMA model isolates daily cyclical trends beautifully, but it is completely blind to the sudden, non-linear 14:00 spike.")
    fig.add_trace(go.Scatter(x=time_labels, y=imputed_data, name="Imputed Signal", line=dict(color="#00CC96", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_labels, y=arima_forecast, name="ARIMA Baseline (Linear)", 
                             line=dict(color="#AB63FA", width=2.5)), row=1, col=1)

# STEP 4: Linear Residual Extraction
elif step == "Step 4: Linear Residual Extraction":
    st.sidebar.info("💡 **Notice:** By subtracting ARIMA predictions from our data, we isolate the pure 'chaotic shocks' (residuals) for our deep learning model.")
    fig.add_trace(go.Scatter(x=time_labels, y=imputed_data, name="Imputed Signal", line=dict(color="#454545", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_labels, y=arima_forecast, name="ARIMA Baseline", line=dict(color="#AB63FA", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_labels, y=residuals, name="Extracted Residuals (ε_t)", 
                             line=dict(color="#FFA15A", width=2)), row=2, col=1)

# STEP 5: Final Hybrid Model
elif step == "Step 5: Final Hybrid Model (BiLSTMA+Attention)":
    st.sidebar.info("💡 **Notice:** The BiLSTMA layer handles the isolated residuals perfectly. Optimized via Particle Swarm Optimization (PSO), it catches both the cyclic baseline and the sudden anomaly.")
    fig.add_trace(go.Scatter(x=time_labels, y=imputed_data, name="True Traffic (Ground Truth)", line=dict(color="#00CC96", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_labels, y=hybrid_forecast, name="PSO-Hybrid Forecast", 
                             line=dict(color="#19D3F3", width=2.5, dash="dot")), row=1, col=1)
    fig.add_vrect(x0="13:00", x1="15:00", fillcolor="#19D3F3", opacity=0.15, line_width=0, 
                  annotation_text="Attention Focus Zone", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Scatter(x=time_labels, y=residuals, name="Resolved Residuals", line=dict(color="#19D3F3", width=1)), row=2, col=1)

# Display chart in Streamlit
st.plotly_chart(fig, use_container_width=True)