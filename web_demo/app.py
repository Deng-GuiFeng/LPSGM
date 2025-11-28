# -*- coding: utf-8 -*-
"""
app.py

Streamlit web application for the LPSGM (Large Polysomnography Model) project demo.

This application provides an interactive interface for loading polysomnography (PSG) EDF files,
selecting and mapping EEG/EOG/EMG channels, preprocessing signals, performing sleep staging inference,
and visualizing the PSG signals alongside inferred sleep stages. It also supports exporting
hypnogram results and detailed sleep metrics in TXT and Excel formats.

Key functionalities:
- EDF file upload and example file loading
- Flexible channel mapping with single or differential channel configurations
- Signal preprocessing and visualization with dynamic downsampling
- Sleep staging inference using the LPSGM backend or synthetic inference for debugging
- Display of sleep staging metrics and duration distribution pie chart
- Export of sleep staging results and metrics for offline analysis

This file serves as the main user interface and orchestrates interactions with utility functions
and the inference backend within the LPSGM project.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import os
import io
from io import BytesIO

from utils import load_sig, pre_process, get_ch_names, calculate_hypnogram_metrics
from inference_backend import inference_recodings

# --- Page Configuration ---
st.set_page_config(
    page_title="LPSGM Sleep Staging System",
    # page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styles - Professional Business Style with Compact Layout ---
st.markdown("""
<style>
    .main {
        padding-top: 0.5rem; /* Reduce top padding */
    }
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Dark gray text */
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; /* Business font */
    }
    .header-container {
        background-color: #ffffff; /* White background */
        padding: 0.8rem; /* Reduce padding */
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 0.8rem; /* Reduce bottom margin */
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .header-container h1 {
        color: #2c3e50; /* Dark blue title */
        font-size: 1.8rem; /* Reduce font size */
        margin-bottom: 0.1rem; /* Reduce bottom margin */
    }
    .header-container p {
        color: #7f8c8d; /* Gray subtitle */
        font-size: 0.8rem; /* Reduce font size */
    }
    .section-header {
        background-color: #ffffff;
        padding: 0.5rem 0.8rem; /* Reduce padding */
        border-left: 4px solid #3498db; /* Blue accent line */
        margin: 0.6rem 0; /* Reduce margin */
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: #2c3e50;
        font-size: 1rem; /* Reduce font size */
        font-weight: 600;
    }
    .stButton > button {
        background-color: #3498db; /* Blue button */
        color: white;
        border: none;
        padding: 0.4rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        transition: background-color 0.2s ease, transform 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: none;
    }
    .stSelectbox > div > div {
        border-radius: 4px;
        border: 1px solid #cccccc;
    }
    .stTextInput > div > div > input {
        border-radius: 4px;
        border: 1px solid #cccccc;
    }
    .stInfo {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        color: #2c3e50;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    .stSuccess {
        background-color: #e6ffe6;
        border-left: 4px solid #27ae60;
        color: #27ae60;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    .stWarning {
        background-color: #fff8e6;
        border-left: 4px solid #f39c12;
        color: #f39c12;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    .stPlotlyChart {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 0.8rem;
    }
    .stSlider .st-bd {
        padding-top: 0.5rem;
    }
    .stSlider .st-bb {
        margin-bottom: 0.5rem;
    }
    .stNumberInput {
        margin-bottom: 0.5rem;
    }
    h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stDataFrame {
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'edf_file_path' not in st.session_state:
    st.session_state.edf_file_path = None
if 'available_channels' not in st.session_state:
    st.session_state.available_channels = []
if 'channel_mapping' not in st.session_state:
    st.session_state.channel_mapping = {}
if 'processed_signals' not in st.session_state:
    st.session_state.processed_signals = None
if 'hypnogram' not in st.session_state:
    st.session_state.hypnogram = None
if 'selected_channel_tag' not in st.session_state:
    st.session_state.selected_channel_tag = None
if 'timescale_s_page' not in st.session_state:
    st.session_state.timescale_s_page = 600 # Default 600 seconds per page for signal display
if 'current_time_offset_s' not in st.session_state:
    st.session_state.current_time_offset_s = 0.0 # Current time offset in seconds (float)

# --- Constants Definition ---
SUPPORTED_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2", "E1", "E2", "Chin"]
EPOCH_DURATION_S = 30  # Duration of one epoch in seconds
RESAMPLE_RATE = 100    # Signal resampling rate in Hz
EXAMPLE_EDF_PATH = "web_demo/example.edf"  # Path to example EDF file
TEMP_UPLOAD_PATH = "web_demo/temp_uploaded.edf"  # Temporary path for uploaded EDF files


def inference(sig_dict, debug=False):
    """
    Perform sleep staging inference on preprocessed signals.

    Args:
        sig_dict (dict): Dictionary of preprocessed signals keyed by channel name.
        debug (bool): If True, perform synthetic inference for debugging purposes.

    Returns:
        list: Hypnogram list of sleep stages per epoch.
    """
    if debug:
        hypnogram = synthetic_inference(sig_dict)
        return hypnogram
    hypnogram = inference_recodings(sig_dict)
    return hypnogram

    
# --- Helper Functions ---
def synthetic_inference(sig_dict):
    """
    Simulate sleep staging inference by generating random hypnogram results.

    Args:
        sig_dict (dict): Dictionary of signals keyed by channel name.

    Returns:
        list: Synthetic hypnogram list of sleep stages per epoch.
    """
    for ch_name, ch_sig in sig_dict.items():
        n_epochs, _ = ch_sig.shape
        break
    
    hypnogram = []
    current_stage = 'W'
    for i in range(n_epochs):
        # Generate synthetic sleep stages with rough proportions
        if i < n_epochs * 0.1: 
            current_stage = 'W'
        elif i < n_epochs * 0.3:
            current_stage = random.choice(['W', 'N1', 'N2'])
        elif i < n_epochs * 0.8:
            current_stage = random.choice(['N1', 'N2', 'N3', 'R'])
        else:
            current_stage = random.choice(['N1', 'N2', 'R', 'W'])
        
        hypnogram.append(current_stage)
    
    import time
    time.sleep(5) # Simulate processing delay

    return hypnogram

def load_edf_file(file_path):
    """
    Load an EDF file, update session state with available channels and reset related states.

    Args:
        file_path (str): Path to the EDF file to load.
    """
    try:
        st.session_state.edf_file_path = file_path
        st.session_state.available_channels = get_ch_names(file_path)  # Extract channel names from EDF
        st.session_state.channel_mapping = {}  # Clear previous channel mappings
        st.session_state.processed_signals = None  # Clear processed signals
        st.session_state.hypnogram = None  # Clear previous hypnogram
        st.success(f"✅ File loaded successfully! Detected {len(st.session_state.available_channels)} channels")
    except Exception as e:
        st.error(f"❌ File reading failed: {str(e)}")

def display_signals():
    """
    Load raw signals according to channel mapping, preprocess them, and update session state.
    """
    try:
        with st.spinner("Loading and processing signals..."):
            channel_map_for_load_sig = {}
            # Prepare channel mapping for load_sig function, ensuring tuple format
            for target_ch, mapping in st.session_state.channel_mapping.items():
                if isinstance(mapping, tuple):
                    channel_map_for_load_sig[target_ch] = (mapping,)
                else:
                    channel_map_for_load_sig[target_ch] = (mapping,)
            
            start_time, sig_dict = load_sig(st.session_state.edf_file_path, channel_map_for_load_sig)
            # Preprocess signals: resample and optionally notch filter (disabled here)
            st.session_state.processed_signals = pre_process(sig_dict, resample_rate=RESAMPLE_RATE, notch=False)
            st.session_state.current_time_offset_s = 0.0 # Reset time offset to start
            st.success("✅ Signal loading and preprocessing completed!")
            
    except Exception as e:
        st.error(f"❌ Signal processing failed: {str(e)}")

def perform_inference():
    """
    Perform sleep staging inference on preprocessed signals and update session state.
    """
    try:
        with st.spinner("Performing sleep staging inference..."):
            st.session_state.hypnogram = inference(st.session_state.processed_signals)
            st.success("✅ Sleep staging inference completed!")
            
    except Exception as e:
        st.error(f"❌ Inference failed: {str(e)}")

# --- Fusion: Add XLSX export functionality ---
def create_excel_export():
    """
    Create an Excel file in memory containing hypnogram data and detailed sleep metrics.

    Returns:
        BytesIO: In-memory bytes buffer containing the Excel file.
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write hypnogram data to 'Hypnogram' sheet
        if st.session_state.hypnogram:
            hypno_df = pd.DataFrame({
                'Epoch': range(1, len(st.session_state.hypnogram) + 1),
                'Sleep_Stage': st.session_state.hypnogram
            })
            hypno_df.to_excel(writer, sheet_name='Hypnogram', index=False)
        
        # Write detailed hypnogram metrics to 'Hypnogram_Metrics' sheet
        if st.session_state.hypnogram:
            metrics = calculate_hypnogram_metrics(st.session_state.hypnogram)

            def v(x):
                # Normalize value for Excel export, replace NaN with empty string
                if isinstance(x, float) and np.isnan(x):
                    return ''
                return x

            # Build a comprehensive, grouped metrics table (Category, Metric, Value)
            rows = []
            # Sleep Latency (min)
            rows.append(["Sleep Latency (min)", "", ""])  # header row
            rows.append(["", "N1", round(v(metrics.get("n1_latency", np.nan)), 1) if v(metrics.get("n1_latency", np.nan)) != '' else ''])
            rows.append(["", "N2", round(v(metrics.get("n2_latency", np.nan)), 1) if v(metrics.get("n2_latency", np.nan)) != '' else ''])
            rows.append(["", "N3", round(v(metrics.get("n3_latency", np.nan)), 1) if v(metrics.get("n3_latency", np.nan)) != '' else ''])
            rows.append(["", "REM", round(v(metrics.get("rem_latency", np.nan)), 1) if v(metrics.get("rem_latency", np.nan)) != '' else ''])

            # Sleep Duration (min)
            rows.append(["Sleep Duration (min)", "", ""])  # header row
            rows.append(["", "TST (Total Sleep Time)", round(v(metrics.get("tst", np.nan)), 1) if v(metrics.get("tst", np.nan)) != '' else ''])
            rows.append(["", "REM", round(v(metrics.get("rem_duration", np.nan)), 1) if v(metrics.get("rem_duration", np.nan)) != '' else ''])
            rows.append(["", "NREM", round(v(metrics.get("nrem_duration", np.nan)), 1) if v(metrics.get("nrem_duration", np.nan)) != '' else ''])
            rows.append(["", "SWS (N3)", round(v(metrics.get("sws_duration", np.nan)), 1) if v(metrics.get("sws_duration", np.nan)) != '' else ''])

            # Sleep Stages
            rows.append(["Sleep Stages", "", ""])  # header row
            # W (SPT)
            rows.append(["W (SPT)", "Episodes (#)", int(v(metrics.get("wake_episodes", 0))) if v(metrics.get("wake_episodes", 0)) != '' else ''])
            rows.append(["W (SPT)", "Duration (min)", round(v(metrics.get("wake_duration", np.nan)), 1) if v(metrics.get("wake_duration", np.nan)) != '' else ''])
            # R
            rows.append(["R", "Duration (min)", round(v(metrics.get("rem_duration", np.nan)), 1) if v(metrics.get("rem_duration", np.nan)) != '' else ''])
            rows.append(["R", "TST (%)", round(v(metrics.get("rem_percentage", np.nan)), 1) if v(metrics.get("rem_percentage", np.nan)) != '' else ''])
            # N1
            rows.append(["N1", "Duration (min)", round(v(metrics.get("n1_duration", np.nan)), 1) if v(metrics.get("n1_duration", np.nan)) != '' else ''])
            rows.append(["N1", "TST (%)", round(v(metrics.get("n1_percentage", np.nan)), 1) if v(metrics.get("n1_percentage", np.nan)) != '' else ''])
            # N2
            rows.append(["N2", "Duration (min)", round(v(metrics.get("n2_duration", np.nan)), 1) if v(metrics.get("n2_duration", np.nan)) != '' else ''])
            rows.append(["N2", "TST (%)", round(v(metrics.get("n2_percentage", np.nan)), 1) if v(metrics.get("n2_percentage", np.nan)) != '' else ''])
            # N3
            rows.append(["N3", "Duration (min)", round(v(metrics.get("n3_duration", np.nan)), 1) if v(metrics.get("n3_duration", np.nan)) != '' else ''])
            rows.append(["N3", "TST (%)", round(v(metrics.get("n3_percentage", np.nan)), 1) if v(metrics.get("n3_percentage", np.nan)) != '' else ''])
            # Note: Summary metrics like SE and SOL are not included to follow the reference format strictly

            metrics_df2 = pd.DataFrame(rows, columns=["Category", "Metric", "Value"])
            metrics_df2.to_excel(writer, sheet_name='Hypnogram_Metrics', index=False)
    
    output.seek(0)
    return output

def export_results():
    """
    Provide download buttons for exporting sleep staging results as TXT and Excel files.
    """
    try:
        if st.session_state.hypnogram:
            hypno_text = '\n'.join(st.session_state.hypnogram)
            # Download button for hypnogram TXT file
            st.download_button(
                label="📥 Download Sleep Staging Results (TXT)",
                data=hypno_text,
                file_name="hypnogram.txt",
                mime="text/plain",
                key="download_hypno"
            )
            
            # Download button for detailed Excel export
            excel_data = create_excel_export()
            st.download_button(
                label="Save Results",
                data=excel_data.getvalue(),
                file_name="sleep_staging_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_xlsx"
            )
            
            st.success("✅ Result export functionality is ready!")
        else:
            st.warning("⚠️ No inference results available for export.")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def get_downsampled_data(signal_data, target_points=2000):
    """
    Downsample signal data to reduce the number of points for plotting.

    Args:
        signal_data (np.ndarray): 1D array of signal values.
        target_points (int): Desired maximum number of points after downsampling.

    Returns:
        tuple: (downsampled_data (np.ndarray), downsampled_indices (np.ndarray))
    """
    n_points = len(signal_data)
    if n_points <= target_points:
        # No downsampling needed
        return signal_data, np.arange(n_points)
    
    # Calculate sampling step interval
    step = n_points // target_points
    if step == 0:  # Avoid division by zero
        return signal_data, np.arange(n_points)

    # Downsample by taking every 'step'-th point
    downsampled_data = signal_data[::step]
    downsampled_indices = np.arange(n_points)[::step]
    return downsampled_data, downsampled_indices

def display_signal_plot():
    """
    Display PSG signals and inferred sleep stages with interactive controls for timescale and time offset.
    """
    st.markdown('<div class="section-header"><h3>📊 Signal Display Area</h3></div>', unsafe_allow_html=True)
    
    if not st.session_state.processed_signals:
        st.info("Please load and process signals first.")
        return
    
    # Calculate total signal duration in seconds
    first_channel_data = list(st.session_state.processed_signals.values())[0]
    total_samples = first_channel_data.shape[0] * first_channel_data.shape[1]  # Total sample points across epochs
    total_duration_s = total_samples / RESAMPLE_RATE  # Total duration in seconds

    # Layout columns for timescale input and time axis slider
    col_ts, col_tx = st.columns([1, 4])

    with col_ts:
        st.session_state.timescale_s_page = st.number_input(
            "Timescale (seconds/page)", 
            min_value=10, 
            max_value=int(total_duration_s), 
            value=st.session_state.timescale_s_page,
            step=10,
            help="Set the time range displayed per page (seconds)",
            key="timescale_input"
        )

    # Calculate maximum time offset to prevent overflow beyond signal duration
    max_offset_s = max(0.0, total_duration_s - st.session_state.timescale_s_page)
    
    # Callback to update time offset when slider changes
    def update_time_offset():
        st.session_state.current_time_offset_s = st.session_state.slider_value

    with col_tx:
        st.slider(
            "Time Axis (seconds)", 
            min_value=0.0, 
            max_value=max_offset_s, 
            value=float(st.session_state.current_time_offset_s), 
            step=st.session_state.timescale_s_page / 100,  # Step size is 1% of page duration
            format="%.1f s",
            help="Drag to adjust the start time of displayed signals",
            key="slider_value",  # Separate key to avoid input bouncing
            on_change=update_time_offset  # Update time offset on slider change
        )

    # Calculate sample indices for current display window
    start_s = st.session_state.current_time_offset_s
    end_s = start_s + st.session_state.timescale_s_page
    start_sample = int(start_s * RESAMPLE_RATE)
    end_sample = int(end_s * RESAMPLE_RATE)

    # Prepare subplots: one row per channel plus one for sleep stages if available
    n_channels = len(st.session_state.processed_signals)
    has_hypnogram = st.session_state.hypnogram is not None
    total_rows = n_channels + (1 if has_hypnogram else 0)
    
    subplot_titles = list(st.session_state.processed_signals.keys())
    if has_hypnogram:
        subplot_titles.append("Sleep Stages")
    
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02
    )
    
    # Define colors for signal traces
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    # Plot each channel's signal
    for i, (ch_name, ch_data_epochs) in enumerate(st.session_state.processed_signals.items()):
        # Flatten epoch data to a continuous 1D signal
        ch_data_flat = ch_data_epochs.flatten()
        
        # Extract signal segment for current display window
        display_data_raw = ch_data_flat[start_sample:end_sample]

        # Downsample signal for efficient plotting
        downsampled_data, downsampled_indices = get_downsampled_data(display_data_raw, target_points=2000)  # Target max 2000 points
        
        # Calculate time axis values relative to overall signal start
        time_axis = (downsampled_indices + start_sample) / RESAMPLE_RATE 
        
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=downsampled_data,
                mode='lines',
                name=ch_name,
                line=dict(color=colors[i % len(colors)], width=1),
                showlegend=False,
                hoverinfo='x+y+name'  # Show x, y, and channel name on hover
            ),
            row=i+1, col=1
        )
        # Disable zoom and pan for signal plots to maintain fixed view
        fig.update_xaxes(fixedrange=True, row=i+1, col=1)
        fig.update_yaxes(fixedrange=True, row=i+1, col=1)

    # Plot sleep stages as a separate subplot if available
    if has_hypnogram:
        # Convert time window to epoch indices
        start_epoch_idx = int(start_s / EPOCH_DURATION_S)
        end_epoch_idx = int(end_s / EPOCH_DURATION_S)
        
        hypno_display = st.session_state.hypnogram[start_epoch_idx:end_epoch_idx]
        # Map sleep stage strings to numeric values for plotting
        stage_mapping = {'W': 4, 'R': 3, 'N1': 2, 'N2': 1, 'N3': 0}
        hypno_numeric = [stage_mapping[stage] for stage in hypno_display]
        
        # Time axis for hypnogram (epoch-based, scaled to seconds)
        time_axis_hypno = np.arange(len(hypno_numeric)) * EPOCH_DURATION_S + start_s
        
        fig.add_trace(
            go.Scatter(
                x=time_axis_hypno,
                y=hypno_numeric,
                mode='lines',
                name='Sleep Stages',
                line=dict(color='red', width=2, shape='hv'),  # Horizontal-vertical step lines
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.05)',  # Light red fill under curve
                showlegend=False,
                hoverinfo='x+y+name'
            ),
            row=n_channels+1, col=1
        )
        
        # Configure y-axis for sleep stages with labels and fixed range
        fig.update_yaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['N3', 'N2', 'N1', 'REM', 'Wake'],
            row=n_channels+1, col=1,
            fixedrange=True,  # Disable zoom and pan
            range=[-0.5, 4.5]  # Fixed y-axis range to show all stages
        )
    
    # Update overall figure layout with compact sizing and styling
    fig.update_layout(
        height=120 * total_rows + 50,  # Adjust height based on number of rows
        title_text="PSG Signals and Sleep Stages",
        title_x=0.5,  # Center title
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=80),  # Margins with extra bottom space
        font=dict(family="Segoe UI", size=10, color="#333333"),
        hovermode="x unified"  # Unified hover label across subplots
    )
    
    # Set unified x-axis range for all subplots
    fig.update_xaxes(range=[start_s, end_s])
    
    # Show x-axis labels only on the bottom subplot
    for i in range(1, total_rows + 1):
        if i == total_rows:
            fig.update_xaxes(title_text="Time (seconds)", row=i, col=1)
        else:
            fig.update_xaxes(title_text="", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
    # Display sleep staging statistics and export options if hypnogram is available
    if has_hypnogram:
        st.markdown("### 📊 Statistics and Export")
        metrics = calculate_hypnogram_metrics(st.session_state.hypnogram)

        # Layout: left column for metrics table, right column for duration distribution pie chart
        col_table, col_chart = st.columns([1, 1])

        def _styled_combined_table(df):
            """
            Style the combined metrics table with centered text, bold headers, and subtle separators.

            Args:
                df (pd.DataFrame): DataFrame containing metrics.

            Returns:
                pd.Styler: Styled DataFrame for display.
            """
            def highlight_headers(row):
                # Highlight header rows where 'Metric' column is empty
                if row["Metric"] == "":
                    return ["background-color: #f5f7fa; font-weight:600; color:#2c3e50;" for _ in row]
                return ["" for _ in row]

            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            styler = (
                df.style
                .apply(highlight_headers, axis=1)
                .format({c: "{:.1f}" for c in num_cols})
                .set_properties(**{"text-align": "left"})
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "left")]},
                    {"selector": "tbody td", "props": [("vertical-align", "middle"), ("text-align", "left")]}
                ])
            )
            return styler

        # Helper functions to safely convert metrics values
        def safe_val(x):
            # Return numeric value or np.nan if NaN
            if isinstance(x, float) and np.isnan(x):
                return np.nan
            return x

        def to_float_or_nan(x):
            # Convert to float or np.nan if conversion fails
            if isinstance(x, float) and np.isnan(x):
                return np.nan
            try:
                return float(x)
            except Exception:
                return np.nan

        # Build combined metrics table rows with categories and values
        rows = []
        # Sleep Latency (min)
        rows.append({"Category": "Sleep Latency (min)", "Metric": "", "Value": np.nan})
        rows.append({"Category": "", "Metric": "N1", "Value": to_float_or_nan(safe_val(metrics.get("n1_latency", np.nan)))})
        rows.append({"Category": "", "Metric": "N2", "Value": to_float_or_nan(safe_val(metrics.get("n2_latency", np.nan)))})
        rows.append({"Category": "", "Metric": "N3", "Value": to_float_or_nan(safe_val(metrics.get("n3_latency", np.nan)))})
        rows.append({"Category": "", "Metric": "REM", "Value": to_float_or_nan(safe_val(metrics.get("rem_latency", np.nan)))})

        # Sleep Duration (min)
        rows.append({"Category": "Sleep Duration (min)", "Metric": "", "Value": np.nan})
        rows.append({"Category": "", "Metric": "TST (Total Sleep Time)", "Value": to_float_or_nan(safe_val(metrics.get("tst", np.nan)))})
        rows.append({"Category": "", "Metric": "REM", "Value": to_float_or_nan(safe_val(metrics.get("rem_duration", np.nan)))})
        rows.append({"Category": "", "Metric": "NREM", "Value": to_float_or_nan(safe_val(metrics.get("nrem_duration", np.nan)))})
        rows.append({"Category": "", "Metric": "SWS (N3)", "Value": to_float_or_nan(safe_val(metrics.get("sws_duration", np.nan)))})

        # Sleep Stages
        rows.append({"Category": "Sleep Stages", "Metric": "", "Value": np.nan})
        rows.append({"Category": "W (SPT)", "Metric": "Episodes (#)", "Value": to_float_or_nan(safe_val(metrics.get("wake_episodes", np.nan)))})
        rows.append({"Category": "W (SPT)", "Metric": "Duration (min)", "Value": to_float_or_nan(safe_val(metrics.get("wake_duration", np.nan)))})
        rows.append({"Category": "R", "Metric": "Duration (min)", "Value": to_float_or_nan(safe_val(metrics.get("rem_duration", np.nan)))})
        rows.append({"Category": "R", "Metric": "TST (%)", "Value": to_float_or_nan(safe_val(metrics.get("rem_percentage", np.nan)))})
        rows.append({"Category": "N1", "Metric": "Duration (min)", "Value": to_float_or_nan(safe_val(metrics.get("n1_duration", np.nan)))})
        rows.append({"Category": "N1", "Metric": "TST (%)", "Value": to_float_or_nan(safe_val(metrics.get("n1_percentage", np.nan)))})
        rows.append({"Category": "N2", "Metric": "Duration (min)", "Value": to_float_or_nan(safe_val(metrics.get("n2_duration", np.nan)))})
        rows.append({"Category": "N2", "Metric": "TST (%)", "Value": to_float_or_nan(safe_val(metrics.get("n2_percentage", np.nan)))})
        rows.append({"Category": "N3", "Metric": "Duration (min)", "Value": to_float_or_nan(safe_val(metrics.get("n3_duration", np.nan)))})
        rows.append({"Category": "N3", "Metric": "TST (%)", "Value": to_float_or_nan(safe_val(metrics.get("n3_percentage", np.nan)))})

        # Note: Not including SE or SOL rows in the combined table per reference image

        combined_df = pd.DataFrame(rows, columns=["Category", "Metric", "Value"])

        # Convert numeric values to formatted strings for UI display with left alignment
        combined_df_display = combined_df.copy()
        def _fmt_val(x):
            try:
                return "" if pd.isna(x) else f"{float(x):.1f}"
            except Exception:
                return ""  # fallback for any unexpected types
        combined_df_display["Value"] = combined_df_display["Value"].apply(_fmt_val)

        with col_table:
            st.subheader("Hypnogram Metrics")
            st.dataframe(
                _styled_combined_table(combined_df_display),
                width="stretch",
                height=500
            )

        with col_chart:
            st.subheader("Duration Distribution")
            # Prepare data for pie chart of sleep stage durations
            def val(v):
                return np.nan if (isinstance(v, float) and (np.isnan(v))) else v
            stage_df = pd.DataFrame({
                "Sleep Stage": ["Wake", "N1", "N2", "N3", "REM"],
                "Duration (min)": [
                    round(val(metrics.get("wake_duration", np.nan)), 1),
                    round(val(metrics.get("n1_duration", np.nan)), 1),
                    round(val(metrics.get("n2_duration", np.nan)), 1),
                    round(val(metrics.get("n3_duration", np.nan)), 1),
                    round(val(metrics.get("rem_duration", np.nan)), 1),
                ]
            })
            # Display pie chart only if total duration is non-zero
            if not stage_df.empty and pd.to_numeric(stage_df["Duration (min)"], errors='coerce').fillna(0).sum() > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=stage_df["Sleep Stage"],
                    values=stage_df["Duration (min)"],
                    hole=0.35,
                    textinfo='label+percent',
                    textfont=dict(size=11, color="#333333"),
                    marker=dict(
                        colors=['#f8f9fa', '#e3f2fd', '#e1f5fe', '#e0f2f1', '#fce4ec'],
                        line=dict(color='#ffffff', width=2)
                    ),
                    showlegend=False
                )])
                fig_pie.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=30, b=20),
                    font=dict(family="Segoe UI", size=10, color="#333333"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, width="stretch", key=f"sleep_distribution_{len(st.session_state.hypnogram)}")
            else:
                st.info("Generating chart...")


# --- Main Application Logic ---
def main():
    """
    Main function to run the Streamlit application interface.
    """
    st.markdown("""
    <div class="header-container">
        <h1>LPSGM Sleep Staging System</h1>
        <p>Demo for paper LPSGM (<a href="https://www.medrxiv.org/content/10.1101/2024.12.11.24318815v3" target="_blank" rel="noopener noreferrer">Preprint</a>    <a href="https://github.com/Deng-GuiFeng/LPSGM" target="_blank" rel="noopener noreferrer">Github</a>)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # EDF file loading section
    st.markdown('<div class="section-header"><h3>📁 EDF File Loading</h3></div>', unsafe_allow_html=True)
    
    col_upload, col_example = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Select EDF File",
            type=['edf'],
            help="Please upload a PSG polysomnography EDF format file",
            key="file_uploader"
        )
    
    with col_example:
        if st.button("Use Example File", type="secondary", key="example_file_btn"):
            load_edf_file(EXAMPLE_EDF_PATH)
            
    # Handle uploaded EDF file and save temporarily
    if uploaded_file is not None and st.session_state.edf_file_path != TEMP_UPLOAD_PATH:
        with open(TEMP_UPLOAD_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_edf_file(TEMP_UPLOAD_PATH)
    
    # Display currently loaded file name
    if st.session_state.edf_file_path:
        display_file_name = os.path.basename(st.session_state.edf_file_path)
        st.info(f"Currently loaded file: **{display_file_name}**")

    if st.session_state.edf_file_path and st.session_state.available_channels:
        # Channel selection and mapping section
        st.markdown('<div class="section-header"><h3>⚙️ Channel Selection and Mapping</h3></div>', unsafe_allow_html=True)
        
        st.info("""
        **How to Configure Channel Mapping:**
        
        LPSGM uses 9 standard channels for sleep staging and supports flexible configurations (1-9 channels).
        The system automatically handles missing channels, so you only need to map the channels available in your EDF file.
        
        **Standard Channels:**
        - **EEG (brain activity)**: F3, F4, C3, C4, O1, O2
        - **EOG (eye movement)**: E1, E2  
        - **EMG (muscle activity)**: Chin
        
        **Mapping Instructions:**
        1. Click a channel button below (e.g., C3, E1) to configure it
        2. Choose the mapping type based on your EDF file format:
           - **Single Channel**: Your EDF has pre-referenced channels like 'C3-M2' or 'EOG-L' → Select the channel name directly
           - **Differential Channel**: Your EDF has separate electrodes like 'C3' and 'M2' → Select both electrodes to compute their difference (C3 minus M2)
        3. Repeat for each available channel in your EDF file
        4. Channels not mapped will be treated as missing (the model handles this automatically)
        
        **Recommended**: Map at least C3, C4, E1, E2 for optimal sleep staging performance. More channels generally improve accuracy.
        
        **For the Example File**: Use single channel mapping for F3→'EEG F3-LER', F4→'EEG F4-LER', C3→'EEG C3-LER', C4→'EEG C4-LER', O1→'EEG O1-LER', O2→'EEG O2-LER', E1→'EOG Left Horiz', E2→'EOG Right Horiz', and differential mapping for Chin→('EMG Chin1', 'EMG Chin2').
        """)
        
        # Display channel tag buttons for mapping selection
        tag_cols = st.columns(9)
        for i, tag in enumerate(SUPPORTED_CHANNELS):
            with tag_cols[i]:
                mapped = tag in st.session_state.channel_mapping
                if mapped:
                    mapping = st.session_state.channel_mapping[tag]
                    if isinstance(mapping, tuple):
                        display_text = f"✅ {tag}: \n{mapping[0]}|{mapping[1]}"
                    else:
                        display_text = f"✅ {tag}: \n{mapping}"
                    button_type = "primary"
                else:
                    display_text = f"➕ {tag}: \nNot Mapped"
                    button_type = "secondary"
                    
                if st.button(display_text, key=f"tagbtn_{tag}", use_container_width=True, type=button_type):
                    st.session_state["active_tag"] = tag

        # Channel configuration panel for selected tag
        if "active_tag" in st.session_state and st.session_state.active_tag:
            tag = st.session_state.active_tag
            
            with st.expander(f"Configure Channel: {tag}", expanded=True):
                col_type, col_ch1, col_ch2, col_btn = st.columns([2, 2, 2, 1])
                
                with col_type:
                    map_type = st.radio("Mapping Type", ["Single Channel", "Differential Channel"], key=f"maptype_{tag}")
                
                if map_type == "Single Channel":
                    with col_ch1:
                        ch_options = [""] + st.session_state.available_channels
                        ch_idx = st.selectbox("Select Channel", range(len(ch_options)), 
                                            format_func=lambda x: ch_options[x] if ch_options[x] else "Please select...", 
                                            key=f"sel_single_{tag}")
                        selected_ch = ch_options[ch_idx] if ch_idx > 0 else None
                    with col_btn:
                        if st.button("Confirm", key=f"save_single_{tag}"):
                            if selected_ch:
                                st.session_state.channel_mapping[tag] = selected_ch
                                st.success(f"{tag} → {selected_ch}")
                                del st.session_state["active_tag"]
                                st.rerun()
                else:
                    with col_ch1:
                        ch_options = [""] + st.session_state.available_channels
                        ch1_idx = st.selectbox("Channel 1", range(len(ch_options)),
                                             format_func=lambda x: ch_options[x] if ch_options[x] else "Please select...",
                                             key=f"sel_pair1_{tag}")
                        selected_ch1 = ch_options[ch1_idx] if ch1_idx > 0 else None
                    with col_ch2:
                        ch2_idx = st.selectbox("Channel 2", range(len(ch_options)),
                                             format_func=lambda x: ch_options[x] if ch_options[x] else "Please select...", 
                                             key=f"sel_pair2_{tag}")
                        selected_ch2 = ch_options[ch2_idx] if ch2_idx > 0 else None
                    with col_btn:
                        if st.button("Confirm", key=f"save_pair_{tag}"):
                            if selected_ch1 and selected_ch2 and selected_ch1 != selected_ch2:
                                st.session_state.channel_mapping[tag] = (selected_ch1, selected_ch2)
                                st.success(f"{tag} → {selected_ch1}|{selected_ch2}")
                                del st.session_state["active_tag"]
                                st.rerun()
                            else:
                                st.error("Two channels must be different and both selected")

        # Operations section with buttons for displaying signals, inference, and exporting results
        st.markdown('<div class="section-header"><h3>📌 Operations</h3></div>', unsafe_allow_html=True)
        col_op1, col_op2, col_op3 = st.columns(3)
        
        with col_op1:
            if st.button("📊 Display Signals", use_container_width=True):
                if st.session_state.channel_mapping:
                    display_signals()
                else:
                    st.warning("⚠️ Please select at least one channel mapping first")
        
        with col_op2:
            if st.button("🚀 Start Inference", use_container_width=True):
                if st.session_state.processed_signals:
                    perform_inference()
                else:
                    st.warning("⚠️ Please display signals first")
        
        with col_op3:
            # If hypnogram results exist, provide direct Excel export button
            if st.session_state.hypnogram:
                excel_data = create_excel_export()
                st.download_button(
                    label="💾 Export Results",
                    data=excel_data.getvalue(),
                    file_name="sleep_staging_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="export_direct"
                )
            else:
                # Otherwise, show warning if export button is pressed prematurely
                if st.button("💾 Export Results", use_container_width=True):
                    st.warning("⚠️ Please complete inference first")
        
        # Display signal plots and sleep staging visualization
        display_signal_plot()

if __name__ == "__main__":
    main()
