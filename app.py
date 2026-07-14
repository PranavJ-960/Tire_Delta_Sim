"""
Streamlit Web UI for F1 Pit Strategy Simulator
Run via: streamlit run app.py
"""

import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from explore import load_race, clean_laps, attach_tire_age
from optimizer import (
    compute_pit_loss,
    fit_degradation_models,
    optimize_strategy,
    compare_driver_to_optimal,
    format_strategy
)

st.set_page_config(page_title="F1 Pit Strategy Simulator", layout="wide")

st.title("🏎️ F1 Pit Strategy Simulator")
st.markdown("Analyze tire degradation models and evaluate team strategic performance using real-world telemetry from the OpenF1 API.")

# --------------------------------------------------------
# DATA INGESTION LAYOUT
# --------------------------------------------------------
st.sidebar.header("Race Selection Settings")

# Detect which races have already been pulled locally in the /data directory
data_dir = "data"
local_races = []
if os.path.exists(data_dir):
    # Folders look like "2024_Italy" or "2024_Belgium"
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for folder in folders:
        parts = folder.split('_', 1)
        if len(parts) == 2:
            local_races.append(f"{parts[0]} - {parts[1].replace('_', ' ')}")

local_races = sorted(list(set(local_races)))

if not local_races:
    st.sidebar.warning("No pulled race folders detected. Run data_pipeline.py or season_pull.py first!")
    st.stop()

selected_race = st.sidebar.selectbox("Choose a pulled Grand Prix", local_races)
year_str, country_str = selected_race.split(" - ")
year = int(year_str)
country = country_str

# --------------------------------------------------------
# CORE CALCULATIONS ENGINE
# --------------------------------------------------------
race = load_race(year, country)
laps_cleaned = clean_laps(race["laps"])
laps_with_tires = attach_tire_age(laps_cleaned, race["stints"])
total_laps = int(race["laps"]["lap_number"].max())
pit_loss = compute_pit_loss(race["pits"])
models = fit_degradation_models(laps_with_tires)
result = optimize_strategy(total_laps, models, pit_loss)

# --------------------------------------------------------
# DASHBOARD DISPLAY METRICS
# --------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric(label="Race Track Distance", value=f"{total_laps} Laps")
col2.metric(label="Calculated Pit Lane Loss", value=f"{pit_loss:.2f} seconds")
col3.metric(label="Theoretical Ideal Strategy Window", value=result["strategy"][0][0])

st.subheader("🏁 Mathematically Optimal Strategy")
st.info(f"**Optimal Path:** {format_strategy(result['strategy'])}  \n**Estimated Strategy Time Floor:** {result['total_time']/60:.2f} minutes")

# --------------------------------------------------------
# VISUALIZATION ROW
# --------------------------------------------------------
st.markdown("---")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📈 Fitted Degradation Linear Curves")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Generate scatter plot points vs model trendlines
    colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray') 
    ax.spines['left'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    for comp in models.keys():
        df_comp = laps_with_tires[(laps_with_tires["compound"] == comp) & (laps_with_tires["tire_age"] > 0)]
        if df_comp.empty: continue
        
        c = colors.get(comp, "cyan")
        # Scatter actual data points lightly
        ax.scatter(df_comp["tire_age"], df_comp["lap_duration"], color=c, alpha=0.15, label=None)
        
        # Plot structural regression lines
        ages = np.arange(1, int(df_comp["tire_age"].max()) + 5)
        fit_line = models[comp]["base_time"] + (models[comp]["deg_rate"] * ages)
        ax.plot(ages, fit_line, color=c, linewidth=2.5, label=f"{comp} (+{models[comp]['deg_rate']:.3f}s/l)")

    ax.set_xlabel("Tire Age (Laps Completed)")
    ax.set_ylabel("Lap Time Pace (Seconds)")
    ax.legend(facecolor="#2e2e2e", labelcolor="white")
    st.pyplot(fig)

with right_col:
    st.subheader("⏱️ Team Pit Wall Variance Table")
    st.markdown("How actual race strategies rank relative to our pure mathematical model baseline:")
    
    available_drivers = sorted(race["stints"]["driver_number"].unique())
    comparisons = []
    for dn in available_drivers:
        comp_data = compare_driver_to_optimal(
            dn, race["stints"], total_laps, models, pit_loss, result["total_time"]
        )
        if comp_data:
            comparisons.append({
                "Driver #": f"#{comp_data['driver_number']}",
                "Actual Strategic Timeline": format_strategy(comp_data["actual_strategy"]),
                "Model Delta": f"+{comp_data['delta_seconds']:.2f}s" if comp_data['delta_seconds'] > 0 else f"{comp_data['delta_seconds']:.2f}s"
            })
            
    df_table = pd.DataFrame(comparisons).sort_values(by="Model Delta")
    st.dataframe(df_table, use_container_width=True, hide_index=True)