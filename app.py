import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from explore import load_race, clean_laps, attach_tire_age
from optimizer import (
    compute_pit_loss,
    build_lap_weather,
    fit_degradation_models,
    optimize_strategy,
    compare_driver_to_optimal,
    format_strategy,
    get_driver_adjusted_predict
)

st.set_page_config(page_title="F1 Pit Strategy Simulator", layout="wide")

st.title("🏎️ F1 Pit Strategy Simulator")
st.markdown("Analyze tire degradation models and evaluate team strategic performance using real-world telemetry.")

# --------------------------------------------------------
# CACHED COMPUTATION LAYERS (Fixes Bottlenecks)
# --------------------------------------------------------
@st.cache_data
def get_cached_race_context(year: int, country: str):
    race_data = load_race(year, country)
    laps_c = clean_laps(race_data["laps"])
    laps_t = attach_tire_age(laps_c, race_data["stints"])
    tot_laps = int(race_data["laps"]["lap_number"].max())
    p_loss = compute_pit_loss(race_data["pits"])
    l_weather = build_lap_weather(race_data["laps"], race_data["weather"])
    deg_models = fit_degradation_models(laps_t, circuit=country)
    return race_data, laps_t, tot_laps, p_loss, l_weather, deg_models

@st.cache_data
def get_cached_optimization(tot_laps, _deg_models, p_loss, l_weather, driver_sel):
    return optimize_strategy(tot_laps, _deg_models, p_loss, l_weather, driver_number=driver_sel)

# --------------------------------------------------------
# DATA INGESTION LAYOUT
# --------------------------------------------------------
st.sidebar.header("Race Selection Settings")
data_dir = "data"
local_races = []
if os.path.exists(data_dir):
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for folder in folders:
        parts = folder.split('_', 1)
        if len(parts) == 2:
            local_races.append(f"{parts[0]} - {parts[1].replace('_', ' ')}")

local_races = sorted(list(set(local_races)))

if not local_races:
    st.sidebar.warning("No pulled race folders detected.")
    st.stop()

selected_race = st.sidebar.selectbox("Choose a Grand Prix", local_races)
year_str, country_str = selected_race.split(" - ")
year, country = int(year_str), country_str

# Load Race telemetry through cached layer
race, laps_with_tires, total_laps, pit_loss, lap_weather, models = get_cached_race_context(year, country)

# Driver Focus Configuration Filter
available_drivers = sorted(race["stints"]["driver_number"].unique())
focused_driver = st.sidebar.selectbox("Optimize strategy specifically for Driver:", [None] + available_drivers)

# Compute Strategy Floor Outputs
result = get_cached_optimization(total_laps, models, pit_loss, lap_weather, focused_driver)

# --------------------------------------------------------
# DASHBOARD DISPLAY METRICS
# --------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric(label="Race Track Distance", value=f"{total_laps} Laps")
col2.metric(label="Calculated Pit Lane Loss", value=f"{pit_loss:.2f} seconds")
col3.metric(label="Focused Analysis Subject", value=f"#{focused_driver}" if focused_driver else "Global Field Optimum")

st.subheader("🏁 Mathematically Optimal Strategy Plan")
if result["strategy"]:
    st.info(f"**Optimal Path:** {format_strategy(result['strategy'])}  \n**Estimated Strategy Time Floor:** {result['total_time']/60:.2f} minutes")
else:
    st.error("Could not optimize a valid strategy configuration for this race profile.")

# --------------------------------------------------------
# VISUALIZATION ROW (Fixes Plot Bug)
# --------------------------------------------------------
st.markdown("---")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📈 Custom Degradation Curve Projections")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white", "INTERMEDIATE": "green", "WET": "blue"}
    
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    for spine in ax.spines.values(): spine.set_color('gray')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    d_profiles = models.get("_drivers", {})
    d_prof = d_profiles.get(focused_driver) if focused_driver else None

    for comp in [c for c in models.keys() if c != "_drivers"]:
        df_comp = laps_with_tires[(laps_with_tires["compound"] == comp) & (laps_with_tires["tire_age"] > 0)]
        if df_comp.empty: continue
        
        c = colors.get(comp, "cyan")
        ax.scatter(df_comp["tire_age"], df_comp["lap_duration"] - df_comp["lap_number"]*models[comp]["fuel_effect_per_lap"], color=c, alpha=0.08, label=None)
        
        max_age = int(df_comp["tire_age"].max())
        ages = np.arange(1, max_age + 3 if max_age > 1 else 15)
        
        # Bugfix applied: evaluate using functional prediction logic rather than forcing straight line formulas
        pred_func = get_driver_adjusted_predict(models[comp]["predict"], d_prof, comp)
        fit_curve = pred_func(ages)
        
        ax.plot(ages, fit_curve, color=c, linewidth=2.5, label=f"{comp}")

    ax.set_xlabel("Tire Age (Laps Completed)")
    ax.set_ylabel("Fuel-Corrected Pace (Seconds)")
    ax.legend(facecolor="#2e2e2e", labelcolor="white")
    st.pyplot(fig)
    plt.close(fig)

with right_col:
    st.subheader("⏱️ Strategic Variance Grid")
    comparisons = []
    for dn in available_drivers:
        comp_data = compare_driver_to_optimal(dn, race["stints"], total_laps, models, pit_loss, result["total_time"], lap_weather)
        if comp_data:
            comparisons.append({
                "Driver": f"#{comp_data['driver_number']}",
                "Actual Strategic Timeline": format_strategy(comp_data["actual_strategy"]),
                "Model Delta": f"{comp_data['delta_seconds']:+.2f}s"
            })
            
    if comparisons:
        df_table = pd.DataFrame(comparisons)
        df_table["sort_val"] = df_table["Model Delta"].str.replace("s", "").astype(float)
        df_table = df_table.sort_values(by="sort_val").drop(columns=["sort_val"])
        st.dataframe(df_table, width="stretch", hide_index=True)
    else:
        st.write("No active team strategy timelines resolved.")