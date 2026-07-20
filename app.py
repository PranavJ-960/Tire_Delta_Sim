"""
Day 5: Streamlit dashboard. Pick a pulled race, see the optimal strategy
vs what teams actually did, plus the degradation curves behind it.

Usage:
    python data_pipeline.py --year 2024 --country Italy   # pull at least one race first
    streamlit run app.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from explore import load_race, clean_laps, attach_tire_age
from optimizer import (
    compute_pit_loss,
    build_lap_weather,
    race_has_rain,
    fit_degradation_models,
    optimize_strategy,
    compare_driver_to_optimal,
    reconstruct_actual_strategy,
    format_strategy,
    DRY_COMPOUNDS,
    WET_COMPOUNDS,
)
from season_priors import load_season_priors
from monte_carlo import run_monte_carlo, SAFETY_CAR_PROBABILITY

st.set_page_config(page_title="F1 Pit Strategy Simulator", layout="wide")
st.title("🏎️ F1 Pit Strategy Simulator")
st.caption(
    "Compares the theoretically optimal pit strategy against what teams "
    "actually did, using real lap, tire, and weather data from OpenF1."
)


@st.cache_data
def get_race_context(year: int, country: str):
    race = load_race(year, country)
    laps_clean = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps_clean, race["stints"])
    total_laps = int(race["laps"]["lap_number"].max())
    pit_loss = compute_pit_loss(race["pits"])
    lap_weather = build_lap_weather(race["laps"], race["weather"])
    season_priors = load_season_priors(year)
    models = fit_degradation_models(laps_with_tires, season_priors=season_priors)
    return race, laps_with_tires, total_laps, pit_loss, lap_weather, models, season_priors is not None


@st.cache_data
def get_optimal(total_laps, models, pit_loss, lap_weather):
    return optimize_strategy(total_laps, models, pit_loss, lap_weather)


# --- Sidebar: pick a pulled race ---
st.sidebar.header("Race Selection")
data_dir = "data"
races_available = []
if os.path.exists(data_dir):
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)) and "_" in folder:
            year_part, country_part = folder.split("_", 1)
            races_available.append((year_part, country_part.replace("_", " ")))

if not races_available:
    st.sidebar.warning("No pulled races found in ./data/")
    st.info(
        "No race data yet. Pull one first:\n\n"
        "```\npython data_pipeline.py --year 2024 --country Italy\n```\n\n"
        "Then refresh this page."
    )
    st.stop()

races_available = sorted(set(races_available))
labels = [f"{y} {c}" for y, c in races_available]
choice = st.sidebar.selectbox("Choose a Grand Prix", labels)
year_str, country_str = choice.split(" ", 1)
year, country = int(year_str), country_str

race, laps_with_tires, total_laps, pit_loss, lap_weather, models, has_season_priors = get_race_context(year, country)
optimal = get_optimal(total_laps, models, pit_loss, lap_weather)

if not has_season_priors:
    st.caption(
        "⚠️ No season priors found for this year — compounds with no laps "
        "this race are excluded from the optimizer rather than estimated. "
        "Run `python season_priors.py --year " + str(year) + "` after pulling "
        "more races from this season to improve coverage."
    )

# --- Top metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Race Distance", f"{total_laps} laps")
col2.metric("Est. Pit Lane Loss", f"{pit_loss:.1f}s")
col3.metric("Rain During Race", "Yes 🌧️" if optimal["wet_race"] else "No ☀️")

st.subheader("🏁 Theoretically Optimal Strategy")
st.info(
    f"**{format_strategy(optimal['strategy'])}**  \n"
    f"Estimated total race time: **{optimal['total_time']/60:.2f} minutes**"
)

st.markdown("---")
left, right = st.columns([1, 1])

# --- Degradation chart ---
with left:
    st.subheader("📈 Tire Degradation by Compound")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {
        "SOFT": "#e10600", "MEDIUM": "#ffd700", "HARD": "#e0e0e0",
        "INTERMEDIATE": "#43b02a", "WET": "#0067ad",
    }
    for comp, model in models.items():
        df_comp = laps_with_tires[
            (laps_with_tires["compound"] == comp) & (laps_with_tires["tire_age"] > 0)
        ]
        max_age = int(df_comp["tire_age"].max()) if not df_comp.empty else 15
        ages = list(range(1, max(max_age, 5) + 1))
        fitted = [model["base_time"] + model["deg_rate"] * a for a in ages]
        ax.plot(ages, fitted, label=comp, color=colors.get(comp, "gray"), linewidth=2)
        if not df_comp.empty:
            ax.scatter(df_comp["tire_age"], df_comp["lap_duration"],
                       color=colors.get(comp, "gray"), alpha=0.15, s=15)
    ax.set_xlabel("Tire age (laps)")
    ax.set_ylabel("Lap time (s)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- Actual vs optimal table ---
with right:
    st.subheader("⏱️ Actual vs Optimal (per driver)")
    available_drivers = sorted(race["stints"]["driver_number"].unique())
    rows = []
    for dn in available_drivers:
        comp = compare_driver_to_optimal(
            dn, race["stints"], total_laps, models, pit_loss,
            optimal["total_time"], lap_weather,
        )
        if comp:
            rows.append({
                "Driver #": comp["driver_number"],
                "Actual Strategy": format_strategy(comp["actual_strategy"]),
                "Delta vs Optimal": f"{comp['delta_seconds']:+.1f}s",
                "_sort": comp["delta_seconds"],
            })
    if rows:
        df = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.write("No stint data available for this race.")

st.markdown("---")
st.subheader("🎲 Monte Carlo: How Often Does Each Strategy Actually Win?")
st.caption(
    "Instead of a single deterministic answer, runs each candidate through "
    "many randomized trials — real per-lap noise from each compound's own "
    "fitted scatter, plus a simple random-safety-car model — and reports "
    "how often each one actually comes out fastest."
)

available_drivers = sorted(race["stints"]["driver_number"].unique())
mc_drivers = st.multiselect(
    "Include these drivers' real strategies as candidates",
    available_drivers, default=available_drivers[:3],
)
n_trials = st.slider("Number of trials", 500, 5000, 2000, step=500)

if st.button("Run Monte Carlo Comparison"):
    candidates = {}
    if optimal["by_stop_count"][1]["strategy"]:
        candidates["Optimal 1-stop"] = optimal["by_stop_count"][1]["strategy"]
    if optimal["by_stop_count"][2]["strategy"]:
        candidates["Optimal 2-stop"] = optimal["by_stop_count"][2]["strategy"]
    for dn in mc_drivers:
        strat = reconstruct_actual_strategy(race["stints"], dn)
        if strat:
            candidates[f"Driver #{dn} actual"] = strat

    with st.spinner(f"Running {n_trials} trials per candidate..."):
        mc_results = run_monte_carlo(candidates, total_laps, models, pit_loss, lap_weather, n_trials=n_trials)

    sorted_results = sorted(mc_results.items(), key=lambda kv: -kv[1]["win_pct"])

    mc_left, mc_right = st.columns([1, 1])
    with mc_left:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        labels = [label for label, _ in sorted_results]
        win_pcts = [r["win_pct"] for _, r in sorted_results]
        ax.barh(labels[::-1], win_pcts[::-1], color="#e10600")
        ax.set_xlabel("Win rate (%)")
        ax.set_xlim(0, 100)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with mc_right:
        table_rows = [{
            "Strategy": label,
            "Win Rate": f"{r['win_pct']:.1f}%",
            "Median Time": f"{r['median_min']:.2f} min",
            "P10-P90": f"{r['p10_min']:.2f}-{r['p90_min']:.2f} min",
        } for label, r in sorted_results]
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    st.caption(
        f"Safety car probability per trial: {SAFETY_CAR_PROBABILITY:.0%} (documented "
        "estimate, not fitted). Candidates share the same random draws per trial, "
        "so differences reflect genuine strategy gaps, not lucky random numbers."
    )

st.markdown("---")
st.caption(
    "Model scope: linear tire degradation per compound, real weather-based "
    "wrong-tire penalties, fixed pit-lane loss. The strategy above ignores "
    "safety cars and traffic; the Monte Carlo section adds a simplified "
    "random safety car model, but track position/traffic are still not "
    "modeled anywhere on this page — see README for details."
)