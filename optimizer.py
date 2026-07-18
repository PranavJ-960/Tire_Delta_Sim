"""
Day 3-5: degradation model, weather-aware strategy simulation, optimizer,
and actual-vs-optimal comparison.

Usage:
    python data_pipeline.py --year 2024 --country Italy   # pull data first
    python optimizer.py --year 2024 --country Italy
"""

import argparse
import numpy as np
import pandas as pd
from explore import load_race, clean_laps, attach_tire_age

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
ALL_COMPOUNDS = DRY_COMPOUNDS + WET_COMPOUNDS

# Applied per lap when a car is on the wrong tire family for the conditions.
# These are deliberately large so the optimizer avoids these combos outright
# rather than "mostly" avoiding them.
WRONG_TIRE_PENALTY_SEC = 15.0


# --------------------------------------------------------------------------
# Pit loss
# --------------------------------------------------------------------------

def compute_pit_loss(pits_df: pd.DataFrame) -> float:
    """
    Typical time lost to a pit stop, in seconds. OpenF1's pit_duration is
    time spent stationary in the pit box — NOT the full pit-lane time loss
    (which also includes the speed-limited drive through the lane and the
    time lost relative to staying flat-out on track). We add a fixed
    estimate for that lane-transit cost on top of the stationary time.
    """
    LANE_TRANSIT_ESTIMATE_SEC = 20.0  # rough, track-independent estimate

    if pits_df.empty or "pit_duration" not in pits_df.columns:
        return 22.0 + LANE_TRANSIT_ESTIMATE_SEC
    valid_pits = pits_df[pits_df["pit_duration"].between(1.0, 50.0)]
    if valid_pits.empty:
        return 22.0 + LANE_TRANSIT_ESTIMATE_SEC
    stationary = float(valid_pits["pit_duration"].median())
    return stationary + LANE_TRANSIT_ESTIMATE_SEC


# --------------------------------------------------------------------------
# Degradation model (linear, per compound)
# --------------------------------------------------------------------------

def fit_degradation_models(laps_with_tires: pd.DataFrame) -> dict:
    """
    Fits LapTime = base_time + deg_rate * tire_age per compound, using
    ordinary linear regression (np.polyfit, degree 1).

    tire_age == 0 laps are dropped before fitting — that row is either the
    standing-start lap (pack bunching, no DRS trains yet) or a pit out-lap,
    both artificially slow for reasons unrelated to tire wear.
    """
    models = {}
    compounds = laps_with_tires["compound"].dropna().unique() if not laps_with_tires.empty else []
    global_q1 = float(laps_with_tires["lap_duration"].quantile(0.1)) if not laps_with_tires.empty else 90.0

    for comp in compounds:
        df_comp = laps_with_tires[laps_with_tires["compound"] == comp]
        df_comp = df_comp[df_comp["tire_age"] > 0]
        if len(df_comp) < 5:
            continue
        slope, intercept = np.polyfit(df_comp["tire_age"], df_comp["lap_duration"], 1)
        if slope < 0:
            slope = 0.01  # track evolution/fuel burn outweighed wear in-sample; clip to keep sim sane
        models[comp] = {"base_time": float(intercept), "deg_rate": float(slope)}

    # Fallback profiles for compounds that didn't run (e.g. no rain -> no WET data)
    fallback_defaults = {
        "SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.05,
        "INTERMEDIATE": 0.06, "WET": 0.04,
    }
    for comp, default_slope in fallback_defaults.items():
        if comp not in models:
            models[comp] = {"base_time": global_q1, "deg_rate": default_slope}

    return models


# --------------------------------------------------------------------------
# Weather join — THE FIX: laps use 'date_start', weather uses 'date'
# --------------------------------------------------------------------------

def build_lap_weather(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict:
    """
    Maps each lap_number -> {rainfall, track_temperature, air_temperature}
    by matching lap start time to the nearest preceding weather sample.

    OpenF1 field names (confirmed from their docs):
      - laps.date_start   <- lap's own timestamp column
      - weather.date      <- weather sample's timestamp column
    Using 'date' for both (an easy assumption to make) silently returns
    an empty dict, which is why weather never influenced anything before.
    """
    if laps_df.empty or weather_df.empty:
        return {}
    if "date_start" not in laps_df.columns or "date" not in weather_df.columns:
        return {}

    fields = [f for f in ("rainfall", "track_temperature", "air_temperature") if f in weather_df.columns]
    if not fields or "lap_number" not in laps_df.columns:
        return {}

    laps = laps_df[["date_start", "lap_number"]].copy().rename(columns={"date_start": "date"})
    weather = weather_df[["date", *fields]].copy()

    # format="ISO8601" is required here: OpenF1 timestamps mix precision
    # (some rows have fractional seconds, some don't), and pandas'
    # format-inference can silently produce NaT for every row after the
    # first if it locks onto the wrong format from row 0.
    laps["date"] = pd.to_datetime(laps["date"], errors="coerce", utc=True, format="ISO8601")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce", utc=True, format="ISO8601")
    laps = laps.dropna(subset=["date"]).sort_values("date")
    weather = weather.dropna(subset=["date"]).sort_values("date")

    merged = pd.merge_asof(
        laps, weather, on="date", direction="backward",
        tolerance=pd.Timedelta(minutes=20),
    )

    by_lap = merged.groupby("lap_number")[fields].mean()
    return {int(lap): row.dropna().to_dict() for lap, row in by_lap.iterrows()}


def is_wet_lap(lap_number: int, lap_weather: dict) -> bool:
    """OpenF1 reports rainfall as 0/1. Treat >=1 as 'wet enough to matter'."""
    sample = lap_weather.get(lap_number, {})
    return float(sample.get("rainfall", 0.0)) >= 1.0


def race_has_rain(lap_weather: dict) -> bool:
    return any(is_wet_lap(lap, lap_weather) for lap in lap_weather)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_strategy(
    strategy: list, total_laps: int, models: dict, pit_loss: float,
    lap_weather: dict | None = None,
) -> float:
    """
    strategy: e.g. [('MEDIUM', 20), ('HARD', 52)] = Medium laps 1-20, Hard 21-52.

    Adds a wrong-tire-for-conditions penalty per lap using real weather data:
    slicks (SOFT/MEDIUM/HARD) on a wet lap, or wets/inters on a dry lap,
    each cost WRONG_TIRE_PENALTY_SEC extra — this is what lets the optimizer
    actually choose intermediates during rain instead of ignoring weather.

    Known scope cut: ignores safety cars and in-race fuel burn.
    """
    lap_weather = lap_weather or {}
    total_time = (len(strategy) - 1) * pit_loss
    current_lap = 1

    for compound, end_lap in strategy:
        model = models.get(compound, models.get("MEDIUM"))
        stint_laps = end_lap - current_lap + 1
        for age in range(1, stint_laps + 1):
            lap_num = current_lap + age - 1
            lap_time = model["base_time"] + model["deg_rate"] * age

            wet = is_wet_lap(lap_num, lap_weather)
            if wet and compound in DRY_COMPOUNDS:
                lap_time += WRONG_TIRE_PENALTY_SEC
            elif not wet and compound in WET_COMPOUNDS:
                lap_time += WRONG_TIRE_PENALTY_SEC

            total_time += lap_time
        current_lap = end_lap + 1

    return total_time


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

def optimize_strategy(
    total_laps: int, models: dict, pit_loss: float, lap_weather: dict | None = None,
) -> dict:
    """
    Brute-forces 1-stop and 2-stop combinations. Compound universe includes
    wet-weather tires so the optimizer can actually pick them when
    lap_weather shows rain. The "must use 2 different compounds" FIA rule
    is only enforced for fully dry races (rain races are exempt in reality).
    """
    lap_weather = lap_weather or {}
    wet_race = race_has_rain(lap_weather)
    best_time, best_strategy = float("inf"), None

    # 1-stop
    for c1 in ALL_COMPOUNDS:
        for c2 in ALL_COMPOUNDS:
            if c1 == c2 and not wet_race:
                continue
            for pit1 in range(5, total_laps - 5):
                strat = [(c1, pit1), (c2, total_laps)]
                t = simulate_strategy(strat, total_laps, models, pit_loss, lap_weather)
                if t < best_time:
                    best_time, best_strategy = t, strat

    # 2-stop
    for c1 in ALL_COMPOUNDS:
        for c2 in ALL_COMPOUNDS:
            for c3 in ALL_COMPOUNDS:
                if len(set([c1, c2, c3])) < 2 and not wet_race:
                    continue
                for pit1 in range(5, total_laps - 10):
                    for pit2 in range(pit1 + 5, total_laps - 5):
                        strat = [(c1, pit1), (c2, pit2), (c3, total_laps)]
                        t = simulate_strategy(strat, total_laps, models, pit_loss, lap_weather)
                        if t < best_time:
                            best_time, best_strategy = t, strat

    return {"strategy": best_strategy, "total_time": best_time, "wet_race": wet_race}


# --------------------------------------------------------------------------
# Actual vs optimal comparison
# --------------------------------------------------------------------------

def reconstruct_actual_strategy(stints_df: pd.DataFrame, driver_number: int) -> list:
    """Turn one driver's stints.csv rows into [(compound, end_lap), ...]."""
    driver_stints = (
        stints_df[stints_df["driver_number"] == driver_number]
        .dropna(subset=["compound", "lap_end"])
        .sort_values("lap_start")
    )
    return [(str(row["compound"]).upper(), int(row["lap_end"])) for _, row in driver_stints.iterrows()]


def compare_driver_to_optimal(
    driver_number: int, stints_df: pd.DataFrame, total_laps: int,
    models: dict, pit_loss: float, optimal_time: float, lap_weather: dict | None = None,
) -> dict:
    actual_strategy = reconstruct_actual_strategy(stints_df, driver_number)
    if not actual_strategy:
        return None
    actual_time = simulate_strategy(actual_strategy, total_laps, models, pit_loss, lap_weather)
    return {
        "driver_number": driver_number,
        "actual_strategy": actual_strategy,
        "actual_time": actual_time,
        "optimal_time": optimal_time,
        "delta_seconds": actual_time - optimal_time,
    }


def format_strategy(strategy: list) -> str:
    parts, start = [], 1
    for compound, end_lap in strategy:
        parts.append(f"{compound}(L{start}-{end_lap})")
        start = end_lap + 1
    return " -> ".join(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Strategy Optimizer")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--drivers", type=int, nargs="*", default=None,
                         help="Driver numbers to compare, e.g. --drivers 1 44 16. "
                              "Defaults to the first 5 drivers found.")
    args = parser.parse_args()

    race = load_race(args.year, args.country)
    laps_cleaned = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps_cleaned, race["stints"])
    total_laps = int(race["laps"]["lap_number"].max())

    pit_loss = compute_pit_loss(race["pits"])
    lap_weather = build_lap_weather(race["laps"], race["weather"])

    print(f"--- Strategy Analysis for {args.year} {args.country} GP ---")
    print(f"Total Race Distance: {total_laps} Laps")
    print(f"Calculated Pit Lane Loss: {pit_loss:.2f} seconds")
    print(f"Weather data mapped for {len(lap_weather)} laps")
    print(f"Rain detected during this race: {race_has_rain(lap_weather)}\n")

    print("Fitting tire degradation profiles...")
    models = fit_degradation_models(laps_with_tires)
    for comp, stats in models.items():
        print(f"  {comp}: Base Pace = {stats['base_time']:.2f}s | Deg Rate = +{stats['deg_rate']:.3f}s/lap")

    print("\nRunning strategy simulation (brute-forcing combinations)...")
    result = optimize_strategy(total_laps, models, pit_loss, lap_weather)

    print("\n================ THEORETICALLY OPTIMAL STRATEGY ================")
    print(format_strategy(result["strategy"]))
    print(f"Estimated Total Racing Time: {result['total_time']/60:.2f} minutes")
    print("================================================================")

    print("\n================ ACTUAL vs OPTIMAL (per driver) ================")
    driver_numbers = args.drivers or sorted(race["stints"]["driver_number"].unique())[:5]
    comparisons = []
    for dn in driver_numbers:
        comp = compare_driver_to_optimal(
            dn, race["stints"], total_laps, models, pit_loss, result["total_time"], lap_weather
        )
        if comp:
            comparisons.append(comp)

    if comparisons:
        for c in sorted(comparisons, key=lambda x: x["delta_seconds"]):
            print(f"Driver #{c['driver_number']}: {format_strategy(c['actual_strategy'])}")
            print(f"    Estimated actual time: {c['actual_time']/60:.2f} min "
                  f"({c['delta_seconds']:+.1f}s vs optimal)")
    else:
        print("No stint data found for the requested drivers.")
    print("==================================================================")