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
    time spent stationary in the pit box, not the full pit-lane loss
    (entry, speed-limited transit, exit, versus staying flat-out on track).

    KNOWN LIMITATION: the extra lane-transit cost genuinely varies a lot
    by track (Monaco's pit lane costs far more than Spa's), and OpenF1
    doesn't expose it directly. 8s is a conservative, track-agnostic
    estimate (real values range roughly 5-12s at most circuits, with
    outliers like Monaco much higher) — it will under-estimate total pit
    loss at long-pit-lane tracks. If you want per-track accuracy, look up
    the circuit's real pit lane loss (widely published after each race)
    and pass it in instead of relying on this default.
    """
    LANE_TRANSIT_ESTIMATE_SEC = 8.0

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

# Used only when a joint fit isn't possible (fewer than 2 compounds with
# real data this race — can't separate compound identity from session
# progress with just one compound). Same rough literature estimate as
# before, now clearly labeled as the degraded fallback path, not the norm.
DEFAULT_PROGRESS_SLOPE = -0.05

# Typical per-lap noise (driver variability, traffic, small track changes)
# when a compound's own residual std can't be computed (e.g. season_prior/
# global_fallback compounds with no real local laps). Used by monte_carlo.py.
DEFAULT_RESIDUAL_STD = 0.35


def fit_joint_race_model(laps_with_tires: pd.DataFrame, min_laps_per_compound: int = 5) -> dict | None:
    """
    Jointly fits, in a SINGLE least-squares solve across ALL compounds with
    enough data in one race:

        lap_duration = base_time[compound] + deg_rate[compound] * tire_age
                        + progress_slope * lap_number

    progress_slope is SHARED across every compound and captures the
    combined effect of fuel burning off AND track evolution (rubber going
    down) — both cause "laps get faster as the race goes on," regardless
    of which tire is fitted. Estimating it from real data (instead of
    assuming a fixed constant) is what actually removes the confound: a
    fixed guess of e.g. 0.05s/lap isn't enough to correct for the fact that
    SOFT stints run early (heavy fuel, low track evolution) and HARD
    stints run late (light fuel, high track evolution) in nearly every
    race — a systematic pattern that persists even after averaging across
    many races, since it's not random noise.

    Returns None if fewer than 2 compounds have >= min_laps_per_compound
    real laps (can't jointly separate compound pace from session
    progress with only one compound), or if the design matrix is
    degenerate. Caller should fall back to the simpler per-compound fit.
    """
    if laps_with_tires.empty or "lap_number" not in laps_with_tires.columns:
        return None

    df = laps_with_tires[laps_with_tires["tire_age"] > 0].dropna(subset=["compound", "lap_duration", "lap_number"])
    counts = df["compound"].value_counts()
    usable = sorted(counts[counts >= min_laps_per_compound].index.tolist())
    if len(usable) < 2:
        return None
    df = df[df["compound"].isin(usable)]

    n_c = len(usable)
    intercept_cols = [(df["compound"] == c).astype(float).to_numpy() for c in usable]
    age_cols = [((df["compound"] == c).astype(float) * df["tire_age"]).to_numpy() for c in usable]
    lap_num_col = df["lap_number"].to_numpy(dtype=float)

    X = np.column_stack(intercept_cols + age_cols + [lap_num_col])
    y = df["lap_duration"].to_numpy(dtype=float)

    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None  # degenerate design, bail to fallback

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    base_times = coeffs[:n_c]
    deg_rates = coeffs[n_c:2 * n_c]
    progress_slope = float(coeffs[-1])

    residuals = y - X @ coeffs  # actual - predicted, per lap

    result = {"_progress_slope": progress_slope}
    for i, c in enumerate(usable):
        slope = float(deg_rates[i])
        if slope < 0:
            slope = 0.01  # residual noise outweighed wear in-sample; clip to keep sim sane
        comp_mask = (df["compound"] == c).to_numpy()
        comp_residual_std = float(np.std(residuals[comp_mask])) if comp_mask.sum() > 1 else DEFAULT_RESIDUAL_STD
        result[c] = {
            "base_time": float(base_times[i]),
            "deg_rate": slope,
            "lap_count": int(counts[c]),
            "residual_std": comp_residual_std,
        }
    return result


def fit_degradation_models(laps_with_tires: pd.DataFrame, season_priors: dict | None = None) -> dict:
    """
    Fits per-compound tire pace using fit_joint_race_model() when possible
    (>=2 compounds with real data this race, so session-progress effects
    can be separated from compound identity — see that function's
    docstring for why this matters). Falls back to independent
    per-compound fits with DEFAULT_PROGRESS_SLOPE when only one (or zero)
    compounds have real local data.

    season_priors (optional): output of season_priors.compute_season_priors().
    Used ONLY for compounds with no real local data this race. Estimates
    that compound's pace as this race's real MEDIUM pace + the compound's
    typical season-wide offset from MEDIUM (itself computed via the same
    joint-fit approach, race by race, before averaging). Falls back to a
    crude fastest-lap-anchored estimate if no season_priors are supplied
    or the compound isn't in them — that fallback is flagged
    'global_fallback' and excluded from the optimizer's search.

    Each model carries 'data_source': 'race_fit' | 'season_prior' | 'global_fallback'.
    """
    models = {}
    progress_slope = DEFAULT_PROGRESS_SLOPE

    joint = fit_joint_race_model(laps_with_tires)
    if joint is not None:
        progress_slope = joint.pop("_progress_slope")
        for comp, fit in joint.items():
            models[comp] = {
                **fit, "progress_slope": progress_slope,
                "has_real_data": True, "data_source": "race_fit",
            }
    else:
        # Fallback: independent per-compound fit, assuming DEFAULT_PROGRESS_SLOPE.
        # Less trustworthy — only hit when this race doesn't have enough
        # compound variety to jointly estimate session progress.
        compounds = laps_with_tires["compound"].dropna().unique() if not laps_with_tires.empty else []
        for comp in compounds:
            df_comp = laps_with_tires[laps_with_tires["compound"] == comp]
            df_comp = df_comp[df_comp["tire_age"] > 0]
            if len(df_comp) < 5:
                continue
            adjusted = df_comp["lap_duration"] - progress_slope * df_comp["lap_number"]
            slope, intercept = np.polyfit(df_comp["tire_age"], adjusted, 1)
            if slope < 0:
                slope = 0.01
            predicted = intercept + slope * df_comp["tire_age"]
            residual_std = float((adjusted - predicted).std()) if len(df_comp) > 1 else DEFAULT_RESIDUAL_STD
            models[comp] = {
                "base_time": float(intercept), "deg_rate": float(slope), "progress_slope": progress_slope,
                "has_real_data": True, "lap_count": int(len(df_comp)), "data_source": "race_fit",
                "residual_std": residual_std,
            }

    global_q1 = float(laps_with_tires["lap_duration"].quantile(0.1)) if not laps_with_tires.empty else 90.0
    fallback_default_slopes = {
        "SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.05,
        "INTERMEDIATE": 0.06, "WET": 0.04,
    }
    medium_base = models.get("MEDIUM", {}).get("base_time")

    for comp, default_slope in fallback_default_slopes.items():
        if comp in models:
            continue

        offset = None
        prior_slope = None
        if season_priors and comp in season_priors.get("relative_offset_vs_medium", {}):
            offset = season_priors["relative_offset_vs_medium"][comp]
            prior_slope = season_priors.get("avg_deg_rate", {}).get(comp)

        if offset is not None and medium_base is not None:
            models[comp] = {
                "base_time": medium_base + offset,
                "deg_rate": float(prior_slope) if prior_slope else default_slope,
                "progress_slope": progress_slope,
                "has_real_data": False, "lap_count": 0,
                "data_source": "season_prior",
                "residual_std": DEFAULT_RESIDUAL_STD,
            }
        else:
            models[comp] = {
                "base_time": global_q1, "deg_rate": float(default_slope), "progress_slope": progress_slope,
                "has_real_data": False, "lap_count": 0,
                "data_source": "global_fallback",
                "residual_std": DEFAULT_RESIDUAL_STD,
            }

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

    Predicted lap time = base_time + deg_rate*age + progress_slope*lap_number,
    matching exactly the model fit_degradation_models() fit (joint
    regression when possible — see fit_joint_race_model docstring for why
    progress_slope is estimated from data rather than assumed).

    Also adds a wrong-tire-for-conditions penalty per lap using real weather
    data: slicks on a wet lap, or wets/inters on a dry lap, each cost
    WRONG_TIRE_PENALTY_SEC extra — this is what lets the optimizer actually
    choose intermediates during rain instead of ignoring weather.

    Known scope cut: ignores safety cars and traffic/dirty air.
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
            lap_time += model.get("progress_slope", DEFAULT_PROGRESS_SLOPE) * lap_num

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

def _get_usable_compounds(models: dict) -> list:
    usable = [
        c for c in ALL_COMPOUNDS
        if models.get(c, {}).get("data_source") in ("race_fit", "season_prior")
    ]
    return usable if usable else list(ALL_COMPOUNDS)


def optimize_one_stop(total_laps: int, models: dict, pit_loss: float, lap_weather: dict | None = None) -> dict:
    """Best 1-stop strategy only. See optimize_strategy for shared details."""
    lap_weather = lap_weather or {}
    wet_race = race_has_rain(lap_weather)
    usable_compounds = _get_usable_compounds(models)
    best_time, best_strategy = float("inf"), None

    for c1 in usable_compounds:
        for c2 in usable_compounds:
            if c1 == c2 and not wet_race:
                continue
            for pit1 in range(5, total_laps - 5):
                strat = [(c1, pit1), (c2, total_laps)]
                t = simulate_strategy(strat, total_laps, models, pit_loss, lap_weather)
                if t < best_time:
                    best_time, best_strategy = t, strat

    return {"strategy": best_strategy, "total_time": best_time}


def optimize_two_stop(total_laps: int, models: dict, pit_loss: float, lap_weather: dict | None = None) -> dict:
    """Best 2-stop strategy only. See optimize_strategy for shared details."""
    lap_weather = lap_weather or {}
    wet_race = race_has_rain(lap_weather)
    usable_compounds = _get_usable_compounds(models)
    best_time, best_strategy = float("inf"), None

    for c1 in usable_compounds:
        for c2 in usable_compounds:
            for c3 in usable_compounds:
                if len(set([c1, c2, c3])) < 2 and not wet_race:
                    continue
                for pit1 in range(5, total_laps - 10):
                    for pit2 in range(pit1 + 5, total_laps - 5):
                        strat = [(c1, pit1), (c2, pit2), (c3, total_laps)]
                        t = simulate_strategy(strat, total_laps, models, pit_loss, lap_weather)
                        if t < best_time:
                            best_time, best_strategy = t, strat

    return {"strategy": best_strategy, "total_time": best_time}


def optimize_strategy(
    total_laps: int, models: dict, pit_loss: float, lap_weather: dict | None = None,
) -> dict:
    """
    Brute-forces 1-stop and 2-stop combinations and returns the overall
    best, plus each stop-count's own best under 'by_stop_count' — useful
    as distinct candidates for monte_carlo.py rather than only a single
    global answer. Compound universe includes wet-weather tires so the
    optimizer can actually pick them when lap_weather shows rain. The
    "must use 2 different compounds" FIA rule is only enforced for fully
    dry races (rain races are exempt in reality).

    IMPORTANT: only searches compounds with data_source in
    ('race_fit', 'season_prior') — i.e. compounds backed by either real
    laps this race, or a season-wide relative-pace prior anchored to this
    race's own MEDIUM pace. 'global_fallback' compounds (no local data AND
    no season prior available) are excluded, since their pace estimate is
    untrustworthy. If every compound is global_fallback (degenerate case,
    e.g. testing on a tiny synthetic dataset), falls back to searching
    everything so the function never returns an empty result.
    """
    one_stop = optimize_one_stop(total_laps, models, pit_loss, lap_weather)
    two_stop = optimize_two_stop(total_laps, models, pit_loss, lap_weather)

    candidates = [c for c in (one_stop, two_stop) if c["strategy"] is not None]
    best = min(candidates, key=lambda c: c["total_time"]) if candidates else {"strategy": None, "total_time": float("inf")}

    return {
        "strategy": best["strategy"],
        "total_time": best["total_time"],
        "wet_race": race_has_rain(lap_weather or {}),
        "by_stop_count": {1: one_stop, 2: two_stop},
    }


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
    import json
    import os

    parser = argparse.ArgumentParser(description="F1 Strategy Optimizer")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--drivers", type=int, nargs="*", default=None,
                         help="Driver numbers to compare, e.g. --drivers 1 44 16. "
                              "Defaults to the first 5 drivers found.")
    parser.add_argument("--no-season-priors", action="store_true",
                         help="Ignore data/season_priors_<year>.json even if it exists.")
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

    season_priors = None
    priors_path = os.path.join("data", f"season_priors_{args.year}.json")
    if not args.no_season_priors and os.path.exists(priors_path):
        with open(priors_path) as f:
            season_priors = json.load(f)
        print(f"Loaded season priors from {priors_path}\n")
    else:
        print(f"No season priors found at {priors_path} "
              f"(run season_priors.py to build one) - untested compounds will "
              f"be excluded from the optimizer instead of estimated.\n")

    print("Fitting tire degradation profiles...")
    models = fit_degradation_models(laps_with_tires, season_priors=season_priors)
    for comp, stats in models.items():
        if stats["data_source"] == "race_fit":
            source = f"{stats['lap_count']} real laps this race"
        elif stats["data_source"] == "season_prior":
            source = "season prior (no local data)"
        else:
            source = "NO DATA - excluded from optimizer"
        print(f"  {comp}: Base Pace = {stats['base_time']:.2f}s | Deg Rate = +{stats['deg_rate']:.3f}s/lap | ({source})")

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