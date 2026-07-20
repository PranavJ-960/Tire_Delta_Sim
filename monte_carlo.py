"""
Monte Carlo strategy comparison.

Instead of one deterministic "optimal" strategy, this runs several
candidate strategies through thousands of randomized race simulations and
reports how often each one actually wins — a more honest claim than a
single point estimate, and one professional strategy groups actually use
in practice (see README's "Known scope cuts" for why: real races have
uncertainty this model can't otherwise represent).

Two sources of randomness, both tied to real data or well-known effects:
1. Per-lap noise, drawn from each compound's OWN fitted residual std
   (fit_joint_race_model computes this from real lap-time scatter — not
   a made-up number).
2. A simple safety car model: with SAFETY_CAR_PROBABILITY chance, one
   random safety car occurs at a random lap. Any strategy with a
   scheduled pit stop within SC_PIT_WINDOW laps of it gets a discounted
   pit loss (pitting under a safety car is well known to be far cheaper,
   since the whole field is slowed). This is a deliberately simple model
   of a real effect, not a precise reproduction — see README.

Usage:
    python data_pipeline.py --year 2024 --country Italy
    python monte_carlo.py --year 2024 --country Italy
"""

import argparse
import numpy as np
import pandas as pd

from explore import load_race, clean_laps, attach_tire_age
from optimizer import (
    compute_pit_loss, build_lap_weather, fit_degradation_models,
    optimize_strategy, is_wet_lap, DRY_COMPOUNDS, WET_COMPOUNDS,
    WRONG_TIRE_PENALTY_SEC, DEFAULT_RESIDUAL_STD, format_strategy,
    reconstruct_actual_strategy,
)

# Rough real-world frequency: a safety car (or VSC) appears in roughly a
# third to a half of F1 races. 0.35 is a reasonable, documented estimate,
# not a fitted value — there's no per-race SC probability data available
# from OpenF1 to calibrate this more precisely.
SAFETY_CAR_PROBABILITY = 0.35
SC_PIT_WINDOW = 3            # laps around the SC event that count as "caught the window"
SC_PIT_LOSS_MULTIPLIER = 0.35  # pitting under SC costs ~1/3 of normal pit loss


def simulate_strategy_once(
    strategy: list, total_laps: int, models: dict, pit_loss: float,
    lap_weather: dict, rng: np.random.Generator, safety_car_lap: int | None,
) -> float:
    """
    One randomized run of a strategy: same lap-time model as
    optimizer.simulate_strategy, plus per-lap gaussian noise from each
    compound's own residual_std, plus a discounted pit loss for any stop
    that lands within SC_PIT_WINDOW laps of safety_car_lap (if one occurred
    this trial).
    """
    total_time = 0.0
    current_lap = 1

    for compound, end_lap in strategy:
        model = models.get(compound, models.get("MEDIUM"))
        stint_laps = end_lap - current_lap + 1
        for age in range(1, stint_laps + 1):
            lap_num = current_lap + age - 1
            lap_time = model["base_time"] + model["deg_rate"] * age
            lap_time += model.get("progress_slope", -0.05) * lap_num
            lap_time += rng.normal(0, model.get("residual_std", DEFAULT_RESIDUAL_STD))

            wet = is_wet_lap(lap_num, lap_weather)
            if wet and compound in DRY_COMPOUNDS:
                lap_time += WRONG_TIRE_PENALTY_SEC
            elif not wet and compound in WET_COMPOUNDS:
                lap_time += WRONG_TIRE_PENALTY_SEC

            total_time += lap_time

        # Pit stop at the end of this stint (except the final stint)
        if end_lap < total_laps:
            this_pit_loss = pit_loss
            if safety_car_lap is not None and abs(end_lap - safety_car_lap) <= SC_PIT_WINDOW:
                this_pit_loss = pit_loss * SC_PIT_LOSS_MULTIPLIER
            total_time += this_pit_loss

        current_lap = end_lap + 1

    return total_time


def run_monte_carlo(
    candidates: dict, total_laps: int, models: dict, pit_loss: float,
    lap_weather: dict | None = None, n_trials: int = 3000, seed: int = 42,
) -> dict:
    """
    candidates: {label: strategy_list}, e.g.
        {"Optimal 1-stop": [...], "Optimal 2-stop": [...], "Driver #1 actual": [...]}

    Returns per-label: win_count, win_pct, mean/median/p10/p90 finish time.
    Every candidate is run through the SAME n_trials random draws (same
    noise seed per trial across strategies) so comparisons are apples to
    apples — this is the standard "common random numbers" variance
    reduction technique, not just cosmetic.
    """
    lap_weather = lap_weather or {}
    rng = np.random.default_rng(seed)
    labels = list(candidates.keys())
    times = {label: np.empty(n_trials) for label in labels}
    wins = {label: 0 for label in labels}

    for trial in range(n_trials):
        safety_car_lap = None
        if rng.random() < SAFETY_CAR_PROBABILITY:
            safety_car_lap = int(rng.integers(8, max(9, total_laps - 8)))

        trial_times = {}
        for label, strategy in candidates.items():
            if strategy is None:
                trial_times[label] = float("inf")
                continue
            t = simulate_strategy_once(
                strategy, total_laps, models, pit_loss, lap_weather, rng, safety_car_lap,
            )
            times[label][trial] = t
            trial_times[label] = t

        winner = min(trial_times, key=trial_times.get)
        wins[winner] += 1

    results = {}
    for label in labels:
        arr = times[label]
        results[label] = {
            "win_pct": 100.0 * wins[label] / n_trials,
            "mean_min": float(np.mean(arr)) / 60,
            "median_min": float(np.median(arr)) / 60,
            "p10_min": float(np.percentile(arr, 10)) / 60,
            "p90_min": float(np.percentile(arr, 90)) / 60,
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo strategy comparison")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--drivers", type=int, nargs="*", default=None,
                         help="Driver numbers whose actual strategy to include as a candidate. "
                              "Defaults to the first 3 drivers found.")
    args = parser.parse_args()

    import json
    import os

    race = load_race(args.year, args.country)
    laps_cleaned = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps_cleaned, race["stints"])
    total_laps = int(race["laps"]["lap_number"].max())
    pit_loss = compute_pit_loss(race["pits"])
    lap_weather = build_lap_weather(race["laps"], race["weather"])

    season_priors = None
    priors_path = os.path.join("data", f"season_priors_{args.year}.json")
    if os.path.exists(priors_path):
        with open(priors_path) as f:
            season_priors = json.load(f)

    models = fit_degradation_models(laps_with_tires, season_priors=season_priors)
    opt = optimize_strategy(total_laps, models, pit_loss, lap_weather)

    candidates = {}
    if opt["by_stop_count"][1]["strategy"]:
        candidates["Optimal 1-stop"] = opt["by_stop_count"][1]["strategy"]
    if opt["by_stop_count"][2]["strategy"]:
        candidates["Optimal 2-stop"] = opt["by_stop_count"][2]["strategy"]

    driver_numbers = args.drivers or sorted(race["stints"]["driver_number"].unique())[:3]
    for dn in driver_numbers:
        strat = reconstruct_actual_strategy(race["stints"], dn)
        if strat:
            candidates[f"Driver #{dn} actual"] = strat

    print(f"--- Monte Carlo Strategy Comparison: {args.year} {args.country} GP ---")
    print(f"Running {args.trials} randomized trials per candidate "
          f"(safety car probability: {SAFETY_CAR_PROBABILITY:.0%})\n")
    for label, strat in candidates.items():
        print(f"  {label}: {format_strategy(strat)}")

    results = run_monte_carlo(candidates, total_laps, models, pit_loss, lap_weather, n_trials=args.trials)

    print("\n================ RESULTS (sorted by win %) ================")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["win_pct"]):
        print(f"{label}:")
        print(f"    Win rate: {r['win_pct']:.1f}%")
        print(f"    Median time: {r['median_min']:.2f} min "
              f"(P10-P90: {r['p10_min']:.2f}-{r['p90_min']:.2f} min)")
    print("=============================================================")