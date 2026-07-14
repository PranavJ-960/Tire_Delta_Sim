"""
Usage:
    python data_pipeline.py --year 2024 --country Italy   # pull data first
    python optimizer.py --year 2024 --country Italy
"""

import argparse
import numpy as np
import pandas as pd
from explore import load_race, clean_laps, attach_tire_age


def compute_pit_loss(pits_df: pd.DataFrame) -> float:
    """
    Typical time loss from a pit stop. Uses the median 'pit_duration'
    (OpenF1's pit_duration is total time lost in the pit lane).
    Defaults to 22.0s if no usable data exists.
    """
    if pits_df.empty or "pit_duration" not in pits_df.columns:
        return 22.0
    valid_pits = pits_df[pits_df["pit_duration"] < 50.0]  # drop retirements/red flags
    if valid_pits.empty:
        return 22.0
    return float(valid_pits["pit_duration"].median())


def fit_degradation_models(laps_with_tires: pd.DataFrame) -> dict:
    """
    Fits a linear model (LapTime = base_time + deg_rate * age) per compound.

    CRITICAL FIX: tire_age == 0 laps are dropped before fitting. That row
    is either the standing-start lap (pack bunching, no DRS trains yet) or
    a pit out-lap — both are artificially slow for reasons that have
    nothing to do with tire wear, and drag the fitted intercept down. 
    Age 1 onward represents genuine green-flag pace.
    """
    models = {}
    compounds = laps_with_tires["compound"].dropna().unique()
    global_q1 = laps_with_tires["lap_duration"].quantile(0.1)

    for comp in compounds:
        df_comp = laps_with_tires[laps_with_tires["compound"] == comp]
        df_comp = df_comp[df_comp["tire_age"] > 0]  # Drop out-lap and start-lap noise
        if len(df_comp) < 10:
            continue

        slope, intercept = np.polyfit(df_comp["tire_age"], df_comp["lap_duration"], 1)
        if slope < 0:
            # Negative slope usually means track evolution / fuel burn
            # outweighing tire wear in the sample. Clip to keep the sim sane.
            slope = 0.01

        models[comp] = {"base_time": float(intercept), "deg_rate": float(slope)}

    # Fallback profiles if a compound didn't run enough laps to fit cleanly
    for fallback_comp, default_slope in [("SOFT", 0.12), ("MEDIUM", 0.08), ("HARD", 0.05)]:
        if fallback_comp not in models:
            models[fallback_comp] = {"base_time": global_q1, "deg_rate": default_slope}

    return models


def simulate_strategy(strategy: list, total_laps: int, models: dict, pit_loss: float) -> float:
    """
    Calculates total race duration given a specific compound strategy layout.
    strategy: e.g. [('MEDIUM', 20), ('HARD', 52)] = Medium laps 1-20, Hard 21-52.
    
    NOTE: This focuses entirely on structural tire degradation + pit loss. 
    Safety cars, VSCs, and linear fuel-burn pace evolution are explicitly excluded.
    """
    total_time = 0.0
    current_lap = 1
    stops = len(strategy) - 1
    total_time += stops * pit_loss

    for compound, end_lap in strategy:
        stint_laps = end_lap - current_lap + 1
        model = models.get(compound, models.get("MEDIUM"))
        
        # Consistent Index Alignment: Since age=0 is excluded from the regression fit, 
        # range(1, stint_laps + 1) matches the physical progression of green-flag racing laps.
        for age in range(1, stint_laps + 1):
            lap_time = model["base_time"] + (model["deg_rate"] * age)
            total_time += lap_time
        current_lap = end_lap + 1

    return total_time


def optimize_strategy(total_laps: int, models: dict, pit_loss: float) -> dict:
    """Brute-force explores 1-stop and 2-stop combinations across SOFT/MEDIUM/HARD."""
    best_time = float("inf")
    best_strategy = None
    compounds = ["SOFT", "MEDIUM", "HARD"]

    # --- 1-STOP PERMUTATIONS ---
    for c1 in compounds:
        for c2 in compounds:
            if c1 == c2:  # F1 Sporting Regs: Must use 2 unique compounds if dry
                continue
            for pit1 in range(5, total_laps - 5):
                strat = [(c1, pit1), (c2, total_laps)]
                t = simulate_strategy(strat, total_laps, models, pit_loss)
                if t < best_time:
                    best_time, best_strategy = t, strat

    # --- 2-STOP PERMUTATIONS ---
    for c1 in compounds:
        for c2 in compounds:
            for c3 in compounds:
                if len(set([c1, c2, c3])) < 2:  # Verify compound variety rule
                    continue
                for pit1 in range(5, total_laps - 10):
                    for pit2 in range(pit1 + 5, total_laps - 5):
                        strat = [(c1, pit1), (c2, pit2), (c3, total_laps)]
                        t = simulate_strategy(strat, total_laps, models, pit_loss)
                        if t < best_time:
                            best_time, best_strategy = t, strat

    return {"strategy": best_strategy, "total_time": best_time}


def reconstruct_actual_strategy(stints_df: pd.DataFrame, driver_number: int) -> list:
    """
    Extracts a driver's historical stints from stints.csv and formats them into
    the standard strategy list structure [(compound, end_lap), ...] expected by the simulator.
    """
    driver_stints = (
        stints_df[stints_df["driver_number"] == driver_number]
        .sort_values("lap_start")
    )
    return [(str(row["compound"]).upper(), int(row["lap_end"])) for _, row in driver_stints.iterrows()]


def compare_driver_to_optimal(
    driver_number: int, stints_df: pd.DataFrame, total_laps: int,
    models: dict, pit_loss: float, optimal_time: float,
) -> dict:
    """
    Reconstructs a real driver's race timeline, executes it inside our mathematical 
    degradation framework, and calculates the variance relative to the absolute optimum.
    """
    actual_strategy = reconstruct_actual_strategy(stints_df, driver_number)
    if not actual_strategy:
        return None
        
    # Standardize edge case where final stint end_lap slightly mismatches due to data truncations
    if actual_strategy[-1][1] != total_laps:
        actual_strategy[-1] = (actual_strategy[-1][0], total_laps)
        
    actual_time = simulate_strategy(actual_strategy, total_laps, models, pit_loss)
    return {
        "driver_number": driver_number,
        "actual_strategy": actual_strategy,
        "actual_time": actual_time,
        "optimal_time": optimal_time,
        "delta_seconds": actual_time - optimal_time,
    }


def format_strategy(strategy: list) -> str:
    """Converts a strategy list into a clean string layout for standard printing."""
    parts = []
    start = 1
    for compound, end_lap in strategy:
        parts.append(f"{compound} (L{start}-{end_lap})")
        start = end_lap + 1
    return " ➔ ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Strategy Optimizer Backend")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--drivers", type=int, nargs="*", default=None,
                        help="Driver numbers to compare, e.g. --drivers 1 44 16. "
                             "Defaults to the first 5 unique drivers found.")
    args = parser.parse_args()

    # Data ingestion layer
    race = load_race(args.year, args.country)
    laps_cleaned = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps_cleaned, race["stints"])

    total_laps = int(race["laps"]["lap_number"].max())
    pit_loss = compute_pit_loss(race["pits"])

    print(f"--- Strategy Analysis Suite: {args.year} {args.country} GP ---")
    print(f"Total Race Distance: {total_laps} Laps")
    print(f"Calculated Pit Lane Loss Window: {pit_loss:.2f} seconds\n")

    print("Fitting compound-specific linear regression profiles...")
    models = fit_degradation_models(laps_with_tires)
    for comp, stats in models.items():
        print(f"  {comp:<7} | Intercept (Base Pace): {stats['base_time']:.3f}s | Slope (Deg Rate): +{stats['deg_rate']:.4f}s/lap")

    print("\nRunning permutations engine (Brute-forcing math strategies)...")
    result = optimize_strategy(total_laps, models, pit_loss)

    print("\n================ THEORETICALLY OPTIMAL STRATEGY ================")
    print(format_strategy(result["strategy"]))
    print(f"Calculated Racing Frame: {result['total_time']/60:.2f} minutes")
    print("================================================================")

    # --- Day 4 Optimization Analysis ---
    print("\n================ DETAILED VARIANCE BREAKDOWN (ACTUAL VS OPTIMAL) ================")
    available_drivers = sorted(race["stints"]["driver_number"].unique())
    driver_numbers = args.drivers if args.drivers is not None else available_drivers[:5]
    
    comparisons = []
    for dn in driver_numbers:
        comp = compare_driver_to_optimal(
            dn, race["stints"], total_laps, models, pit_loss, result["total_time"]
        )
        if comp:
            comparisons.append(comp)

    if comparisons:
        # Sort by the delta to show who got closest to the theoretical performance floor
        for c in sorted(comparisons, key=lambda x: x["delta_seconds"]):
            print(f"Driver #{c['driver_number']:<2}: {format_strategy(c['actual_strategy'])}")
            print(f"   Simulated Expected Pace: {c['actual_time']/60:.2f} min ({c['delta_seconds']:+.2f}s relative to optimum)")
    else:
        print("No strategy profiles resolved for target driver keys.")
    print("=================================================================================\n")