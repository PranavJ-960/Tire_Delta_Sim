"""
Pools every pulled race for a season to compute a stable, physically
grounded relative pace offset per compound vs MEDIUM (fuel-corrected).

Why this exists: a single race often doesn't run every compound (e.g. a
dry race never runs WET tires), so that race alone can't tell you how fast
WET "would have been." Pooling many races gives a much more reliable
answer to "how much slower is HARD than MEDIUM, on average" than either
guessing a constant or anchoring to a single race's fastest lap.

Prerequisite: you need multiple races already pulled via data_pipeline.py
or season_pull.py before this has anything useful to compute from.

Usage:
    python season_pull.py --year 2024                # pull races first
    python season_priors.py --year 2024               # then build priors
    python optimizer.py --year 2024 --country Italy   # auto-loads them
"""

import argparse
import json
import os
import numpy as np

from explore import load_race, clean_laps, attach_tire_age
from optimizer import fit_joint_race_model


def fit_race_compounds(year: int, country: str, data_dir: str = "data") -> dict:
    """
    Per-compound fit for a single race, keyed by compound, using the same
    joint session-progress regression as optimizer.fit_degradation_models
    (see fit_joint_race_model's docstring — this is what correctly
    separates compound pace from fuel-burn/track-evolution effects,
    instead of assuming a fixed constant).
    """
    race = load_race(year, country, data_dir)
    laps = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps, race["stints"])

    joint = fit_joint_race_model(laps_with_tires)
    if joint is None:
        return {}
    joint.pop("_progress_slope", None)
    return joint


def compute_season_priors(year: int, data_dir: str = "data", reference_age: int = 5) -> dict:
    """
    Loops every pulled race for `year`, fits each compound present, and
    computes each compound's average pace offset from MEDIUM at a fixed
    reference tire age (so stints of different lengths are compared fairly).

    Returns:
      {
        "relative_offset_vs_medium": {"SOFT": -1.2, "HARD": 1.4, "WET": 14.1, ...},
        "avg_deg_rate": {"SOFT": 0.11, "HARD": 0.05, ...},
        "races_used": {"SOFT": 8, "HARD": 12, "WET": 2, ...},  # how many races contributed
        "year": 2024,
      }

    A compound only gets an offset if at least one race had BOTH that
    compound and MEDIUM with real fits — offsets are always relative to
    that race's own MEDIUM pace before averaging, which cancels out
    track-to-track pace differences (Monza vs Monaco) and isolates the
    tire-to-tire gap.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No '{data_dir}/' folder found — pull some races first.")

    races = []
    for folder in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, folder)) or "_" not in folder:
            continue
        year_part, country_part = folder.split("_", 1)
        if year_part == str(year):
            races.append(country_part.replace("_", " "))

    if not races:
        raise ValueError(f"No pulled races found for {year} in {data_dir}/")

    offsets_by_compound = {}   # compound -> list of (offset from MEDIUM at reference_age)
    slopes_by_compound = {}    # compound -> list of deg_rate

    for country in races:
        try:
            fits = fit_race_compounds(year, country, data_dir)
        except Exception as e:
            print(f"  Skipping {country}: {e}")
            continue

        if "MEDIUM" not in fits:
            continue
        medium_pace_at_ref = fits["MEDIUM"]["base_time"] + fits["MEDIUM"]["deg_rate"] * reference_age

        for comp, fit in fits.items():
            pace_at_ref = fit["base_time"] + fit["deg_rate"] * reference_age
            offsets_by_compound.setdefault(comp, []).append(pace_at_ref - medium_pace_at_ref)
            slopes_by_compound.setdefault(comp, []).append(fit["deg_rate"])

    relative_offset_vs_medium = {c: float(np.mean(v)) for c, v in offsets_by_compound.items()}
    avg_deg_rate = {c: float(np.mean(v)) for c, v in slopes_by_compound.items()}
    races_used = {c: len(v) for c, v in offsets_by_compound.items()}

    return {
        "year": year,
        "reference_age": reference_age,
        "relative_offset_vs_medium": relative_offset_vs_medium,
        "avg_deg_rate": avg_deg_rate,
        "races_used": races_used,
    }


def save_season_priors(priors: dict, data_dir: str = "data") -> str:
    path = os.path.join(data_dir, f"season_priors_{priors['year']}.json")
    with open(path, "w") as f:
        json.dump(priors, f, indent=2)
    return path


def load_season_priors(year: int, data_dir: str = "data") -> dict | None:
    path = os.path.join(data_dir, f"season_priors_{year}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build season-wide compound pace priors")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    priors = compute_season_priors(args.year, args.data_dir)

    print(f"\nSeason priors for {args.year} (relative to MEDIUM at tire_age={priors['reference_age']}):")
    for comp, offset in sorted(priors["relative_offset_vs_medium"].items(), key=lambda kv: kv[1]):
        n = priors["races_used"][comp]
        slope = priors["avg_deg_rate"][comp]
        sign = "+" if offset >= 0 else ""
        print(f"  {comp}: {sign}{offset:.2f}s vs MEDIUM | deg_rate {slope:.3f}s/lap | from {n} race(s)")

    if "MEDIUM" not in priors["relative_offset_vs_medium"] or priors["relative_offset_vs_medium"].get("MEDIUM") != 0.0:
        pass  # MEDIUM's own offset is trivially 0 and may not appear depending on data; that's expected

    path = save_season_priors(priors, args.data_dir)
    print(f"\nSaved to {path}")
    print("optimizer.py will now auto-load this file for any race in the same year.")