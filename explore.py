"""
Usage:
    python data_pipeline.py --year 2024 --country Italy   # pull data first
    python explore.py --year 2024 --country Italy         # then explore it
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_race(year: int, country: str, data_dir: str = "data") -> dict:
    race_dir = os.path.join(data_dir, f"{year}_{country.replace(' ', '_')}")
    return {
        "laps": pd.read_csv(os.path.join(race_dir, "laps.csv")),
        "stints": pd.read_csv(os.path.join(race_dir, "stints.csv")),
        "pits": pd.read_csv(os.path.join(race_dir, "pit_stops.csv")),
        "weather": pd.read_csv(os.path.join(race_dir, "weather.csv")),
    }


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Drop laps that will wreck a degradation fit: in/out laps, laps under
    safety car (no lap-time reliability), and missing lap times.
    """
    df = laps.copy()
    df = df.dropna(subset=["lap_duration"])
    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] == False]  # noqa: E712
    return df


def attach_tire_age(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """
    Join each lap to the tire stint it belongs to, so every lap has a
    compound + tire age (laps since that tire was fitted).
    """
    rows = []
    for _, stint in stints.iterrows():
        mask = (
            (laps["driver_number"] == stint["driver_number"])
            & (laps["lap_number"] >= stint["lap_start"])
            & (laps["lap_number"] <= stint["lap_end"])
        )
        chunk = laps[mask].copy()
        chunk["compound"] = stint["compound"]
        chunk["tire_age"] = chunk["lap_number"] - stint["lap_start"] + stint["tyre_age_at_start"]
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True) if rows else laps


def fit_degradation(laps_with_tires: pd.DataFrame):
    """
    Very simple per-compound degradation: median lap time by tire age.
    Good enough for a first pass — swap for a linear/poly fit once you
    trust the data.
    """
    summary = (
        laps_with_tires.groupby(["compound", "tire_age"])["lap_duration"]
        .median()
        .reset_index()
    )
    return summary


def plot_degradation(summary: pd.DataFrame, out_path: str = "degradation.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for compound, group in summary.groupby("compound"):
        ax.plot(group["tire_age"], group["lap_duration"], marker="o", label=compound)
    ax.set_xlabel("Tire age (laps)")
    ax.set_ylabel("Median lap time (s)")
    ax.set_title("Tire degradation by compound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", type=str, required=True)
    args = parser.parse_args()

    race = load_race(args.year, args.country)
    laps = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps, race["stints"])

    print(f"Loaded {len(laps_with_tires)} clean laps across "
          f"{laps_with_tires['compound'].nunique()} compounds.")

    summary = fit_degradation(laps_with_tires)
    print("\nMedian lap time by compound + tire age (first 15 rows):")
    print(summary.head(15).to_string(index=False))

    plot_degradation(summary)

    print(f"\nActual pit stops this race: {len(race['pits'])}")
    print(race["pits"][["driver_number", "lap_number", "pit_duration"]].to_string(index=False))