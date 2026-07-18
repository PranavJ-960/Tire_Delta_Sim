"""
Data loading + cleaning utilities, shared by optimizer.py and app.py.

Usage (standalone exploration):
    python explore.py --year 2024 --country Italy
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
    Drop laps that would wreck a degradation fit: missing lap times and
    pit in/out laps (artificially slow for reasons unrelated to tire wear).
    Known scope cut: this does NOT filter safety car / VSC laps. OpenF1's
    laps endpoint doesn't reliably flag those, so degradation fits can be
    a little noisy during SC periods. Worth knowing, not worth blocking on.
    """
    if laps.empty:
        return laps.copy()
    df = laps.copy()
    df = df.dropna(subset=["lap_duration"])
    for col in ("is_pit_out_lap", "is_pit_in_lap"):
        if col in df.columns:
            df = df[df[col] == False]  # noqa: E712
    return df


def attach_tire_age(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """
    Join each lap to the tire stint it belongs to, so every lap has a
    compound + tire_age (laps completed on that set of tires).
    NaN tyre_age_at_start (happens for the very first stint sometimes)
    is treated as 0 — a fresh set.
    """
    if laps.empty or stints.empty:
        return laps.copy()

    rows = []
    for _, stint in stints.iterrows():
        mask = (
            (laps["driver_number"] == stint["driver_number"])
            & (laps["lap_number"] >= stint["lap_start"])
            & (laps["lap_number"] <= stint["lap_end"])
        )
        chunk = laps[mask].copy()
        if chunk.empty:
            continue
        start_age = stint["tyre_age_at_start"] if pd.notna(stint["tyre_age_at_start"]) else 0
        chunk["compound"] = str(stint["compound"]).upper()
        chunk["tire_age"] = chunk["lap_number"] - stint["lap_start"] + start_age
        rows.append(chunk)

    return pd.concat(rows, ignore_index=True) if rows else laps.copy()


def fit_degradation_summary(laps_with_tires: pd.DataFrame) -> pd.DataFrame:
    """Median lap time by compound + tire age — quick sanity-check view."""
    df = laps_with_tires[laps_with_tires["tire_age"] > 0]  # drop start/out-lap noise
    return df.groupby(["compound", "tire_age"])["lap_duration"].median().reset_index()


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
    plt.close(fig)
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

    summary = fit_degradation_summary(laps_with_tires)
    print("\nMedian lap time by compound + tire age (first 15 rows):")
    print(summary.head(15).to_string(index=False))

    plot_degradation(summary)