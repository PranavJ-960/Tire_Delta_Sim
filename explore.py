import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Historical fallback used only when telemetry cannot identify a fuel effect.
FUEL_EFFECT_PER_LAP = -0.035

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
    Centralized data cleaning layer. Drops laps that wreck degradation fits:
    missing lap durations, pit entry/exit laps, safety car windows, and non-green periods.
    """
    if laps.empty:
        return laps.copy()
    
    df = laps.copy()
    df = df.dropna(subset=["lap_duration"])
    
    # Modern boolean flags
    for col in ("is_pit_out_lap", "is_pit_in_lap"):
        if col in df.columns:
            df = df[df[col] == False]

    # Handle abnormal race control operational conditions
    if "track_status" in df.columns:
        df = df[df["track_status"].astype(str).str.strip() == "1"]
    elif "lap_status" in df.columns:
        df = df[df["lap_status"].astype(str).str.lower() == "valid"]
        
    for column in ("is_safety_car", "is_virtual_safety_car", "is_vsc"):
        if column in df.columns:
            df = df[df[column] != True]
            
    return df

def attach_tire_age(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    if laps.empty or stints.empty:
        raise ValueError("Critical Telemetry Loss: Empty datasets passed to pipeline.")
        
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
            
        # Fixed Point 3: Fallback boundary checks for NaN starting ages
        start_age = stint["tyre_age_at_start"] if pd.notna(stint["tyre_age_at_start"]) else 0
        
        chunk["compound"] = stint["compound"]
        chunk["tire_age"] = chunk["lap_number"] - stint["lap_start"] + start_age
        rows.append(chunk)
        
    if not rows:
        raise RuntimeError("Stint Realignment Failed: No valid compound allocations resolved.")
    return pd.concat(rows, ignore_index=True)

def fit_degradation(laps_with_tires: pd.DataFrame):
    return laps_with_tires.groupby(["compound", "tire_age"])["lap_duration"].median().reset_index()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", type=str, required=True)
    args = parser.parse_args()

    race = load_race(args.year, args.country)
    laps = clean_laps(race["laps"])
    laps_with_tires = attach_tire_age(laps, race["stints"])

    summary = fit_degradation(laps_with_tires)
    print(summary.head(15).to_string(index=False))