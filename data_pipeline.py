"""
F1 Strategy Simulator — Data Pipeline
--------------------------------------
Pulls lap times, pit stops, tire stints, and weather for a given race
from the free OpenF1 API (https://openf1.org, no API key required).

Usage:
    python data_pipeline.py --year 2024 --country Italy

This will:
  1. Find the Race session for that Grand Prix
  2. Pull lap-by-lap data (lap time, tire compound, tire age)
  3. Pull pit stop data (who stopped, when, how long it cost)
  4. Pull weather data (track temp, air temp, rainfall)
  5. Save everything to ./data/<year>_<country>/ as CSV files

Run this once per race you want to analyze. To pull a whole season,
see season_pull.py, which just calls this in a loop.
"""

import argparse
import os
import time
import requests
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"


def _get(endpoint: str, params: dict) -> list:
    """GET a single OpenF1 endpoint with basic retry on rate-limit (429)."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed to fetch {endpoint} after retries")


def find_race_session(year: int, country: str) -> dict:
    """Find the Race session key for a given year + country name."""
    sessions = _get(
        "sessions",
        {"year": year, "session_type": "Race", "country_name": country},
    )
    if not sessions:
        raise ValueError(f"No race session found for {country} {year}")
    return sessions[0]


def get_laps(session_key: int) -> pd.DataFrame:
    """Lap-by-lap data: lap time, driver, lap number."""
    data = _get("laps", {"session_key": session_key})
    return pd.DataFrame(data)


def get_stints(session_key: int) -> pd.DataFrame:
    """Tire stints: compound used, lap range, starting tire age per driver."""
    data = _get("stints", {"session_key": session_key})
    return pd.DataFrame(data)


def get_pit_stops(session_key: int) -> pd.DataFrame:
    """Pit stop events: driver, lap number, pit duration."""
    data = _get("pit", {"session_key": session_key})
    return pd.DataFrame(data)


def get_weather(session_key: int) -> pd.DataFrame:
    """Weather samples through the race: air/track temp, rainfall, wind."""
    data = _get("weather", {"session_key": session_key})
    return pd.DataFrame(data)


def get_race_data(year: int, country: str, out_dir: str = "data") -> dict:
    """
    Pull all data for one race and save to CSV.
    Returns a dict of DataFrames for immediate use too.
    """
    print(f"Finding race session: {country} {year}...")
    session = find_race_session(year, country)
    session_key = session["session_key"]
    meeting_name = session.get("meeting_name", country)
    print(f"  Found: {meeting_name} (session_key={session_key})")

    print("  Pulling laps...")
    laps = get_laps(session_key)
    print("  Pulling stints (tire compounds)...")
    stints = get_stints(session_key)
    print("  Pulling pit stops...")
    pits = get_pit_stops(session_key)
    print("  Pulling weather...")
    weather = get_weather(session_key)

    race_dir = os.path.join(out_dir, f"{year}_{country.replace(' ', '_')}")
    os.makedirs(race_dir, exist_ok=True)

    laps.to_csv(os.path.join(race_dir, "laps.csv"), index=False)
    stints.to_csv(os.path.join(race_dir, "stints.csv"), index=False)
    pits.to_csv(os.path.join(race_dir, "pit_stops.csv"), index=False)
    weather.to_csv(os.path.join(race_dir, "weather.csv"), index=False)

    print(f"  Saved to {race_dir}/")

    return {
        "session": session,
        "laps": laps,
        "stints": stints,
        "pits": pits,
        "weather": weather,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull one race's data from OpenF1")
    parser.add_argument("--year", type=int, required=True, help="e.g. 2024")
    parser.add_argument("--country", type=str, required=True, help="e.g. Italy")
    parser.add_argument("--out", type=str, default="data", help="output folder")
    args = parser.parse_args()

    get_race_data(args.year, args.country, args.out)