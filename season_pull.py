"""
Pull an entire season's worth of races, reusing data_pipeline.get_race_data().

Usage:
    python season_pull.py --year 2024
"""

import argparse
import time
import requests
from data_pipeline import get_race_data, BASE_URL


def list_race_countries(year: int) -> list:
    """Get every country that hosted a Race session in a given year."""
    resp = requests.get(
        f"{BASE_URL}/sessions",
        params={"year": year, "session_type": "Race"},
        timeout=30,
    )
    resp.raise_for_status()
    sessions = resp.json()
    seen = set()
    countries = []
    for s in sessions:
        c = s.get("country_name")
        if c and c not in seen:
            seen.add(c)
            countries.append(c)
    return countries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull a full season from OpenF1")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    countries = list_race_countries(args.year)
    print(f"Found {len(countries)} races in {args.year}: {countries}\n")

    results = {}
    for country in countries:
        try:
            results[country] = get_race_data(args.year, country, args.out)
        except Exception as e:
            print(f"  FAILED on {country}: {e}")
        time.sleep(1)  # be polite to the free tier (3 req/s, 30 req/min)

    print(f"\nDone. Pulled {len(results)}/{len(countries)} races successfully.")