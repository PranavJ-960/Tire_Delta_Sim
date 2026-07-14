import pandas as pd
import pytest


@pytest.fixture
def race_frames():
    """Small, schema-compatible race data used by loader and analysis tests."""
    laps = pd.DataFrame(
        {
            "driver_number": [44, 44, 44, 44, 44, 44],
            "lap_number": [1, 2, 3, 4, 5, 6],
            "lap_duration": [90.0, 90.1, 90.2, 90.3, 90.4, 90.5],
            "is_pit_out_lap": [False, False, False, False, False, False],
            "is_pit_in_lap": [False, False, False, False, False, False],
            "track_status": [1, 1, 1, 1, 1, 1],
        }
    )
    stints = pd.DataFrame(
        {
            "driver_number": [44, 44],
            "lap_start": [1, 4],
            "lap_end": [3, 6],
            "compound": ["SOFT", "HARD"],
            "tyre_age_at_start": [1, 0],
        }
    )
    pits = pd.DataFrame({"driver_number": [44], "pit_duration": [24.0]})
    weather = pd.DataFrame({"date": ["2024-01-01T00:00:00Z"], "rainfall": [0.0]})
    return {"laps": laps, "stints": stints, "pits": pits, "weather": weather}


@pytest.fixture
def stored_race(tmp_path, race_frames):
    """Write the four CSV files in the directory layout expected by load_race."""
    race_dir = tmp_path / "2024_Test_Land"
    race_dir.mkdir()
    race_frames["laps"].to_csv(race_dir / "laps.csv", index=False)
    race_frames["stints"].to_csv(race_dir / "stints.csv", index=False)
    race_frames["pits"].to_csv(race_dir / "pit_stops.csv", index=False)
    race_frames["weather"].to_csv(race_dir / "weather.csv", index=False)
    return tmp_path
