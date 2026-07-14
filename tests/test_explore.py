import pandas as pd
from pandas.testing import assert_frame_equal

from explore import attach_tire_age, clean_laps, load_race


def test_load_race_reads_all_expected_csvs(stored_race, race_frames):
    race = load_race(2024, "Test Land", data_dir=str(stored_race))

    assert set(race) == {"laps", "stints", "pits", "weather"}
    assert_frame_equal(race["laps"], race_frames["laps"])
    assert_frame_equal(race["stints"], race_frames["stints"])
    assert_frame_equal(race["pits"], race_frames["pits"])
    assert_frame_equal(race["weather"], race_frames["weather"])


def test_clean_laps_excludes_missing_pit_and_non_green_laps():
    laps = pd.DataFrame(
        {
            "lap_number": [1, 2, 3, 4, 5],
            "lap_duration": [90.0, None, 91.0, 92.0, 93.0],
            "is_pit_out_lap": [False, False, True, False, False],
            "is_pit_in_lap": [False, False, False, True, False],
            "track_status": [1, 1, 1, 1, 2],
        }
    )

    cleaned = clean_laps(laps)

    assert cleaned["lap_number"].tolist() == [1]
    assert cleaned["lap_duration"].tolist() == [90.0]


def test_attach_tire_age_assigns_compound_and_initial_age(race_frames):
    attached = attach_tire_age(race_frames["laps"], race_frames["stints"])

    assert attached["compound"].tolist() == ["SOFT", "SOFT", "SOFT", "HARD", "HARD", "HARD"]
    assert attached["tire_age"].tolist() == [1, 2, 3, 0, 1, 2]


def test_attach_tire_age_returns_input_when_no_stints(race_frames):
    empty_stints = race_frames["stints"].iloc[0:0]

    assert attach_tire_age(race_frames["laps"], empty_stints) is race_frames["laps"]
