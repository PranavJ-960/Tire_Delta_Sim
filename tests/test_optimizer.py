import pandas as pd
import pytest

from explore import FUEL_EFFECT_PER_LAP
from optimizer import (
    compute_fuel_effect,
    compute_pit_loss,
    compute_track_evolution,
    compute_tyre_degradation,
    compute_tyre_cliff,
    compute_tyre_warmup,
    compute_weather_penalty,
    build_lap_weather,
    estimate_fuel_model,
    fit_degradation_models,
    optimize_strategy,
    reconstruct_actual_strategy,
    simulate_single_lap,
    simulate_strategy,
)


def test_compute_pit_loss_uses_median_and_ignores_large_values():
    pits = pd.DataFrame({"pit_duration": [20.0, 24.0, 30.0, 60.0]})

    assert compute_pit_loss(pits) == 24.0


def test_fit_degradation_models_recovers_fuel_corrected_linear_slope():
    rows = []
    for driver in (1, 2, 3):
        for age in range(1, 5):
            rows.append(
                {
                    "driver_number": driver,
                    "compound": "MEDIUM",
                    "tire_age": age,
                    "lap_number": age,
                    "lap_duration": 90.0 + 0.1 * age + FUEL_EFFECT_PER_LAP * age,
                }
            )

    models = fit_degradation_models(pd.DataFrame(rows))

    assert models["MEDIUM"]["base_time"] == pytest.approx(90.0)
    assert models["MEDIUM"]["deg_rate"] == pytest.approx(0.1)
    assert {"SOFT", "HARD", "INTERMEDIATE"}.issubset(models)


def test_estimate_fuel_model_separates_race_lap_and_tyre_age():
    rows = []
    for stint_start in (1, 12, 24):
        for age in range(1, 11):
            lap = stint_start + age - 1
            rows.append({"compound": "MEDIUM", "tire_age": age, "lap_number": lap, "lap_duration": 90 + 0.10 * age - 0.04 * lap})

    fuel_model = estimate_fuel_model(pd.DataFrame(rows), circuit="Test Circuit")

    assert fuel_model["source"] == "telemetry_fit"
    assert fuel_model["effect_per_lap"] == pytest.approx(-0.04)
    assert fuel_model["average_stint_improvement"] < 0


def test_estimate_fuel_model_uses_circuit_override_before_telemetry():
    model = estimate_fuel_model(pd.DataFrame(), circuit="Monza", circuit_coefficients={"Monza": -0.03})

    assert model == {"effect_per_lap": -0.03, "source": "circuit_override", "circuit": "Monza"}


def test_simulate_strategy_applies_wet_penalty_to_slicks():
    models = {"MEDIUM": {"base_time": 90.0, "deg_rate": 0.0}}

    dry_time = simulate_strategy([("MEDIUM", 2)], 2, models, 0.0, {1: 0, 2: 0})
    wet_time = simulate_strategy([("MEDIUM", 2)], 2, models, 0.0, {1: 1, 2: 1})

    assert wet_time > dry_time


def test_weather_penalties_are_continuous_and_create_intermediate_crossover():
    model = {"base_time": 90.0}
    dry = {1: {"rainfall": 0.0, "track_temperature": 35, "air_temperature": 25, "humidity": 40}}
    damp = {1: {"rainfall": 0.5, "track_temperature": 28, "air_temperature": 20, "humidity": 80}}
    wet = {1: {"rainfall": 1.0, "track_temperature": 22, "air_temperature": 18, "humidity": 95}}

    assert compute_weather_penalty("MEDIUM", 1, dry, model) < compute_weather_penalty("MEDIUM", 1, damp, model) < compute_weather_penalty("MEDIUM", 1, wet, model)
    assert compute_weather_penalty("INTERMEDIATE", 1, dry, model) > compute_weather_penalty("INTERMEDIATE", 1, damp, model) > compute_weather_penalty("INTERMEDIATE", 1, wet, model)
    assert compute_weather_penalty("INTERMEDIATE", 1, wet, model) < compute_weather_penalty("MEDIUM", 1, wet, model)


def test_build_lap_weather_aligns_all_available_weather_fields():
    laps = pd.DataFrame({"lap_number": [1, 2], "date": ["2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"]})
    weather = pd.DataFrame({"date": ["2024-01-01T00:00:30Z"], "rainfall": [0.4], "track_temperature": [31], "air_temperature": [23], "humidity": [70]})

    result = build_lap_weather(laps, weather)

    assert result[1] == {"rainfall": 0.4, "track_temperature": 31.0, "air_temperature": 23.0, "humidity": 70.0}


def test_lap_helpers_preserve_current_simulation_components():
    model = {"base_time": 90.0, "deg_rate": 0.1}

    assert compute_fuel_effect(20) == 0.0
    assert compute_tyre_warmup("MEDIUM", 1) == 0.0
    assert compute_track_evolution(20) == 0.0
    assert compute_tyre_degradation(model, 3) == pytest.approx(90.3)
    wet_penalty = compute_weather_penalty("MEDIUM", 2, {2: 1}, model)
    assert wet_penalty > 0
    assert simulate_single_lap("MEDIUM", model, 3, 2, {2: 1}) == pytest.approx(90.3 + wet_penalty)


def test_optimize_strategy_returns_lowest_one_stop_plan_when_stops_are_expensive():
    models = {
        "SOFT": {"base_time": 1.0, "deg_rate": 0.0},
        "MEDIUM": {"base_time": 100.0, "deg_rate": 0.0},
        "HARD": {"base_time": 100.0, "deg_rate": 0.0},
        "INTERMEDIATE": {"base_time": 100.0, "deg_rate": 0.0},
    }

    result = optimize_strategy(20, models, pit_loss=1_000.0, lap_weather={})

    # Soft is fastest, so the minimum permitted five-lap Medium stint comes
    # first and Soft covers the remaining 15 laps.
    assert result["strategy"] == [("MEDIUM", 5), ("SOFT", 20)]
    assert result["total_time"] < float("inf")


def test_fit_degradation_models_uses_a_callable_spline_for_large_samples():
    laps = pd.DataFrame(
        {
            "compound": ["SOFT"] * 20,
            "tire_age": list(range(1, 21)),
            "lap_number": list(range(1, 21)),
            "lap_duration": [90 + 0.04 * age + 0.002 * age**2 + FUEL_EFFECT_PER_LAP * age for age in range(1, 21)],
        }
    )

    model = fit_degradation_models(laps)["SOFT"]

    assert model["model_type"] == "spline"
    assert callable(model["predict"])
    assert isinstance(model["predict"](10), float)
    assert {"warmup_end", "optimal_window", "degradation_start", "cliff_age", "cliff_rate"}.issubset(model["lifecycle"])


def test_spline_lifecycle_infers_a_cliff_from_accelerating_degradation():
    ages = list(range(1, 31))
    laps = pd.DataFrame(
        {
            "compound": ["HARD"] * len(ages),
            "tire_age": ages,
            "lap_number": ages,
            "lap_duration": [90 + 0.02 * age + max(age - 20, 0) ** 2 * 0.08 + FUEL_EFFECT_PER_LAP * age for age in ages],
        }
    )

    model = fit_degradation_models(laps)["HARD"]

    assert model["lifecycle"]["cliff_age"] is not None
    assert compute_tyre_cliff(model, 30) > 0


def test_reconstruct_actual_strategy_sorts_stints_and_fills_missing_final_lap():
    stints = pd.DataFrame(
        {
            "driver_number": [7, 7, 8],
            "lap_start": [10, 1, 1],
            "lap_end": [None, 9, 5],
            "compound": ["HARD", "MEDIUM", "SOFT"],
        }
    )

    assert reconstruct_actual_strategy(stints, 7, total_laps=20) == [
        ("MEDIUM", 9),
        ("HARD", 20),
    ]
