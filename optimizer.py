import argparse
from collections.abc import Mapping
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from explore import FUEL_EFFECT_PER_LAP, attach_tire_age, clean_laps, load_race

PACE_PENALTIES = {"SLICKS_ON_WET": 28.0, "WETS_ON_DRY": 7.0, "INTERS_DRY_OFFSET": 14.5}
WEATHER_FIELDS = ("rainfall", "track_temperature", "air_temperature", "humidity")
CIRCUIT_FUEL_COEFFICIENTS = {}

class LinearPredictor:
    def __init__(self, intercept: float, slope: float, compound: str = "MEDIUM"):
        self.intercept = float(intercept)
        floor_rates = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03, "INTERMEDIATE": 0.04, "WET": 0.05}
        self.slope = max(float(slope), floor_rates.get(str(compound).upper(), 0.05))

    def __call__(self, age):
        value = self.intercept + self.slope * np.asarray(age)
        return float(value) if np.ndim(value) == 0 else value


class SplinePredictor:
    def __init__(self, spline, observed_ages: np.ndarray, compound: str = "MEDIUM"):
        self.spline = spline
        self.max_observed_age = float(np.max(observed_ages)) if len(observed_ages) > 0 else 15.0
        
        tail_ages = np.linspace(max(1.0, self.max_observed_age - 2), self.max_observed_age, 5)
        tail_paces = spline(tail_ages)
        
        floor_rates = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04, "INTERMEDIATE": 0.05, "WET": 0.06}
        fitted_slope = (tail_paces[-1] - tail_paces[0]) / (tail_ages[-1] - tail_ages[0])
        
        self.extrapolation_slope = max(fitted_slope, floor_rates.get(str(compound).upper(), 0.05))
        self.boundary_value = float(spline(self.max_observed_age))

    def __call__(self, age):
        age_arr = np.asarray(age)
        if np.ndim(age_arr) == 0:
            if age_arr <= self.max_observed_age:
                return float(self.spline(age_arr))
            return float(self.boundary_value + self.extrapolation_slope * (age_arr - self.max_observed_age))
            
        out = np.zeros_like(age_arr, dtype=float)
        mask_interpolated = age_arr <= self.max_observed_age
        mask_extrapolated = ~mask_interpolated
        
        out[mask_interpolated] = self.spline(age_arr[mask_interpolated])
        out[mask_extrapolated] = self.boundary_value + self.extrapolation_slope * (age_arr[mask_extrapolated] - self.max_observed_age)
        return out


class DriverAdjustedPredictor:
    def __init__(self, global_predict, offset: float, mgmt: float):
        self.global_predict = global_predict
        self.offset = float(offset)
        self.mgmt = float(mgmt)

    def __call__(self, age):
        age_arr = np.asarray(age)
        base_val = self.global_predict(0)
        deg_val = self.global_predict(age_arr) - base_val
        return base_val + self.offset + (deg_val * self.mgmt)


def _remove_iqr_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if len(df) < 4:
        return df
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return df
    return df[df[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]


def compute_pit_loss(pits_df: pd.DataFrame) -> float:
    if pits_df.empty or "pit_duration" not in pits_df.columns:
        return 22.0
    valid_pits = pits_df[pits_df["pit_duration"].between(15.0, 50.0)]
    return 22.0 if valid_pits.empty else float(valid_pits["pit_duration"].median())


def estimate_fuel_model(laps_with_tires: pd.DataFrame, circuit: str | None = None, circuit_coefficients: dict | None = None) -> dict:
    coefficients = CIRCUIT_FUEL_COEFFICIENTS | (circuit_coefficients or {})
    configured = coefficients.get(circuit) if circuit else None
    if configured is not None:
        effect = configured.get("effect_per_lap") if isinstance(configured, Mapping) else configured
        return {"effect_per_lap": float(effect), "source": "circuit_override", "circuit": circuit}

    required = {"lap_duration", "lap_number", "tire_age", "compound"}
    if laps_with_tires.empty or not required.issubset(laps_with_tires.columns):
        return {"effect_per_lap": FUEL_EFFECT_PER_LAP, "source": "fallback", "circuit": circuit}
    
    df = clean_laps(laps_with_tires)
    df = df[df["tire_age"] > 0]
    df = _remove_iqr_outliers(df, "lap_duration")
    if len(df) < 20 or (df["lap_number"] - df["tire_age"]).nunique() < 2:
        return {"effect_per_lap": FUEL_EFFECT_PER_LAP, "source": "fallback", "circuit": circuit}

    try:
        compound_dummies = pd.get_dummies(df["compound"].astype(str), drop_first=True, dtype=float)
        age_terms = compound_dummies.mul(df["tire_age"], axis=0)
        design = np.column_stack((np.ones(len(df)), df["lap_number"], df["tire_age"], age_terms))
        if np.linalg.matrix_rank(design) < design.shape[1]:
            return {"effect_per_lap": FUEL_EFFECT_PER_LAP, "source": "fallback", "circuit": circuit}
        coefficient = float(np.linalg.lstsq(design, df["lap_duration"].to_numpy(dtype=float), rcond=None)[0][1])
        if not -0.5 < coefficient < 0:
            return {"effect_per_lap": FUEL_EFFECT_PER_LAP, "source": "fallback", "circuit": circuit}
        return {"effect_per_lap": coefficient, "source": "telemetry_fit", "circuit": circuit}
    except Exception:
        return {"effect_per_lap": FUEL_EFFECT_PER_LAP, "source": "fallback", "circuit": circuit}


def fit_degradation_models(laps_with_tires: pd.DataFrame, circuit: str | None = None, circuit_coefficients: dict | None = None) -> dict:
    models = {}
    fuel_model = estimate_fuel_model(laps_with_tires, circuit, circuit_coefficients)
    fuel_effect = fuel_model["effect_per_lap"]
    
    if laps_with_tires.empty:
        compounds, global_q1 = [], 90.0
    else:
        compounds = laps_with_tires["compound"].dropna().unique()
        global_q1 = laps_with_tires["lap_duration"].quantile(0.1)
        global_q1 = float(global_q1) if np.isfinite(global_q1) else 90.0

    cleaned_base = clean_laps(laps_with_tires)

    for compound in compounds:
        df = cleaned_base[cleaned_base["compound"] == compound].copy()
        df = df[df["tire_age"] > 0]
        if len(df) < 2:
            continue
            
        df["fuel_corrected_lap_duration"] = df["lap_duration"] - df["lap_number"] * fuel_effect
        df = _remove_iqr_outliers(df, "fuel_corrected_lap_duration")
        if len(df) < 2:
            continue

        ages = df["tire_age"].to_numpy(dtype=float)
        times = df["fuel_corrected_lap_duration"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(ages, times, 1)
        
        predict = LinearPredictor(intercept, slope, compound=compound)
        model_type = "linear"
        
        by_age = df.groupby("tire_age", as_index=False)["fuel_corrected_lap_duration"].median()
        if len(df) >= 15 and len(by_age) >= 4:
            x = by_age["tire_age"].to_numpy(dtype=float)
            y = by_age["fuel_corrected_lap_duration"].to_numpy(dtype=float)
            try:
                smoothing = max(len(y) * float(np.var(y)) * 0.25, 1e-9)
                spline_obj = UnivariateSpline(x, y, k=min(3, len(x) - 1), s=smoothing)
                predict = SplinePredictor(spline_obj, ages, compound=compound)
                model_type = "spline"
            except Exception:
                pass

        models[str(compound).upper()] = {
            "base_time": float(predict(0)),
            "deg_rate": max(float(predict(2) - predict(1)), 0.01),
            "predict": predict,
            "model_type": model_type,
            "fuel_effect_per_lap": fuel_effect,
            "fuel_model_source": fuel_model["source"],
        }

    for compound, default_slope in (("SOFT", 0.12), ("MEDIUM", 0.08), ("HARD", 0.05), ("INTERMEDIATE", 0.02)):
        if compound not in models:
            models[compound] = {
                "base_time": float(global_q1),
                "deg_rate": float(default_slope),
                "predict": LinearPredictor(global_q1, default_slope, compound=compound),
                "model_type": "fallback",
                "fuel_effect_per_lap": fuel_effect,
                "fuel_model_source": fuel_model["source"]
            }
    
    models["_drivers"] = fit_driver_profiles(laps_with_tires, models, fuel_effect)
    return models


def fit_driver_profiles(laps_df: pd.DataFrame, global_models: dict, fuel_effect: float) -> dict:
    profiles = {}
    if laps_df.empty or "driver_number" not in laps_df.columns:
        return profiles
    cleaned = clean_laps(laps_df)
    if cleaned.empty:
        return profiles

    team_col = next((c for c in ("team_name", "constructor_name", "team") if c in cleaned.columns), None)
    
    residuals = []
    for _, row in cleaned.iterrows():
        comp = str(row["compound"]).upper()
        if comp in global_models and row["tire_age"] > 0:
            corrected = row["lap_duration"] - (row["lap_number"] * fuel_effect)
            base_pred = global_models[comp]["predict"](row["tire_age"])
            residuals.append({
                "driver": row["driver_number"],
                "team": row[team_col] if team_col else "GLOBAL",
                "compound": comp,
                "res": corrected - base_pred,
                "age": row["tire_age"]
            })
            
    if not residuals:
        return profiles
        
    res_df = pd.DataFrame(residuals)
    global_consistency = float(res_df["res"].std()) if len(res_df) > 1 else 0.5

    for driver, d_group in res_df.groupby("driver"):
        team_id = d_group["team"].iloc[0]
        t_group = res_df[res_df["team"] == team_id]
        driver_compounds = {}
        
        for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            d_comp = d_group[d_group["compound"] == comp]
            t_comp = t_group[t_group["compound"] == comp]
            g_comp = res_df[res_df["compound"] == comp]
            
            if len(d_comp) >= 5: pace_offset = float(d_comp["res"].mean())
            elif len(t_comp) >= 10: pace_offset = float(t_comp["res"].mean())
            else: pace_offset = float(g_comp["res"].mean()) if not g_comp.empty else 0.0
                
            if len(d_comp) >= 8 and d_comp["age"].nunique() > 1:
                cov = np.cov(d_comp["age"], d_comp["res"])[0][1]
                var = np.var(d_comp["age"])
                management = 1.0 + float(cov / (var + 1e-5))
            elif len(t_comp) >= 15 and t_comp["age"].nunique() > 1:
                cov = np.cov(t_comp["age"], t_comp["res"])[0][1]
                var = np.var(t_comp["age"])
                management = 1.0 + float(cov / (var + 1e-5))
            else: management = 1.0

            consistency = float(d_comp["res"].std()) if len(d_comp) > 3 else (float(d_group["res"].std()) if len(d_group) > 3 else global_consistency)
            
            driver_compounds[comp] = {
                "pace_offset": pace_offset,
                "tyre_management": np.clip(management, 0.5, 2.0),
                "consistency": max(consistency, 0.05)
            }
        profiles[driver] = driver_compounds
    return profiles


def get_driver_adjusted_predict(global_predict, driver_profile, compound):
    if not driver_profile or compound not in driver_profile:
        return global_predict
    prof = driver_profile[compound]
    return DriverAdjustedPredictor(global_predict, prof["pace_offset"], prof["tyre_management"])


def compute_fuel_effect(lap_number: int, fuel_effect_per_lap: float | None = None) -> float:
    return 0.0 if fuel_effect_per_lap is None else lap_number * fuel_effect_per_lap


def build_lap_weather(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict:
    """Combines laps and weather telemetry safely using cross-timezone parsing."""
    if laps_df.empty or weather_df.empty or "date" not in laps_df.columns or "date" not in weather_df.columns:
        return {}
        
    available = [field for field in WEATHER_FIELDS if field in weather_df.columns]
    if not available or "lap_number" not in laps_df.columns:
        return {}
        
    laps = laps_df[["date", "lap_number"]].copy()
    weather = weather_df[["date", *available]].copy()
    
    # Force alignment by setting everything explicitly to UTC Datetime formats
    laps["date"] = pd.to_datetime(laps["date"], errors='coerce', utc=True)
    weather["date"] = pd.to_datetime(weather["date"], errors='coerce', utc=True)
    
    laps = laps.dropna(subset=["date"]).sort_values("date")
    weather = weather.dropna(subset=["date"]).sort_values("date")
    
    # 15-minute tolerance ensures that timezone drift won't drop values
    merged = pd.merge_asof(
        laps, 
        weather, 
        on="date", 
        direction="backward", 
        tolerance=pd.Timedelta(minutes=15)
    )
    
    by_lap = merged.groupby("lap_number").agg({f: ("max" if f == "rainfall" else "mean") for f in available})
    return {int(l): {f: float(v) for f, v in r.items() if pd.notna(v)} for l, r in by_lap.iterrows()}


def _unit_interval(value, default: float = 0.0) -> float:
    try:
        # OpenF1 records rainfall binary status as integers (0 or 1)
        number = float(value)
    except (TypeError, ValueError):
        return default
    return 1.0 if number >= 1.0 else 0.0


def _weather_conditions(lap_number: int, lap_weather: dict | None) -> dict:
    sample = (lap_weather or {}).get(lap_number, {})
    if not isinstance(sample, Mapping):
        sample = {"rainfall": sample}
    
    rainfall = _unit_interval(sample.get("rainfall", 0.0))
    air = float(sample.get("air_temperature")) if pd.notna(sample.get("air_temperature")) else 20.0
    track = float(sample.get("track_temperature")) if pd.notna(sample.get("track_temperature")) else air + 5.0
    return {"wetness": rainfall, "track_temperature": track, "air_temperature": air}


def compute_weather_penalty(compound: str, lap_number: int, lap_weather: dict | None, model: dict | None = None) -> float:
    """Applies cross-over performance limits depending on binary track wetness."""
    conditions = _weather_conditions(lap_number, lap_weather)
    is_wet = conditions["wetness"] > 0.5
    
    # Dry Slick Tyre execution on a real wet track surface
    if is_wet and compound in ["SOFT", "MEDIUM", "HARD"]:
        return 20.0  # Lose 20s per lap slipping around on rain
        
    # Wet/Intermediate tyre execution on a perfectly dry asphalt surface
    if not is_wet and compound in ["INTERMEDIATE", "WET"]:
        return 15.0 if compound == "INTERMEDIATE" else 25.0
        
    return 0.0


def compute_tyre_warmup(compound: str, tyre_age: int, model: dict | None = None) -> float:
    return 0.0


def compute_track_evolution(lap_number: int) -> float:
    return 0.0


def compute_tyre_degradation(model: dict, tyre_age: int) -> float:
    predict = model.get("predict")
    if callable(predict):
        return float(predict(tyre_age))
    return float(model["base_time"] + model["deg_rate"] * tyre_age)


def compute_tyre_cliff(model: dict, tyre_age: int) -> float:
    lifecycle = model.get("lifecycle")
    if not lifecycle or lifecycle["cliff_age"] is None or tyre_age <= lifecycle["cliff_age"]:
        return 0.0
    return float((tyre_age - lifecycle["cliff_age"]) * lifecycle["cliff_rate"])


def simulate_single_lap(compound: str, model: dict, tyre_age: int, lap_number: int, lap_weather: dict | None = None) -> float:
    return (
        compute_tyre_degradation(model, tyre_age)
        + compute_fuel_effect(lap_number, model.get("fuel_effect_per_lap"))
        + compute_weather_penalty(compound, lap_number, lap_weather, model)
        + compute_tyre_warmup(compound, tyre_age, model)
        + compute_tyre_cliff(model, tyre_age)
        + compute_track_evolution(lap_number)
    )


def simulate_strategy(strategy: list, total_laps: int, models: dict, pit_loss: float, lap_weather: dict | None = None, driver_number: int | None = None) -> float:
    total_time = (len(strategy) - 1) * pit_loss
    current_lap = 1
    
    d_profiles = models.get("_drivers", {})
    d_prof = d_profiles.get(driver_number) if driver_number is not None else None

    for compound, end_lap in strategy:
        base_model = models.get(compound, models.get("MEDIUM"))
        model = base_model.copy()
        if d_prof and compound in d_prof:
            model["predict"] = get_driver_adjusted_predict(base_model["predict"], d_prof, compound)
            
        for age in range(1, end_lap - current_lap + 2):
            lap_num = current_lap + age - 1
            total_time += simulate_single_lap(compound, model, age, lap_num, lap_weather)
        current_lap = end_lap + 1
    return total_time


def optimize_strategy(total_laps: int, models: dict, pit_loss: float, lap_weather: dict | None = None, driver_number: int | None = None) -> dict:
    best_time, best_strategy = float("inf"), None
    compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]
    
    # Scans mapping to verify if wet weather variables exist anywhere in the telemetry feed
    is_wet_race = False if not lap_weather else any((x.get("rainfall", 0.0) >= 1.0) for x in lap_weather.values())
    
    for c1 in compounds:
        for c2 in compounds:
            for pit1 in range(5, total_laps - 5):
                strategy = [(c1, pit1), (c2, total_laps)]
                if len({c1, c2}) < 2 and not is_wet_race:
                    continue
                time_cost = simulate_strategy(strategy, total_laps, models, pit_loss, lap_weather, driver_number=driver_number)
                if time_cost < best_time:
                    best_time, best_strategy = time_cost, strategy

    for c1 in compounds:
        for c2 in compounds:
            for c3 in compounds:
                for pit1 in range(5, total_laps - 10):
                    for pit2 in range(pit1 + 5, total_laps - 5):
                        strategy = [(c1, pit1), (c2, pit2), (c3, total_laps)]
                        if (c1 == c2 or c2 == c3) and (c2 in ["HARD", "INTERMEDIATE", "WET"]):
                            continue
                        if len({c1, c2, c3}) < 2 and not is_wet_race:
                            continue
                        time_cost = simulate_strategy(strategy, total_laps, models, pit_loss, lap_weather, driver_number=driver_number)
                        if time_cost < best_time:
                            best_time, best_strategy = time_cost, strategy
                            
    return {"strategy": best_strategy, "total_time": best_time}


def reconstruct_actual_strategy(stints_df: pd.DataFrame, driver_number: int, total_laps: int | None = None) -> list:
    stints = stints_df[stints_df["driver_number"] == driver_number].copy()
    if stints.empty: return []
    stints = stints.dropna(subset=["compound"]).sort_values("lap_start")
    if total_laps is not None:
        stints["lap_end"] = stints["lap_end"].fillna(total_laps)
    return [(str(row["compound"]).upper(), int(row["lap_end"])) for _, row in stints.dropna(subset=["lap_end"]).iterrows()]


def compare_driver_to_optimal(driver_number: int, stints_df: pd.DataFrame, total_laps: int, models: dict, pit_loss: float, optimal_time: float, lap_weather: dict | None = None) -> dict:
    actual_strategy = reconstruct_actual_strategy(stints_df, driver_number, total_laps)
    if not actual_strategy: return None
    actual_time = simulate_strategy(actual_strategy, total_laps, models, pit_loss, lap_weather, driver_number=driver_number)
    return {"driver_number": driver_number, "actual_strategy": actual_strategy, "actual_time": actual_time, "optimal_time": optimal_time, "delta_seconds": actual_time - optimal_time}


def format_strategy(strategy: list) -> str:
    parts, start = [], 1
    for compound, end_lap in strategy:
        parts.append(f"{compound}(L{start}-{end_lap})")
        start = end_lap + 1
    return " -> ".join(parts)