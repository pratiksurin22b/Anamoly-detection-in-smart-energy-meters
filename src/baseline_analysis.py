import pandas as pd
from typing import Optional, Dict, Any, Tuple


def extract_representative_week(series: pd.Series, start: Optional[pd.Timestamp] = None) -> pd.Series:
    series = series.sort_index()
    if start is None:
        start = series.index.min()
    end = start + pd.Timedelta(days=7)
    week = series.loc[start:end]
    return week


def estimate_background_load(series: pd.Series, night_hours: Tuple[int, int] = (1, 5)) -> Dict[str, Any]:
    s = series.copy()
    df = s.to_frame("p_plus")
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    mask = (df["hour"] >= night_hours[0]) & (df["hour"] < night_hours[1])
    night = df[mask]
    daily_min = night.groupby("date")["p_plus"].min()
    result = {
        "daily_minima": daily_min,
        "weekly_background_mean": float(daily_min.mean()) if not daily_min.empty else None,
        "weekly_background_std": float(daily_min.std()) if len(daily_min) > 1 else None,
        "method": f"min_{night_hours[0]}-{night_hours[1]}h",
    }
    return result


def estimate_peak_load(series: pd.Series,
                        morning_hours: Tuple[int, int] = (6, 9),
                        evening_hours: Tuple[int, int] = (17, 22)) -> Dict[str, Any]:
    s = series.copy()
    df = s.to_frame("p_plus")
    df["date"] = df.index.date
    df["hour"] = df.index.hour

    def window_max(start_h: int, end_h: int) -> pd.Series:
        m = (df["hour"] >= start_h) & (df["hour"] < end_h)
        return df[m].groupby("date")["p_plus"].max()

    morning = window_max(*morning_hours)
    evening = window_max(*evening_hours)
    daily_max = df.groupby("date")["p_plus"].max()

    result = {
        "daily_morning_peaks": morning,
        "daily_evening_peaks": evening,
        "daily_max_peaks": daily_max,
        "mean_morning_peak": float(morning.mean()) if not morning.empty else None,
        "mean_evening_peak": float(evening.mean()) if not evening.empty else None,
        "mean_daily_max": float(daily_max.mean()) if not daily_max.empty else None,
    }
    return result


def compute_weekly_baseline(series: pd.Series) -> Dict[str, Any]:
    bg = estimate_background_load(series)
    peaks = estimate_peak_load(series)
    return {"background": bg, "peaks": peaks}

