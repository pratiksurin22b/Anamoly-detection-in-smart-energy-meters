import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def set_time_index(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.set_index(timestamp_col)
    return df


def sort_and_deduplicate(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    df = df.sort_index()
    if not df.index.is_unique:
        if agg == "mean":
            df = df.groupby(level=0).mean(numeric_only=True)
        elif agg == "last":
            df = df.groupby(level=0).last()
        elif agg == "first":
            df = df.groupby(level=0).first()
    return df


def check_frequency_and_gaps(df: pd.DataFrame, expected_freq: str = "1S") -> Dict[str, Any]:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return {"error": "Index must be DatetimeIndex with at least 2 rows"}

    deltas = idx.to_series().diff().dropna()
    observed_median = deltas.median()

    full_range = pd.date_range(start=idx.min(), end=idx.max(), freq=expected_freq)
    missing = full_range.difference(idx)

    summary = {
        "expected_freq": expected_freq,
        "observed_median_delta": observed_median,
        "num_missing": int(missing.size),
        "missing_ratio": float(missing.size / len(full_range)) if len(full_range) > 0 else 0.0,
        "start": idx.min(),
        "end": idx.max(),
    }
    return summary


def align_to_frequency(df: pd.DataFrame, freq: str = "1S", method: str = "ffill") -> pd.DataFrame:
    df = df.sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)
    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "interpolate":
        df = df.interpolate(method="time", limit_direction="both")
    return df


def get_p_plus_series(df: pd.DataFrame, col: str = "p_plus") -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in DataFrame")
    return df[col]


# ---------- Phase 2: missing data handling + multi-resolution datasets ----------


def _impute_small_gaps(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
    small_gap_max: str = "5min",
) -> pd.DataFrame:
    """Reindex to a regular 1-second grid and interpolate P+ only across small gaps.

    - small gaps: <= small_gap_max
    - large gaps: values are set to NaN so interpolation does not bridge them
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if p_col not in df.columns:
        raise KeyError(f"Column {p_col} not found in DataFrame")

    df = df.sort_index()
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=base_freq)
    df_full = df.reindex(full_index)

    # Local gaps on reindexed grid
    deltas = df_full.index.to_series().diff()
    small_gap_max_td = pd.to_timedelta(small_gap_max)
    mask_small = (deltas <= small_gap_max_td) | deltas.isna()

    p = df_full[p_col].copy()
    # Positions after large gaps -> do not bridge with interpolation
    large_gap_positions = (~mask_small).values
    p.iloc[large_gap_positions] = np.nan

    # Time-based interpolation only across small gaps
    p_interp = p.interpolate(method="time", limit_direction="both")
    df_full[p_col] = p_interp

    return df_full


def _drop_large_gap_days(df: pd.DataFrame, large_gap_min: str = "1H") -> pd.DataFrame:
    """Drop entire days that contain a time gap >= large_gap_min."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    df = df.sort_index()
    deltas = df.index.to_series().diff().dropna()
    large_gap_min_td = pd.to_timedelta(large_gap_min)

    # Timestamps where a large gap starts (current timestamp)
    large_gap_times = deltas[deltas >= large_gap_min_td].index
    if large_gap_times.empty:
        return df

    days_to_drop = large_gap_times.normalize().unique()
    mask_keep = ~df.index.normalize().isin(days_to_drop)
    return df.loc[mask_keep].copy()


def preprocess_missing_data(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
    small_gap_max: str = "5min",
    large_gap_min: str = "1H",
) -> pd.DataFrame:
    """Phase 2 preprocessing:

    1) Reindex to 1-second grid and interpolate P+ over small gaps only.
    2) Drop whole days that have at least one large gap.
    """
    df_interp = _impute_small_gaps(
        df,
        p_col=p_col,
        base_freq=base_freq,
        small_gap_max=small_gap_max,
    )
    df_clean = _drop_large_gap_days(df_interp, large_gap_min=large_gap_min)
    return df_clean


def create_daily_aggregates(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
) -> pd.DataFrame:
    """Dataset A: daily aggregates for tree-based models (e.g., Random Forest).

    Features:
    - mean_power, min_power, max_power, std_power, median_power
    - q10, q90
    - energy_kWh (assuming P+ in Watts and base_freq sampling)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if p_col not in df.columns:
        raise KeyError(f"Column {p_col} not found in DataFrame")

    series = df[p_col]
    sample_sec = pd.to_timedelta(base_freq).total_seconds()

    agg = series.resample("1D").agg(
        mean_power="mean",
        min_power="min",
        max_power="max",
        std_power="std",
        median_power="median",
        q10=lambda x: x.quantile(0.1),
        q90=lambda x: x.quantile(0.9),
        energy_Wh=lambda x: x.sum() * (sample_sec / 3600.0),
    )
    agg["energy_kWh"] = agg["energy_Wh"] / 1000.0
    agg = agg.drop(columns=["energy_Wh"])
    agg = agg.dropna(how="all")

    return agg


def create_high_res_sequences(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    resample_freq: str = "1T",
) -> pd.DataFrame:
    """Dataset B: high-resolution sequences for LSTM.

    - Resample P+ to resample_freq (e.g. 1T or 5T) using mean.
    - Interpolate tiny remaining gaps on this coarser grid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if p_col not in df.columns:
        raise KeyError(f"Column {p_col} not found in DataFrame")

    series = df[p_col]
    seq = series.resample(resample_freq).mean()
    seq = seq.interpolate(method="time", limit_direction="both")

    return seq.to_frame(name=p_col)


def build_multi_resolution_datasets(
    df_clean: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
    seq_freq: str = "1T",
):
    """Convenience wrapper.

    Returns:
    - daily_df: Dataset A (daily aggregates)
    - seq_df: Dataset B (high-res sequences)
    """
    daily_df = create_daily_aggregates(df_clean, p_col=p_col, base_freq=base_freq)
    seq_df = create_high_res_sequences(df_clean, p_col=p_col, resample_freq=seq_freq)
    return daily_df, seq_df

