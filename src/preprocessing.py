import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple


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


# ------------------------- Phase 3: Anomaly Simulation -------------------------

def initialize_master_labels(df_master: pd.DataFrame, p_col: str = "p_plus") -> pd.DataFrame:
    """Create backup and label columns on a clean 1-second df_master.

    - Assumes df_master is already 1-second, no missing timestamps, sorted.
    """
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")

    df = df_master.copy()
    if "label" not in df.columns:
        df["label"] = 0  # Normal
    else:
        df["label"] = 0

    backup_col = f"{p_col}_original"
    if backup_col not in df.columns:
        df[backup_col] = df[p_col]

    return df


def inject_drift(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    frac: float = 0.15,
    max_factor: float = 1.10,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Inject drift (Label 1) over a continuous block covering ~frac of samples.

    - Picks a contiguous window of size round(frac * N).
    - Multiplies P+ by a factor that increases linearly from 1.0 to max_factor.
    - Sets label to 1 in that window (will be overwritten by later, more severe labels).
    """
    rng = np.random.default_rng(random_state)
    n = len(df)
    if n == 0:
        return df

    block_size = int(round(frac * n))
    block_size = max(1, min(block_size, n))

    start_idx = rng.integers(0, n - block_size + 1)
    end_idx = start_idx + block_size

    window_index = df.index[start_idx:end_idx]
    factors = np.linspace(1.0, max_factor, num=block_size)

    df = df.copy()
    df.loc[window_index, p_col] = df.loc[window_index, p_col].values * factors
    df.loc[window_index, "label"] = 1

    return df


def inject_fraud(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    target_frac: float = 0.10,
    min_hours: int = 2,
    max_hours: int = 12,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Inject fraud (Label 2) in scattered windows totaling ~target_frac of samples.

    - Creates multiple random windows (2-12 hours each).
    - Within each window, multiplies P+ by 0.1 (90% bypass).
    - Sets label to 2 in those windows (overwrites drift but not faults).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex for fraud injection.")

    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    rng = np.random.default_rng(random_state)
    one_hour = pd.Timedelta(hours=1)
    total_target = int(round(target_frac * n))
    affected = 0

    while affected < total_target:
        hours = rng.integers(min_hours, max_hours + 1)
        window_len = int(hours * 3600)  # 1-second resolution
        if window_len <= 0 or window_len > n:
            break

        start = rng.integers(0, n - window_len + 1)
        end = start + window_len
        idx_window = df.index[start:end]

        df.loc[idx_window, p_col] = df.loc[idx_window, p_col] * 0.1
        df.loc[idx_window, "label"] = 2
        affected += window_len

    return df


def inject_faults(
    df: pd.DataFrame,
    p_col: str = "p_plus",
    target_frac: float = 0.05,
    spike_value: float = 50.0,
    min_stuck_minutes: int = 15,
    max_stuck_minutes: int = 30,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Inject faults (Label 3) using spikes and stuck-value windows.

    - Spikes: random single seconds set to spike_value.
    - Stuck values: random windows (15-30 mins) where P+ is locked to the
      first value in the window.
    - Label is set to 3 (most severe) and overwrites 1 and 2.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    rng = np.random.default_rng(random_state)
    target_samples = int(round(target_frac * n))

    # Logic A: spikes (~half of target samples)
    num_spikes = max(1, target_samples // 2)
    spike_indices = rng.choice(n, size=num_spikes, replace=False)
    spike_index_labels = df.index[spike_indices]

    df.loc[spike_index_labels, p_col] = spike_value
    df.loc[spike_index_labels, "label"] = 3

    # Logic B: stuck values windows (~remaining target)
    remaining = target_samples - num_spikes
    one_minute = pd.Timedelta(minutes=1)

    while remaining > 0:
        minutes = int(rng.integers(min_stuck_minutes, max_stuck_minutes + 1))
        window_len = minutes * 60  # seconds
        if window_len <= 0 or window_len > n:
            break

        start = int(rng.integers(0, n - window_len + 1))
        end = start + window_len
        idx_window = df.index[start:end]

        first_val = df.loc[idx_window[0], p_col]
        df.loc[idx_window, p_col] = first_val
        df.loc[idx_window, "label"] = 3

        remaining -= window_len

    return df


# ---------------------- Phase 3b: Rule-based anomaly labelling ----------------------


def apply_rule_based_labels(
    df_master: pd.DataFrame,
    p_col: str = "p_plus",
    drift_window: str = "7D",
    drift_thresh: float = 0.20,
    fraud_quantile: float = 0.01,
    fraud_ratio_thresh: float = 0.2,
    spike_z_thresh: float = 6.0,
    stuck_window_sec: int = 30 * 60,
    stuck_tol: float = 1e-3,
) -> pd.DataFrame:
    """Assign labels 0–3 on df_master using deterministic rules on existing data.

    Labels (in increasing severity / priority):
    - 0: Normal
    - 1: Drift-like behavior (sustained increase over a long window)
    - 2: Fraud-like under-reporting (sustained periods unusually low vs history)
    - 3: Fault-like behavior (strong spikes or stuck readings)

    Priority: 3 > 2 > 1 > 0
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")

    df = df_master.copy()

    # Start all as normal
    df["label"] = 0

    s = df[p_col].astype(float)

    # --- Fault-like spikes (Label 3) ---
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        mad = 1e-6
    z = (s - median) / (1.4826 * mad)
    mask_spike = z.abs() >= spike_z_thresh

    # --- Fault-like stuck segments (Label 3) ---
    # approximate by comparing each point to a short rolling window
    window_size = min(stuck_window_sec, max(5, len(df)))
    rolling_min = s.rolling(window_size, min_periods=5).min()
    rolling_max = s.rolling(window_size, min_periods=5).max()
    mask_stuck = (rolling_max - rolling_min).abs() <= stuck_tol

    mask_fault = mask_spike | mask_stuck
    df.loc[mask_fault, "label"] = 3

    # --- Fraud-like under-reporting (Label 2) ---
    # Define a "too low" threshold based on global distribution
    low_thresh = s.quantile(fraud_quantile)

    # 15-minute windows: if a large share of points are extremely low, flag as fraud-like
    freq_sec = 1.0
    span = int(15 * 60 / freq_sec)
    if span < 2:
        span = 2
    low_flag = (s <= low_thresh).astype(int)
    low_ratio = low_flag.rolling(span, min_periods=span // 2).mean()
    mask_fraud = (low_ratio >= fraud_ratio_thresh) & (df["label"] < 3)
    df.loc[mask_fraud, "label"] = 2

    # --- Drift-like behavior (Label 1) ---
    # Long-window rolling mean and compare to long-term median
    rolling_long = s.rolling(drift_window, min_periods=int(0.5 * pd.Timedelta(drift_window).total_seconds())).mean()
    baseline_long = rolling_long.median()
    if np.isnan(baseline_long):
        baseline_long = s.median()
    rel_change = (rolling_long - baseline_long) / (baseline_long + 1e-6)
    mask_drift = (rel_change >= drift_thresh) & (df["label"] < 2)
    df.loc[mask_drift, "label"] = 1

    return df


def build_master_from_clean(
    df_clean: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
) -> pd.DataFrame:
    """Build df_master from cleaned data without injecting synthetic anomalies.

    - Ensures a perfect 1-second grid between min and max timestamps.
    - Keeps only the continuous p_col signal; no anomaly labels are assigned here
      for the new 15-minute relative pipeline.
    """
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        raise ValueError("df_clean must have a DatetimeIndex.")

    df_master = df_clean.copy().sort_index()
    full_index = pd.date_range(df_master.index.min(), df_master.index.max(), freq=base_freq)
    df_master = df_master.reindex(full_index)

    # For the new relative 15-minute pipeline we do NOT initialize any label here.
    # Old simulation pipeline functions that rely on labels still exist below but
    # are considered deprecated and are no longer used from main.py.
    return df_master


def build_master_with_anomalies(
    df_clean: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
    drift_frac: float = 0.15,
    fraud_frac: float = 0.10,
    fault_frac: float = 0.05,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Create df_master with labeled anomalies in the required priority order.

    Order: drift (1) -> fraud (2) -> faults (3), so faults win overlaps.
    """
    # Start from pure master based on cleaned data
    df_master = build_master_from_clean(df_clean, p_col=p_col, base_freq=base_freq)

    # Inject anomalies in order of severity
    df_master = inject_drift(
        df_master,
        p_col=p_col,
        frac=drift_frac,
        max_factor=1.10,
        random_state=random_state,
    )
    df_master = inject_fraud(
        df_master,
        p_col=p_col,
        target_frac=fraud_frac,
        min_hours=2,
        max_hours=12,
        random_state=random_state,
    )
    df_master = inject_faults(
        df_master,
        p_col=p_col,
        target_frac=fault_frac,
        spike_value=50.0,
        min_stuck_minutes=15,
        max_stuck_minutes=30,
        random_state=random_state,
    )

    return df_master


def create_daily_labeled_dataset(
    df_master: pd.DataFrame,
    p_col: str = "p_plus",
) -> pd.DataFrame:
    """Create Dataset A: daily labeled summary from df_master.

    - Aggregates P+ with daily statistics.
    - Aggregates label using daily max (if any second is abnormal, the day is abnormal).
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")
    if "label" not in df_master.columns:
        raise KeyError("Column 'label' not found in df_master")

    sample = df_master[p_col]
    daily_stats = sample.resample("1D").agg(
        min_power="min",
        mean_power="mean",
        max_power="max",
        std_power="std",
    )
    daily_label = df_master["label"].resample("1D").max()

    df_daily = daily_stats
    df_daily["label"] = daily_label
    df_daily = df_daily.dropna(how="all")
    return df_daily


def resample_master_for_lstm(
    df_master: pd.DataFrame,
    p_col: str = "p_plus",
    freq: str = "1T",
) -> pd.DataFrame:
    """Create Dataset B base: resampled P+ and labels for LSTM.

    - P+ is averaged over `freq`.
    - label is max over `freq` (to preserve any anomaly in the window).
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")
    if "label" not in df_master.columns:
        raise KeyError("Column 'label' not found in df_master")

    p_resampled = df_master[p_col].resample(freq).mean()
    label_resampled = df_master["label"].resample(freq).max()

    df_seq = pd.DataFrame({p_col: p_resampled, "label": label_resampled})
    df_seq = df_seq.dropna(how="all")
    return df_seq


def create_lstm_windows(
    df_seq_labeled: pd.DataFrame,
    p_col: str = "p_plus",
    window_size: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for LSTM from labeled sequence data.

    - X: shape (num_samples, window_size, 1) with P+ values.
    - y: shape (num_samples,) with max label in each window.
    """
    if p_col not in df_seq_labeled.columns or "label" not in df_seq_labeled.columns:
        raise KeyError("df_seq_labeled must contain both P+ and label columns")

    values = df_seq_labeled[p_col].values.astype(float)
    labels = df_seq_labeled["label"].values.astype(int)

    n = len(df_seq_labeled)
    if n <= window_size:
        return np.empty((0, window_size, 1)), np.empty((0,), dtype=int)

    X_list = []
    y_list = []
    for start in range(0, n - window_size + 1):
        end = start + window_size
        window_vals = values[start:end]
        window_labels = labels[start:end]
        X_list.append(window_vals.reshape(-1, 1))
        y_list.append(window_labels.max())

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y


# ---------------------- Phase 3c: Multi-resolution simulation datasets ----------------------


def compute_label_distribution(series: pd.Series) -> Dict[int, int]:
    """Return counts of each integer label in a label series."""
    vc = series.value_counts().sort_index()
    return {int(k): int(v) for k, v in vc.items()}


def build_master_simulated(
    df_clean: pd.DataFrame,
    p_col: str = "p_plus",
    base_freq: str = "1S",
    drift_day_frac: float = 0.15,
    fraud_sec_frac: float = 0.10,
    fault_sec_frac: float = 0.05,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Create a 1-second master dataset with simulated drift, fraud, and faults.

    - Drift (1): applied on a contiguous block of ~drift_day_frac days as a daily ramp.
    - Fraud (2): applied in scattered 2-12 hour windows totaling ~fraud_sec_frac seconds.
    - Fault (3): applied as spikes and stuck windows totaling ~fault_sec_frac seconds.

    This wraps existing inject_* functions but ensures we start from a clean
    1-second master with labels initialized to 0.
    """
    df_master = build_master_from_clean(df_clean, p_col=p_col, base_freq=base_freq)

    # Inject anomalies in order: drift -> fraud -> faults
    df_master = inject_drift(
        df_master,
        p_col=p_col,
        frac=drift_sec_frac_from_day_frac(df_master, drift_day_frac),
        max_factor=1.10,
        random_state=random_state,
    )
    df_master = inject_fraud(
        df_master,
        p_col=p_col,
        target_frac=fraud_sec_frac,
        min_hours=2,
        max_hours=12,
        random_state=random_state,
    )
    df_master = inject_faults(
        df_master,
        p_col=p_col,
        target_frac=fault_sec_frac,
        spike_value=50.0,
        min_stuck_minutes=10,
        max_stuck_minutes=30,
        random_state=random_state,
    )

    return df_master


def drift_sec_frac_from_day_frac(df_master: pd.DataFrame, day_frac: float) -> float:
    """Approximate seconds fraction given a target fraction of days for drift.

    We convert the desired fraction of days into an approximate fraction of
    total seconds, assuming days are roughly equal length.
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    days = df_master.index.normalize().unique()
    if len(days) == 0:
        return 0.0
    return float(day_frac)


def build_15min_lstm_dataset(
    df_master: pd.DataFrame,
    p_col: str = "p_plus",
    block_freq: str = "15T",
    fault_sec_threshold: int = 10,
) -> pd.DataFrame:
    """Create Level-2 (15-minute) dataset for LSTM from simulated master.

    Features per block:
    - mean, max, std of P+ over the block.

    Label per block (priority):
    - 3 (Fault) if block has >= fault_sec_threshold seconds with label 3.
    - 2 (Fraud) if any second in the block has label 2 and not already 3.
    - 0 otherwise.

    (Drift is handled at daily RF level and not as a separate class here.)
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")
    if "label" not in df_master.columns:
        raise KeyError("Column 'label' not found in df_master")

    p = df_master[p_col]
    labels = df_master["label"]

    grouped = p.resample(block_freq)
    f_mean = grouped.mean()
    f_max = grouped.max()
    f_std = grouped.std()

    # Count number of seconds with each label in each block
    fault_count = (labels == 3).astype(int).resample(block_freq).sum()
    fraud_count = (labels == 2).astype(int).resample(block_freq).sum()

    block_label = pd.Series(0, index=f_mean.index, dtype=int)
    block_label[fault_count >= fault_sec_threshold] = 3
    block_label[(block_label < 3) & (fraud_count > 0)] = 2

    df_15 = pd.DataFrame({
        "mean": f_mean,
        "max": f_max,
        "std": f_std,
        "label": block_label,
    })
    df_15 = df_15.dropna(subset=["mean"])  # drop empty blocks
    return df_15


def build_daily_rf_dataset(
    df_master: pd.DataFrame,
    p_col: str = "p_plus",
) -> pd.DataFrame:
    """Create Level-3 (daily) dataset for Random Forest from simulated master.

    Features:
    - min_power, mean_power, max_power, std_power (daily summary of P+).

    Label:
    - 1 if any second in the day has label 1 (drift day), else 0.
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")
    if "label" not in df_master.columns:
        raise KeyError("Column 'label' not found in df_master")

    p = df_master[p_col]

    daily_stats = p.resample("1D").agg(
        min_power="min",
        mean_power="mean",
        max_power="max",
        std_power="std",
    )

    daily_label_raw = df_master["label"].resample("1D").apply(lambda x: 1 if 1 in x.values else 0)
    daily_stats["label"] = daily_label_raw.astype(int)
    daily_stats = daily_stats.dropna(how="all")
    return daily_stats


def create_15min_aggregates(
    df_master: pd.DataFrame,
    p_col: str = "p_plus",
    freq: str = "15T",
) -> pd.DataFrame:
    """Aggregate master data to 15-minute features for relative labeling.

    Inputs
    ------
    df_master : DataFrame
        Time-indexed data at 1-second (or 1-minute) resolution containing p_col.
    p_col : str
        Name of the active power column (default "p_plus").
    freq : str
        Resampling frequency, "15T" by default.

    Output
    ------
    df_15 : DataFrame indexed by 15-minute timestamps with columns:
        - P_mean: mean of p_col in the block
        - P_max: max of p_col in the block
        - P_std: std of p_col in the block
        - n_samples: number of samples in block
        - day_of_week: 0-6 (Mon-Sun)
        - hour: 0-23
        - weekend: 1 if Saturday/Sunday else 0
    """
    if not isinstance(df_master.index, pd.DatetimeIndex):
        raise ValueError("df_master must have a DatetimeIndex.")
    if p_col not in df_master.columns:
        raise KeyError(f"Column {p_col} not found in df_master")

    df = df_master[[p_col]].copy()
    grouped = df.resample(freq)
    P_mean = grouped[p_col].mean()
    P_max = grouped[p_col].max()
    P_std = grouped[p_col].std()
    n_samples = grouped[p_col].count()

    df_15 = pd.DataFrame(
        {
            "P_mean": P_mean,
            "P_max": P_max,
            "P_std": P_std,
            "n_samples": n_samples,
        }
    )
    # Drop completely empty blocks
    df_15 = df_15.dropna(subset=["P_mean"]).copy()

    # Time-of-week features
    df_15["day_of_week"] = df_15.index.dayofweek
    df_15["hour"] = df_15.index.hour
    df_15["weekend"] = df_15["day_of_week"].isin([5, 6]).astype(int)

    return df_15


def compute_15min_baseline(df_15: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic baseline per (day_of_week, hour, weekend flag) using medians.

    For each unique combination of (day_of_week, hour, weekend) we compute:
    - Baseline_Mean: median of P_mean
    - Baseline_Max: median of P_max

    Weekday vs weekend is handled explicitly via the `weekend` flag so, for
    example, Saturday 10:00 and Monday 10:00 do not share the same baseline.
    """
    if not isinstance(df_15.index, pd.DatetimeIndex):
        raise ValueError("df_15 must have a DatetimeIndex.")

    df = df_15.copy()

    # Ensure time-of-week columns exist (idempotent if already created)
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df.index.dayofweek
    if "hour" not in df.columns:
        df["hour"] = df.index.hour
    if "weekend" not in df.columns:
        df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    group_cols = ["day_of_week", "hour", "weekend"]
    baseline = (
        df.groupby(group_cols)[["P_mean", "P_max"]]
        .median()
        .rename(columns={"P_mean": "Baseline_Mean", "P_max": "Baseline_Max"})
    )

    df = df.merge(
        baseline,
        left_on=group_cols,
        right_index=True,
        how="left",
    )

    return df


def apply_relative_anomaly_labels_15min(
    df_15: pd.DataFrame,
    fault_ratio_thresh: float = 5.0,
    fraud_ratio_thresh: float = 0.1,
) -> pd.DataFrame:
    """Label 15-minute blocks using relative anomalies vs dynamic baseline.

    New columns
    -----------
    ratio_mean : float
        P_mean / Baseline_Mean (NaN where Baseline_Mean <= 0 or missing).
    ratio_max : float
        P_max / Baseline_Max (NaN where Baseline_Max <= 0 or missing).
    label_rel : int
        Final label in {0, 2, 3}, where
        - 0 = normal
        - 2 = fraud-like (mean much lower than baseline)
        - 3 = fault-like spike (max much higher than baseline)

    Priority is 3 (fault) > 2 (fraud). Drift is never considered here.
    """
    required = ["P_mean", "P_max", "Baseline_Mean", "Baseline_Max"]
    missing = [c for c in required if c not in df_15.columns]
    if missing:
        raise KeyError(f"df_15 is missing required columns: {missing}")

    df = df_15.copy()

    # Avoid divide-by-zero and negative baselines
    baseline_mean = df["Baseline_Mean"].where(df["Baseline_Mean"] > 0)
    baseline_max = df["Baseline_Max"].where(df["Baseline_Max"] > 0)

    df["ratio_mean"] = df["P_mean"] / baseline_mean
    df["ratio_max"] = df["P_max"] / baseline_max

    # Start with all blocks marked as normal
    df["label_rel"] = 0

    # Fraud-like: mean much smaller than baseline
    mask_fraud = df["ratio_mean"] < fraud_ratio_thresh
    mask_fraud &= df["ratio_mean"].notna()
    df.loc[mask_fraud, "label_rel"] = 2

    # Fault-like: max much larger than baseline (higher priority)
    mask_fault = df["ratio_max"] > fault_ratio_thresh
    mask_fault &= df["ratio_max"].notna()
    df.loc[mask_fault, "label_rel"] = 3

    # No drift, no combined label column; the relative label is final.
    return df

