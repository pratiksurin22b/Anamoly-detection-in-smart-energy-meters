import os

from . import config
from .io_utils import load_premises_csv
from .preprocessing import (
    set_time_index,
    sort_and_deduplicate,
    check_frequency_and_gaps,
    get_p_plus_series,
    preprocess_missing_data,
    build_multi_resolution_datasets,
)
from .baseline_analysis import extract_representative_week, compute_weekly_baseline
from .visualization import plot_week_p_plus, plot_daily_p_plus, plot_hourly_p_plus

def summarize_preprocessing(df_raw, df_clean, p_col="p_plus"):
    # Basic shapes and ranges
    print("\n--- Phase 2 Preprocessing Summary ---")
    print(f"Raw rows:   {len(df_raw)}")
    print(f"Clean rows: {len(df_clean)}")

    if len(df_raw) > 0:
        print(f"Raw index range:   {df_raw.index.min()}  ->  {df_raw.index.max()}")
    if len(df_clean) > 0:
        print(f"Clean index range: {df_clean.index.min()}  ->  {df_clean.index.max()}")

    # Daily coverage
    raw_days = df_raw.index.normalize().unique() if len(df_raw) > 0 else []
    clean_days = df_clean.index.normalize().unique() if len(df_clean) > 0 else []
    print(f"Raw days:   {len(raw_days)}")
    print(f"Clean days: {len(clean_days)}")
    print(f"Days dropped due to large gaps: {len(raw_days) - len(clean_days)}")

    # NaNs in P+ before vs after
    if p_col in df_raw.columns:
        raw_nans = df_raw[p_col].isna().sum()
        print(f"NaN count in raw {p_col}:   {raw_nans}")
    if p_col in df_clean.columns:
        clean_nans = df_clean[p_col].isna().sum()
        print(f"NaN count in clean {p_col}: {clean_nans}")

    print("--- End of Preprocessing Summary ---\n")

def run_pipeline_for_house(house_id: str = "house_1") -> None:
    csv_name = config.HOUSE_FILES[house_id]
    csv_path = os.path.join(config.DATA_DIR, csv_name)

    # Step 1: Data ingestion
    df = load_premises_csv(csv_path)

    # Step 2: Time-series alignment prep (no heavy imputation yet)
    df = set_time_index(df)
    df = sort_and_deduplicate(df)
    freq_info = check_frequency_and_gaps(df, expected_freq=config.TARGET_FREQ)
    print(f"Frequency summary for {house_id}:", freq_info)

    # Phase 2: preprocessing & data cleaning (gap-aware)
    df_clean = preprocess_missing_data(
        df,
        p_col="p_plus",
        base_freq=config.TARGET_FREQ,
        small_gap_max="5min",
        large_gap_min="1H",
    )
    summarize_preprocessing(df, df_clean, p_col="p_plus")
    # Build multi-resolution datasets (Random Forest + LSTM)
    daily_df, seq_df = build_multi_resolution_datasets(
        df_clean,
        p_col="p_plus",
        base_freq=config.TARGET_FREQ,
        seq_freq="1T",  # change to "5T" for 5-minute if you prefer
    )

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save processed datasets
    daily_path = os.path.join(config.OUTPUT_DIR, f"{house_id}_daily_aggregates.csv")
    seq_path = os.path.join(config.OUTPUT_DIR, f"{house_id}_highres_1min.csv")
    daily_df.to_csv(daily_path, index=True)
    seq_df.to_csv(seq_path, index=True)

    # Step 3: Baseline visualization on cleaned data
    p_plus_clean = get_p_plus_series(df_clean)
    week = extract_representative_week(p_plus_clean)

    # Weekly plot
    week_fig_path = os.path.join(config.OUTPUT_DIR, f"{house_id}_pplus_week.png")
    plot_week_p_plus(week, week_fig_path, title=f"{house_id} Weekly P+ Load (cleaned)")

    # Daily plots (one per day)
    daily_dir = os.path.join(config.OUTPUT_DIR, f"{house_id}_daily")
    plot_daily_p_plus(week, daily_dir, prefix=house_id)

    # Hourly plots (one per hour per day)
    hourly_dir = os.path.join(config.OUTPUT_DIR, f"{house_id}_hourly")
    plot_hourly_p_plus(week, hourly_dir, prefix=house_id)

    baseline = compute_weekly_baseline(week)
    print(f"Baseline summary for {house_id}:", baseline)


if __name__ == "__main__":
    run_pipeline_for_house("house_1")
