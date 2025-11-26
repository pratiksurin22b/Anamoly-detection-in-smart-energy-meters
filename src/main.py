import os

from . import config
from .io_utils import load_premises_csv
from .preprocessing import (
    set_time_index,
    sort_and_deduplicate,
    check_frequency_and_gaps,
    get_p_plus_series,
    preprocess_missing_data,
    build_master_from_clean,
    create_15min_aggregates,
    compute_15min_baseline,
    apply_relative_anomaly_labels_15min,
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
    """Run the end-to-end pipeline for a single house.

    This version focuses on the new 15-minute relative labeling method:
    - Works on cleaned data aggregated to 15-minute blocks.
    - Builds a dynamic baseline per (day_of_week, hour, weekend).
    - Assigns labels in {0, 2, 3}: 0=normal, 2=fraud-like low usage, 3=fault/spike.
    - Outputs a single CSV: `<house_id>_15min_relative.csv` in OUTPUT_DIR.
    """
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

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Build master dataset on a regular grid (no labels here for the new pipeline)
    df_master = build_master_from_clean(
        df_clean,
        p_col="p_plus",
        base_freq=config.TARGET_FREQ,
    )

    # Phase 3: 15-minute relative labeling on real data
    print("\n--- Running Phase 3: 15-minute relative labeling (0=normal, 2=fraud, 3=fault) ---")
    df_15 = create_15min_aggregates(df_master, p_col="p_plus", freq="15T")
    df_15 = compute_15min_baseline(df_15)
    df_15 = apply_relative_anomaly_labels_15min(df_15)

    rel_15_path = os.path.join(config.OUTPUT_DIR, f"{house_id}_15min_relative.csv")
    df_15.to_csv(rel_15_path, index=True)

    print("15-min relative labels (label_rel):", df_15["label_rel"].value_counts().sort_index())

    # Step 4: Baseline visualization on cleaned data (independent of labels)
    p_plus_clean = get_p_plus_series(df_clean)
    week = extract_representative_week(p_plus_clean)

    week_fig_path = os.path.join(config.OUTPUT_DIR, f"{house_id}_pplus_week.png")
    plot_week_p_plus(week, week_fig_path, title=f"{house_id} Weekly P+ Load (cleaned)")

    daily_dir = os.path.join(config.OUTPUT_DIR, f"{house_id}_daily")
    plot_daily_p_plus(week, daily_dir, prefix=house_id)

    hourly_dir = os.path.join(config.OUTPUT_DIR, f"{house_id}_hourly")
    plot_hourly_p_plus(week, hourly_dir, prefix=house_id)

    baseline = compute_weekly_baseline(week)
    print(f"Baseline summary for {house_id}:", baseline)


if __name__ == "__main__":
    run_pipeline_for_house("house_1")
