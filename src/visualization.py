import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_week_p_plus(series: pd.Series, output_path: str, title: str = "Weekly P+ Load") -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("P+ (Total Active Power Import)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_daily_p_plus(series: pd.Series, output_dir: str, prefix: str = "house_1") -> None:
    """Plot one graph per day for the given week-long P+ series.

    Each day gets its own PNG for easier visual inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = series.to_frame("p_plus")
    df["date"] = df.index.date

    for day, group in df.groupby("date"):
        day_series = group["p_plus"]
        plt.figure(figsize=(10, 3))
        plt.plot(day_series.index, day_series.values, linewidth=0.7)
        plt.title(f"{prefix} P+ on {day}")
        plt.xlabel("Time")
        plt.ylabel("P+ (Total Active Power Import)")
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{prefix}_pplus_{day}.png")
        plt.savefig(filename)
        plt.close()


def plot_hourly_p_plus(series: pd.Series, output_dir: str, prefix: str = "house_1") -> None:
    """Plot one graph per hour for the given week-long P+ series.

    This will create up to 24 * number_of_days graphs for fine-grained inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = series.to_frame("p_plus")
    df["date"] = df.index.date
    df["hour"] = df.index.hour

    for (day, hour), group in df.groupby(["date", "hour"]):
        hour_series = group["p_plus"]
        if hour_series.empty:
            continue
        plt.figure(figsize=(6, 3))
        plt.plot(hour_series.index, hour_series.values, linewidth=0.8)
        plt.title(f"{prefix} P+ on {day} at {hour:02d}:00")
        plt.xlabel("Time")
        plt.ylabel("P+ (Total Active Power Import)")
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{prefix}_pplus_{day}_h{hour:02d}.png")
        plt.savefig(filename)
        plt.close()

