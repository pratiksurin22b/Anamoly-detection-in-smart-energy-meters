import pandas as pd
from . import config


def load_premises_csv(path: str) -> pd.DataFrame:
    """Load a PREMISES-style CSV and standardize key columns.

    - Keeps timestamp, P+, P-, Gas, Water where available.
    - Renames them to logical names used downstream.
    """
    df = pd.read_csv(path)

    col_map = {}
    for logical, raw in [
        ("timestamp", config.TIMESTAMP_COL),
        ("p_plus", config.P_PLUS_COL),
        ("p_minus", config.P_MINUS_COL),
        ("gas", config.GAS_COL),
        ("water", config.WATER_COL),
    ]:
        if raw in df.columns:
            col_map[raw] = logical

    df = df.rename(columns=col_map)
    return df

