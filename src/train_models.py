import os
from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from . import config


def _ensure_training_log_dir() -> str:
    """Ensure the directory for training logs exists and return its path."""
    log_dir = os.path.join(config.OUTPUT_DIR, "training_models_info")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _start_run_log(model_name: str, house_id: str) -> Tuple[str, list]:
    """Create a new log file path and initial header lines for a run.

    Returns (log_path, lines_list) where lines_list accumulates strings.
    """
    log_dir = _ensure_training_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{house_id}_{timestamp}.txt"
    log_path = os.path.join(log_dir, filename)

    header = [
        f"Model: {model_name}",
        f"House: {house_id}",
        f"Timestamp: {timestamp}",
        "" ,
    ]
    return log_path, header


def _write_log(log_path: str, lines: list) -> None:
    """Write accumulated lines to the given log file path."""
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def load_15min_relative(house_id: str = "house_1") -> pd.DataFrame:
    """Load the 15-minute relative-labeled dataset created by main.py."""
    path = os.path.join(config.OUTPUT_DIR, f"{house_id}_15min_relative.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"15-min relative file not found: {path}. Run src.main first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def make_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split based on index order."""
    df = df.sort_index()
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    return df_train, df_val, df_test


def prepare_block_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare features and labels for block-level models (Random Forest).

    Features (per spec):
    - P_mean, P_max, P_std, Baseline_Mean, Baseline_Max

    Target:
    - label_rel in {0, 2, 3}
    """
    required = ["P_mean", "P_max", "P_std", "Baseline_Mean", "Baseline_Max", "label_rel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df for RF training: {missing}")

    feature_cols = ["P_mean", "P_max", "P_std", "Baseline_Mean", "Baseline_Max"]
    X = df[feature_cols].values.astype(float)
    y = df["label_rel"].values.astype(int)
    return X, y, feature_cols


def train_random_forest_edge(house_id: str = "house_1") -> None:
    """Experiment 1: Edge benchmark using Random Forest on single 15-min blocks.

    - Inputs: P_mean, P_max, P_std, Baseline_Mean, Baseline_Max
    - Target: label_rel in {0,2,3}
    - Split: time-based train/val/test
    - Prints classification report and confusion matrix on test set.
    - Automatically logs configuration and results to a text file.
    """
    df = load_15min_relative(house_id)
    df_train, df_val, df_test = make_train_val_test_split(df)

    X_train, y_train, feature_cols = prepare_block_features(df_train)
    X_test, y_test, _ = prepare_block_features(df_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)

    print("\n=== Random Forest Edge Benchmark (single 15-min blocks) ===")
    print("Features:", feature_cols)
    print("Test set size:", len(y_test))
    print("Label distribution in test set:", pd.Series(y_test).value_counts().sort_index())

    print("\nClassification report (labels: 0=normal, 2=fraud, 3=fault):")
    report = classification_report(y_test, y_pred, labels=[0, 2, 3])
    print(report)

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 2, 3])
    print(cm)

    # Per-class accuracy (percentage of correctly classified samples for each label)
    print("\nPer-class accuracy (% correctly classified in each true class):")
    true_labels = [0, 2, 3]
    per_class_acc = {}
    for i, lbl in enumerate(true_labels):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = 100.0 * correct / total if total > 0 else 0.0
        per_class_acc[lbl] = acc
        print(f"Label {lbl}: {acc:.2f}% (correct {correct} / {total})")

    # --- Logging to text file ---
    log_path, lines = _start_run_log("RandomForest", house_id)
    lines.append("Training/testing split: 70% train, 15% val, 15% test (chronological)")
    lines.append(f"Features: {feature_cols}")
    lines.append("RF hyperparameters:")
    lines.append(f"  n_estimators=200, max_depth=None, class_weight='balanced_subsample', random_state=42")
    lines.append("")
    lines.append(f"Test set size: {len(y_test)}")
    lines.append("Label distribution in test set (true y):")
    lines.append(str(pd.Series(y_test).value_counts().sort_index()))
    lines.append("")
    lines.append("Classification report:")
    lines.append(report)
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    lines.append(str(cm))
    lines.append("")
    lines.append("Per-class accuracy (% correctly classified in each true class):")
    for lbl, acc in per_class_acc.items():
        lines.append(f"  Label {lbl}: {acc:.2f}%")

    _write_log(log_path, lines)
    print(f"\nRF run logged to: {log_path}")


def make_sequences(
    df: pd.DataFrame,
    seq_len: int = 96,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a time-ordered 15-min dataframe into sequences for LSTM.

    Each sequence is `seq_len` consecutive 15-min blocks with features:
    - P_mean, P_max, P_std, Baseline_Mean, Baseline_Max

    The sequence label is the max label_rel in the window (3 > 2 > 0).
    Returns X of shape (num_seq, seq_len, 5), y of shape (num_seq,).
    """
    required = ["P_mean", "P_max", "P_std", "Baseline_Mean", "Baseline_Max", "label_rel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df for LSTM sequences: {missing}")

    feats = ["P_mean", "P_max", "P_std", "Baseline_Mean", "Baseline_Max"]
    values = df[feats].values.astype(float)
    labels = df["label_rel"].values.astype(int)

    n = len(df)
    if n < seq_len:
        return np.empty((0, seq_len, len(feats))), np.empty((0,), dtype=int)

    X_list = []
    y_list = []
    for start in range(0, n - seq_len + 1):
        end = start + seq_len
        window_vals = values[start:end]
        window_labels = labels[start:end]

        X_list.append(window_vals)
        y_list.append(window_labels.max())

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y


def _map_labels_to_indices(y: np.ndarray) -> Tuple[np.ndarray, dict, dict]:
    """Map labels {0,2,3} to contiguous indices {0,1,2} for Keras.

    Returns mapped_y, label_to_idx, idx_to_label.
    """
    unique = sorted(int(v) for v in np.unique(y))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    mapped = np.vectorize(label_to_idx.get)(y)
    return mapped, label_to_idx, idx_to_label


def train_lstm_cloud(house_id: str = "house_1", seq_len: int = 96, epochs: int = 10, batch_size: int = 64) -> None:
    """Experiment 2: Cloud LSTM using 24-hour sequences of 15-min blocks.

    - Inputs: sequences of length seq_len (default 96 = 24h) of
      [P_mean, P_max, P_std, Baseline_Mean, Baseline_Max].
    - Target: max label_rel in the sequence (0,2,3).
    - Split: time-based train/val/test, then sequence generation per split.
    - Automatically logs configuration and results to a text file.
    """
    # Lazy import to avoid TensorFlow cost if only RF is used
    import tensorflow as tf
    from tensorflow.keras import layers, models

    df = load_15min_relative(house_id)
    df_train, df_val, df_test = make_train_val_test_split(df)

    # Build sequences on each split separately (to avoid leakage)
    X_train, y_train = make_sequences(df_train, seq_len=seq_len)
    X_val, y_val = make_sequences(df_val, seq_len=seq_len)
    X_test, y_test = make_sequences(df_test, seq_len=seq_len)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise RuntimeError("Not enough data to build LSTM sequences. Try shorter seq_len.")

    # Standardize features across all timesteps jointly using train set
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    def scale_sequences(X: np.ndarray) -> np.ndarray:
        X2d = X.reshape(-1, n_features)
        X2d_scaled = scaler.transform(X2d)
        return X2d_scaled.reshape(X.shape)

    X_train_scaled = scale_sequences(X_train)
    X_val_scaled = scale_sequences(X_val)
    X_test_scaled = scale_sequences(X_test)

    # Map labels to contiguous indices for softmax
    y_train_mapped, label_to_idx, idx_to_label = _map_labels_to_indices(y_train)
    y_val_mapped = np.vectorize(label_to_idx.get)(y_val)
    y_test_mapped = np.vectorize(label_to_idx.get)(y_test)

    n_classes = len(label_to_idx)

    # Build LSTM model
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n=== LSTM Cloud Model (24h sequences) ===")
    print("Train sequences:", X_train_scaled.shape[0])
    print("Val sequences:", X_val_scaled.shape[0])
    print("Test sequences:", X_test_scaled.shape[0])
    print("Label mapping (original -> index):", label_to_idx)

    history = model.fit(
        X_train_scaled,
        y_train_mapped,
        validation_data=(X_val_scaled, y_val_mapped),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Evaluate on test set
    y_prob = model.predict(X_test_scaled)
    y_pred_idx = y_prob.argmax(axis=1)

    # Map indices back to original labels {0,2,3}
    idx_to_label_arr = np.vectorize(idx_to_label.get)
    y_pred = idx_to_label_arr(y_pred_idx)

    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification report (sequence labels: 0=normal, 2=fraud, 3=fault):")
    report = classification_report(y_test, y_pred, labels=sorted(label_to_idx.keys()))
    print(report)

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(label_to_idx.keys()))
    print(cm)

    print("\nPer-class accuracy for sequences (% correctly classified in each true class):")
    per_class_acc = {}
    for i, lbl in enumerate(sorted(label_to_idx.keys())):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = 100.0 * correct / total if total > 0 else 0.0
        per_class_acc[lbl] = acc
        print(f"Label {lbl}: {acc:.2f}% (correct {correct} / {total})")

    # --- Logging to text file ---
    log_path, lines = _start_run_log("LSTM", house_id)
    lines.append("Training/testing split: 70% train, 15% val, 15% test (chronological)")
    lines.append(f"Sequence length (15-min blocks): {seq_len}")
    lines.append(f"Number of features per timestep: {n_features}")
    lines.append("LSTM hyperparameters:")
    lines.append("  LSTM units=64, dense_units=64, dropout=0.3, optimizer='adam',")
    lines.append(f"  loss='sparse_categorical_crossentropy', epochs={epochs}, batch_size={batch_size}")
    lines.append("")
    lines.append(f"Train sequences: {X_train_scaled.shape[0]}")
    lines.append(f"Val sequences: {X_val_scaled.shape[0]}")
    lines.append(f"Test sequences: {X_test_scaled.shape[0]}")
    lines.append("Label mapping (original -> index):")
    lines.append(str(label_to_idx))
    lines.append("")
    lines.append("Classification report:")
    lines.append(report)
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    lines.append(str(cm))
    lines.append("")
    lines.append("Per-class accuracy for sequences (% correctly classified in each true class):")
    for lbl, acc in per_class_acc.items():
        lines.append(f"  Label {lbl}: {acc:.2f}%")

    _write_log(log_path, lines)
    print(f"\nLSTM run logged to: {log_path}")


def run_both_models(house_id: str = "house_1") -> None:
    """Convenience helper to run RF edge benchmark and LSTM cloud model."""
    train_random_forest_edge(house_id)
    train_lstm_cloud(house_id, seq_len=96, epochs=10, batch_size=64)


if __name__ == "__main__":
    # By default, just run the RF experiment to keep it fast.
    # For full comparison incl. LSTM, call run_both_models("house_1")
    train_random_forest_edge("house_1")
