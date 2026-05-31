import os
import pandas as pd

EVOSR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evosr_llm_data")


def load_data(file_paths, input_cols=None, target_col=None):
    """Load CSV data from train/test files.

    Parameters
    ----------
    file_paths : dict with ``train_data`` and ``test_data`` keys.
    input_cols, target_col : column names for EvoSR semantic data.
        When both are provided, uses them; otherwise auto-detects from
        CSV columns (checks for ``x1``/``y``, then falls back to
        "last column = target" heuristic).
    """
    df_train = pd.read_csv(file_paths["train_data"])
    df_test = pd.read_csv(file_paths["test_data"])

    if input_cols is not None and target_col is not None:
        X_train = df_train[list(input_cols)].to_numpy()
        y_train = df_train[target_col].to_numpy()
        X_test = df_test[list(input_cols)].to_numpy()
        y_test = df_test[target_col].to_numpy()
    else:
        cols = df_train.columns.tolist()
        if "x1" in cols:
            inp = [c for c in cols if c.startswith("x")]
            X_train = df_train[inp].to_numpy()
            y_train = df_train["y"].to_numpy()
            X_test = df_test[inp].to_numpy()
            y_test = df_test["y"].to_numpy()
        else:
            X_train = df_train.iloc[:, :-1].to_numpy()
            y_train = df_train.iloc[:, -1].to_numpy()
            X_test = df_test.iloc[:, :-1].to_numpy()
            y_test = df_test.iloc[:, -1].to_numpy()

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def build_evosr_paths(meta_entry):
    """Build ``train_data`` / ``test_data`` paths for an EvoSR dataset."""
    return {
        "train_data": os.path.join(EVOSR_ROOT, meta_entry["data_group"], meta_entry["name"], "train.csv"),
        "test_data": os.path.join(EVOSR_ROOT, meta_entry["data_group"], meta_entry["name"], "test_id.csv"),
    }


def resolve_evosr(meta_entry, config=None):
    """Convenience: return (file_paths, input_cols, target_col) for an EvoSR
    dataset (func >= 11), or ``None, None, None`` for original benchmarks.

    Each wrapper's ``train_one`` can use this to detect and handle EvoSR data::

        fp, inp, tgt = resolve_evosr(metadata[function_id], config)
        if fp is not None:
            file_paths = fp
        else:
            file_paths = build_file_paths(dataset_dir, function_id)
        X_train, y_train, X_test, y_test = load_data(file_paths, inp, tgt)
    """
    if meta_entry.get("input_cols") and meta_entry.get("target_col"):
        return (
            build_evosr_paths(meta_entry),
            meta_entry["input_cols"],
            meta_entry["target_col"],
        )
    return None, None, None
