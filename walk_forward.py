from collections.abc import Callable

import numpy as np
import pandas as pd

from config import INITIAL_TRAIN_SIZE, BLOCK_SIZE, DEFAULT_HORIZON


def walk_forward_splits(
    n_rows: int,
    initial_train_size: int = INITIAL_TRAIN_SIZE,
    block_size: int = BLOCK_SIZE,
    horizon: int = DEFAULT_HORIZON,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if initial_train_size + horizon - 1 >= n_rows:
        raise ValueError("Not enough rows to make a test.")
    result = []
    test_start = initial_train_size + horizon - 1
    while test_start < n_rows:
        train_range = np.arange(0, test_start - horizon + 1)
        test_end = min(n_rows, test_start + block_size)
        test_range = np.arange(test_start, test_end)
        result.append((train_range, test_range))
        test_start += block_size
    return result


def run_walk_forward(
    model: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], pd.Series],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.Series:
    preds = []
    for train, test in splits:
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        result = model(X_train, y_train, X_test)
        preds.append(result)
    return pd.concat(preds)