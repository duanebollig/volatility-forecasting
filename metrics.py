import pandas as pd
import numpy as np

def compute_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Indexes aren't aligned. Computed values may be inaccurate.")
    diff_y = (y_true - y_pred) ** 2
    rmse = np.sqrt(np.mean(diff_y))
    return rmse


def compute_qlike(y_true: pd.Series, y_pred: pd.Series) -> float:
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Indexes aren't aligned. Computed values may be inaccurate.")
    if (y_true <= 0).any():
        raise ValueError("y_true is nonpositive. Target series values cannot be <= 0.")
    if (y_pred <= 0).any():
            raise ValueError("y_pred is nonpositive. Forecast series values cannot be <= 0.")
    true_over_pred = y_true / y_pred
    qlike = (true_over_pred) - np.log(true_over_pred) - 1
    return np.mean(qlike)
