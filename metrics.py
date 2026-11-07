import pandas as pd
import numpy as np

def compute_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Indexes aren't aligned, Computed values may be inaccurate.")
    diff_y = (y_true - y_pred) ** 2
    rmse = np.sqrt(np.mean(diff_y))
    return rmse
