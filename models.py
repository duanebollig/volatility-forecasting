import numpy as np
import pandas as pd


def naive_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    return X_test["weekly"]


def har_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    A_train = np.column_stack((
        np.ones(len(X_train)),
        X_train[["daily", "weekly", "monthly"]].to_numpy(),
    ))
    beta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
    A_test = np.column_stack((
        np.ones(len(X_test)),
        X_test[["daily", "weekly", "monthly"]].to_numpy(),
    ))
    preds = pd.Series(A_test @ beta, index=X_test.index)
    preds = preds.clip(lower=y_train.min())
    return preds
