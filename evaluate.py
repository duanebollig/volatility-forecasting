from collections.abc import Callable

import pandas as pd

from config import HIGH_VOL_QUANTILE
from features import build_dataset
from walk_forward import walk_forward_splits, run_walk_forward
from models import naive_predict, har_predict, lgbm_predict
from metrics import compute_rmse, compute_qlike


models = {
    "naive": naive_predict,
    "har": har_predict,
    "lgbm": lgbm_predict,
}


def get_predictions(
    models: dict[str, Callable[[pd.DataFrame, pd.Series, pd.DataFrame], pd.Series]],
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = build_dataset()
    splits = walk_forward_splits(len(X))
    preds = {}
    for name, model in models.items():
        preds[name] = run_walk_forward(model, X, y, splits)
    df_preds = pd.DataFrame(preds)
    return df_preds, y.loc[df_preds.index]


def build_table(
    preds: pd.DataFrame,
    y_true: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, y_pred in preds.items():
        rows.append({
            "model": name,
            "rmse": compute_rmse(y_true, y_pred),
            "qlike": compute_qlike(y_true, y_pred),
            "n": len(y_pred),
        })
    return pd.DataFrame(rows)


def build_quantile_table(
    preds: pd.DataFrame,
    y_true: pd.Series,
    quantile: float = HIGH_VOL_QUANTILE,
) -> pd.DataFrame:
    threshold = y_true.quantile(quantile)
    threshold_mask = y_true >= threshold
    preds_top = preds[threshold_mask]
    preds_rest = preds[~threshold_mask]
    y_true_top = y_true[threshold_mask]
    y_true_rest = y_true[~threshold_mask]
    rows = []
    for name in preds.columns:
        rows.append({
            "model": name,
            "rmse_top": compute_rmse(y_true_top, preds_top[name]),
            "rmse_rest": compute_rmse(y_true_rest, preds_rest[name]),
            "qlike_top": compute_qlike(y_true_top, preds_top[name]),
            "qlike_rest": compute_qlike(y_true_rest, preds_rest[name]),
            "bias_top": (preds_top[name] - y_true_top).mean(),
            "bias_rest": (preds_rest[name] - y_true_rest).mean(),
            "n_top": len(preds_top[name]),
            "n_rest": len(preds_rest[name]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    preds, y_true = get_predictions(models)
    table = build_table(preds, y_true)
    quantile_table = build_quantile_table(preds, y_true)
    print(f"Overall ({len(y_true)} out-of-sample forecasts)")
    print(table.to_string(index=False))
    print(f"\nTop {1 - HIGH_VOL_QUANTILE:.0%} target rows "
          f"(y >= {HIGH_VOL_QUANTILE * 100:.0f}th percentile) vs. rest")
    print(quantile_table.to_string(index=False))
