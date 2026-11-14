from collections.abc import Callable

import pandas as pd

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

if __name__ == "__main__":
    preds, y_true = get_predictions(models)
    table = build_table(preds, y_true)
    print(table)
