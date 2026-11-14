from collections.abc import Callable

import pandas as pd

from features import build_dataset
from walk_forward import walk_forward_splits, run_walk_forward
from models import naive_predict
from metrics import compute_rmse, compute_qlike


models = {
    "naive": naive_predict,
}

def build_table(
    models: dict[str, Callable[[pd.DataFrame, pd.Series, pd.DataFrame], pd.Series]],
) -> pd.DataFrame:
    X, y = build_dataset()
    splits = walk_forward_splits(len(X))
    rows = []
    for name, model in models.items():
        y_pred = run_walk_forward(model, X, y, splits)
        y_true = y.loc[y_pred.index]
        rows.append({
            "model": name,
            "rmse": compute_rmse(y_true, y_pred),
            "qlike": compute_qlike(y_true, y_pred),
            "n": len(y_pred),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    table = build_table(models)
    print(table)
