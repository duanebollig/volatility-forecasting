from features import build_dataset
from walk_forward import walk_forward_splits, run_walk_forward
from models import naive_predict
from metrics import compute_rmse, compute_qlike


if __name__ == "__main__":
    X, y = build_dataset()
    splits = walk_forward_splits(len(X))
    naive_pred = run_walk_forward(naive_predict, X, y, splits)
    y_true = y.iloc[naive_pred.index]
    rmse = compute_rmse(y_true, naive_pred)
    qlike = compute_qlike(y_true, naive_pred)
    print(rmse)
    print(qlike)
