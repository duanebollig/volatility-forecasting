import numpy as np
import pandas as pd
import pytest

from metrics import compute_rmse, compute_qlike


def test_rmse_is_accurate():
    y_true = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 2, 4])
    assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt(1/3))


def test_rmse_zero_error_returns_zero():
    y = pd.Series([1, 2, 3])
    assert compute_rmse(y, y) == pytest.approx(0)


def test_rmse_mismatched_index_raises():
    y_true = pd.Series([1, 2, 3], index=[0, 1, 2])
    y_pred = pd.Series([1, 2, 4], index=[1,2,3])
    with pytest.raises(ValueError, match="aren't aligned"):
        compute_rmse(y_true, y_pred)


def test_qlike_mismatched_index_raises():
    y_true = pd.Series([1, 2, 3], index=[0, 1, 2])
    y_pred = pd.Series([1, 2, 4], index=[1,2,3])
    with pytest.raises(ValueError, match="aren't aligned"):
        compute_qlike(y_true, y_pred)


def test_qlike_nonpositive_target_raises():
    y_true = pd.Series([1, 0, 2])
    y_pred = pd.Series([1, 2, 2])
    with pytest.raises(ValueError, match="y_true is nonpositive"):
        compute_qlike(y_true, y_pred)


def test_qlike_nonpositive_forecast_raises():
    y_true = pd.Series([1, 2, 2])
    y_pred = pd.Series([1, 0, 2])
    with pytest.raises(ValueError,  match="y_pred is nonpositive"):
        compute_qlike(y_true, y_pred)


def test_qlike_is_accurate():
    y_true = pd.Series([1, 2, 2])
    y_pred = pd.Series([1, 2, 1])
    assert compute_qlike(y_true, y_pred) == pytest.approx((1 - np.log(2)) / 3)


def test_qlike_perfect_forecast_returns_zero():
    y = pd.Series([1, 2, 2])
    assert compute_qlike(y, y) == pytest.approx(0)


def test_qlike_penalizes_underprediction_more():
    y_true = pd.Series([2, 2, 2])
    y_pred_lower = pd.Series([1, 1, 1])
    y_pred_higher = pd.Series([3, 3, 3])
    assert compute_qlike(y_true, y_pred_lower) > compute_qlike(y_true, y_pred_higher)


def test_qlike_is_scale_invariant():
    y_true = pd.Series([1, 2, 2])
    y_pred = pd.Series([1, 2, 1])
    assert (
        compute_qlike(3 * y_true, 3 * y_pred)
        == pytest.approx(compute_qlike(y_true, y_pred))
    )