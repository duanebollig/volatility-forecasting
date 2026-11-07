import numpy as np
import pandas as pd
import pytest

from metrics import compute_rmse


def test_rmse_is_accurate():
    y_t = pd.Series([1, 2, 3])
    y_p = pd.Series([1, 2, 4])
    assert compute_rmse(y_t, y_p) == pytest.approx(np.sqrt(1/3))


def test_mismatched_index_raises():
    y_t = pd.Series([1, 2, 3], index=[0, 1, 2])
    y_p = pd.Series([1, 2, 4], index=[1,2,3])
    with pytest.raises(ValueError, match="aren't aligned"):
        compute_rmse(y_t, y_p)


def test_zero_error_returns_zero():
    y = pd.Series([1, 2, 3])
    assert compute_rmse(y, y) == pytest.approx(0)