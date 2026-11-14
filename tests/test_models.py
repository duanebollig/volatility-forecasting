import pandas as pd
import numpy as np
import pytest

from models import naive_predict, har_predict, lgbm_predict


@pytest.fixture
def x_test():
    return pd.DataFrame(
        {
            "daily": [6, 8, 10],
            "weekly": [4, 5, 6],
            "monthly": [1, 2, 3],
        },
        index=[10, 11, 12]
    )

@pytest.fixture
def x_train_har():
    return pd.DataFrame(
        {
            "daily": [1, 4, 2, 3, 5, 7],
            "weekly": [6, 5, 2, 3, 4, 1],
            "monthly": [2, 3, 1, 4, 5, 6],
        }
    )

@pytest.fixture
def y_train_har(x_train_har):
    return x_train_har @ np.array([2.0, 3.0, 4.0]) + 1

@pytest.fixture
def x_train_lgbm():
    n = 100
    return pd.DataFrame(
        {
            "daily": np.arange(n) + 0.0,
            "weekly": np.arange(n) + 5.0,
            "monthly": np.arange(n) + 22.0,
        }
    )

@pytest.fixture
def y_train_lgbm(x_train_lgbm):
    return x_train_lgbm @ np.array([2.0, 3.0, 4.0]) + 1


def test_naive_returns_weekly_column(x_test):
    result = naive_predict(pd.DataFrame(), pd.Series(dtype=float), x_test)
    expected = pd.Series(
        [4, 5, 6],
        index=[10, 11, 12],
        name="weekly",
    )
    pd.testing.assert_series_equal(result, expected)


def test_naive_index_matches_x_test(x_test):
    result = naive_predict(pd.DataFrame(), pd.Series(dtype=float), x_test)
    assert result.index.equals(x_test.index)


def test_naive_ignores_training_data(x_test):
    rand_td = pd.DataFrame(
        {
            "a": [2, 3],
            "b": [4, 5],
        }
    )
    result_1 = naive_predict(pd.DataFrame(), pd.Series(dtype=float), x_test)
    result_2 = naive_predict(rand_td, pd.Series([2.0, 3.0], dtype=float), x_test)
    pd.testing.assert_series_equal(result_1, result_2)


def test_har_output_index_matches_test_index(x_train_har, y_train_har, x_test):
    result = har_predict(x_train_har, y_train_har, x_test)
    assert x_test.index.equals(result.index)


def test_har_recovers_known_equation(x_train_har, y_train_har, x_test):
    result = har_predict(x_train_har, y_train_har, x_test)
    pd.testing.assert_series_equal(result, x_test @ np.array([2.0, 3.0, 4.0]) + 1, check_exact=False)


def test_har_floors_predictions_below_train_min(x_train_har, x_test):
    y_train_har = x_train_har @ np.array([-5, 0, 0]) + 100
    result = har_predict(x_train_har, y_train_har, x_test)
    assert np.isclose(y_train_har.min(), result).any()
    assert (result >= y_train_har.min()).all()


def test_lgbm_output_index_matches_test(x_train_har, y_train_har, x_test):
    result = lgbm_predict(x_train_har, y_train_har, x_test)
    assert x_test.index.equals(result.index)


def test_lgbm_predictions_vary_inside_training_range(x_train_lgbm, y_train_lgbm):
    x_test = pd.DataFrame({
        "daily": [25, 90],
        "weekly": [25, 90],
        "monthly": [25, 90],
    })
    result = lgbm_predict(x_train_lgbm, y_train_lgbm, x_test)
    assert result.iloc[0] != result.iloc[1]


def test_lgbm_flat_outside_training_range(x_train_lgbm, y_train_lgbm):
    x_test = pd.DataFrame({
        "daily": [1000, 100000, -1000, -100000],
        "weekly": [1000, 100000, -1000, -100000],
        "monthly": [1000, 100000, -1000, -100000],
    })
    result = lgbm_predict(x_train_lgbm, y_train_lgbm, x_test)
    assert result.iloc[0] == result.iloc[1]
    assert result.iloc[2] == result.iloc[3]
    assert result.iloc[0] > result.iloc[2]
