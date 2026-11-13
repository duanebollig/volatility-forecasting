import pandas as pd
import pytest

from models import naive_predict


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
