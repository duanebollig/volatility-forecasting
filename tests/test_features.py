import numpy as np
import pandas as pd

from features import forward_variance, har_features, build_dataset


def make_variance(n: int = 30) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(np.arange(1, n + 1, dtype=float), index=dates)


def test_target_first_row_is_mean_of_next_5():
    variance = make_variance()
    target = forward_variance(variance, horizon=5)
    assert np.isclose(target.iloc[0], 4.0)


def test_target_excludes_today():
    variance = make_variance()
    target = forward_variance(variance, horizon=5)
    assert not np.isclose(target.iloc[0], 3.0)


def test_target_last_rows_are_nan():
    variance = make_variance()
    target = forward_variance(variance, horizon=5)
    assert not pd.isna(target.iloc[-6])
    assert target.iloc[-5:].isna().all()


def test_weekly_feature_is_mean_of_last_5():
    variance = make_variance()
    features = har_features(variance)
    assert np.isclose(features["weekly"].iloc[4], 3.0)
    assert pd.isna(features["weekly"].iloc[3])


def test_monthly_feature_is_mean_of_last_22():
    variance = make_variance()
    features = har_features(variance)
    assert np.isclose(features["monthly"].iloc[21], 11.5)
    assert pd.isna(features["monthly"].iloc[20])


def test_features_ignore_future_values():
    variance = make_variance()
    before = har_features(variance)
    changed = variance.copy()
    changed.iloc[25] = 1000
    after = har_features(changed)
    assert not np.isclose(before.iloc[25], after.iloc[25]).any()
    assert np.allclose(before.iloc[:25], after.iloc[:25], equal_nan=True)


def test_target_ignores_today_and_far_future():
    variance = make_variance()
    before = forward_variance(variance, horizon=5)
    changed = variance.copy()
    changed.iloc[25] = 1000
    changed.iloc[0] = 1000
    after = forward_variance(changed, horizon=5)
    assert np.isclose(before.iloc[0], after.iloc[0])
    changed_2 = variance.copy()
    changed_2.iloc[3] = 1000
    after_2 = forward_variance(changed_2, horizon=5)
    assert not np.isclose(before.iloc[0], after_2.iloc[0])


def test_build_dataset_is_aligned():
    X, y = build_dataset()
    assert X.index.equals(y.index)
    assert not y.isna().any()
    assert not X.isna().any().any()
    assert isinstance(y, pd.Series)
