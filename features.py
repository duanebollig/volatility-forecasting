from typing import cast

import numpy as np
import pandas as pd

from config import DATA_DIR, DEFAULT_TICKER, DEFAULT_HORIZON


def load_data(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / f"{ticker.lower()}.csv",
        index_col="Date",
        parse_dates=True
    )
    return df


def log_returns(prices: pd.Series) -> pd.Series:
    prev_prices = prices.shift(1)
    result = np.log(prices) - np.log(prev_prices)
    return cast(pd.Series, result)


def squared_returns(returns: pd.Series) -> pd.Series:
    return returns ** 2


def har_features(variance: pd.Series) -> pd.DataFrame:
    result = {
        "daily": variance,
        "weekly": variance.rolling(5).mean(),
        "monthly": variance.rolling(22).mean(),
    }
    return pd.DataFrame(result)


def forward_variance(variance: pd.Series, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    return variance.rolling(horizon).mean().shift(-horizon)


def build_dataset(ticker: str = DEFAULT_TICKER) -> tuple[pd.DataFrame, pd.Series]:
    df = load_data(ticker)
    returns = log_returns(df["Close"])
    squared = squared_returns(returns)
    data = har_features(squared)
    target = forward_variance(squared)
    data["target"] = target
    data = data.dropna()
    return data.drop(columns="target"), data["target"]


if __name__ == "__main__":
    X, y = build_dataset()
    print(X)
    print(y)

    df = load_data()
    returns = log_returns(df["Close"])
    squared = squared_returns(returns)
    target = forward_variance(squared)
    assert (
        np.isclose(target.iloc[100], squared.iloc[101:106].mean())
    ), "Given horizon = 5, target should equal mean of t+1,...,t+5"
    assert (
        not np.isclose(target.iloc[100], squared.iloc[100:105].mean())
    ), "Given horizon = 5, target should not equal mean of t,t+1,...,t+4"
    assert (
        target.iloc[-5:].isna().all()
    ), "Given horizon = 5, last 5 entries in target should be NaN."
    assert (
        not pd.isna(target.iloc[-6])
    ), "Given horizon = 5, the 6th to last entry in target should not be NaN."
