from typing import cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIR, DEFAULT_TICKER


def load_data(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{ticker.lower()}.csv", index_col="Date", parse_dates=True)
    return df


def log_returns(prices: pd.Series) -> pd.Series:
    prev_prices = prices.shift(1)
    result = np.log(prices) - np.log(prev_prices)
    return cast(pd.Series, result)


def squared_returns(returns: pd.Series) -> pd.Series:
    return returns ** 2


if __name__ == "__main__":
    df = load_data()
    returns = log_returns(df["Close"])
    squared = squared_returns(returns)
    print(squared.idxmax())
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(squared), lw=0.5)
    ax.set_yscale("log")
    ax.set_title("Daily Volatility")
    ax.set_ylabel("daily vol, log return")
    plt.show()
