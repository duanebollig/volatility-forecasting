from typing import cast

import numpy as np
import pandas as pd

from config import DATA_DIR, DEFAULT_TICKER


def load_data(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{ticker.lower()}.csv", index_col="Date", parse_dates=True)
    return df


def log_returns_close(sr: pd.Series) -> pd.Series:
    prev_sr = sr.shift(1)
    log_returns = np.log(sr) - np.log(prev_sr)
    log_returns = cast(pd.Series, log_returns)
    return log_returns


if __name__ == "__main__":
    df = load_data()
    print(log_returns_close(df["Close"]).head())
