from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from features import build_dataset
from config import FIGURES_DIR, DEFAULT_TICKER


def plot_volatility(vol: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(vol, lw=1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title)
    ax.set_ylabel("Annualized volatility")
    fig.savefig(path)


if __name__ == "__main__":
    ticker = DEFAULT_TICKER
    X, _ = build_dataset(ticker)
    plot_volatility(
        np.sqrt(X["monthly"]) * np.sqrt(252),
        f"{ticker} 22-day realized volatility (annualized)",
        FIGURES_DIR / f"{ticker.lower()}_realized_vol.png",
    )
