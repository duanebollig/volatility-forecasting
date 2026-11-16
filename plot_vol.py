from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import DateFormatter
import numpy as np

from features import build_dataset
from evaluate import models, get_predictions
from config import FIGURES_DIR, DEFAULT_TICKER


def to_annual_vol[T: (pd.Series, pd.DataFrame)](x: T) -> T:
    """
    Daily variance in, annualized volatility out.
    """
    return np.sqrt(x) * np.sqrt(252)


def plot_volatility(vol: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(vol, lw=1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title)
    ax.set_ylabel("Annualized volatility")
    fig.savefig(path)


def plot_preds_on_vol(y_true: pd.Series, model_preds: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_true, lw=1.5, color="black", label="Realized (next 5 days)")
    for col in model_preds:
        ax.plot(model_preds[col], lw=1, label=col)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    ticker = DEFAULT_TICKER
    X, y = build_dataset(ticker)
    plot_volatility(
        to_annual_vol(X["monthly"]),
        f"{ticker} 22-day realized volatility (annualized)",
        FIGURES_DIR / f"{ticker.lower()}_realized_vol.png",
    )
    preds, y_true = get_predictions(models)
    plot_preds_on_vol(
        to_annual_vol(y_true).loc["2020-01":"2020-06"],
        to_annual_vol(preds.drop(columns="Naive")).loc["2020-01":"2020-06"],
        f"{ticker} 5-day volatility: forecasts vs. realized (Jan-Jun 2020)",
        FIGURES_DIR / f"{ticker.lower()}_forecasts_2020.png",
    )
