import yfinance as yf

from config import DATA_DIR, DEFAULT_TICKER, DEFAULT_START


def download_data(ticker: str = DEFAULT_TICKER, start: str = DEFAULT_START) -> None:
    df = yf.download(ticker.upper(), start=start, multi_level_index=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r} starting {start},"
                          " check ticker or connection.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / f"{ticker.lower()}.csv")
    print("Download successful.")


if __name__ == "__main__":
    download_data()
