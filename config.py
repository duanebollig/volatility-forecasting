from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
FIGURES_DIR = Path(__file__).parent / "figures"
DEFAULT_TICKER = "SPY"
DEFAULT_START = "2000-01-01"
DEFAULT_HORIZON = 5
INITIAL_TRAIN_SIZE = 1400 # dot.com bubble and following 2003 - mid 2005's calm
BLOCK_SIZE = 22 # trading-monthly refit, useful because HAR refitting is cheap