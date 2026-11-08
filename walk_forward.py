import numpy as np

from config import INITIAL_TRAIN_SIZE, BLOCK_SIZE, DEFAULT_HORIZON


def walk_forward_splits(
    n_rows: int,
    initial_train_size: int = INITIAL_TRAIN_SIZE,
    block_size: int = BLOCK_SIZE,
    horizon: int = DEFAULT_HORIZON,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if initial_train_size + horizon - 1 >= n_rows:
        raise ValueError("Not enough rows to make a test.")
    result = []
    test_start = initial_train_size + horizon - 1
    while test_start < n_rows:
        train_range = np.arange(0, test_start - horizon + 1)
        test_end = min(n_rows, test_start + block_size)
        test_range = np.arange(test_start, test_end)
        result.append((train_range, test_range))
        test_start += block_size
    return result
