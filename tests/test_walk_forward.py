import numpy as np
import pytest

from walk_forward import walk_forward_splits


@pytest.fixture
def splits():
    return walk_forward_splits(
        n_rows=100, initial_train_size=20,
        block_size=10, horizon=5,
    )


def test_no_room_for_test_raises():
    n_rows = 24
    initial = 20
    horizon = 5
    with pytest.raises(ValueError, match="enough"):
        walk_forward_splits(n_rows=n_rows, initial_train_size=initial, horizon=horizon)


def test_first_split_matches(splits):
    train, test = splits[0]
    assert np.array_equal(train, np.arange(0, 20))
    assert np.array_equal(test, np.arange(24, 34))


def test_number_of_splits(splits):
    assert len(splits) == 8


def test_gap_is_exactly_horizon():
    horizon = 5
    splits = walk_forward_splits(
        n_rows=100, initial_train_size=20,
        block_size=10, horizon=horizon,
    )
    for train, test in splits:
        assert train[-1] + horizon == test[0]


def test_train_always_starts_at_zero(splits):
    for train, _ in splits:
        assert train[0] == 0


def test_test_blocks_cover_all_rows_once(splits):
    test_list = [i[1] for i in splits]
    assert np.array_equal(np.concatenate(test_list), np.arange(24, 100))


def test_short_last_block_is_kept(splits):
    _, test = splits[-1]
    assert np.array_equal(test, np.arange(94, 100))


def test_horizon_sets_the_gap():
    horizon = 22
    splits = walk_forward_splits(
        n_rows=100, initial_train_size=20,
        block_size=10, horizon=horizon,
    )
    train, test = splits[0]
    assert test[0] == 41
    assert np.array_equal(train, np.arange(0, 20))


def test_real_config_shapes():
    splits = walk_forward_splits(n_rows=6719)
    assert len(splits) == 242
    _, first_test = splits[0]
    assert first_test[0] == 1404
    _, last_test = splits[-1]
    assert last_test[-1] == 6718


def test_loop_stops_when_test_start_passes_end():
    splits = walk_forward_splits(
        n_rows=92, initial_train_size=20,
        block_size=10, horizon=5
    )
    assert len(splits) == 7
    _, test = splits[-1]
    assert np.array_equal(test, np.arange(84, 92))
