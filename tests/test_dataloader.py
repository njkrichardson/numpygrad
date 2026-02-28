"""Tests for utils/data/dataset.py and utils/data/dataloader.py."""

import numpy as np
import pytest

from numpygrad.utils.data.dataloader import DataLoader
from numpygrad.utils.data.dataset import Dataset, TensorDataset

# ---------------------------------------------------------------------------
# TensorDataset
# ---------------------------------------------------------------------------


def test_tensor_dataset_len():
    X = np.arange(10.0).reshape(5, 2)
    y = np.arange(5)
    ds = TensorDataset(X, y)
    assert len(ds) == 5


def test_tensor_dataset_getitem():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    ds = TensorDataset(X, y)
    x0, y0 = ds[0]
    np.testing.assert_array_equal(x0.data, [1.0, 2.0])
    assert int(y0.data) == 0


def test_tensor_dataset_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        TensorDataset(np.ones((3, 2)), np.ones(4))


# ---------------------------------------------------------------------------
# DataLoader — basic iteration
# ---------------------------------------------------------------------------


def test_dataloader_basic_iteration():
    X = np.arange(12.0).reshape(6, 2)
    y = np.arange(6)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 3
    for xb, _ in batches:
        assert xb.shape[0] == 2


def test_dataloader_len_no_drop():
    ds = TensorDataset(np.ones((10, 2)), np.ones(10))
    loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=False)
    assert len(loader) == 4  # ceil(10/3)


def test_dataloader_len_drop_last():
    ds = TensorDataset(np.ones((10, 2)), np.ones(10))
    loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=True)
    assert len(loader) == 3  # floor(10/3)


def test_dataloader_drop_last_drops_partial():
    ds = TensorDataset(np.ones((7, 2)), np.ones(7))
    loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0][0].shape[0] == 4


def test_dataloader_shuffle_covers_all_samples():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=3, shuffle=True)

    seen = []
    for _, yb in loader:
        seen.extend(yb.data.tolist())
    assert sorted(seen) == list(range(10))


def test_dataloader_invalid_batch_size():
    ds = TensorDataset(np.ones((4, 2)), np.ones(4))
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(ds, batch_size=0)


# ---------------------------------------------------------------------------
# DataLoader — generic Dataset (per-sample __getitem__ path)
# ---------------------------------------------------------------------------


class SimpleDataset(Dataset):
    """Dataset without .data/.targets attributes — forces generic stacking path.

    Returns Python lists and ints so that _to_numpy falls through to np.asarray.
    """

    def __init__(self, n: int):
        self._X = np.arange(float(n * 2)).reshape(n, 2).tolist()  # Python lists
        self._y = list(range(n))  # Python ints

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, index: int):
        return self._X[index], self._y[index]


def test_dataloader_generic_dataset():
    ds = SimpleDataset(6)
    loader = DataLoader(ds, batch_size=3, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2
    for xb, yb in batches:
        assert xb.shape == (3, 2)
        assert yb.shape == (3,)


def test_dataloader_iter_on_iterator():
    """Calling iter() on a _DataLoaderIter should return self (line 29)."""
    ds = TensorDataset(np.ones((4, 2)), np.ones(4))
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    it = iter(loader)
    assert iter(it) is it


class ArrayItemDataset(Dataset):
    """Generic dataset that returns Array objects — exercises _to_numpy item.data path."""

    def __init__(self, n: int):
        import numpygrad as npg

        self._X = [npg.array(np.array([float(i), float(i) + 0.5])) for i in range(n)]
        self._y = [npg.array(np.array(float(i))) for i in range(n)]

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, index: int):
        return self._X[index], self._y[index]


def test_dataloader_generic_dataset_array_items():
    """Generic dataset returning Array objects hits _to_numpy(item).data branch."""
    ds = ArrayItemDataset(4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    for xb, yb in batches:
        assert xb.shape == (2, 2)
        assert yb.shape == (2,)
