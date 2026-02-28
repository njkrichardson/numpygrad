import numpy as np

from numpygrad.utils.data.dataset import Dataset


def _can_batch_by_slice(dataset: Dataset) -> bool:
    """True if dataset exposes .data and .targets for slice-based batching."""
    return hasattr(dataset, "data") and hasattr(dataset, "targets")


def _to_numpy(item: object) -> np.ndarray:
    """Extract numpy array from Array or return as-is if already ndarray."""
    if hasattr(item, "data"):
        return item.data
    return np.asarray(item)


class _DataLoaderIter:
    """Iterator that yields batches from a DataLoader."""

    def __init__(self, loader: "DataLoader"):
        self.loader = loader
        self.indices = np.arange(len(loader.dataset))
        if loader.shuffle:
            np.random.shuffle(self.indices)
        self.position = 0

    def __iter__(self) -> "_DataLoaderIter":
        return self

    def __next__(self) -> tuple:
        dataset = self.loader.dataset
        batch_size = self.loader.batch_size
        n = len(self.indices)

        if self.position >= n:
            raise StopIteration

        end = self.position + batch_size
        if self.loader.drop_last and end > n:
            raise StopIteration
        end = min(end, n)
        idx_batch = self.indices[self.position : end]
        self.position = end

        if _can_batch_by_slice(dataset):
            batch_x = dataset.data[idx_batch]
            batch_y = dataset.targets[idx_batch]
            return batch_x, batch_y

        # Generic path: per-sample __getitem__ and stack
        xs = [dataset[i][0] for i in idx_batch]
        ys = [dataset[i][1] for i in idx_batch]
        import numpygrad as npg

        batch_x = npg.array(np.stack([_to_numpy(x) for x in xs]))
        batch_y = npg.array(np.stack([_to_numpy(y) for y in ys]))
        return batch_x, batch_y


class DataLoader:
    """
    Loads batches from a Dataset with optional shuffling, in a PyTorch-like way.

    Datasets that expose .data and .targets as arrays (e.g. TensorDataset) are
    batched by slicing for efficiency; others use per-sample __getitem__ and stacking.

    Example:
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for x_batch, y_batch in loader:
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
            ...
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> _DataLoaderIter:
        return _DataLoaderIter(self)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last and n % self.batch_size != 0:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
