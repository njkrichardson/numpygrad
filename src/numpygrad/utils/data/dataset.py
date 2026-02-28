from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Dataset(ABC):
    """Base class for datasets used with DataLoader.
    Subclass and implement __len__ and __getitem__.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, Any]: ...


class TensorDataset(Dataset):
    """
    Dataset of (data, targets) from two array-likes, similar to
    torch.utils.data.TensorDataset.

    Accepts np.ndarray or list; stores data and targets as numpygrad arrays so the
    DataLoader can batch by slicing without per-sample conversion or stacking.
    """

    def __init__(self, data: np.ndarray | list[Any], targets: np.ndarray | list[Any]):
        import numpygrad as npg

        data_np = np.asarray(data)
        targets_np = np.asarray(targets)
        if len(data_np) != len(targets_np):
            raise ValueError("data and targets must have the same length")
        self.data = npg.array(data_np)
        self.targets = npg.array(targets_np)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.data[index], self.targets[index]
