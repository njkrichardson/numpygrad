import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

import numpygrad as npg
from numpygrad.utils.data import TensorDataset

Log = npg.Log(__name__)

DATA_DIR = Path(__file__).parent / "data"

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download(url, dest):
    if os.path.exists(dest):
        Log.debug(f"  {dest.name} already exists, skipping.")
        return
    Log.info(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def parse_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[0:4], "big") == 2051  # magic
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)


def parse_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[0:4], "big") == 2049  # magic
    n = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data[8:], dtype=np.uint8).reshape(n)


def load_mnist(data_dir=DATA_DIR):
    data_dir.mkdir(exist_ok=True)

    paths = {}
    for key, filename in FILES.items():
        dest = data_dir / filename
        download(BASE_URL + filename, dest)
        paths[key] = dest

    return {
        "train_images": parse_images(paths["train_images"]),
        "train_labels": parse_labels(paths["train_labels"]),
        "test_images": parse_images(paths["test_images"]),
        "test_labels": parse_labels(paths["test_labels"]),
    }


class MNIST(TensorDataset):
    def __init__(self, split: str = "train", data_dir=DATA_DIR):
        raw = load_mnist(data_dir)
        if split == "train":
            images = raw["train_images"].reshape(-1, 1, 28, 28)
            labels = raw["train_labels"]
        else:
            images = raw["test_images"].reshape(-1, 1, 28, 28)
            labels = raw["test_labels"]
        super().__init__(images, labels)


if __name__ == "__main__":
    data = load_mnist()
    for k, v in data.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
