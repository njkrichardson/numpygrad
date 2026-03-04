import urllib.request
from pathlib import Path

import numpygrad as npg
from numpygrad.utils.data import Dataset

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SHAKESPEARE_URI = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"


def load_shakespeare() -> str:
    if not SHAKESPEARE_PATH.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URI, SHAKESPEARE_PATH)

    with open(SHAKESPEARE_PATH) as f:
        return f.read()


class ShakespeareDataset(Dataset):
    def __init__(self, context_size: int = 4):
        text = load_shakespeare()
        self.context_size = context_size
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.stoi = {char: i for i, char in enumerate(self.vocab)}
        self.itos = {i: char for i, char in enumerate(self.vocab)}
        self.data = npg.array([self.stoi[char] for char in text])

    def decode(self, tokens: npg.array) -> str:
        tokens_list = tokens.squeeze().tolist()
        return "".join([self.itos[token] for token in tokens_list])

    def __len__(self) -> int:
        return self.data.size - self.context_size

    def __getitem__(self, index: int) -> tuple[npg.array, npg.array]:
        return self.data[index : index + self.context_size], self.data[
            index + 1 : index + self.context_size + 1
        ]
