import urllib.request
from pathlib import Path

import numpygrad as npg
from examples.gpt2.tokenizer import BPETokenizer
from numpygrad.utils.data import Dataset

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SHAKESPEARE_URI = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"
BPE_CACHE_PATH = DATA_DIR / "bpe_256.json"


def load_shakespeare() -> str:
    if not SHAKESPEARE_PATH.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URI, SHAKESPEARE_PATH)

    with open(SHAKESPEARE_PATH) as f:
        return f.read()


class ShakespeareDataset(Dataset):
    def __init__(self, context_size: int = 4, tokenizer: str = "bpe"):
        text = load_shakespeare()
        self.context_size = context_size

        if tokenizer == "bpe":
            if BPE_CACHE_PATH.exists():
                self.tokenizer = BPETokenizer.load(BPE_CACHE_PATH)
            else:
                self.tokenizer = BPETokenizer(num_merges=256)
                self.tokenizer.train(text)
                self.tokenizer.save(BPE_CACHE_PATH)
            self.vocab_size = self.tokenizer.vocab_size
            self.data = npg.array(self.tokenizer.encode(text))
        else:
            vocab = sorted(set(text))
            self.vocab_size = len(vocab)
            stoi = {ch: i for i, ch in enumerate(vocab)}
            itos = {i: ch for i, ch in enumerate(vocab)}
            self.tokenizer = None
            self._itos = itos
            self.data = npg.array([stoi[ch] for ch in text])

    def decode(self, tokens: npg.array) -> str:
        tokens_list = tokens.squeeze().tolist()
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens_list)
        return "".join(self._itos[t] for t in tokens_list)

    def __len__(self) -> int:
        return self.data.size - self.context_size

    def __getitem__(self, index: int) -> tuple[npg.array, npg.array]:
        return self.data[index : index + self.context_size], self.data[
            index + 1 : index + self.context_size + 1
        ]
