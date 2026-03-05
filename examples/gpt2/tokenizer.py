import json
from pathlib import Path


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:], strict=False):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, num_merges: int = 256):
        self.num_merges = num_merges
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, str] = {}

    def train(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        char_to_id = {ch: i for i, ch in enumerate(chars)}
        ids = [char_to_id[ch] for ch in text]

        for _i in range(self.num_merges):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.__getitem__)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append(pair)
            ids = _merge(ids, pair, new_id)

    def encode(self, text: str) -> list[int]:
        char_to_id = {v: k for k, v in self.vocab.items() if len(v) == 1}
        ids = [char_to_id[ch] for ch in text]
        base_size = len(self.vocab) - len(self.merges)
        for i, pair in enumerate(self.merges):
            ids = _merge(ids, pair, base_size + i)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path | str) -> None:
        data = {
            "num_merges": self.num_merges,
            "merges": self.merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(num_merges=data["num_merges"])
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.vocab = {int(k): v for k, v in data["vocab"].items()}
        return tok
