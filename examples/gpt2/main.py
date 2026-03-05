import argparse
import dataclasses
from pathlib import Path

import numpy as np

import numpygrad as npg
import numpygrad.nn as nn
from examples.gpt2.data import ShakespeareDataset
from numpygrad.utils.data import DataLoader
from numpygrad.utils.io import load_checkpoint, save_checkpoint

npg.manual_seed(0)
Log = npg.Log(__name__)

parser = argparse.ArgumentParser()

parser.add_argument_group("architecture")
parser.add_argument("--context-size", type=int, default=64)
parser.add_argument("--num-blocks", type=int, default=6)
parser.add_argument("--num-heads", type=int, default=6)
parser.add_argument("--embedding-dim", type=int, default=288)
parser.add_argument("--dropout", type=float, default=0.0)

parser.add_argument_group("training")
parser.add_argument("--num-steps", type=int, default=2_048)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--report-every", type=int, default=25)

parser.add_argument_group("sampling")
parser.add_argument("--initial-sample-length", type=int, default=256)
parser.add_argument("--final-sample-length", type=int, default=1_024)
parser.add_argument("--temperature", type=float, default=1.0)

parser.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")

parser.add_argument_group("checkpointing")
parser.add_argument("--checkpoint-dir", type=str, default=None)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--resume", type=str, default=None)


@dataclasses.dataclass
class GPT2Config:
    context_size: int = 32
    num_blocks: int = 4
    num_heads: int = 4
    embedding_dim: int = 64
    dropout: float = 0.0
    vocab_size: int | None = None  # from the dataset


class CausalAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads: "
                f"{config.embedding_dim} % {config.num_heads} != 0"
            )
        self.head_dim = config.embedding_dim // config.num_heads

        self.config = config

        # query, key, value projections for all heads (batched)
        self.in_proj = nn.Linear(
            config.embedding_dim, 3 * config.embedding_dim, bias=False
        )  # 3 since we have q, k, v
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # causal mask buffer
        self.causal_mask = npg.triu(npg.ones((config.context_size, config.context_size)), k=1).view(
            1, 1, config.context_size, config.context_size
        )

    def forward(self, x: npg.array) -> npg.array:
        B, C, _ = x.shape  # batch size, context size, embedding dim

        x = self.in_proj(x)
        q, k, v = x.split(self.config.embedding_dim, dim=2)
        q = q.view(B, C, self.config.num_heads, self.head_dim).transpose(1, 2)  # B, H, C, head_dim
        k = k.view(B, C, self.config.num_heads, self.head_dim).transpose(1, 2)  # B, H, C, head_dim
        v = v.view(B, C, self.config.num_heads, self.head_dim).transpose(1, 2)  # B, H, C, head_dim

        # attention
        attn_scores = q @ k.transpose(-2, -1) / npg.sqrt(self.head_dim)  # B, H, C, C
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :C, :C], -float("inf"))
        attn_weights = npg.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        x = attn_weights @ v  # B, H, C, head_dim
        x = x.transpose(1, 2).reshape(B, C, self.config.embedding_dim)
        x = self.out_proj(x)
        x = self.residual_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Config, dilation: int = 4):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.embedding_dim, dilation * config.embedding_dim, bias=False)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(
            dilation * config.embedding_dim, config.embedding_dim, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: npg.array) -> npg.array:
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class Gpt2Block(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
    ):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalAttention(config)
        self.ffn = MLP(config)

    def forward(self, x: npg.array) -> npg.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
    ):
        super().__init__()
        self.config = config
        assert config.vocab_size is not None

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output_ln = nn.LayerNorm(config.embedding_dim)
        self.blocks = nn.Sequential(
            *[Gpt2Block(config) for _ in range(config.num_blocks)]
        )  # TODO: make sequential iterable

        # TODO: add parameter initializer strategies
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, x: npg.array) -> npg.array:
        B, C = x.shape
        if C > self.config.context_size:
            raise ValueError(
                f"Context size must be greater than or equal to the number"
                f"of tokens: {C} > {self.config.context_size}"
            )

        positions = npg.arange(C)

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = self.dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.output_ln(x)
        logits = self.output_projection(x)
        return logits


@npg.no_grad()
def sample(
    net: GPT2, context: npg.array | None = None, max_new_tokens: int = 100, temperature: float = 1.0
) -> npg.array:
    if context is None:
        context = npg.zeros((1, 1), dtype=npg.int64)
    for _ in range(max_new_tokens):
        logits = net(context[:, -net.config.context_size :]) / temperature
        next_token_probs = npg.softmax(logits[:, -1, :], axis=-1)
        idx_next = np.random.choice(
            np.arange(next_token_probs.shape[1]), p=next_token_probs.numpy().squeeze()
        )
        next_token = npg.array([idx_next]).view(1, 1)
        context = npg.cat((context, next_token), axis=-1)
    return context


def main(args: argparse.Namespace):
    config = GPT2Config(
        context_size=args.context_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )

    dataset = ShakespeareDataset(config.context_size, tokenizer=args.tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config.vocab_size = dataset.vocab_size
    net = GPT2(config)

    Log.info(
        f"Initial sample: {dataset.decode(sample(net, max_new_tokens=args.initial_sample_length, temperature=args.temperature))}"  # noqa: E501
    )

    @npg.no_grad()
    def estimate_loss(num_batches: int = 32, loader=None):
        if loader is None:
            loader = dataloader
        losses = []
        for _ in range(num_batches):
            x, target = next(iter(loader))
            logits = net(x)
            loss = nn.cross_entropy_loss(logits, target)
            losses.append(loss.item())
        return npg.mean(npg.array(losses)).item()

    optimizer = npg.optim.AdamW(net.parameters(), lr=1e-3)

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        Log.info(f"Resumed from step {ckpt['step']}")

    for step, (x, target) in enumerate(dataloader):
        if step < start_step:
            continue

        optimizer.zero_grad()
        logits = net(x)
        loss = nn.cross_entropy_loss(logits, target)
        loss.backward()
        optimizer.step()

        if step % args.report_every == 0:
            Log.info(f"Step {step}: loss={estimate_loss():.4f}")

        if args.checkpoint_dir and step % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"checkpoint_{step:06d}.pkl"
            save_checkpoint(ckpt_path, step=step, model=net, optimizer=optimizer)

        if step >= args.num_steps:
            break

    if args.checkpoint_dir:
        ckpt_path = Path(args.checkpoint_dir) / f"checkpoint_{step:06d}.pkl"
        save_checkpoint(ckpt_path, step=step, model=net, optimizer=optimizer)

    Log.info(
        f"Final sample: {dataset.decode(sample(net, max_new_tokens=args.final_sample_length, temperature=args.temperature))}"  # noqa: E501
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
