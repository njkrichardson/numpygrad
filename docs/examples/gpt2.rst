GPT-2 Character Language Model
================================

Source: ``examples/gpt2/main.py``

Overview
--------

A GPT-2-style transformer trained as a character-level language model on
Shakespeare's complete works. The architecture mirrors the original GPT-2
paper: stacked transformer blocks with causal (masked) self-attention,
LayerNorm, and GeLU activations — all implemented in pure NumPy using
numpygrad.

Running
-------

::

    python -m examples.gpt2.main              # downloads data, trains with defaults
    python -m examples.gpt2.main --help       # see all options

Selected options:

- ``--context-size`` — sequence length seen by the model (default 64)
- ``--num-blocks`` — number of transformer blocks (default 6)
- ``--num-heads`` — attention heads per block (default 6)
- ``--embedding-dim`` — model width (default 288)
- ``--num-steps`` — training steps (default 2 048)
- ``--batch-size`` — mini-batch size (default 32)
- ``--temperature`` — sampling temperature for generation (default 1.0)

Sample output
-------------

A sample from the GPT-2 model before training:

.. code-block:: text

    AH,.u,I.znUsANK.MK.L.J'NCzWv'rAnsyMJD.en.HJtnM.eB3J?LsnTA.h.zN.MN.MWN.Fwz.MNdYOsL MaMN.TN.JCMMNqZiMJenfP.$wZNvhvN'MNMNM.,M.LKlYZ.Tcx.UhaVMNgZ$nTYM.zmsebEUTcMNMCchQNEsx.c WTA.sW?.M?YpMN SMwAQDN!Hb.MvUxM yqWMNq.M.Uq.Zc.Mw,MGV.mJkNDnUwMN.MUMcHLZ'x.ME,ebF!.MaM

And 45 minutes later, it speaks at least some kind of proto-English language! Lesson learned, LLMs should not be trained on CPU.

.. code-block:: text

    COMINIUS:
    Nine, I two it-ay because
    Till 'twere but equied big last!' Blanger, un withouts,
    Thy friend rude, or wantouch Romeo. Here have kneels: ounis, keep you
    To be even find and full of second worse: I ree say!

    DUCHESS OF YORK:
    Nay, be thing?

    STINGS:
    The assible tender of the breast,
    To fall the absence in this tower itself abiliory.
    Will you be blessing fancy,
    To give outward upon. Mark for your Naples;
    Upon you have not past a meritent woman:
    The prottle golden person, have good, hour some do break
    together, our love; and pray.'d you
    I mean not to be dvertender, use themselves:
    Ah, our spirits are not: accentible
    'Twise; and our enemyour as most say.

    LUCENTIO:
    Prithee, my life, he hath with you, and saved
    upon him. Here is receive against the head.
    I pray you'll saw 'Come hither come;
    Not mark the people I dower to ceaear
    The rashly of his audible, fair suns ever
    Their apt in thine eyes and weary not;
    My lord, is not only to too; compine. So, fare you
    And you are all to be friend and she will, let us giveli to friends?

    BENVOLIO:
    Alas is rememberance!

    ESTIAN:
    Ah, the thing that I trouble him, hear not with me.

    EDWARD:
    Call it in Rutland: we have not singled key,
    Yet I shall so, in my exe,
    He shall away on the wagow of yourselve?' do you
    Even o' the brother; but whereify hearese cadies
    The stewes of the fac officient counter.

    ROMEO:
    Is it a content our tongue.

    VINCENTIO:
    Ay, man, you should love your mind.

    BICA:
    Well, the ribunless lackers, convenient is awed upon,
    Even together, remembers his silvance and drink you!

    QUEEN ELIZABETH:
    Lords, her hath all of wealer inchange in post
    All fellow of these his own is fainted ofun,
    And she dismiss'd thy children, twi herable withcland.

    ROMEO:
    I dare thee more let than thou hats from the gits in art:
    Nay, let up, so, in the birdshires.

    Nurse:
    Romer, you may again Margaret,
    I have stood like to Choot, when may have it fought
    To like to spea

Architecture
------------

The model follows the standard GPT-2 design:

.. code-block:: none

    Tokens → Embedding + Positional Embedding
        → Dropout
        → N × TransformerBlock
            LayerNorm → CausalAttention → residual
            LayerNorm → MLP (GeLU) → residual
        → LayerNorm
        → Linear (vocabulary projection)

Code walkthrough
----------------

**Config**

All architectural hyperparameters live in a single dataclass::

    @dataclasses.dataclass
    class GPT2Config:
        context_size:  int   = 64
        num_blocks:    int   = 6
        num_heads:     int   = 6
        embedding_dim: int   = 288
        dropout:       float = 0.0
        vocab_size:    int | None = None   # set from dataset

**Causal self-attention**

Q, K, V are produced by a single fused projection then split::

    class CausalAttention(nn.Module):
        def __init__(self, config):
            self.in_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            # static upper-triangular causal mask (1 = masked)
            self.causal_mask = npg.triu(npg.ones((T, T)), k=1).view(1, 1, T, T)

        def forward(self, x):
            B, C, _ = x.shape
            q, k, v = self.in_proj(x).split(embed_dim, dim=2)
            # reshape: (B, C, H, head_dim) → (B, H, C, head_dim)
            q = q.view(B, C, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, C, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, C, num_heads, head_dim).transpose(1, 2)

            scores  = q @ k.transpose(-2, -1) / npg.sqrt(head_dim)
            scores  = scores.masked_fill(self.causal_mask[:, :, :C, :C], float("-inf"))
            weights = npg.softmax(scores, axis=-1)
            x = (weights @ v).transpose(1, 2).reshape(B, C, embed_dim)
            return self.out_proj(x)

**MLP block with GeLU**

::

    class MLP(nn.Module):
        def __init__(self, config, dilation=4):
            self.up_proj   = nn.Linear(embed_dim, dilation * embed_dim, bias=False)
            self.gelu      = nn.GELU()
            self.down_proj = nn.Linear(dilation * embed_dim, embed_dim, bias=False)

        def forward(self, x):
            return self.down_proj(self.gelu(self.up_proj(x)))

**Transformer block (pre-LayerNorm)**

::

    class Gpt2Block(nn.Module):
        def forward(self, x):
            x = x + self.attn(self.ln1(x))   # attention residual
            x = x + self.ffn(self.ln2(x))    # FFN residual
            return x

**Full model — token + position embeddings**

::

    class GPT2(nn.Module):
        def forward(self, x):           # x: (B, C) integer token ids
            positions = npg.arange(C)
            x = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
            x = self.blocks(x)
            return self.output_projection(self.output_ln(x))   # (B, C, vocab_size)

**Loss — 3D logits**

``cross_entropy_loss`` flattens ``(B, T, V)`` logits internally::

    logits = net(x)                               # (B, T, vocab_size)
    loss   = nn.cross_entropy_loss(logits, target)  # target: (B, T)
    loss.backward()

**Autoregressive sampling**

::

    @npg.no_grad()
    def sample(net, context, max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = net(context[:, -context_size:]) / temperature
            probs  = npg.softmax(logits[:, -1, :], axis=-1)
            idx    = np.random.choice(vocab_size, p=probs.numpy().squeeze())
            context = npg.cat((context, npg.array([[idx]])), axis=-1)
        return context

**Training loop**

::

    optimizer = npg.optim.AdamW(net.parameters(), lr=1e-3)
    for step, (x, target) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = net(x)
        loss   = nn.cross_entropy_loss(logits, target)
        loss.backward()
        optimizer.step()

New primitives used
-------------------

This example exercises several numpygrad features added alongside it:

- ``nn.LayerNorm`` — layer normalisation with learnable affine parameters
- ``nn.Embedding`` — token and position embedding lookup tables
- ``nn.Dropout`` — inverted dropout; a no-op after ``model.eval()``
- ``nn.GELU`` — GeLU activation module (tanh approximation)
- ``nn.Sequential`` — chains the transformer blocks
- ``nn.init`` — parameter initialisation helpers
- ``npg.triu`` — creates the static upper-triangular causal mask
- ``npg.split`` — splits the fused Q/K/V projection into three tensors
- ``masked_fill`` — applies the causal mask (broadcasts 2D mask over 4D scores)
- ``cross_entropy_loss`` — accepts ``(B, T, V)`` logits without manual reshape
