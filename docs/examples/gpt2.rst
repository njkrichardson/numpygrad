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

    Farthink's birry-day, down nor portuou.
    What cames me farewells like, to a times on game.

    MENRY BOPHELLAND:
    Now, good matters herein;
    Ere I of thee huse's now from olding be,
    Be shall and not live directed to half a garment-speed in
    A pervecred temple: and power for'd sweetms
    on that Warwick he is cusey.

    BENVOLIO:
    Supplish a stindrer age, gentle Horn; for I
    Beggary thoughts, parting unjury'st comes; he endure
    To these obedy the glound of such leisurarly
    Eschadienle! Pomfortunes
    And yet sorrow to delass Iitch in Friendsded,
    Twhen I first his cause idle is house
    cardon is cares
    Thine shoronge a coldler corning good.

    NORTHUMBERLAND:
    I sear our too other: darrel, have being preRtify
    earth dissure flesh. My follows! Olden heavy face
    Music and light moy owers of bring appleased.

    ROMEOSS:
    Out, and yet give so sighout.

    BLUHEMOOP:
    The bear free of almoster ire, to with strangewn remembaw
    and infectsdoms famely argood! Who less her one thou?

    ANGELO:
    And 'Clown with says such as again a marriage.

    PARIS:
    You prais

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