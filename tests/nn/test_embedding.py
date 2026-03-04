import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import numpygrad as npg
from numpygrad.nn.embedding import Embedding
from tests.configuration import FLOAT_DISTRIBUTION, check_equality

MIN_VOCAB = 2
MAX_VOCAB = 20
MIN_DIM = 1
MAX_DIM = 16
MIN_SEQ = 1
MAX_SEQ = 10


@st.composite
def configuration(draw):
    num_embeddings = draw(st.integers(min_value=MIN_VOCAB, max_value=MAX_VOCAB))
    embedding_dim = draw(st.integers(min_value=MIN_DIM, max_value=MAX_DIM))
    seq_len = draw(st.integers(min_value=MIN_SEQ, max_value=MAX_SEQ))
    indices = np.random.randint(0, num_embeddings, size=(seq_len,))
    return num_embeddings, embedding_dim, indices


@given(configuration())
def test_embedding_forward(config):
    num_embeddings, embedding_dim, indices = config
    weight = FLOAT_DISTRIBUTION((num_embeddings, embedding_dim)).astype(np.float64)

    emb = Embedding(num_embeddings, embedding_dim)
    emb.weight.data = weight

    torch_emb = torch.nn.Embedding(num_embeddings, embedding_dim)
    torch_emb.weight.data = torch.from_numpy(weight)

    y = emb(npg.array(indices))
    yt = torch_emb(torch.from_numpy(indices).long())

    check_equality(y.data, yt.detach().numpy())


@settings(deadline=None)
@given(configuration())
def test_embedding_backward(config):
    num_embeddings, embedding_dim, indices = config
    weight = FLOAT_DISTRIBUTION((num_embeddings, embedding_dim)).astype(np.float64)

    emb = Embedding(num_embeddings, embedding_dim)
    emb.weight.data = weight

    torch_emb = torch.nn.Embedding(num_embeddings, embedding_dim)
    torch_emb.weight.data = torch.from_numpy(weight)

    y = emb(npg.array(indices))
    y.backward()

    yt = torch_emb(torch.from_numpy(indices).long())
    yt.sum().backward()

    assert emb.weight.grad is not None
    check_equality(emb.weight.grad, torch_emb.weight.grad.numpy())
