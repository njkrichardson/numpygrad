import numpy as np
import pytest

import numpygrad as npg
import numpygrad.nn.init as init


def _tensor(shape):
    return npg.array(np.zeros(shape), requires_grad=True)


def test_uniform_range():
    t = _tensor((1000,))
    init.uniform_(t, low=-2.0, high=3.0)
    assert np.all(t.data >= -2.0) and np.all(t.data <= 3.0)


def test_normal_stats():
    t = _tensor((10000,))
    init.normal_(t, mean=1.0, std=2.0)
    assert abs(t.data.mean() - 1.0) < 0.1
    assert abs(t.data.std() - 2.0) < 0.1


def test_zeros_():
    t = _tensor((5, 5))
    init.zeros_(t)
    np.testing.assert_array_equal(t.data, 0.0)


def test_ones_():
    t = _tensor((5, 5))
    init.ones_(t)
    np.testing.assert_array_equal(t.data, 1.0)


def test_kaiming_uniform_range():
    fan_in = 64
    t = _tensor((fan_in, 128))
    init.kaiming_uniform_(t, mode="fan_in", nonlinearity="relu")
    gain = np.sqrt(2.0)
    bound = np.sqrt(3.0) * gain / np.sqrt(fan_in)
    assert np.all(t.data >= -bound) and np.all(t.data <= bound)


def test_kaiming_uniform_variance():
    # Var of Uniform(-b, b) = b^2/3; gain^2/(3*fan) for relu
    fan_in = 512
    t = _tensor((fan_in, 512))
    init.kaiming_uniform_(t, mode="fan_in", nonlinearity="relu")
    gain = np.sqrt(2.0)
    expected_var = gain**2 / (3.0 * fan_in)
    assert abs(t.data.var() - expected_var) < 0.05


def test_kaiming_normal_std():
    fan_in = 256
    t = _tensor((fan_in, 256))
    init.kaiming_normal_(t, mode="fan_in", nonlinearity="relu")
    gain = np.sqrt(2.0)
    expected_std = gain / np.sqrt(fan_in)
    assert abs(t.data.std() - expected_std) < 0.05


def test_kaiming_fan_out():
    fan_out = 128
    t = _tensor((64, fan_out))
    init.kaiming_uniform_(t, mode="fan_out", nonlinearity="relu")
    gain = np.sqrt(2.0)
    bound = np.sqrt(3.0) * gain / np.sqrt(fan_out)
    assert np.all(t.data >= -bound) and np.all(t.data <= bound)


def test_xavier_uniform_range():
    fan_in, fan_out = 64, 128
    t = _tensor((fan_in, fan_out))
    init.xavier_uniform_(t)
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    assert np.all(t.data >= -bound) and np.all(t.data <= bound)


def test_xavier_normal_std():
    fan_in, fan_out = 128, 256
    t = _tensor((fan_in, fan_out))
    init.xavier_normal_(t)
    expected_std = np.sqrt(2.0 / (fan_in + fan_out))
    assert abs(t.data.std() - expected_std) < 0.02


def test_inplace_returns_same_object():
    t = _tensor((4, 4))
    for fn in [
        lambda x: init.uniform_(x),
        lambda x: init.normal_(x),
        lambda x: init.zeros_(x),
        lambda x: init.ones_(x),
        lambda x: init.kaiming_uniform_(x),
        lambda x: init.kaiming_normal_(x),
        lambda x: init.xavier_uniform_(x),
        lambda x: init.xavier_normal_(x),
    ]:
        result = fn(t)
        assert result is t


def test_4d_tensor_fan():
    # Conv2d: (out_channels, in_channels, kH, kW)
    out_c, in_c, kH, kW = 16, 8, 3, 3
    t = _tensor((out_c, in_c, kH, kW))
    receptive = kH * kW
    expected_fan_in = in_c * receptive
    expected_fan_out = out_c * receptive
    fan_in, fan_out = init._calculate_fan(t)
    assert fan_in == expected_fan_in
    assert fan_out == expected_fan_out


def test_invalid_ndim_raises():
    t = _tensor((4, 4, 4))  # 3D
    with pytest.raises(ValueError, match="fan requires 2D or 4D"):
        init._calculate_fan(t)
