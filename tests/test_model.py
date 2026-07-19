# Unit tests for model.py.

# Each test corresponds to a "done when" check from the exercise roadmap.
# Run with: pytest tests/


import numpy as np
import pytest
from mnist_nn.model import init_parameters, forward_pass


def test_init_params_shapes():
    params = init_parameters()
    for name, arr in params.items():
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")

    assert params["W1"].shape == (784, 128)
    assert params["b1"].shape == (1, 128)
    assert params["W2"].shape == (128, 10)
    assert params["b2"].shape == (1, 10)

    # sanity: biases should be exactly zero
    assert np.all(params["b1"] == 0)
    assert np.all(params["b2"] == 0)

    # sanity: weights should NOT be zero (symmetry problem check)
    assert not np.all(params["W1"] == 0)
    assert not np.all(params["W2"] == 0)


def test_forward_pass_output_shape():
    params = init_parameters()
    X = np.random.randn(32, 784).astype(np.float32)
    cache = forward_pass(X, params)
    assert cache["A2"].shape == (32, 10)
    assert np.allclose(cache["A2"].sum(axis=1), 1.0, atol=1e-5)


# def test_backward_gradient_shapes():
#     params = init_parameters()
#     X = np.random.randn(32, 784).astype(np.float32)
#     Y = np.eye(10, dtype=np.float32)[np.random.randint(0, 10, size=32)]
#     cache = forward_pass(X, params)
#     grads = backward(X, Y, params, cache)
#     assert grads["dW1"].shape == params["W1"].shape
#     assert grads["db1"].shape == params["b1"].shape
#     assert grads["dW2"].shape == params["W2"].shape
#     assert grads["db2"].shape == params["b2"].shape