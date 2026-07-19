import numpy as np

def relu(z):
    return np.maximum(0,z)

def softmax(z, axis = 1):
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def init_parameters(input_size = 784, hidden_size = 128, output_size = 10) -> dict[str, np.ndarray]:
    
    #   Initialize weights and biases for a 2-layer network. W1 uses He initialization (scaled for ReLU); W2 uses a smaller-scale random init suited to a softmax output layer. Biases start at zero. Right now i will be using HE init as that number is also very small
    
    W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size), dtype=np.float32)

    W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size), dtype=np.float32)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_pass(X : np.ndarray, params : dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    
    # Full forward pass. Caches every intermediate for use in backward().
    
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = softmax(Z2)
    
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        
    
if __name__ == "__main__":
    params = init_parameters()
    for name, arr in params.items():
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
        print()
        print(arr)
        print()

    assert params["W1"].shape == (784, 128)
    assert params["b1"].shape == (1, 128)
    assert params["W2"].shape == (128, 10)
    assert params["b2"].shape == (1, 10)

    # sanity check: biases should be exactly zero
    assert np.all(params["b1"] == 0)
    assert np.all(params["b2"] == 0)

    # sanity check: weights should NOT be zero (symmetry problem check) so that we dont feed "W = 0" through the nn
    assert not np.all(params["W1"] == 0)
    assert not np.all(params["W2"] == 0)
    
    X = np.random.randn(32, 784).astype(np.float32)
    cache = forward_pass(X, params)
    
    for name, arr in cache.items():
        print(f"{name} = {arr}")
        print()
    
    assert cache["A2"].shape == (32, 10)
    assert np.allclose(cache["A2"].sum(axis=1), 1.0, atol=1e-5)

    print("All checks passed.")

