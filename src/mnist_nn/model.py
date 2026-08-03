import numpy as np

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    # Returns 1 for elements > 0, and 0 otherwise
    return (z > 0).astype(z.dtype)

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

def backward_pass(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray], cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    m = X.shape[0]
    
    delta2 = (cache["A2"] - y) / m
    
    dW2 = cache["A1"].T @ delta2
    db2 = np.sum(delta2, axis= 0, keepdims=True)
    
    dA1 = delta2 @ params["W2"].T
    delta1 = dA1 * relu_derivative(cache["Z1"])
    
    dW1 = X.T @ delta1
    db1 = np.sum(delta1, axis= 0, keepdims=True)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
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
    Y = np.eye(10, dtype=np.float32)[np.random.randint(0, 10, size=32)]
    cache = forward_pass(X, params)
    
    for name, arr in cache.items():
        print()
        print(f"{name} = {arr}")
        print()
    
    assert cache["A2"].shape == (32, 10)
    assert np.allclose(cache["A2"].sum(axis=1), 1.0, atol=1e-5)
    
    grads = backward_pass(X, Y, params, cache)
    
    for name, arr in grads.items():
        print()
        print(f"{name} = {arr}")
        print()
    
    for name, arr in params.items():
        print(f"{name} : {arr.dtype}")
        
    for name, arr in cache.items():
        print(f"{name} : {arr.dtype}")
    
    for name, arr in grads.items():
        print(f"{name} : {arr.dtype}")
    
    assert grads["dW1"].shape == params["W1"].shape
    assert grads["db1"].shape == params["b1"].shape
    assert grads["dW2"].shape == params["W2"].shape
    assert grads["db2"].shape == params["b2"].shape


    print("All checks passed.")

