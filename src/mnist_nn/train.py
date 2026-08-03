import numpy as np
from mnist_nn.model import init_parameters, forward_pass, backward_pass
from mnist_nn.losses import cross_entropy_loss

def train_step(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray], learning_rate: np.float32) -> tuple[dict[str, np.ndarray], np.float32]:
    
    # forward pass
    cache = forward_pass(X, params)
    
    # Cross entropy loss after the forward pass to check how much loss has decreased after each epoch 
    loss = cross_entropy_loss(y, cache["A2"])
    
    # backward pass to get the derivative for the sgd update 
    derivatives = backward_pass(X, y, params, cache)
    
    # sgd update on W1, b1, W2, b2

    # summarized 4 lines of code below into one for loop:
        # params["W1"] -= (derivatives["dW1"] * learning_rate)
        # params["b1"] -= (derivatives["db1"] * learning_rate)
        # params["W2"] -= (derivatives["dW2"] * learning_rate)
        # params["b2"] -= (derivatives["db2"] * learning_rate)
    
    updated_params = {}
    
    for name,arr in params.items():
        updated_params[name] = arr - (derivatives[f"d{name}"] * learning_rate)
    
    return updated_params, loss


if __name__ == "__main__":
    params = init_parameters()
    X = np.random.randn(16, 784).astype(np.float32)
    Y = np.eye(10, dtype=np.float32)[np.random.randint(0, 10, size=16)]

    losses = []
    for _ in range(30):
        params, loss = train_step(X, Y, params, learning_rate=0.5)
        losses.append(loss)

    print(f"loss[0]  = {losses[0]:.4f}")
    print(f"loss[-1] = {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"
    print("All checks passed.")