import numpy as np

def cross_entropy_loss(y: np.ndarray, A_L: np.ndarray) -> np.float32:
    eps = np.finfo(A_L.dtype).eps
    A_L_clipped = np.clip(A_L, eps, 1.0 - eps)
    loss_sample = np.sum(-(y * np.log(A_L_clipped)), axis=-1, keepdims=True)
    return (np.sum(loss_sample)) / np.float32(A_L_clipped.shape[0])
    
    
    
if __name__ == "__main__":

    from model import init_parameters, forward_pass
    
    params = init_parameters()
    X = np.random.randn(32, 784).astype(np.float32)
    Y = np.eye(10, dtype=np.float32)[np.random.randint(0, 10, size=32)]
    cache = forward_pass(X, params)
    
    print(f"Y = {Y}, dtype = {Y.dtype}")
    
    for name, arr in cache.items():
        if name == "A2":
            print()
            print(f"{name} = {arr}")
            print()
            
            
    CEL = cross_entropy_loss(Y,cache["A2"])
    print(f"cross entropy loss: {CEL}; dtype = {CEL.dtype}")
        

    