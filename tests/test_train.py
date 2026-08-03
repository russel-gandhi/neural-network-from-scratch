import numpy as np 

from mnist_nn.model import init_parameters
from mnist_nn.train import train_step

def test_last_step_loss_is_less_than_0th_step():
    params = init_parameters()
    X = np.random.randn(16,784).astype(np.float32)
    Y = np.eye(10, dtype=np.float32)[np.random.randint(0,10, size=16)]
    
    losses = []
    
    for _ in range(30):
        params, loss = train_step(X, Y, params, learning_rate=0.5)
        losses.append(loss)

    assert losses[-1] < losses[0], "loss did not decrease"
    
    
def test_if_loss_output_is_scalar_and_loss_is_less_than_zero():
    params = init_parameters()
    X = np.random.randn(16,784).astype(np.float32)
    Y = np.eye(10, dtype=np.float32)[np.random.randint(0,10, size=16)]
    
    losses = []
    
    for _ in range(30):
        params, loss = train_step(X, Y, params, learning_rate=0.5)
        losses.append(loss)
    
    assert all(np.isscalar(loss) for loss in losses), "The loss is not a scalar"
    assert all(loss > 0 for loss in losses), "loss < 0"

def test_if_loss_output_is_float32():
    params = init_parameters()
    X = np.random.randn(16,784).astype(np.float32)
    Y = np.eye(10, dtype=np.float32)[np.random.randint(0,10, size=16)]
    
    losses = []
    
    for _ in range(30):
        params, loss = train_step(X, Y, params, learning_rate=0.5)
        losses.append(loss)
    
    assert all(np.asarray(loss).dtype == np.float32), "Dtype of loss output is not float32"
    
    


    

