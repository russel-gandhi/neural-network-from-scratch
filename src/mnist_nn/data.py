from sklearn.datasets import fetch_openml
import numpy as np 

NUM_CLASSES = 10
MAX_PIXEL = 255.0

def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    mnist = fetch_openml('mnist_784', as_frame=False)
    X,y = mnist.data, mnist.target
    y = y.astype(int)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return (X_train,X_test,y_train,y_test)
    
def preprocessing_images(X: np.ndarray) -> np.ndarray:
    
    assert X.ndim == 2, "Input array must have 2 dimensions"

    return X.astype(np.float32) / MAX_PIXEL

def one_hot_encode(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
       
    assert y.min() >= 0 and y.max() < num_classes, "labels out of range"
    
    return np.eye(num_classes, dtype=np.float32)[y]

if __name__ == "__main__":
    X_train,X_test,y_train,y_test = load_mnist()
    
    X_train = preprocessing_images(X_train)
    X_test = preprocessing_images(X_test)

    Y_train = one_hot_encode(y_train)
    Y_test = one_hot_encode(y_test)
    
    print(X_train.shape)
    print(Y_train.shape)
    
    assert X_train.shape == (60000, 784)
    assert Y_train.shape == (60000, 10)
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert Y_train[0].sum() == 1

    print("All checks passed.")

    
    