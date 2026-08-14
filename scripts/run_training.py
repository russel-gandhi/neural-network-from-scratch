import numpy as np
from mnist_nn.data import load_mnist, preprocessing_images, one_hot_encode
from mnist_nn.model import init_parameters, forward_pass
from mnist_nn.train import train_step

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Load the mnist dataset into a train-test split
X_train,X_test,y_train,y_test = load_mnist()

# preprocess the images - normalize the pixels and flatten the images 
X_train = preprocessing_images(X_train)

X_test = preprocessing_images(X_test)

print("Size of X_train", X_train.shape)
print("Size of X_test", X_test.shape)

# OHE the labels 
y_train = one_hot_encode(y_train)

y_test = one_hot_encode(y_test)

print("Size of y_train", y_train.shape)
print("Size of y_test", y_test.shape)

# Initialize the parameters for the first time 
params = init_parameters()


# Initialize the training parameters 
epoch = 30
learning_rate = 0.01
batch_size = 16

for i in range(epoch):
    # Data Loading for each epoch 

    shuffled_X,shuffled_Y = unison_shuffled_copies(X_train,y_train)

    number_of_samples = X_train.shape[0]

    batches = []
    losses = []

    for j in range(int(np.ceil(number_of_samples / batch_size))):
        batches.append((shuffled_X[batch_size * j : batch_size * (j + 1)], shuffled_Y[batch_size * j : batch_size * (j + 1)]))
    
    for X, Y in batches:
        params , loss = train_step(X,Y, params, learning_rate)
        losses.append(loss)
        
    average_epoch_loss = np.average(losses)
    
    print(f"Average loss of epoch {i + 1} =", average_epoch_loss)
    
def predict(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    
    cache = forward_pass(X, params)
    probs = cache["A2"]
    predictions = np.argmax(probs, axis=1)
    
    return predictions

preds = predict(X_test, params) 

y_test_class_indices = np.argmax(y_test,axis=1)

correct = preds == y_test_class_indices

accuracy = correct.sum() / len(correct)

print("Accuracy:", accuracy *100, "%")


import matplotlib.pyplot as plt 

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    image = X_test[i].reshape(28, 28)

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"P: {preds[i]} | A: {y_test_class_indices[i]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()


wrong_indices = np.where(preds != y_test_class_indices)[0]

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, index in zip(axes.flat, wrong_indices[:10]):
    image = X_test[index].reshape(28, 28)

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"P: {preds[index]} | A: {y_test_class_indices[index]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()