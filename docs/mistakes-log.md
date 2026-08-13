# Mistakes Log

A running record of real bugs hit while building this project, how they were
diagnosed, and how they were fixed. Kept intentionally — debugging your own
code is most of the actual work in ML engineering, and this is a more honest
record of that process than a repo that only ever shows working code.

---

## `data.py`

### 1. `load_mnist()` returned a `set` instead of a `tuple`

```python
# before
return {X_train, X_test, y_train, y_test}
```

Curly braces create a **set**, not a tuple. Two problems:
- Sets are unordered, so unpacking with `X_train, X_test, y_train, y_test = load_mnist()`
  would not be guaranteed to preserve order even if it worked.
- NumPy arrays are **unhashable**, so this actually crashes outright:
  `TypeError: unhashable type: 'numpy.ndarray'`.

**Fix:**
```python
return (X_train, X_test, y_train, y_test)
```

### 2. Incorrect return type hint on `load_mnist()`

```python
# before
def load_mnist() -> np.ndarray:
```

The function returns four arrays, not one.

**Fix:**
```python
def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

### 3. `fetch_openml` returned pandas objects, not NumPy arrays

Calling `fetch_openml('mnist_784')` without `as_frame=False` returns
`mnist.target` as a pandas `Series` with dtype `category`. Categorical data
has no inherent numeric ordering, so calling `.min()` / `.max()` on it later
raised:

```
TypeError: Categorical is not ordered for operation min
```

**Fix:** force plain NumPy arrays at the source:
```python
mnist = fetch_openml('mnist_784', as_frame=False)
```

### 4. `y.min` / `y.max` called without parentheses

```python
# before
assert y.min >= 0 and y.max < num_classes, "labels out of range"
```

Without `()`, this refers to the bound *method object* itself, not its
return value — comparing a method object to an int raises:
```
TypeError: '>=' not supported between instances of
'builtin_function_or_method' and 'int'
```

**Fix:**
```python
assert y.min() >= 0 and y.max() < num_classes, "labels out of range"
```

### 5. Design note — one-hot labels overwrote the integer labels

```python
# before
y_train = one_hot_encode(y_train)
```

This works for the shape/sum assertions, but discards the original integer
labels. They're needed later for accuracy computation and the confusion
matrix (both compare predicted vs. true *class indices*, not one-hot
vectors). Kept both instead:

```python
Y_train = one_hot_encode(y_train)  # one-hot, capital Y
# y_train (integer labels) is left untouched
```

---

## `model.py`

### 6. `forward()` originally returned only `A2`, discarding all cached intermediates

```python
# before
def forward_pass(X, params) -> np.ndarray:
    ...
    return A2
```

The backward pass needs `Z1`, `A1`, and `Z2` — not just the final output —
to compute `dW2` (needs `A1`), `delta1` (needs `Z1`, via `relu_derivative`),
and so on (see docs/derivation.md). Returning only `A2` would force
recomputing the entire forward pass inside `backward()`, defeating the
purpose of caching described in the algorithm outline.

**Fix:** return a dict of every intermediate:
```python
return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```

### 7. Invalid type hint: `dict[np.ndarray]`

```python
# before
def forward_pass(X: np.ndarray, params: dict[str, np.ndarray]) -> dict[np.ndarray]:
```

A `dict` type hint needs two type parameters — key type and value type.
`dict[np.ndarray]` only supplies one, which raises a `TypeError` at
runtime when Python evaluates the annotation (dict generics are not
lazily evaluated here).

**Fix:**
```python
-> dict[str, np.ndarray]
```

### 8. Function/module naming mismatch with the rest of the project

`forward_pass` / `init_parameters` were used locally, but the test suite
(`tests/test_model.py`) and other modules (`train.py`, `scripts/`) import
`forward` and `init_params` specifically. Renamed to match, since a name
mismatch causes an `ImportError`, not a logic bug — easy to miss since the
function itself runs fine standalone.

### 9. Activation functions originally nested inside `forward_pass`

```python
# before
def forward_pass(X, params):
    def relu(z): ...
    def softmax(z): ...
    ...
```

This works, but `relu_derivative` (needed independently in the backward
pass, Ex 7) has nowhere sensible to live if `relu` itself is nested and
inaccessible elsewhere. Moved `relu`, `relu_derivative`, and `softmax` to
module level, matching the rest of the codebase's structure.

### 10. Missing 1/m batch averaging in the backward pass

```python
# before
delta2 = cache["A2"] - y
```

`m` was computed (`m = X.shape[0]`) but never used anywhere. Since every
downstream gradient (`dW2`, `db2`, `dA1`, `delta1`, `dW1`, `db1`) is linear
in `delta2`, this meant every returned gradient was `m` times larger than
it should be for a batch-averaged loss — a silent scaling bug that passes
every shape assertion (since shape doesn't depend on scale) and would only
surface as unstable/diverging training, far downstream of the actual bug.

**Fix:** divide once, at the source, and let it propagate:
```python
delta2 = (cache["A2"] - y) / m
```

### 11. `dA1` used the wrong layer's weights

```python
# before
dA1 = delta2 @ params["W1"].T
```

`dA1` (gradient flowing back into layer 1's *output*) must flow through
the weights connecting layer 1 to layer 2 — i.e. `W2` — not `W1`. `W1.T`
has shape `(128, 784)`, which isn't even multiplicable against `delta2`
`(m, 10)`, so this would have raised a shape-mismatch error immediately
on the first run with real biases (it happened to not crash only because
an earlier draft coincidentally hadn't been run against real shapes yet).

**Fix:**
```python
dA1 = delta2 @ params["W2"].T   # (m,10) @ (10,128) -> (m,128), matches A1
```

### 12. `db2`/`db1` summed the wrong array and wrong axis

```python
# before
db2 = np.sum(params["b2"], axis=1, keepdims=True)
```

Two separate mistakes stacked: summing `params["b2"]` (the bias itself)
instead of `delta2` (the error signal), and summing `axis=1` (across the
10 output neurons) instead of `axis=0` (across the batch). `db` must be
the column-sum of `delta` — collapsing the *batch* axis, leaving one value
per neuron.

**Fix:**
```python
db2 = np.sum(delta2, axis=0, keepdims=True)
```

### 13. `relu_derivative` silently upcast gradients to float64

```python
# before
def relu_derivative(z):
    return np.where(z > 0, 1, 0)
```

`np.where(cond, 1, 0)` uses plain Python ints, which NumPy defaults to an
integer dtype (not float32). Multiplying `dA1 (float32) * relu_derivative(Z1) (int)`
triggers NumPy's type-promotion rules, upcasting the result to float64.
Every gradient built from that point on (`delta1`, `dW1`, `db1`) silently
inherited float64, while `dW2`/`db2` (which never touch `relu_derivative`)
stayed float32 — a real, easy-to-miss dtype inconsistency across the
gradient dict, only caught by explicitly printing `.dtype` on every array.

**Fix:** derive the dtype from the input instead of hardcoding one:
```python
def relu_derivative(z):
    return (z > 0).astype(z.dtype)
```

The final `__main__` block in `model.py` now asserts every array in
`params`, `cache`, and `grads` is float32, specifically to catch this
class of bug automatically going forward.

---

## `run_training.py`

### 14. Incorrect preprocessing loop — reassigning a loop variable does nothing

```python
# before
for arr in {X_train, X_test}:
    arr = preprocessing_images(arr)
```

Two bugs stacked: `{}` creates a set (unhashable for numpy arrays, crashes
immediately), and even if it didn't, reassigning `arr` inside the loop only
rebinds the local name — `X_train` and `X_test` in the outer scope are
unchanged. NumPy arrays are mutable objects but assignment is not mutation.

**Fix:** assign back to each variable explicitly:
```python
X_train = preprocessing_images(X_train)
X_test  = preprocessing_images(X_test)
```

### 15. Wrong batch-count calculation

```python
# before
number_of_samples // batch_size + 1
```

Fails when the dataset divides evenly — `128 // 16 + 1 = 9` batches when
there are exactly 8. The correct formula is `ceil(n / batch_size)`, which
handles both exact and non-exact division:

```python
import math
num_batches = math.ceil(number_of_samples / batch_size)
```

### 16. Applying ceil after floor division discards the remainder

```python
# also considered
math.ceil(number_of_samples // batch_size)
```

Floor division already throws away the remainder before ceil sees it:
`math.ceil(100 // 16) = math.ceil(6) = 6`, but `math.ceil(100 / 16) = 7`.
The fix is to use true division (`/`) inside ceil, not floor division (`//`).

### 17. Thought the last (partial) batch needed special-casing

Initially believed slicing `X[start:end]` would raise an out-of-bounds
error when `end > len(X)`. Python slicing safely stops at the array boundary
— `X[96:112]` on a 100-element array returns `X[96:100]` without error. The
general formula `start = batch_size * j; end = batch_size * (j + 1)` works
for every batch including the last partial one, with no special case needed.

### 18. Used `{}` to store a batch pair

```python
# before
batches.append({X_batch, y_batch})
```

Curly braces create a set. NumPy arrays are unhashable, so this crashes with
`TypeError`. A tuple is the right structure for pairing two arrays:

```python
batches.append((X_batch, y_batch))
```

### 19. Major parameter-update bug — discarding updates between batches

```python
# before
param, loss = train_step(X, Y, params, learning_rate)
# then continued using the old `params` for the next batch
```

`train_step` returns `updated_params` as a new dict — it does not mutate
`params` in place. Assigning to a different name (`param` vs `params`) meant
every batch trained against the original initialization. The loss appeared to
decrease (the batch being trained on was fitting), but the global `params`
never moved.

**Fix:** assign back to the same name so each batch inherits the previous
batch's updates:
```python
params, loss = train_step(X, Y, params, learning_rate)
```

The correct data flow is:
```
params_0 -> batch 1 -> params_1
params_1 -> batch 2 -> params_2
params_2 -> batch 3 -> params_3
```

### 20. Shuffling once before all epochs instead of once per epoch

Shuffling once means every epoch sees batches in the same fixed order. The
model can learn to exploit the ordering rather than the actual signal —
certain classes always appear together in early batches, others always in
late batches. Shuffling at the start of each epoch breaks this:

```
epoch
  -> shuffle
  -> batch and train
```

not:
```
shuffle once
  -> train for all epochs
```

### 21. Risk of breaking X-y pairing when shuffling

X and y cannot be shuffled independently — each image must stay paired with
its correct label. The fix is to generate one permutation index array and
apply it to both:

```python
p = np.random.permutation(len(X))
X_shuffled, Y_shuffled = X[p], Y[p]
```

### 22. Variable naming: `epoch = 30` when the intended meaning is the count

```python
# before
epoch = 30
for i in range(epoch):
```

Using `epoch` for the count means the loop variable can't also be called
`epoch` without shadowing the outer name. The cleaner convention:

```python
epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
```

### 23. Storing all batches before training instead of generating on the fly

```python
batches = []
for j in range(num_batches):
    batches.append((X[...], Y[...]))
for X_batch, Y_batch in batches:
    ...
```

This works for MNIST (fits in memory) but doubles the memory footprint
unnecessarily. The scalable pattern is to generate each batch, train, and
discard it immediately — no list needed:

```python
for j in range(num_batches):
    X_batch = X[batch_size * j : batch_size * (j + 1)]
    Y_batch = Y[batch_size * j : batch_size * (j + 1)]
    params, loss = train_step(X_batch, Y_batch, params, lr)
```

### 24. Averaging batch losses without weighting for batch size

```python
average_epoch_loss = np.average(losses)
```

If the final batch is smaller than `batch_size`, it contributes the same
weight to the average as a full batch. A sample-weighted average is
conceptually more correct:

```python
# weighted by actual batch size
epoch_loss = sum(l * s for l, s in zip(losses, sizes)) / total_samples
```

For MNIST with batch_size=16 the difference is tiny (one batch of ≤16 vs
60,000 samples), but the distinction matters for smaller datasets or large
batch sizes.

### 25. Using the test set to choose hyperparameters

```
batch size 64 -> 96.33%
batch size 16 -> 97.33%
```

Comparing test accuracy across hyperparameter settings means the test set
influenced the choice of batch size — it is no longer a held-out measure of
generalization. The correct setup:

```
train set       -> train model
validation set  -> choose hyperparameters
test set        -> final evaluation only, once
```

### 26. Not controlling randomness when comparing experiments

Comparing two runs with different batch sizes also mixes in differences from
random initialization and random shuffling. For a fair comparison, fix the
seed before both runs:

```python
np.random.seed(42)
```

### 27. Thinking argmax should be part of the forward pass

The forward pass must return full softmax probabilities, not a class index,
because backprop needs the full distribution to compute `delta2 = A2 - Y`.
Argmax is only applied at prediction time, after training:

```
training:   forward -> softmax probabilities -> loss -> backprop
prediction: forward -> softmax probabilities -> argmax -> class
```

### 28. Confusing cross-entropy loss with percentage error

A loss of `0.125` does not mean `12.5%` error. Cross-entropy is an
information-theoretic quantity (nats or bits), not a fraction of
misclassified examples. Accuracy is a separate metric computed from
predicted class indices, not from the loss value.

### 29. One-hot encoding test labels unnecessarily

`y_test` was one-hot encoded and then converted back via `argmax` for
accuracy. One-hot encoding is needed for the loss function during training —
the test labels only need to be integer class indices for accuracy
computation. Keeping `y_test` as integers avoids the round-trip.

### 30. No validation set

Currently using train/test split only. Because hyperparameters (batch size,
learning rate, epochs, architecture) were tuned by looking at test accuracy,
the test set is no longer a clean measure of generalization. A proper setup:

```
60,000 train  ->  50,000 train + 10,000 validation (for tuning)
10,000 test   ->  untouched until final evaluation
```
