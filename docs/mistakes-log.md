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

---

*(This file will grow as later exercises — backprop, training loop,
evaluation — surface their own bugs.)*
