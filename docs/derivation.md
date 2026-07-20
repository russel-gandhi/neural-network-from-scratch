# Derivation

Full derivation of the forward pass, loss, and backward pass used in
`src/mnist_nn/model.py` and `losses.py`. Row-convention throughout: each
row of a batch matrix is one training example.

## Forward pass

```
Z1 = X  @ W1 + b1      # (m,784)@(784,128) -> (m,128)
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2      # (m,128)@(128,10)  -> (m,10)
A2 = softmax(Z2)       # row-wise
```

Implemented in `model.py::forward()`. Every intermediate (`Z1`, `A1`, `Z2`,
`A2`) is cached and returned, not just `A2` — the backward pass needs all
of them (see below), and recomputing them there would defeat the point of
caching described in the general algorithm.

### Weight initialization

`W1` uses He initialization, `W1 ~ N(0, 2/n_in)`, i.e.
`np.random.randn(n_in, n_out) * sqrt(2 / n_in)`. This specifically
compensates for ReLU zeroing out roughly half its inputs — without the
factor of 2, variance shrinks layer by layer through a deep ReLU network.
Xavier/Glorot init (`1/n_in`, no factor of 2) is the standard choice for
sigmoid/tanh instead, since those activations are roughly linear near 0
and don't discard half their input variance the way ReLU does.

Biases are initialized to zero. Weights are not, to avoid the *symmetry
problem*: if every weight starts identical (e.g. all zero), every neuron
in a layer computes the same output and receives the same gradient
forever, so the layer never differentiates its neurons regardless of how
long training runs.

### Numerical stability in softmax

```
softmax(z)_k = exp(z_k - max(z)) / sum_k' exp(z_k' - max(z))
```

Subtracting the row max before exponentiating does not change the
mathematical result — algebraically, `exp(z_k - c) / sum exp(z_k' - c) =
e^{-c} exp(z_k) / (e^{-c} sum exp(z_k'))`, and the `e^{-c}` factor cancels
top and bottom for any constant `c`. Choosing `c = max(z)` keeps the
largest exponent at exactly 0 and every other exponent `<= 0`, preventing
`np.exp` overflow that raw `exp(z)` could hit for large logits.

## Loss: cross-entropy as a special case of KL divergence

Let `p` be the true label distribution (one-hot) and `q = A2` the predicted
distribution. KL divergence:

```
D_KL(p || q) = sum_k p_k log(p_k / q_k)
             = sum_k p_k log(p_k) - sum_k p_k log(q_k)
             = -H(p)                - H(p, q)
```

`-H(p)` doesn't depend on model parameters (p is fixed data), so it
contributes zero gradient. Minimizing `D_KL(p || q)` over parameters is
therefore identical to minimizing the cross-entropy `H(p, q) = -sum_k p_k
log(q_k)` — this is why cross-entropy loss and KL loss are used
interchangeably in classification.

## The key result: delta at the output layer

Differentiating the loss directly w.r.t. `Z2` (not `A2`, since softmax
isn't elementwise) gives, remarkably:

```
delta2 = A2 - Y
```

Predicted probabilities minus true one-hot labels, elementwise. The
softmax Jacobian and the cross-entropy gradient cancel algebraically to
produce this simple form — see the full symbolic derivation in the project
history / chat log this repo was built from.

## Backward pass, general layer `l`

Given `delta[l] = dLoss/dZ[l]`:

```
dW[l] = A[l-1].T @ delta[l]           # weight gradient
db[l] = column_sum(delta[l])          # bias gradient
dA[l-1] = delta[l] @ W[l].T           # error handed to previous layer
delta[l-1] = dA[l-1] * g'(Z[l-1])     # elementwise; convert activation
                                       # gradient to pre-activation gradient
```

### Implementation, this project's 2-layer case

```
delta2 = (A2 - Y) / m      # 1/m applied once, here; every downstream
                            # gradient inherits it since they're all
                            # linear in delta2 or delta1

dW2 = A1.T @ delta2         # (128,m) @ (m,10)  -> (128,10), matches W2
db2 = column_sum(delta2)    # (1,10), matches b2

dA1 = delta2 @ W2.T         # (m,10) @ (10,128) -> (m,128), matches A1
delta1 = dA1 * relu'(Z1)    # elementwise

dW1 = X.T @ delta1          # (784,m) @ (m,128) -> (784,128), matches W1
db1 = column_sum(delta1)    # (1,128), matches b1
```

Implemented in `model.py::backward()`. Two implementation details worth
being explicit about, since both caused real bugs during development (see
docs/mistakes-log.md, bugs #10-13):

- **The 1/m factor.** Applying it once, at `delta2`, is enough — `dW2`,
  `db2`, `dA1`, `delta1`, `dW1`, `db1` are all linear functions of `delta2`
  (or of `delta1`, itself linear in `delta2`), so the averaging propagates
  automatically. Forgetting this doesn't break any shape check — it's a
  pure scaling bug that only shows up as unstable/diverging training.
- **`relu_derivative`'s dtype.** Must return an array matching its input's
  dtype (`(z > 0).astype(z.dtype)`), not rely on Python's default int
  literals (`np.where(z > 0, 1, 0)`), which silently upcasts every
  downstream float32 gradient to float64 via NumPy's type-promotion rules.

### Why `dW` needs `A[l-1]` but `db` does not

A weight `W[l]_pq` multiplies the incoming activation `A[l-1]_p` in the
forward pass, so its gradient is weighted by how strong that input signal
was: `sum_i delta_iq * A[l-1]_ip`. A bias is only ever *added*, never
multiplied by anything from the previous layer, so its gradient is just
the raw error signal summed over the batch: `sum_i delta_iq`.

### Why the sum over examples

Both `W[l]_pq` and `b[l]_q` are single shared parameters used identically
across all `m` examples in a batch. The multivariable chain rule says: when
one variable affects an output through multiple independent paths (here,
one path per example), sum the contribution from each path. Each example
"votes" on how the shared parameter should move; the total gradient is the
sum of all votes.

### Why transposes appear where they do

Matrix multiplication contracts (sums over) the column-index of the left
operand against the row-index of the right operand. The index being summed
over (the batch index `i`) sits on the row axis of `A[l-1]` and `delta[l]`
in this row-convention, so `A[l-1]` needs transposing to bring `i` into the
contracted position for `dW`. Symmetric reasoning applies to `dA[l-1] =
delta[l] @ W[l].T`, contracting over the current-layer neuron index `q`
instead.

## Column-convention (Z = W @ A + b)

If a resource writes `Z = W @ A + b` instead, every transpose in this
derivation flips sides:

```
dW[l]     = delta[l] @ A[l-1].T     (was A[l-1].T @ delta[l])
db[l]     = row_sum(delta[l])       (was column_sum)
dA[l-1]   = W[l].T @ delta[l]       (was delta[l] @ W[l].T)
delta[l-1] = dA[l-1] * g'(Z[l-1])   (unchanged — elementwise, no matmul)
```

The underlying math is identical; only which physical axis batch examples
sit on changes, which determines which operand needs transposing to bring
the batch index into the position matrix multiplication contracts.

## Gradient checking

`tests/test_gradcheck.py` rebuilds the identical forward pass in PyTorch
with `requires_grad=True`, calls `.backward()`, and asserts the resulting
`.grad` tensors match the NumPy gradients computed above to within
numerical tolerance — an independent, automated verification that this
derivation is implemented correctly.
