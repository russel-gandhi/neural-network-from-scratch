# Derivation

Full derivation of the forward pass, loss, and backward pass used in
`src/mnist_nn/model.py` and `losses.py`. Row-convention throughout: each
row of a batch matrix is one training example ($i$ = example index,
second index = neuron index).

---

## Forward pass

$$Z^{[1]} = X W^{[1]} + b^{[1]} \quad (m,784)@(784,128) \to (m,128)$$
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$
$$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]} \quad (m,128)@(128,10) \to (m,10)$$
$$A^{[2]} = \text{softmax}(Z^{[2]}) \quad \text{(row-wise)}$$

Implemented in `model.py::forward()`. Every intermediate ($Z^{[1]}$,
$A^{[1]}$, $Z^{[2]}$, $A^{[2]}$) is cached and returned, not just $A^{[2]}$
— the backward pass needs all of them (see below), and recomputing them
there would defeat the point of caching described in the general
algorithm.

### Weight initialization

$W^{[1]}$ uses **He initialization**:

$$W^{[1]} \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}}}\right)$$

i.e. `np.random.randn(n_in, n_out) * sqrt(2 / n_in)`. This specifically
compensates for ReLU zeroing out roughly half its inputs — without the
factor of 2, variance shrinks layer by layer through a deep ReLU network.
Xavier/Glorot init ($1/n_{\text{in}}$, no factor of 2) is the standard
choice for sigmoid/tanh instead, since those activations are roughly
linear near 0 and don't discard half their input variance the way ReLU
does.

Biases are initialized to zero. Weights are not, to avoid the *symmetry
problem*: if every weight starts identical (e.g. all zero), every neuron
in a layer computes the same output and receives the same gradient
forever, so the layer never differentiates its neurons regardless of how
long training runs.

### Numerical stability in softmax

$$\text{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\sum_{k'} e^{z_{k'} - \max(z)}}$$

Subtracting the row max before exponentiating does not change the
mathematical result — algebraically:

$$\frac{e^{z_k - c}}{\sum_{k'} e^{z_{k'} - c}} = \frac{e^{-c}\, e^{z_k}}{e^{-c} \sum_{k'} e^{z_{k'}}}$$

and the $e^{-c}$ factor cancels top and bottom for any constant $c$.
Choosing $c = \max(z)$ keeps the largest exponent at exactly $0$ and every
other exponent $\le 0$, preventing `np.exp` overflow that raw `exp(z)`
could hit for large logits.

---

## Loss: cross-entropy as a special case of KL divergence

Let $p$ be the true label distribution (one-hot) and $q = A^{[2]}$ the
predicted distribution. KL divergence:

$$D_{KL}(p \| q) = \sum_k p_k \log\frac{p_k}{q_k} = \sum_k p_k \log p_k - \sum_k p_k \log q_k = -H(p) - H(p,q)$$

$-H(p)$ doesn't depend on model parameters ($p$ is fixed data), so it
contributes zero gradient. Minimizing $D_{KL}(p \| q)$ over parameters is
therefore identical to minimizing the cross-entropy

$$H(p,q) = -\sum_k p_k \log q_k$$

— this is why cross-entropy loss and KL loss are used interchangeably in
classification.

---

## The key result: delta at the output layer

Differentiating the loss directly w.r.t. $Z^{[2]}$ (not $A^{[2]}$, since
softmax isn't elementwise) gives, remarkably:

$$\boxed{\delta^{[2]} = A^{[2]} - Y}$$

Predicted probabilities minus true one-hot labels, elementwise. The
softmax Jacobian and the cross-entropy gradient cancel algebraically to
produce this simple form — see the full symbolic derivation in the
project history / chat log this repo was built from.

---

## Backward pass, general layer $l$

Given $\delta^{[l]} = \dfrac{\partial \text{Loss}}{\partial Z^{[l]}}$:

$$dW^{[l]} = \left(A^{[l-1]}\right)^\top \delta^{[l]} \quad \text{(weight gradient)}$$

$$db^{[l]} = \text{column\_sum}\left(\delta^{[l]}\right) \quad \text{(bias gradient)}$$

$$dA^{[l-1]} = \delta^{[l]} \left(W^{[l]}\right)^\top \quad \text{(error handed to previous layer)}$$

$$\delta^{[l-1]} = dA^{[l-1]} \odot g^{[l-1]\prime}(Z^{[l-1]}) \quad \text{(elementwise; convert activation gradient to pre-activation gradient)}$$

### Why `dW` needs `A[l-1]` but `db` does not

A weight $W^{[l]}_{pq}$ multiplies the incoming activation $A^{[l-1]}_p$ in
the forward pass, so its gradient is weighted by how strong that input
signal was:

$$\sum_i \delta^{[l]}_{iq} \, A^{[l-1]}_{ip}$$

A bias is only ever *added*, never multiplied by anything from the
previous layer, so its gradient is just the raw error signal summed over
the batch:

$$\sum_i \delta^{[l]}_{iq}$$

### Why the sum over examples

Both $W^{[l]}_{pq}$ and $b^{[l]}_q$ are single shared parameters used
identically across all $m$ examples in a batch. The multivariable chain
rule says: when one variable affects an output through multiple
independent paths (here, one path per example), sum the contribution from
each path. Each example "votes" on how the shared parameter should move;
the total gradient is the sum of all votes.

### Why transposes appear where they do

Matrix multiplication contracts (sums over) the column-index of the left
operand against the row-index of the right operand. The index being
summed over (the batch index $i$) sits on the row axis of $A^{[l-1]}$ and
$\delta^{[l]}$ in this row-convention, so $A^{[l-1]}$ needs transposing to
bring $i$ into the contracted position for $dW^{[l]}$. Symmetric reasoning
applies to $dA^{[l-1]} = \delta^{[l]} (W^{[l]})^\top$, contracting over the
current-layer neuron index $q$ instead.

---

## Column-convention ($Z = WA + b$)

If a resource writes $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ instead, every
transpose in this derivation flips sides:

$$dW^{[l]} = \delta^{[l]} \left(A^{[l-1]}\right)^\top \quad \text{(was } (A^{[l-1]})^\top \delta^{[l]} \text{)}$$

$$db^{[l]} = \text{row\_sum}\left(\delta^{[l]}\right) \quad \text{(was column\_sum)}$$

$$dA^{[l-1]} = \left(W^{[l]}\right)^\top \delta^{[l]} \quad \text{(was } \delta^{[l]} (W^{[l]})^\top \text{)}$$

$$\delta^{[l-1]} = dA^{[l-1]} \odot g^{[l-1]\prime}(Z^{[l-1]}) \quad \text{(unchanged — elementwise, no matmul)}$$

The underlying math is identical; only which physical axis batch examples
sit on changes, which determines which operand needs transposing to bring
the batch index into the position matrix multiplication contracts.

---

## Gradient checking

`tests/test_gradcheck.py` rebuilds the identical forward pass in PyTorch
with `requires_grad=True`, calls `.backward()`, and asserts the resulting
`.grad` tensors match the NumPy gradients computed above to within
numerical tolerance — an independent, automated verification that this
derivation is implemented correctly.