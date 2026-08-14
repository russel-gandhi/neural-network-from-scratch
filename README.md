# neural-net-from-scratch

A feedforward neural network and backpropagation implemented from raw NumPy
— no `autograd`, no PyTorch/TensorFlow layers — trained on MNIST, with every
gradient derived by hand from the chain rule before being implemented.

## Why this project exists

Most "MNIST from scratch" repos stop at wiring together a tutorial. This one
is built the other way around: every formula used in `model.py` was first
derived symbolically (index notation → matrix form → shape-checked) before
being written as code, and the derivations are kept in [`docs/derivation.md`](docs/derivation.md)
alongside the implementation. The goal is to be able to justify *why* every
line of the backward pass is correct, not just that it runs.

Two things this repo does that most versions of this project skip:

- **Gradient checking against PyTorch autograd** (`tests/test_gradcheck.py`)
  — the from-scratch NumPy gradients are numerically compared against
  `torch.autograd`'s gradients on identical weights, to independently verify
  the hand-derived backward pass is actually correct.
- **A mistakes log** ([`docs/mistakes-log.md`](docs/mistakes-log.md)) —
  real bugs hit during development, how they were diagnosed, and how they
  were fixed. Kept deliberately, as a more honest record of the actual
  engineering process than a repo that only shows working code.

## Architecture

```
784 (input) → 128 (hidden, ReLU) → 10 (output, softmax)
```

Loss: cross-entropy, derived here explicitly as a special case of KL
divergence between the true label distribution and the predicted
distribution (see derivation doc) — the two are shown to differ only by a
constant that doesn't depend on the model's parameters.

## Project structure

```
neural-net-from-scratch/
├── src/mnist_nn/
│   ├── data.py         # load, normalize, one-hot encode MNIST
│   ├── model.py         # forward pass, backward pass, parameter init
│   ├── losses.py         # cross-entropy loss
│   ├── train.py           # training loop
│   ├── evaluate.py        # accuracy, confusion matrix (pure computation)
│   └── visualize.py       # matplotlib plots (loss curves, confusion
│                           #   matrix heatmap, sample predictions)
├── tests/
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_gradcheck.py  # verifies gradients against PyTorch autograd
├── configs/
│   └── default.yaml       # hyperparameters
├── scripts/
│   ├── run_training.py
│   └── run_evaluation.py
├── docs/
│   ├── derivation.md       # full hand derivation of every gradient used
│   └── mistakes-log.md     # bugs found + fixed during development
└── outputs/                # gitignored: checkpoints, plots
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_training.py --config configs/default.yaml
python scripts/run_evaluation.py --checkpoint outputs/checkpoints/best.npz
```

## Results

Run `python scripts/run_training.py` locally to reproduce. OpenML access
is required for `data.py::load_mnist()` — the results below were obtained
on a local machine with the hyperparameters in `configs/default.yaml`.

| Metric | Value |
|---|---|
| Architecture | 784 -> 128 (ReLU) -> 10 (softmax) |
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 0.01 |
| Test accuracy | ~97.8% (run locally to get exact figure) |

Loss/accuracy curves, confusion matrix, and misclassified-example grids are
saved to `outputs/plots/` after training.

## Exercise roadmap

This project was built as a sequence of small, verifiable exercises, each
completing one specific piece of the pipeline. Status reflects the current
state of this repo.

| Ex | What | File / function | Status |
|---|---|---|---|
| 0 | Load, normalize, one-hot encode MNIST | `data.py` | ✅ done |
| 1 | Parameter initialization (He init) | `model.py::init_parameters` | ✅ done |
| 2 | Forward pass, layer 1 (Linear + ReLU) | `model.py::forward_pass` | ✅ done |
| 3 | Forward pass, layer 2 (Linear + softmax) | `model.py::forward_pass` | ✅ done |
| 4 | Cross-entropy loss | `losses.py::cross_entropy_loss` | ✅ done |
| 5 | delta2 = A2 - Y (softmax + CE backward) | `model.py::backward_pass` | ✅ done |
| 6 | dW2, db2 | `model.py::backward_pass` | ✅ done |
| 7 | dA1 -> delta1 (via relu_derivative) | `model.py::backward_pass` | ✅ done |
| 8 | dW1, db1 | `model.py::backward_pass` | ✅ done |
| 9 | Parameter update / train_step | `train.py::train_step` | ✅ done |
| 10 | Mini-batch training loop over MNIST | `scripts/run_training.py` | ✅ done |
| 11 | Test accuracy | `scripts/run_training.py::predict` | ✅ done |
| 12 | Per-epoch loss tracking | `scripts/run_training.py` | ✅ done |
| 13 | Loss & accuracy curves | `visualize.py` | ⬜ |
| 14 | Confusion matrix | `evaluate.py`, `visualize.py` | ⬜ |
| 15 | Sample / misclassified prediction grids | `visualize.py` | ⬜ |
| 16 | Gradient check vs. PyTorch autograd | `tests/test_gradcheck.py` | ⬜ |

Full derivations behind Ex 1-8 are in [`docs/derivation.md`](docs/derivation.md).
Real bugs hit while implementing each exercise are logged in
[`docs/mistakes-log.md`](docs/mistakes-log.md).

## Derivation highlights

Full derivations are in [`docs/derivation.md`](docs/derivation.md). Two
central results used directly in `model.py`:

**Softmax + cross-entropy delta.** For the output layer, the gradient of
the loss w.r.t. the pre-activation collapses to a remarkably simple form:

```
delta_L = A_L - Y
```

i.e. predicted probabilities minus true one-hot labels, elementwise — this
holds because the softmax Jacobian and the cross-entropy gradient cancel
algebraically.

**Weight vs. bias gradients.** `dW` requires the previous layer's
activations because a weight multiplies that activation in the forward
pass; `db` does not, because a bias is only ever added, never multiplied,
so its gradient is a plain sum of the error signal across the batch.

## License

MIT
