# Unit tests for losses.py.
#
# Each test targets one specific failure mode we hit (or could hit)
# while building cross_entropy_loss.
#
# Run with: pytest tests/

import numpy as np
from mnist_nn.losses import cross_entropy_loss


def test_returns_scalar_not_array():
    # Regression test: cross_entropy_loss should collapse the whole batch
    # down to a single number, not leave a per-example array around.
    Y = np.eye(3, dtype=np.float32)[[0, 1, 2]]
    A2 = np.full((3, 3), 1.0 / 3.0, dtype=np.float32)

    loss = cross_entropy_loss(Y, A2)

    # np.ndim(x) == 0 covers both a plain Python float and a 0-d numpy
    # scalar -- either is fine, an array with shape (m,) or (m,1) is not.
    assert np.ndim(loss) == 0


def test_uniform_prediction_equals_log_num_classes():
    # If the model predicts a uniform distribution over k classes
    # regardless of input, loss = -log(1/k) = log(k) exactly, by
    # definition of cross-entropy. This is a hand-checkable ground truth,
    # independent of the model.
    num_classes = 3
    A2 = np.full((5, num_classes), 1.0 / num_classes, dtype=np.float32)
    Y = np.eye(num_classes, dtype=np.float32)[
        np.random.randint(0, num_classes, size=5)
    ]

    loss = cross_entropy_loss(Y, A2)

    assert np.isclose(loss, np.log(num_classes), atol=1e-4)


def test_confident_correct_gives_lower_loss_than_confident_wrong():
    # Sanity check on ordering: a prediction that's close to the true
    # label should be penalized less than one that's confidently wrong.
    # Also catches an args-swapped bug (Y and A2 in the wrong order),
    # since a swap would generally break this ordering too.
    Y = np.eye(3, dtype=np.float32)[[0, 1, 2]]

    A2_good = np.array(
        [[0.97, 0.02, 0.01],
         [0.02, 0.96, 0.02],
         [0.01, 0.02, 0.97]],
        dtype=np.float32,
    )
    A2_bad = np.array(
        [[0.02, 0.97, 0.01],
         [0.96, 0.02, 0.02],
         [0.97, 0.02, 0.01]],
        dtype=np.float32,
    )

    loss_good = cross_entropy_loss(Y, A2_good)
    loss_bad = cross_entropy_loss(Y, A2_bad)

    assert loss_good < loss_bad


def test_no_nan_or_inf_when_predicted_prob_is_exactly_zero():
    # Regression test for the exact nan bug hit during development:
    # A2 = 0.0 for the true class -> log(0) = -inf -> 0 * -inf = nan
    # without clipping. This test exists specifically so that if the
    # clip is ever removed, this fails instead of training silently
    # producing nan losses.
    Y = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    A2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    loss = cross_entropy_loss(Y, A2)

    assert np.isfinite(loss)