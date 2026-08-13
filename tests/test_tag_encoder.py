import jax
import jax.numpy as jnp
import numpy as np

from audience_predictor.models.losses import bce_soft
from audience_predictor.models.tag_encoder import ScriptTagEncoder


def test_tag_encoder_independent_sigmoid():
    model = ScriptTagEncoder(
        in_dim=8,
        model_dim=16,
        n_tags=5,
        n_heads=2,
        n_inducing=3,
        key=jax.random.key(0),
        pool="isab",
    )
    blocks = jax.random.normal(jax.random.key(1), (6, 8))
    mask = jnp.array([True, True, True, False, False, False])
    pred = model(blocks, mask)
    assert pred.shape == (5,)
    assert float(pred.min()) > 0.0
    assert float(pred.max()) < 1.0
    # Independent heads: values must not be constrained to a simplex.
    assert abs(float(pred.sum()) - 1.0) > 0.05


def test_tag_encoder_mask_ignores_padding():
    model = ScriptTagEncoder(
        in_dim=8, model_dim=16, n_tags=4, n_heads=2, n_inducing=3, key=jax.random.key(2)
    )
    blocks = jax.random.normal(jax.random.key(3), (4, 8))
    mask = jnp.array([True, True, False, False])
    a = model(blocks, mask)
    dirty = blocks.at[3].set(10.0)
    b = model(dirty, mask)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_bce_soft_decreases_on_easy_labels():
    pred = jnp.array([0.9, 0.1])
    target = jnp.array([1.0, 0.0])
    easy = bce_soft(pred, target)
    hard = bce_soft(jnp.array([0.1, 0.9]), target)
    assert float(easy) < float(hard)


def test_tag_mean_encoder_shape():
    from audience_predictor.models.tag_encoder import TagMeanEncoder

    model = TagMeanEncoder(in_dim=8, hidden=16, n_tags=5, key=jax.random.key(4))
    pred = model(jax.random.normal(jax.random.key(5), (8,)))
    assert pred.shape == (5,)
    assert 0.0 < float(pred.sum()) < 5.0
