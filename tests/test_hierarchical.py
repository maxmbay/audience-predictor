import jax
import jax.numpy as jnp
import numpy as np

from audience_predictor.data.narrative import blocks_from_screenplay
from audience_predictor.data.screenplay_parser import STRUCT_KEYS
from audience_predictor.models.hierarchical import NarrativeSetEncoder


def _model(hash_dim=32, model_dim=16, out_dim=8):
    return NarrativeSetEncoder(
        hash_dim=hash_dim,
        model_dim=model_dim,
        out_dim=out_dim,
        n_heads=2,
        n_inducing=4,
        n_struct=len(STRUCT_KEYS),
        key=jax.random.key(0),
    )


def test_set_encoder_shapes_and_mask(sample_screenplay):
    d, a, mask, struct = blocks_from_screenplay(sample_screenplay, hash_dim=32, max_blocks=8)
    model = _model()
    mu, log_var, bias, aux, z_d, delta = model(d[0], a[0], mask[0], struct[0])
    assert mu.shape == (8,)
    assert z_d.shape == (8,)
    assert delta.shape == (8,)
    assert aux.shape == (4,)


def test_padding_does_not_change_pool(sample_screenplay):
    d, a, mask, struct = blocks_from_screenplay(sample_screenplay, hash_dim=32, max_blocks=8)
    model = _model()
    mu1, *_ = model(d[0], a[0], mask[0], struct[0])
    d2 = jnp.asarray(d[0]).at[7].set(100.0)
    mu2, *_ = model(d2, a[0], mask[0], struct[0])
    np.testing.assert_allclose(np.asarray(mu1), np.asarray(mu2), atol=1e-5)


def test_residual_near_zero_when_action_empty(sample_screenplay):
    d, a, mask, struct = blocks_from_screenplay(sample_screenplay, hash_dim=32, max_blocks=8)
    model = _model()
    zeros = jnp.zeros_like(a[0])
    _, _, _, _, z_d, delta = model(d[0], zeros, mask[0], jnp.zeros_like(struct[0]))
    # Delta is a learned map of zeros; it should be small relative to z_d scale.
    assert float(jnp.linalg.norm(delta)) < 5.0 * (float(jnp.linalg.norm(z_d)) + 1.0)


def test_vmap_over_movies(sample_screenplay):
    d, a, mask, struct = blocks_from_screenplay(sample_screenplay, hash_dim=32, max_blocks=8)
    model = _model()
    batched = jax.jit(jax.vmap(model))
    d2 = jnp.concatenate([d, d], axis=0)
    a2 = jnp.concatenate([a, a], axis=0)
    m2 = jnp.concatenate([mask, mask], axis=0)
    s2 = jnp.concatenate([struct, struct], axis=0)
    mu, log_var, bias, aux, z_d, delta = batched(d2, a2, m2, s2)
    assert mu.shape == (2, 8)
    assert log_var.shape == (2,)
