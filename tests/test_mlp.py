import jax
import jax.numpy as jnp
import numpy as np

from audience_predictor.models.mlp import AudienceMLP
from audience_predictor.train.loop import load_checkpoint, predict_mlp, save_checkpoint, train_mlp


def test_mlp_shapes():
    key = jax.random.key(0)
    model = AudienceMLP(in_dim=8, out_dim=4, hidden=16, key=key)
    mu, log_var, bias, aux = model(jnp.ones(8))
    assert mu.shape == (4,)
    assert log_var.shape == ()
    assert bias.shape == ()
    assert aux.shape == (4,)


def test_mlp_vmap_and_jit():
    key = jax.random.key(1)
    model = AudienceMLP(8, 4, 16, key)
    x = jnp.ones((5, 8))
    batched = jax.jit(jax.vmap(model))
    mu, log_var, bias, aux = batched(x)
    assert mu.shape == (5, 4)
    assert log_var.shape == (5,)
    assert aux.shape == (5, 4)


def test_mlp_train_step_decreases_loss(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 6)).astype(np.float32)
    w = rng.normal(size=(6, 4)).astype(np.float32)
    z = x @ w
    bias = rng.normal(size=(32,)).astype(np.float32)
    aux = rng.normal(size=(32, 4)).astype(np.float32)
    idx = np.arange(24)
    model, history = train_mlp(
        x, z, bias, aux, idx,
        hidden=16, epochs=5, lr=0.05, weight_decay=0.0,
        nll_weight=0.05, aux_weight=0.05, bias_weight=0.05, seed=0,
    )
    assert history[-1] < history[0]
    pred = predict_mlp(model, x)
    assert pred["mu"].shape == (32, 4)
    path = tmp_path / "m.eqx"
    save_checkpoint(path, model)
    cloned = AudienceMLP(6, 4, 16, jax.random.key(99))
    restored = load_checkpoint(path, cloned)
    pred2 = predict_mlp(restored, x)
    np.testing.assert_allclose(pred["mu"], pred2["mu"], atol=1e-5)
