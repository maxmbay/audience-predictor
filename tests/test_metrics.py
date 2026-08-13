import numpy as np

from audience_predictor.eval.metrics import cosine, recovery_metrics
from audience_predictor.models.posterior import residual_variance, shrink_posterior


def test_cosine_identical_is_one():
    z = np.ones((3, 4))
    np.testing.assert_allclose(cosine(z, z), 1.0, atol=1e-6)


def test_recovery_metrics_keys():
    pred = np.eye(4)
    true = np.eye(4)
    log_var = np.zeros(4)
    aux = np.linspace(0, 1, 16).reshape(4, 4)
    out = recovery_metrics(pred, true, log_var, aux, aux)
    assert out["cosine_mean"] > 0.99
    assert "nll" in out
    assert "crps" in out
    assert "spearman_p_love" in out


def test_shrink_cold_start_equals_content():
    z = np.ones((2, 3))
    c = np.zeros((2, 3))
    n = np.array([100.0, 0.0])
    post = shrink_posterior(
        z, c, np.zeros(2), np.ones(2), n, np.arange(2),
        n0=50.0, content_var=1.0, collab_mask=np.array([True, False]),
    )
    np.testing.assert_allclose(post.mu[1], 0.0)
    assert post.var[1] > post.var[0]
    assert residual_variance(z, c, np.array([True, False])) == 1.0
