from audience_predictor.config import load_config
from audience_predictor.paths import project_root


def test_config_loads_defaults():
    cfg = load_config()
    assert cfg.representation.embedding_dim == 64
    assert cfg.representation.n_interests == 8
    assert cfg.data.hash_dim > 0
    assert cfg.train.holdout_frac == 0.2


def test_project_root_has_pyproject():
    assert (project_root() / "pyproject.toml").exists()
