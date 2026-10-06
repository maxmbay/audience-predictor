import numpy as np

from audience_predictor.viz.model_card import (
    plot_architecture_dag,
    plot_honesty_ladder,
    plot_product_recall,
    plot_rt_reception,
    plot_two_step,
)


def test_architecture_dag_writes(tmp_path):
    path = plot_architecture_dag(tmp_path / "dag.png")
    assert path.exists()
    assert path.stat().st_size > 1000


def test_honesty_and_two_step_write(tmp_path):
    a = plot_honesty_ladder(
        [("baseline", 0.12, "script"), ("genome", 0.38, "release")],
        tmp_path / "ladder.png",
    )
    b = plot_two_step(0.32, 0.02, 0.02, 0.12, tmp_path / "two.png")
    c = plot_product_recall(
        [("baseline", 0.04, 0.03, "script"), ("genome", 0.2, 0.15, "leak")],
        0.005,
        tmp_path / "product.png",
    )
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.2, 0.8, size=(40, 2))
    true = pred + rng.normal(0, 0.05, size=pred.shape)
    mask = np.ones_like(pred, dtype=bool)
    d = plot_rt_reception(pred, true, mask, tmp_path / "rt.png", spearman=(0.5, 0.3, 0.2))
    assert a.exists() and b.exists() and c.exists() and d.exists()
