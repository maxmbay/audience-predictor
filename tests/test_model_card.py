from audience_predictor.viz.model_card import (
    plot_architecture_dag,
    plot_honesty_ladder,
    plot_product_recall,
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
    assert a.exists() and b.exists() and c.exists()
