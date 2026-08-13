from audience_predictor.viz.model_card import plot_architecture_dag, plot_honesty_ladder, plot_two_step


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
    assert a.exists() and b.exists()
