import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from audience_predictor.data.document import classify_document
from audience_predictor.web.app import create_app
from audience_predictor.web.demos import demo_catalog, load_demo


def test_demo_specs_classify_as_screenplays_or_subtitles():
    for item in demo_catalog():
        _, text = load_demo(item["id"])
        kind = classify_document(text)
        if item["id"] == "window_seat":
            assert kind == "subtitle"
        else:
            assert kind in {"screenplay", "shooting_script"}, (item["id"], kind)
        assert len(text.split()) > 80


def test_coverage_desk_serves_the_title_page():
    client = TestClient(create_app())
    home = client.get("/")
    assert home.status_code == 200
    assert "AUDIENCE PREDICTOR" in home.text
    assert "COVERAGE" in home.text
    demos = client.get("/api/demos")
    assert demos.status_code == 200
    ids = {row["id"] for row in demos.json()}
    assert {"hamlet", "macbeth", "romeo_and_juliet", "the_well", "two_tickets"} <= ids
    assert client.get("/api/health").json()["status"] == "ok"


def test_predict_rejects_empty_pages():
    client = TestClient(create_app())
    res = client.post("/api/predict", json={"text": "   "})
    assert res.status_code == 400


def test_unknown_demo_is_404():
    client = TestClient(create_app())
    res = client.post("/api/predict", json={"demo_id": "not_a_spec"})
    assert res.status_code == 404


def test_health_names_the_scene_judge():
    client = TestClient(create_app())
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["kind"] == "heuristic"


def test_predict_returns_coverage_without_an_audience_head(sample_screenplay):
    client = TestClient(create_app())
    res = client.post("/api/predict", json={"text": sample_screenplay})
    assert res.status_code == 200
    body = res.json()
    assert body["coverage"]["stamp"] in {"pass", "consider", "recommend"}
    assert "turns" in body["coverage"]["scene_means"]
    assert body["coverage"]["n_judged"] == 3
    assert "WHAT IS IT" in client.get("/").text
    home = client.get("/").text
    assert "JEV  ·  SCENE TIME" in home
    assert "jev-plot" in home
    row = body["coverage"]["scenes"][0]
    assert "softmax" in row
    assert abs(sum(row["softmax"].values()) - 1.0) < 1e-3

