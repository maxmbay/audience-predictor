import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from audience_predictor.data.bridge import BridgeError, catalog_value_root
from audience_predictor.data.screenplay_parser import parse_moviesum_xml

FIXTURE = Path(__file__).parent / "fixtures" / "moviesum_scene.xml"


def test_moviesum_xml_extracts_scenes_dialogue_and_action():
    parsed = parse_moviesum_xml(FIXTURE.read_text())
    assert parsed.n_scenes == 2
    assert parsed.scenes[0].interior is True
    assert parsed.scenes[1].interior is False
    assert "KIRA" in parsed.characters
    assert "JON" in parsed.characters
    assert "don't have long" in parsed.scenes[0].dialogue_text
    assert "Steam ghosts" in parsed.scenes[0].action_text
    assert parsed.scenes[0].location


def test_ingest_writes_identifiers_not_script_text(tmp_path, monkeypatch):
    try:
        catalog_value_root()
    except BridgeError:
        pytest.skip("catalog-value not present")

    from audience_predictor.data import moviesum as ms

    raw = tmp_path / "raw"
    gold = tmp_path / "gold"
    raw.mkdir()
    rec = {
        "movie_name": "Toy Story",
        "imdb_id": "tt0114709",
        "script": FIXTURE.read_text(),
        "summary": "WIKIPEDIA LEAK MUST NOT BE WRITTEN",
    }
    (raw / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (raw / "val.jsonl").write_text("")
    (raw / "test.jsonl").write_text("")
    monkeypatch.setattr(ms, "moviesum_raw_dir", lambda: raw)
    monkeypatch.setattr(ms, "gold_processed_dir", lambda: gold)
    monkeypatch.setattr(ms, "provenance_path", lambda: gold / "moviesum_provenance.parquet")
    monkeypatch.setattr(ms, "blocks_path", lambda encoder="hash": gold / "moviesum_blocks.npz")

    frame = ms.ingest_moviesum(hash_dim=32, max_blocks=8)
    assert "script" not in frame.columns
    assert "summary" not in frame.columns
    blob = (gold / "moviesum_provenance.parquet").read_bytes()
    assert b"WIKIPEDIA LEAK MUST NOT BE WRITTEN" not in blob
    assert b"<dialogue>" not in blob
    assert frame.filter(pl.col("movie_row").is_not_null()).height == 1
    blocks = np.load(gold / "moviesum_blocks.npz", allow_pickle=True)
    assert blocks["mask"].any()
