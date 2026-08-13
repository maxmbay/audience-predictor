from pathlib import Path

import numpy as np
import polars as pl
import pytest

from audience_predictor.data.bridge import BridgeError, catalog_value_root
from audience_predictor.data.subtitles import (
    chunk_cues,
    ingest_sublens,
    parse_srt_cues,
    subtitle_block_texts,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.srt"


def test_parse_srt_strips_timecodes_and_keeps_speech(sample_srt):
    cues = parse_srt_cues(sample_srt)
    assert cues == ["We don't have long.", "Then we move before the next window."]


def test_parse_srt_strips_html_and_webvtt():
    text = """WEBVTT

1
00:00:01.000 --> 00:00:03.000
<i>Hello</i> there.

2
00:00:03.500 --> 00:00:05.000
Goodbye.
"""
    assert parse_srt_cues(text) == ["Hello there.", "Goodbye."]


def test_chunk_and_select_spread_across_file():
    cues = [f"line {i} " * 20 for i in range(40)]
    chunks = chunk_cues(cues, max_chars=80)
    assert len(chunks) > 4
    blocks = subtitle_block_texts(cues, max_blocks=4)
    assert len(blocks) == 4
    assert "line 0" in blocks[0]
    assert "line 39" in blocks[-1]


def test_ingest_writes_identifiers_not_subtitle_text(tmp_path, monkeypatch):
    try:
        catalog_value_root()
    except BridgeError:
        pytest.skip("catalog-value not present")

    from audience_predictor.data import subtitles as sl

    raw = tmp_path / "raw"
    silver = tmp_path / "silver"
    raw.mkdir()
    secret = "THIS SUBTITLE MUST NOT BE WRITTEN TO PROVENANCE"
    (raw / "0114709.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n" + secret + "\n"
    )
    nested = raw / "0114709"
    nested.mkdir()
    (nested / "999.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nTo infinity and beyond.\n"
    )
    monkeypatch.setattr(sl, "sublens_raw_dir", lambda: raw)
    monkeypatch.setattr(sl, "silver_processed_dir", lambda: silver)
    monkeypatch.setattr(sl, "provenance_path", lambda: silver / "sublens_provenance.parquet")
    monkeypatch.setattr(sl, "blocks_path", lambda encoder="hash": silver / "sublens_blocks.npz")

    frame = sl.ingest_sublens(hash_dim=32, max_blocks=8)
    assert "text" not in frame.columns
    blob = (silver / "sublens_provenance.parquet").read_bytes()
    assert secret.encode() not in blob
    assert frame["document_type"][0] == "subtitle"
    assert frame["imdb_join"].n_unique() == 1
    assert frame.filter(pl.col("movie_row").is_not_null())["movie_row"].n_unique() == 1
    blocks = np.load(silver / "sublens_blocks.npz", allow_pickle=True)
    assert blocks["mask"].any()
    assert (blocks["document_type"] == "subtitle").any()
    assert not np.asarray(blocks["action"]).any()
