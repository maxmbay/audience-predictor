import numpy as np

from audience_predictor.data.encoder import HASH_ENCODER, HashEncoder, get_encoder
from audience_predictor.data.narrative import blocks_from_parsed, pooled_block_features
from audience_predictor.data.screenplay_parser import parse_screenplay


class FakeEncoder:
    name = "fake"
    dim = 8

    def embed_many(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text.strip():
                out[i, 0] = 1.0
                out[i, 1] = min(len(text), 50) / 50.0
        return out


def test_get_encoder_hash_dim():
    enc = get_encoder(HASH_ENCODER, hash_dim=32)
    assert isinstance(enc, HashEncoder)
    assert enc.dim == 32
    vec = enc.embed_many(["hello there"])
    assert vec.shape == (1, 32)
    assert np.isfinite(vec).all()


def test_blocks_from_parsed_uses_injected_encoder(sample_screenplay):
    parsed = parse_screenplay(sample_screenplay)
    fake = FakeEncoder()
    d, a, mask, struct = blocks_from_parsed(parsed, hash_dim=16, max_blocks=8, encoder=fake)
    assert d.shape == (1, 8, 8)
    assert int(mask.sum()) == 3
    assert d[0, 0, 0] == 1.0


def test_pooled_block_features_respects_mask():
    from audience_predictor.data.narrative import NarrativeBatch

    dialogue = np.zeros((2, 4, 3), dtype=np.float32)
    dialogue[0, 0] = 1.0
    dialogue[0, 1] = 3.0
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, :2] = True
    batch = NarrativeBatch(
        dialogue=dialogue,
        action=np.zeros_like(dialogue),
        mask=mask,
        structure=np.zeros((2, 11), dtype=np.float32),
        document_type=np.array(["screenplay", "unknown"]),
    )
    pooled = pooled_block_features(batch, include_action=False)
    np.testing.assert_allclose(pooled[0], np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(pooled[1], 0.0)
