import numpy as np

from audience_predictor.data.narrative import blocks_from_screenplay
from tests.conftest import SAMPLE_SCREENPLAY


def test_screenplay_blocks_mask_matches_scene_count():
    d, a, mask, struct = blocks_from_screenplay(SAMPLE_SCREENPLAY, hash_dim=16, max_blocks=8)
    assert d.shape == (1, 8, 16)
    assert int(mask.sum()) == 3
    assert not mask[0, 3]
    assert struct.shape[1] > 0
