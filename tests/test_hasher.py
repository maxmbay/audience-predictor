from audience_predictor.data.hasher import hash_embed


def test_hasher_is_deterministic():
    a = hash_embed("INT. ORBITAL CANTEEN - NIGHT", 64)
    b = hash_embed("INT. ORBITAL CANTEEN - NIGHT", 64)
    assert a.shape == (64,)
    assert (a == b).all()


def test_hasher_is_sensitive_to_text():
    a = hash_embed("space opera corridor", 128)
    b = hash_embed("rom-com bakery meet cute", 128)
    assert abs(float(a @ b)) < 0.95


def test_empty_text_is_zero():
    z = hash_embed("   ", 32)
    assert float(z.sum()) == 0.0
