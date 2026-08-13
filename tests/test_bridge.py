import pytest

from audience_predictor.data.bridge import BridgeError, catalog_value_root


def test_catalog_value_sibling_or_error():
    try:
        root = catalog_value_root()
    except BridgeError:
        pytest.skip("catalog-value not present")
    assert (root / "pyproject.toml").exists()
