import numpy as np

from src.features.spectral_gpt import _to_spectralgpt_s2_12, reduce_embeddings_with_pca


def test_to_spectralgpt_s2_12_shapes_and_bounds():
    n, h, w, b = 3, 16, 16, 10
    rng = np.random.default_rng(42)
    chips = rng.normal(loc=1000.0, scale=200.0, size=(n, h, w, b)).astype(np.float32)
    band_names = [
        "red",
        "green",
        "blue",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "nir",
        "nir_2",
        "swir1",
        "swir2",
    ]

    out = _to_spectralgpt_s2_12(chips, band_names=band_names)
    assert out.shape == (n, 12, h, w)
    assert np.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0

    # B1 and B2 are both proxied from blue in this backend.
    assert np.allclose(out[:, 0], out[:, 1], atol=1e-6)


def test_reduce_embeddings_with_pca_shape():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(50, 768)).astype(np.float32)
    y = reduce_embeddings_with_pca(x, n_components=16, random_state=7)
    assert y.shape == (50, 16)
    assert np.isfinite(y).all()
