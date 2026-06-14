"""Test cases for the dimensionality-reduction reducers.
"""
import numpy as np
import dask.array as da
import pytest
from sklearn.pipeline import Pipeline

from pymks.fmks.reducers import tsne, TSNETransformer
from pymks import FlattenTransformer


def test_tsne_shape():
    """t-SNE embeds (n_samples, n_features) into (n_samples, n_components)"""
    x_data = np.random.RandomState(0).random((20, 8))
    assert tsne(x_data, perplexity=5.0, random_state=0).shape == (20, 2)
    assert tsne(x_data, n_components=3, perplexity=5.0, random_state=0).shape == (20, 3)


def test_tsne_deterministic():
    """A fixed random_state gives a reproducible embedding"""
    x_data = np.random.RandomState(1).random((20, 8))
    first = tsne(x_data, perplexity=5.0, random_state=42)
    second = tsne(x_data, perplexity=5.0, random_state=42)
    assert np.allclose(first, second)


def test_tsne_curried():
    """The functional interface is curried like the rest of fmks"""
    x_data = np.random.RandomState(2).random((20, 8))
    embed = tsne(perplexity=5.0, random_state=0)
    assert embed(x_data).shape == (20, 2)


def test_tsne_materializes_dask():
    """A Dask array input is computed and embedded"""
    x_data = da.random.random((20, 8), chunks=(5, 8))
    assert tsne(x_data, perplexity=5.0, random_state=0).shape == (20, 2)


def test_tsne_requires_2d():
    """A non-2D input raises a clear error"""
    with pytest.raises(RuntimeError, match="2D"):
        tsne(np.zeros((2, 3, 3)))


def test_tsne_transformer():
    """The transformer wraps the function for standalone use"""
    x_data = np.random.RandomState(3).random((20, 8))
    out = TSNETransformer(perplexity=5.0, random_state=0).fit_transform(x_data)
    assert out.shape == (20, 2)


def test_tsne_transformer_in_pipeline():
    """The transformer works as the terminal step of a pipeline, fed by
    FlattenTransformer to satisfy the 2D input contract."""
    x_data = np.random.RandomState(4).random((20, 3, 3))
    out = Pipeline(
        [
            ("flatten", FlattenTransformer()),
            ("tsne", TSNETransformer(perplexity=5.0, random_state=0)),
        ]
    ).fit_transform(x_data)
    assert out.shape == (20, 2)
