"""Dimensionality-reduction transformers for PyMKS pipelines.

This module provides manifold-learning reducers that plug into
Scikit-learn pipelines using the same conventions as the rest of
``fmks``: a curried functional core plus a thin, stateless
``BaseEstimator`` / ``TransformerMixin`` wrapper.

Currently t-SNE (:func:`tsne` / :class:`TSNETransformer`) is provided,
wrapping :class:`sklearn.manifold.TSNE`. The reducers expect
**flattened** ``(n_samples, n_features)`` data, so they are placed after
a flattening step such as :class:`~pymks.FlattenTransformer` (and,
typically, a PCA pre-reduction) in a pipeline.
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.manifold import TSNE

from .func import curry


@curry
def tsne(
    x_data, n_components=2, perplexity=30.0, learning_rate="auto", random_state=None
):
    """Embed flattened data into a low-dimensional space with t-SNE.

    Thin curried wrapper around :class:`sklearn.manifold.TSNE`. The
    input is materialized to a Numpy array (so Dask arrays are computed)
    and must be two-dimensional, ``(n_samples, n_features)``, matching
    the Scikit-learn convention. Flatten upstream with
    :class:`~pymks.FlattenTransformer` if needed.

    t-SNE is stochastic and provides no out-of-sample embedding; fix
    ``random_state`` for reproducible results. ``perplexity`` must be
    less than ``n_samples``.

    Args:
      x_data: the data to embed, shape ``(n_samples, n_features)``
      n_components: the dimension of the embedded space
      perplexity: the t-SNE perplexity (must be ``< n_samples``)
      learning_rate: the t-SNE learning rate (``"auto"`` by default)
      random_state: seed for reproducible embeddings

    Returns:
      the embedded data, shape ``(n_samples, n_components)``

    >>> import numpy as np
    >>> x_data = np.random.RandomState(0).random((20, 8))
    >>> tsne(x_data, perplexity=5.0, random_state=0).shape
    (20, 2)

    The embedding dimension is configurable.

    >>> tsne(x_data, n_components=3, perplexity=5.0, random_state=0).shape
    (20, 3)

    A fixed ``random_state`` gives a reproducible embedding.

    >>> a = tsne(x_data, perplexity=5.0, random_state=0)
    >>> b = tsne(x_data, perplexity=5.0, random_state=0)
    >>> assert np.allclose(a, b)

    The input must be 2D.

    >>> tsne(np.zeros((2, 3, 3)))
    Traceback (most recent call last):
    ...
    RuntimeError: t-SNE input must be 2D (n_samples, n_features)

    """
    arr = np.asarray(x_data)
    if arr.ndim != 2:
        raise RuntimeError("t-SNE input must be 2D (n_samples, n_features)")
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
    ).fit_transform(arr)


class TSNETransformer(BaseEstimator, TransformerMixin):
    """t-SNE embedding as a Scikit-learn pipeline step.

    Wraps the :func:`tsne` function. See that for more complete
    documentation. As :class:`sklearn.manifold.TSNE` provides no
    out-of-sample ``transform``, this transformer is stateless and
    recomputes the embedding on every call, so it is intended as the
    terminal step of a pipeline (used via ``fit_transform``).

    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from pymks import FlattenTransformer

    >>> x_data = np.random.RandomState(0).random((20, 3, 3))
    >>> Pipeline([
    ...     ('flatten', FlattenTransformer()),
    ...     ('tsne', TSNETransformer(perplexity=5.0, random_state=0)),
    ... ]).fit_transform(x_data).shape
    (20, 2)

    """

    def __init__(
        self, n_components=2, perplexity=30.0, learning_rate="auto", random_state=None
    ):
        """Instantiate a TSNETransformer

        Args:
          n_components: the dimension of the embedded space
          perplexity: the t-SNE perplexity (must be ``< n_samples``)
          learning_rate: the t-SNE learning rate (``"auto"`` by default)
          random_state: seed for reproducible embeddings

        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, *_):
        """Only necessary to make pipelines work"""
        return self

    def transform(self, x_data):
        """Embed the data with t-SNE

        Args:
          x_data: the data to embed, shape ``(n_samples, n_features)``

        Returns:
          the embedded data, shape ``(n_samples, n_components)``
        """
        return tsne(
            x_data,
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def __sklearn_is_fitted__(self):
        """Stateless transformer; always reports as fitted."""
        return True
