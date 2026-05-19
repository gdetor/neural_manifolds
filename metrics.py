# import gs
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA


eps = np.finfo(np.float32).eps


def covarianceMatrix(X: np.ndarray,
                     neural_activity: bool = True) -> np.ndarray:
    """
    Estimate the covariance matrix. In case the data describe neural activity
    (responses), then the matrix X will be transposed.
    """
    n_samples = X.shape[0]
    R = X.copy()

    if neural_activity:
        R = R.T
        mu = np.mean(R, axis=1).reshape(-1, 1)
        # std = np.std(R, axis=1).reshape(-1, 1)
    else:
        mu = np.mean(R, axis=0).reshape(1, -1)
        # std = np.std(R, axis=0).reshape(1, -1)

    # X_ = (R - mu) / (std + 1e-9)
    X_ = R - mu

    C = X_ @ X_.T
    C /= (n_samples - 1)
    return C


def estimateParticipationRatio(X: np.ndarray,
                               neural_activity: bool = True) -> float:
    """
    Estimate the participation ratio.

    @note Participation ratio is unity if one element is of ...
    """
    cov = covarianceMatrix(X, neural_activity=neural_activity)
    evals = np.linalg.eigvals(cov)
    PR = evals.sum()**2 / (evals**2).sum()
    return PR.real


def explainedVariance(X: np.ndarray, n_components: int = 3):
    X = (X - X.mean()) / X.std()
    embed = PCA(n_components=n_components)
    embed.fit_transform(X)
    return embed.explained_variance_ratio_.cumsum()


def estimateSparseness(X: np.ndarray) -> float:
    """
    Estimate the sparseness of a unit (neuron).

    @note
    """
    sparsns = ((np.mean(X, axis=0))**2 / (np.mean(X**2, axis=0) + 1e-8))
    return 1.0 - sparsns


def screePCA(X: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Estimate the explained variance ratio for the input matrix X by applying a
    principal component analysis (PCA).

    @note Standarise the data before applying the PCA.
    """
    X = (X - X.mean()) / X.std()
    pca = PCA().fit(X)
    return pca.n_components_, pca.explained_variance_ratio_
