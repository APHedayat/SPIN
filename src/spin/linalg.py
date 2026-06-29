"""Offline linear algebra: POD basis construction and QDEIM sampling."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def compute_pod_basis(snapshots, r):
    """Proper Orthogonal Decomposition basis from a snapshot matrix.

    Parameters
    ----------
    snapshots : ndarray, shape (N, K)
        Column-wise snapshot matrix (already centered/scaled as desired).
    r : int
        Number of POD modes to keep.

    Returns
    -------
    Phi : ndarray, shape (N, r)
        The first ``r`` left singular vectors (the POD basis).
    sigma : ndarray, shape (r,)
        The corresponding singular values.
    """
    U, S, _ = np.linalg.svd(snapshots, full_matrices=False)
    return U[:, :r], S[:r]


def qdeim(basis, n_sensors):
    """QDEIM sample indices via a column-pivoted QR of ``basis.T``.

    Parameters
    ----------
    basis : ndarray, shape (N, r)
        Orthonormal basis whose rows are sampled.
    n_sensors : int
        Number of sample (sensor) points ``m`` to select.

    Returns
    -------
    indices : ndarray of int, shape (n_sensors,)
        Selected row indices.
    """
    _, _, P = qr(basis.T, pivoting=True)
    return P[:n_sensors]
