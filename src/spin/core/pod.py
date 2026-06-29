"""Proper Orthogonal Decomposition (POD) for the offline basis.

The reduced basis that initializes every static, baseline-adaptive, and SPIN ROM
is the leading-mode subspace of a thin SVD of the offline snapshot matrix.
"""

from __future__ import annotations

import numpy as np


def compute_pod_basis(snapshots: np.ndarray, r: int):
    """Compute the leading ``r`` POD modes of a snapshot matrix.

    Parameters
    ----------
    snapshots : (N, K) ndarray
        Column-wise snapshot matrix of the full state.
    r : int
        Target reduced dimension.

    Returns
    -------
    Phi : (N, r) ndarray
        Leading ``r`` left singular vectors (the POD basis).
    sigma : (r,) ndarray
        Associated singular values.
    """
    U, S, _ = np.linalg.svd(snapshots, full_matrices=False)
    return U[:, :r], S[:r]
