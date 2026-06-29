"""Hyper-reduction sampling: QDEIM index selection.

The LSPG ROM evaluates its residual and Jacobian at only ``m`` sampled rows.
QDEIM picks those rows via a column-pivoted QR of the basis transpose.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def qdeim(basis: np.ndarray, n_sensors: int) -> np.ndarray:
    """QR-pivoting-based DEIM (QDEIM) index selection.

    Parameters
    ----------
    basis : (N, r) ndarray
        Basis whose row indices are to be sampled (e.g. the POD basis).
    n_sensors : int
        Number of sampling indices ``m`` to return. Must satisfy
        ``n_sensors >= r`` for a well-posed hyper-reduced problem.

    Returns
    -------
    inds : (n_sensors,) ndarray of int
        Row indices of ``basis`` selected by QDEIM.
    """
    _, _, P = qr(basis.T, pivoting=True)
    return P[:n_sensors]
