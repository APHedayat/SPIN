"""Incremental SVD (iSVD) with forgetting.

This single routine is the engine behind *both* SPIN channels:

* the **in-span** update streams the ROM's own prediction through ``isvd`` with
  forgetting factor ``gamma_in`` (the snapshot lies inside the current span, so
  the residual is zero and only the spectrum / in-plane rotation change), and
* the **out-of-span** update streams an occasional full-order correction
  snapshot through ``isvd`` with forgetting factor ``gamma_out`` (the snapshot
  has a component outside the span, so the subspace genuinely moves).

The math is the truncated iSVD-with-forgetting of Brand-type updates
(see Methods, Eqs. 20-23 of the paper).
"""

from __future__ import annotations

import numpy as np


def isvd(V_old, S_old, u_new, forgetting_factor=0.0, r=None, tol=1e-12,
         orthonormalize=True):
    r"""One rank-:math:`r` truncated iSVD update with forgetting.

    Maintains a rank-:math:`r` approximation of an exponentially weighted
    snapshot history. Given the current basis :math:`\Phi` and singular values
    :math:`\Sigma`, and a new (preprocessed) snapshot :math:`y`, it forms the
    small core matrix

    .. math::
        K = \begin{bmatrix} \gamma\,\Sigma & w \\ 0 & \rho \end{bmatrix},

    where :math:`w = \Phi^\top y` is the in-span coefficient, :math:`\rho` is the
    orthogonal-residual norm, and :math:`\gamma` is the forgetting factor, then
    takes the SVD of :math:`K` to rotate/reweight (and possibly extend) the basis.

    Parameters
    ----------
    V_old : ndarray, shape (N, r)
        Current orthonormal basis :math:`\Phi`.
    S_old : ndarray, shape (r,)
        Current singular values :math:`\Sigma`.
    u_new : ndarray, shape (N,)
        Incoming (already-preprocessed) snapshot :math:`y`.
    forgetting_factor : float in [0, 1]
        :math:`\gamma`. ``1`` keeps all past history, ``0`` forgets it entirely.
    r : int or None
        Target rank after the update. ``None`` keeps the current rank.
    tol : float
        Residual norm below which the snapshot is treated as exactly in-span
        (no new direction is appended).
    orthonormalize : bool
        Apply a QR clean-up to the returned basis. The paper uses
        ``True`` for out-of-span updates and ``False`` for in-span updates
        (the in-span update is a pure in-plane rotation and stays orthonormal).

    Returns
    -------
    V_new : ndarray, shape (N, r)
        Updated basis.
    S_new : ndarray, shape (r,)
        Updated singular values.
    """
    U = V_old
    s = np.asarray(S_old).ravel()
    N, Kdim = U.shape
    if r is None:
        r = Kdim

    # In-span coefficient w = Phi^T y and orthogonal residual q = y - Phi w.
    # A least-squares solve is used instead of U.T @ y so the routine is robust
    # even if U has drifted slightly from perfect orthonormality.
    y = u_new.reshape(-1, 1)
    p, _, _, _ = np.linalg.lstsq(U, y, rcond=None)
    p = p.reshape(-1, 1)
    q = y - (U @ p)
    qnorm = float(np.linalg.norm(q))

    if qnorm <= tol:
        # in-span snapshot: zero residual, no new direction
        q = np.zeros((N, 1))
        qnorm = 0.0
        u_perp = np.zeros((N, 1))
    else:
        u_perp = q / qnorm

    # small (r+1) x (r+1) core matrix with forgetting applied to the spectrum
    Kcore = np.zeros((Kdim + 1, Kdim + 1), dtype=U.dtype)
    Kcore[:Kdim, :Kdim] = np.diag(forgetting_factor * s[:Kdim])
    Kcore[:Kdim, Kdim] = p.ravel()
    Kcore[Kdim, Kdim] = qnorm

    Ubar, s_new, _ = np.linalg.svd(Kcore, full_matrices=False)

    # rotate the augmented basis and truncate back to rank r
    U_aug = np.hstack([U, u_perp])           # (N, r+1)
    U_new_full = U_aug @ Ubar                # (N, r+1)

    k_new = min(r, U_new_full.shape[1])
    V_new = U_new_full[:, :k_new]
    S_new = s_new[:k_new].copy()

    if orthonormalize:
        V_new, _ = np.linalg.qr(V_new, mode="reduced")

    return V_new, S_new
