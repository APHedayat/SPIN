"""Diagnostics that quantify how in-span learning reshapes the basis.

Two groups, matching the paper:

* **single-correction diagnostics** (Fig. 3): how well one out-of-span
  correction is absorbed -- residual capture, plane-change angle, correction
  error, and the residual-to-weakest-mode ratio ``eta``.
* **trajectory diagnostics** (Fig. 6): how the basis evolves along the online
  run -- singular-value history, basis reorientation per step, and genuine
  subspace motion.
"""

from __future__ import annotations

import numpy as np


# ======================================================================
# Single-correction diagnostics (Fig. 3)
# ======================================================================
def residual_capture(Phi_post, r_hat):
    r"""Residual capture :math:`\chi = \|\Phi_{\rm post}^\top \hat r\|_2`.

    Fraction of the unit new-information direction :math:`\hat r` that the
    updated basis can represent.
    """
    return float(np.linalg.norm(Phi_post.T @ r_hat))


def principal_angles(P, Q):
    """Principal angles (radians, ascending) between two orthonormal bases."""
    s = np.clip(np.linalg.svd(P.T @ Q, compute_uv=False), -1.0, 1.0)
    return np.arccos(s)[::-1]


def plane_angle_deg(P, Q):
    r"""Largest principal angle (degrees) between ``span(P)`` and ``span(Q)``."""
    return float(np.degrees(principal_angles(P, Q).max()))


def correction_error(Phi_post, q_corr):
    r"""Correction error :math:`\|q_{\rm corr} - \Phi_{\rm post}\Phi_{\rm post}^\top q_{\rm corr}\|_2`.

    The part of the correction snapshot the *updated* subspace cannot represent.
    """
    proj = Phi_post @ (Phi_post.T @ q_corr)
    return float(np.linalg.norm(q_corr - proj))


def weakest_retained_mode(Phi_pre, sigma_pre, q_corr, gamma_out=1.0):
    r"""Weakest retained eigenvalue :math:`\lambda_r(C_{\rm corr})` of the local core.

    With in-plane coefficients :math:`\alpha = \Phi_{\rm pre}^\top q_{\rm corr}`,
    the in-plane block of the iSVD core covariance is
    :math:`C_{\rm corr} = \gamma_{\rm out}^2 \Sigma_{\rm pre}^2 + \alpha\alpha^\top`.
    Its smallest eigenvalue measures the weakest competitor for a rank-:math:`r` slot.
    """
    alpha = Phi_pre.T @ q_corr
    C = (gamma_out ** 2) * np.diag(np.asarray(sigma_pre) ** 2) + np.outer(alpha, alpha)
    return float(np.sort(np.linalg.eigvalsh(C))[0])


def eta_ratio(q_corr, Phi_pre, sigma_pre, gamma_out=1.0):
    r"""Residual-to-weakest-mode ratio :math:`\eta = \rho^2 / \lambda_r(C_{\rm corr})`.

    :math:`\eta \ll 1` means the incoming residual is weak relative to the
    existing core content; :math:`\eta \gg 1` means it can compete for one of the
    retained rank-:math:`r` slots.
    """
    r = q_corr - Phi_pre @ (Phi_pre.T @ q_corr)
    rho_sq = float(np.linalg.norm(r) ** 2)
    lam_r = weakest_retained_mode(Phi_pre, sigma_pre, q_corr, gamma_out)
    return rho_sq / lam_r


# ======================================================================
# Trajectory diagnostics (Fig. 6)
# ======================================================================
def basis_reorientation(Phi_old, Phi_new):
    r"""Basis reorientation :math:`\|I - (\Phi^{n+1})^\top \Phi^n\|_F`.

    Nonzero even when the two bases span the same subspace -- it detects pure
    in-plane rotation, which is the in-span signature.
    """
    M = Phi_new.T @ Phi_old
    return float(np.linalg.norm(np.eye(M.shape[0]) - M, "fro"))


def subspace_change(Phi_old, Phi_new):
    r"""Genuine subspace motion :math:`\max_i \sin\theta_i`.

    The principal-angle sine between consecutive subspaces. Zero (to numerical
    precision) when the subspace is unchanged even if the basis rotated inside
    it -- so a nonzero ``basis_reorientation`` with a zero ``subspace_change`` is
    the fingerprint of a pure in-span update.
    """
    s = np.clip(np.linalg.svd(Phi_old.T @ Phi_new, compute_uv=False), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - np.min(s) ** 2)))


def relative_l2_error(soln, ref):
    r"""Per-time-step relative :math:`L_2` error :math:`\|u^n - u^n_{\rm ref}\|/\|u^n_{\rm ref}\|`.

    ``soln`` and ``ref`` are (N, T) arrays.
    """
    num = np.linalg.norm(soln - ref, axis=0)
    den = np.maximum(np.linalg.norm(ref, axis=0), 1e-15)
    return num / den
