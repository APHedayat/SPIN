"""The closed-form spiral experiment (paper Section 1.2-1.3).

A rank-2 ROM is built from two snapshots of the exact spiral and advanced inside
its initial plane. Before a single out-of-span correction arrives, the ROM
streams its own intermediate predictions through ``s`` in-span iSVD updates. We
then compare how the *same* correction is absorbed by:

* the **baseline adaptive** ROM (correction applied to the original spectrum), vs
* the **SPIN** ROM (correction applied after in-span preconditioning).

The whole experiment is closed form (no time-integration error), so every
difference is caused by the in-span preconditioning of the spectrum.
"""

from __future__ import annotations

import numpy as np

from ...diagnostics import correction_error, plane_angle_deg, residual_capture
from .fom import SpiralModel


def build_spiral_experiment(alpha=0.4, dt=0.35, t0=4.0, gamma_in=0.1, n_inspan=4):
    """Construct the rank-2 spiral ROM and run the in-span + single-correction study.

    Parameters
    ----------
    alpha, dt, t0 : float
        Spiral parameters (paper defaults reproduce Fig. 2/3 exactly).
    gamma_in : float
        In-span forgetting factor. The paper uses an aggressive ``0.1`` here so
        the spectral contraction is clearly visible.
    n_inspan : int
        Number of in-span updates streamed before the correction (paper: 4).

    Returns
    -------
    dict
        All quantities needed for the figures and the reported numbers, including
        per-model ``capture``, ``plane`` (deg), ``correction_error``, the in-plane
        competition (``lam_out/lam_in``, ``eta_out/eta_in``), the bases/spectra,
        and the in-span ellipse sequence.
    """
    spiral = SpiralModel(alpha=alpha, dt=dt, t0=t0)
    G, c = spiral.G, spiral.c

    # exact FOM states q^0 ... q^6
    times, x_FOM = spiral.trajectory(6)

    # rank-2 ROM from snapshots q^1, q^2
    U_init, S_init, _ = np.linalg.svd(x_FOM[:, [1, 2]], full_matrices=False)
    Phi0 = U_init[:, :2]
    Sigma0 = S_init[:2].copy()

    # propagate the ROM inside S_0 (Galerkin on the exact discrete flow)
    Ar = Phi0.T @ G @ Phi0
    cr = Phi0.T @ c
    a_ROM = {2: Phi0.T @ x_FOM[:, 2]}
    for k in range(2, 5):
        a_ROM[k + 1] = Ar @ a_ROM[k] + cr
    u_ROM = {k: Phi0 @ a_ROM[k] for k in a_ROM}

    # in-span preconditioning: covariance recursion C <- gamma^2 C + a a^T
    coeff_seq = [a_ROM[k] for k in [2, 3, 4, 5]][:n_inspan]
    Phis, spectra, covs = _inspan_pre_basis(Phi0, Sigma0, coeff_seq, gamma_in)
    Phi_in_pre = Phis[-1].copy()
    Sigma_in_pre = spectra[-1]

    # single out-of-span correction: one exact FOM step from the ROM state
    u_corr = G @ u_ROM[5] + c
    r_corr = u_corr - Phi0 @ (Phi0.T @ u_corr)
    rho = float(np.linalg.norm(r_corr))
    r_hat = r_corr / rho

    # in-plane coefficients in both pre-update bases
    alpha_out = Phi0.T @ u_corr
    alpha_in = Phi_in_pre.T @ u_corr

    # rank-2 local competition (paper convention gamma_out = 1 for the spiral)
    C_out = np.diag(Sigma0 ** 2) + np.outer(alpha_out, alpha_out)
    C_in = np.diag(Sigma_in_pre ** 2) + np.outer(alpha_in, alpha_in)
    lam_out = float(np.sort(np.linalg.eigvalsh(C_out))[0])
    lam_in = float(np.sort(np.linalg.eigvalsh(C_in))[0])
    eta_out = rho ** 2 / lam_out
    eta_in = rho ** 2 / lam_in

    # apply the same correction to both pre-update states (iSVD, gamma=1)
    Phi_out_post, _ = _isvd_closed_form(Phi0, Sigma0, u_corr, gamma=1.0)
    Phi_in_post, _ = _isvd_closed_form(Phi_in_pre, Sigma_in_pre, u_corr, gamma=1.0)

    return {
        "spiral": spiral, "times": times, "x_FOM": x_FOM,
        "Phi0": Phi0, "Sigma0": Sigma0, "a_ROM": a_ROM, "u_ROM": u_ROM,
        "Phi_in_pre": Phi_in_pre, "Sigma_in_pre": Sigma_in_pre,
        "Phis": Phis, "spectra": spectra, "covs": covs,
        "u_corr": u_corr, "r_hat": r_hat, "rho": rho,
        "alpha_out": alpha_out, "alpha_in": alpha_in,
        "lam_out": lam_out, "lam_in": lam_in, "eta_out": eta_out, "eta_in": eta_in,
        "baseline": {
            "Phi_pre": Phi0, "Phi_post": Phi_out_post,
            "capture": residual_capture(Phi_out_post, r_hat),
            "plane": plane_angle_deg(Phi0, Phi_out_post),
            "correction_error": correction_error(Phi_out_post, u_corr),
            "lambda_r": lam_out, "eta": eta_out,
        },
        "spin": {
            "Phi_pre": Phi_in_pre, "Phi_post": Phi_in_post,
            "capture": residual_capture(Phi_in_post, r_hat),
            "plane": plane_angle_deg(Phi0, Phi_in_post),
            "correction_error": correction_error(Phi_in_post, u_corr),
            "lambda_r": lam_in, "eta": eta_in,
        },
    }


def _inspan_pre_basis(Phi_init, Sigma_init, coeff_seq, gamma):
    """In-span covariance recursion; returns bases, spectra, covariances per step."""
    Cmat = np.diag(Sigma_init ** 2).astype(float)
    spectra = [np.sqrt(np.sort(np.linalg.eigvalsh(Cmat))[::-1])]
    Phis = [Phi_init.copy()]
    covs = [Cmat.copy()]
    for a in coeff_seq:
        Cmat = (gamma ** 2) * Cmat + np.outer(a, a)
        lam, V = np.linalg.eigh(Cmat)
        order = np.argsort(lam)[::-1]
        Sig = np.sqrt(np.clip(lam[order], 0, None))
        Phi_ = Phi_init @ V[:, order]
        for j in range(Phi_.shape[1]):                  # sign-align columns
            if np.dot(Phi_[:, j], Phis[-1][:, j]) < 0:
                Phi_[:, j] *= -1
        Phis.append(Phi_)
        spectra.append(Sig)
        covs.append(Cmat.copy())
    return Phis, spectra, covs


def _isvd_closed_form(Phi, Sigma, y_new, gamma=1.0):
    """Closed-form rank-r iSVD update used by the spiral study (Brand-type)."""
    Phi = np.asarray(Phi)
    Sigma = np.asarray(Sigma)
    y_new = np.asarray(y_new)
    r = Phi.shape[1]
    alpha_ = Phi.T @ y_new
    q = y_new - Phi @ alpha_
    rho = np.linalg.norm(q)
    q_hat = q / rho if rho > 1e-14 else np.zeros_like(q)
    K = np.zeros((r + 1, r + 1))
    K[:r, :r] = np.diag(gamma * Sigma)
    K[:r, r] = alpha_
    K[r, r] = rho
    U_K, S_K, _ = np.linalg.svd(K, full_matrices=False)
    Phi_new = np.column_stack([Phi, q_hat]) @ U_K[:, :r]
    return Phi_new, S_K[:r]
