"""Generic adaptive LSPG ROM with in-span and out-of-span learning.

:class:`SpinROMBase` implements Algorithm 1 of the paper. The online loop is
entirely problem-agnostic: it advances the ROM, and depending on the ``mode`` it

* performs an **out-of-span** update every ``zs`` steps (a single full-order
  operator query, streamed through an iSVD with forgetting ``gamma_out``), and/or
* performs an **in-span** update on every other step (the ROM's *own* prediction,
  streamed through an iSVD with forgetting ``gamma_in``).

The three models in the paper come from the same class via ``mode``:

================  ===============  ==================  ==============================
``mode``          in-span updates  out-of-span         what it is
================  ===============  ==================  ==============================
``"static"``      no               no                  fixed POD basis
``"baseline"``    no               yes                 baseline adaptive ROM
``"spin"``        yes              yes                 **SPIN** (this work)
================  ===============  ==================  ==============================

A concrete subclass supplies the equation through the two hooks inherited from
:class:`~spin.core.rom_base.LSPGROMBase` (``residual_sample``,
``jacobian_sample``) and a ``fom_solver`` exposing ``.step(u, tol, max_iter,
verbose)`` for the out-of-span correction query. See
:mod:`spin.problems.burgers` for a worked template.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .basis_adaptation import isvd
from .rom_base import LSPGROMBase
from .sampling import qdeim

MODES = ("static", "baseline", "spin")


class SpinROMBase(LSPGROMBase):
    """Generic adaptive LSPG + QDEIM ROM with in-span / out-of-span learning.

    Parameters
    ----------
    Phi : (N, r) ndarray
        Initial POD basis.
    sigma : (r,) ndarray
        Singular values associated with ``Phi``.
    p_inds : sequence of int, length ``m``
        Initial QDEIM sampling indices.
    dt : float
        ROM time step.
    fom_solver : object
        A FOM solver exposing ``.step(u, tol, max_iter, verbose) -> u_new`` used
        for the out-of-span correction query (one fine FOM step from the ROM
        state).
    zs : int
        Out-of-span correction interval (a full-order query every ``zs`` steps).
    mode : {"static", "baseline", "spin"}
        Adaptation mode (see the table above).
    gamma_in : float
        In-span iSVD forgetting factor (used only by ``"spin"``).
    gamma_out : float
        Out-of-span iSVD forgetting factor (used by ``"baseline"`` and ``"spin"``).
    """

    def __init__(self, Phi, sigma, p_inds, dt, fom_solver, zs,
                 mode="spin", gamma_in=1.0, gamma_out=0.25):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}.")
        super().__init__(Phi, p_inds, dt)
        self.Sigma = np.asarray(sigma, dtype=float).copy()
        self.fom_solver = fom_solver
        self.zs = int(zs)
        self.mode = mode
        self.gamma_in = gamma_in
        self.gamma_out = gamma_out

    def simulate(self, a0, n_steps, tol=1e-8, max_iter=10, verbose=False,
                 record=False):
        """Run the adaptive ROM (Algorithm 1) and return the full-state trajectory.

        Parameters
        ----------
        a0 : (r,) ndarray
            Initial reduced coordinates.
        n_steps : int
            Number of ROM time steps.
        record : bool
            If ``True``, also return a ``history`` dict logging, per step, the
            basis ``Phi``, singular values ``sigma``, the update kind
            (``"none"`` / ``"in"`` / ``"out"``), and the per-step diagnostics
            ``basis_reorientation`` and ``subspace_change`` (used for the Fig. 6
            panels).

        Returns
        -------
        soln : (N, n_steps+1) ndarray
            Reconstructed full-order ROM states over time.
        history : dict
            Only when ``record=True``.
        """
        from ..diagnostics import basis_reorientation, subspace_change

        u0 = self.Phi @ a0
        soln = [u0.copy()]
        a = a0.copy()
        u_old = u0.copy()                 # last full-order correction state

        do_in = (self.mode == "spin")
        do_out = (self.mode in ("baseline", "spin"))

        history = {"step": [0], "kind": ["init"],
                   "sigma": [self.Sigma.copy()], "Phi": [self.Phi.copy()],
                   "basis_reorientation": [0.0], "subspace_change": [0.0]}

        for step in range(1, n_steps + 1):
            if verbose:
                print(f"step {step}/{n_steps}")
            Phi_before = self.Phi.copy()
            a_new = self.step(a_old=a, tol=tol, max_iter=max_iter)
            u_rom_pred = self.Phi @ a_new

            is_correction = (step % self.zs == 0)
            kind = "none"

            if do_in and not is_correction:
                # ---- in-span update: stream the ROM's own prediction --------
                # zero-residual snapshot -> pure in-plane rotation + reweighting
                self.Phi, self.Sigma = isvd(
                    V_old=self.Phi, S_old=self.Sigma, u_new=u_rom_pred,
                    forgetting_factor=self.gamma_in, r=None, tol=1e-16,
                    orthonormalize=False)
                self.update_sampling()
                a = self.Phi.T @ u_rom_pred
                u_old = u_rom_pred.copy()
                kind = "in"

            elif do_out and is_correction:
                # ---- out-of-span update: one full-order operator query -------
                u_corr = self.fom_solver.step(u_old, tol, max_iter, verbose=False)
                u_old = u_corr.copy()
                self.Phi, self.Sigma = isvd(
                    V_old=self.Phi, S_old=self.Sigma, u_new=u_corr,
                    forgetting_factor=self.gamma_out, r=None, tol=1e-16,
                    orthonormalize=True)
                self.p_inds = qdeim(self.Phi, self.p_inds.size)
                self.update_sampling()
                a = self.Phi.T @ u_rom_pred
                kind = "out"

            else:
                # ---- no adaptation (static, or non-correction baseline step) -
                a = a_new.copy()
                u_old = u_rom_pred.copy()

            soln.append(u_rom_pred.copy())

            if record:
                history["step"].append(step)
                history["kind"].append(kind)
                history["sigma"].append(self.Sigma.copy())
                history["Phi"].append(self.Phi.copy())
                history["basis_reorientation"].append(
                    basis_reorientation(Phi_before, self.Phi))
                history["subspace_change"].append(
                    subspace_change(Phi_before, self.Phi))

        soln = np.array(soln).T
        if record:
            history["step"] = np.array(history["step"])
            history["sigma"] = np.array(history["sigma"])
            history["basis_reorientation"] = np.array(history["basis_reorientation"])
            history["subspace_change"] = np.array(history["subspace_change"])
            return soln, history
        return soln
