"""Adaptive reduced-order models: static, baseline adaptive, and SPIN.

The single driver here is a hyper-reduced LSPG-QDEIM ROM whose online basis can
be adapted through two channels (see Algorithm 1 in the paper):

* **out-of-span** updates: every ``zs`` steps a full-order operator query
  produces a correction snapshot, streamed through an iSVD update with forgetting
  ``gamma_out``. This is the only channel a *baseline adaptive* ROM uses.
* **in-span** updates: on every non-correction step, the ROM's *own* prediction
  is streamed through an iSVD update with forgetting ``gamma_in``. This is the
  extra channel that turns the baseline adaptive ROM into **SPIN**.

Three models are obtained from the same driver via the ``mode`` argument:

==================  ===================  ==================
``mode``            in-span updates      out-of-span updates
==================  ===================  ==================
``"static"``        no                   no
``"baseline"``      no                   yes
``"spin"``          yes                  yes
==================  ===================  ==================

``AdaptiveLSPGROM`` is abstract: a concrete subclass supplies the sampled
(hyper-reduced) residual and Jacobian for its PDE. :class:`BurgersROM` and
:class:`FisherKPPROM` are provided.
"""

from __future__ import annotations

import numpy as np

from .isvd import isvd
from .linalg import qdeim


class AdaptiveLSPGROM:
    """Hyper-reduced LSPG-QDEIM ROM with optional in-span / out-of-span adaptation.

    Parameters
    ----------
    Phi : ndarray, shape (N, r)
        Initial orthonormal POD basis.
    sigma : ndarray, shape (r,)
        Initial singular values associated with ``Phi``.
    p_inds : array-like of int, shape (m,)
        Initial QDEIM sample indices.
    fom_solver : object
        A FOM solver instance (e.g. :class:`spin.models.BurgersSolver`) used for
        the out-of-span correction queries. Must expose ``step(u, ...)``.
    dt : float
        Time step (must match ``fom_solver.dt``).
    zs : int
        Out-of-span correction interval (a full-order query every ``zs`` steps).
    mode : {"static", "baseline", "spin"}
        Adaptation mode (see module docstring).
    gamma_in : float
        In-span iSVD forgetting factor (used only by ``"spin"``).
    gamma_out : float
        Out-of-span iSVD forgetting factor (used by ``"baseline"`` and ``"spin"``).
    """

    def __init__(self, Phi, sigma, p_inds, fom_solver, dt, zs,
                 mode="spin", gamma_in=1.0, gamma_out=0.25):
        if mode not in ("static", "baseline", "spin"):
            raise ValueError("mode must be 'static', 'baseline', or 'spin'.")
        self.Phi = np.array(Phi, dtype=float)
        self.Sigma = np.array(sigma, dtype=float)
        self.p_inds = np.asarray(p_inds, dtype=int)
        self.fom_solver = fom_solver
        self.dt = dt
        self.zs = zs
        self.mode = mode
        self.gamma_in = gamma_in
        self.gamma_out = gamma_out

        self.N, self.r = self.Phi.shape
        self.dx = fom_solver.dx
        self._refresh_sampling_cache()

    # -- sampled operator (PDE-specific) -------------------------------------
    def _residual_sample(self, a, a_old):
        """Return (sampled LSPG residual, sampled spatial operator F) at p_inds.

        The sampled residual is ``Phi_p @ (a - a_old) + dt * R_F`` where ``R_F``
        is the discrete spatial-operator contribution at the sample points (its
        sign is folded in by the subclass so the base ``step`` is uniform).
        """
        raise NotImplementedError

    def _W_sample(self, a):
        """Return the sampled LSPG test operator ``W = Phi_p + dt * dR_F/da``."""
        raise NotImplementedError

    def _refresh_sampling_cache(self):
        """Cache sample-point neighbour indices and the QDEIM pseudoinverse M."""
        self.p_inds = np.asarray(self.p_inds, dtype=int)
        self._ip = (self.p_inds + 1) % self.N
        self._im = (self.p_inds - 1) % self.N
        self._Phi_p = self.Phi[self.p_inds, :]
        self._Phi_up = self.Phi[self._ip, :]
        self._Phi_down = self.Phi[self._im, :]
        self.M = np.linalg.pinv(self._Phi_p)

    # Whether the inner LSPG solve uses a pseudoinverse (True) or a direct solve
    # (False). On the square m = r QDEIM system both are equivalent up to
    # round-off; subclasses pick the convention to match their reference run.
    _use_pinv = True

    # -- one implicit LSPG step (Gauss-Newton on the sampled residual) -------
    def step(self, a_old, tol=1e-20, max_iter=10):
        a = a_old.copy()
        F_samp = None
        for _ in range(max_iter):
            R_samp, F_samp = self._residual_sample(a, a_old)
            if np.linalg.norm(R_samp) < tol:
                break
            W_samp = self._W_sample(a)
            LHS = self.M @ W_samp
            RHS = -self.M @ R_samp
            if self._use_pinv:
                a = a + np.linalg.pinv(LHS) @ RHS
            else:
                a = a + np.linalg.solve(LHS, RHS)
        return a, F_samp

    # -- the SPIN online loop (Algorithm 1) ----------------------------------
    def simulate(self, a0, n_steps, tol=1e-20, max_iter=10, verbose=False,
                 record=False):
        """Run the adaptive ROM for ``n_steps`` and return the trajectory.

        Parameters
        ----------
        record : bool
            If ``True``, also return a ``history`` dict logging, per step, the
            basis ``Phi``, singular values ``sigma``, the update kind
            (``"none"``/``"in"``/``"out"``), the per-step ``basis_reorientation``
            and ``subspace_change`` diagnostics. Used for the Fig. 6 panels.

        Returns
        -------
        soln : ndarray, shape (N, n_steps+1)
            Reconstructed full-order ROM states over time.
        history : dict
            Only returned when ``record=True``.
        """
        from .diagnostics import basis_reorientation, subspace_change

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
            a_new, F_samp = self.step(a_old=a, tol=tol, max_iter=max_iter)
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
                self._refresh_sampling_cache()
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
                self._refresh_sampling_cache()
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
            for k in ("step", "kind", "basis_reorientation", "subspace_change"):
                history[k] = np.array(history[k]) if k != "kind" else history[k]
            history["sigma"] = np.array(history["sigma"])
            return soln, history
        return soln


class BurgersROM(AdaptiveLSPGROM):
    """Adaptive LSPG-QDEIM ROM for viscous Burgers."""

    def __init__(self, *args, nu, **kwargs):
        self.nu = nu
        super().__init__(*args, **kwargs)

    def _residual_sample(self, a, a_old):
        u = self.Phi @ a
        u_p = u[self.p_inds]
        u_up = u[self._ip]
        u_dn = u[self._im]
        upos = u_p >= 0.0
        conv = np.where(upos, u_p * (u_p - u_dn) / self.dx,
                        u_p * (u_up - u_p) / self.dx)
        uxx = (u_up - 2 * u_p + u_dn) / (self.dx ** 2)
        Fp = conv - self.nu * uxx
        return self._Phi_p @ (a - a_old) + self.dt * Fp, Fp

    def _W_sample(self, a):
        u = self.Phi @ a
        u_p = u[self.p_inds]
        u_up = u[self._ip]
        u_dn = u[self._im]
        upos = u_p >= 0.0
        diag0 = np.where(upos, (2 * u_p - u_dn) / self.dx,
                         (u_up - 2 * u_p) / self.dx) + 2 * self.nu / self.dx ** 2
        diagp = np.where(upos, 0.0, u_p / self.dx) - self.nu / self.dx ** 2
        diagm = np.where(upos, -u_p / self.dx, 0.0) - self.nu / self.dx ** 2
        JfPhi = (diag0[:, None] * self._Phi_p
                 + diagm[:, None] * self._Phi_down
                 + diagp[:, None] * self._Phi_up)
        return self._Phi_p + self.dt * JfPhi


class FisherKPPROM(AdaptiveLSPGROM):
    """Adaptive LSPG-QDEIM ROM for Fisher-KPP reaction-diffusion."""

    _use_pinv = False   # reference Fisher-KPP run uses a direct solve

    def __init__(self, *args, D, alpha, **kwargs):
        self.D = D
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def _residual_sample(self, a, a_old):
        # full reconstruction then sample (matches the reference run exactly;
        # algebraically equal to Phi_p @ (a - a_old) - dt * Fp)
        u = self.Phi @ a
        u_old = self.Phi @ a_old
        up = np.roll(u, -1)
        um = np.roll(u, +1)
        g = self.D * (up - 2.0 * u + um) / (self.dx ** 2) + self.alpha * u * (1.0 - u)
        res_full = u - u_old - self.dt * g
        return res_full[self.p_inds], g[self.p_inds]

    def _W_sample(self, a):
        u = self.Phi @ a
        u_p = u[self.p_inds]
        diff = self.D * (self._Phi_up - 2.0 * self._Phi_p + self._Phi_down) / (self.dx ** 2)
        react = (self.alpha * (1.0 - 2.0 * u_p))[:, None] * self._Phi_p
        JfPhi = diff + react
        return self._Phi_p - self.dt * JfPhi
