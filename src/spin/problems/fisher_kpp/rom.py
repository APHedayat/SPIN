"""Static and adaptive LSPG ROMs for the 1D Fisher-KPP equation.

Mirrors :mod:`spin.problems.burgers.rom`: a problem-specific mixin provides the
hyper-reduced residual / Jacobian, then combines with the generic static and
adaptive cores.
"""

from __future__ import annotations

import numpy as np

from ...core.adaptive_rom import SpinROMBase
from ...core.rom_base import LSPGROMBase
from .fom import FisherKPPSolver


class _FisherKPPROMMixin:
    """Problem-specific hyper-reduced residual / Jacobian for Fisher-KPP.

    Expects ``self`` to carry ``Phi, p_inds, dt, D, alpha, dx`` and the cached
    sample neighbour bases (set by ``update_sampling``).
    """

    # the reference Fisher-KPP run uses a direct solve in the inner LSPG step
    _use_pinv = False

    def residual_sample(self, a, a_old):
        # full reconstruction then sample (the reference convention; algebraically
        # equal to Phi_p @ (a - a_old) - dt * Fp)
        u = self.Phi @ a
        u_old = self.Phi @ a_old
        up = np.roll(u, -1)
        um = np.roll(u, +1)
        g = self.D * (up - 2.0 * u + um) / (self.dx ** 2) + self.alpha * u * (1.0 - u)
        res_full = u - u_old - self.dt * g
        return res_full[self.p_inds]

    def jacobian_sample(self, a):
        u = self.Phi @ a
        u_p = u[self.p_inds]
        diff = self.D * (self._Phi_up - 2.0 * self._Phi_p + self._Phi_down) / (self.dx ** 2)
        react = (self.alpha * (1.0 - 2.0 * u_p))[:, None] * self._Phi_p
        JfPhi = diff + react
        return self._Phi_p - self.dt * JfPhi


class FisherKPPLSPGROM(_FisherKPPROMMixin, LSPGROMBase):
    """Static LSPG + QDEIM ROM for periodic 1D Fisher-KPP."""

    def __init__(self, Phi, p_inds, dt, D, alpha, dx):
        self.D = D
        self.alpha = alpha
        self.dx = dx
        super().__init__(Phi, p_inds, dt)


class FisherKPPSpinROM(_FisherKPPROMMixin, SpinROMBase):
    """Adaptive LSPG + QDEIM ROM for Fisher-KPP (static / baseline / SPIN modes).

    Parameters
    ----------
    Phi, sigma, p_inds, dt : see :class:`~spin.core.adaptive_rom.SpinROMBase`.
    D, alpha, dx : float
        Fisher-KPP diffusivity, reaction rate, and grid spacing.
    zs, mode, gamma_in, gamma_out : see :class:`~spin.core.adaptive_rom.SpinROMBase`.
    fom_solver : FisherKPPSolver, optional
        FOM for the out-of-span correction query. Built automatically if ``None``.
    """

    def __init__(self, Phi, sigma, p_inds, dt, D, alpha, dx, zs,
                 mode="spin", gamma_in=1.0, gamma_out=0.25, fom_solver=None):
        self.D = D
        self.alpha = alpha
        self.dx = dx
        if fom_solver is None:
            fom_solver = FisherKPPSolver(Nx=Phi.shape[0], L=dx * Phi.shape[0],
                                         D=D, alpha=alpha, dt=dt)
        super().__init__(Phi, sigma, p_inds, dt, fom_solver, zs,
                         mode=mode, gamma_in=gamma_in, gamma_out=gamma_out)
