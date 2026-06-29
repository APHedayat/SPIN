"""Static and adaptive LSPG ROMs for the 1D viscous Burgers equation.

These concrete ROMs show how to couple a problem-specific hyper-reduction
(residual and Jacobian at the QDEIM sample points) to the generic core
machinery. When writing a ROM for a new equation, mirror this file: implement
``residual_sample`` and ``jacobian_sample`` in a mixin, then combine it with
:class:`~spin.core.rom_base.LSPGROMBase` (static) and
:class:`~spin.core.adaptive_rom.SpinROMBase` (static / baseline / SPIN).
"""

from __future__ import annotations

import numpy as np

from ...core.adaptive_rom import SpinROMBase
from ...core.rom_base import LSPGROMBase
from .fom import BurgersSolver


class _BurgersROMMixin:
    """Problem-specific hyper-reduced residual / Jacobian for Burgers.

    Expects ``self`` to carry ``Phi, p_inds, dt, nu, dx`` and the cached sample
    neighbour bases ``_Phi_p, _Phi_up, _Phi_down`` (set by ``update_sampling``).
    """

    def residual_sample(self, a, a_old):
        u = self.Phi @ a
        u_p = u[self.p_inds]
        u_up = u[self._ip]
        u_dn = u[self._im]
        upos = u_p >= 0.0
        conv = np.where(upos, u_p * (u_p - u_dn) / self.dx,
                        u_p * (u_up - u_p) / self.dx)
        uxx = (u_up - 2 * u_p + u_dn) / (self.dx ** 2)
        Fp = conv - self.nu * uxx
        return self._Phi_p @ (a - a_old) + self.dt * Fp

    def jacobian_sample(self, a):
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


class BurgersLSPGROM(_BurgersROMMixin, LSPGROMBase):
    """Static LSPG + QDEIM ROM for periodic 1D viscous Burgers."""

    def __init__(self, Phi, p_inds, dt, nu, dx):
        self.nu = nu
        self.dx = dx
        super().__init__(Phi, p_inds, dt)


class BurgersSpinROM(_BurgersROMMixin, SpinROMBase):
    """Adaptive LSPG + QDEIM ROM for Burgers (static / baseline / SPIN modes).

    Reuses the Burgers residual/Jacobian (via ``_BurgersROMMixin``) and inherits
    the generic adaptation loop from
    :class:`~spin.core.adaptive_rom.SpinROMBase`.

    Parameters
    ----------
    Phi, sigma, p_inds, dt : see :class:`~spin.core.adaptive_rom.SpinROMBase`.
    nu, dx : float
        Burgers viscosity and grid spacing.
    zs, mode, gamma_in, gamma_out : see :class:`~spin.core.adaptive_rom.SpinROMBase`.
    fom_solver : BurgersSolver, optional
        FOM used for the out-of-span correction query. If ``None``, a fine-step
        :class:`BurgersSolver` (``dt`` matching the ROM) is built automatically.
    """

    def __init__(self, Phi, sigma, p_inds, dt, nu, dx, zs,
                 mode="spin", gamma_in=1.0, gamma_out=0.25, fom_solver=None):
        self.nu = nu
        self.dx = dx
        if fom_solver is None:
            fom_solver = BurgersSolver(Nx=Phi.shape[0], L=dx * Phi.shape[0],
                                       nu=nu, dt=dt)
        super().__init__(Phi, sigma, p_inds, dt, fom_solver, zs,
                         mode=mode, gamma_in=gamma_in, gamma_out=gamma_out)
