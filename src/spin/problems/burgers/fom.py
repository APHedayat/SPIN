r"""Full-order model for the 1D viscous Burgers equation.

The equation

.. math::
    \partial_t u + u\,\partial_x u = \nu\,\partial_{xx} u

is discretized on a uniform periodic grid using a first-order upwind scheme for
the convective term and a second-order centered scheme for the diffusive term.
Time is integrated implicitly with backward Euler; each step is a Newton solve.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class BurgersSolver:
    """Implicit FOM solver for periodic 1D viscous Burgers.

    Parameters
    ----------
    Nx : int
        Number of spatial grid points.
    L : float
        Length of the periodic domain.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    """

    def __init__(self, Nx: int = 256, L: float = 1.0, nu: float = 1e-2, dt: float = 1e-3):
        self.Nx = Nx
        self.L = L
        self.nu = nu
        self.dt = dt
        self.dx = L / Nx
        self.x = np.linspace(0, L, Nx, endpoint=False)

    def initial_condition(self, sigma: float = 0.1) -> np.ndarray:
        """Gaussian pulse centered at ``L/2``."""
        return np.exp(-0.5 * ((self.x - self.L / 2) / sigma) ** 2)

    def _residual(self, u_new, u_old):
        up = np.roll(u_new, -1)
        um = np.roll(u_new, +1)
        conv = np.where(u_new >= 0.0,
                        u_new * (u_new - um) / self.dx,
                        u_new * (up - u_new) / self.dx)
        diff = (up - 2 * u_new + um) / (self.dx ** 2)
        return u_new - u_old + self.dt * (conv - self.nu * diff)

    def _jacobian(self, u):
        N = self.Nx
        dx, dt, nu = self.dx, self.dt, self.nu
        up = np.roll(u, -1)
        um = np.roll(u, +1)
        upos = u >= 0.0
        diag0 = np.ones(N) + dt * (
            np.where(upos, (2 * u - um) / dx, (up - 2 * u) / dx) + 2 * nu / dx ** 2)
        diag_p = dt * (np.where(upos, 0.0, u / dx) - nu / dx ** 2)
        diag_m = dt * (np.where(upos, -u / dx, 0.0) - nu / dx ** 2)
        offsets = [0, 1, -1]
        diags = [diag0, np.roll(diag_p, 1), np.roll(diag_m, -1)]
        return sp.diags(diags, offsets, shape=(N, N), format="csr")

    def step(self, u_old, tol: float = 1e-8, max_iter: int = 10, verbose: bool = False):
        """Advance one backward-Euler step (Newton iteration)."""
        u = u_old.copy()
        for _ in range(max_iter):
            res = self._residual(u, u_old)
            if np.linalg.norm(res) < tol:
                break
            J = self._jacobian(u)
            u = u + spla.spsolve(J, -res)
        return u

    def simulate(self, u0, n_steps: int = 500, tol: float = 1e-8, max_iter: int = 10,
                 verbose: bool = False) -> np.ndarray:
        """March the FOM forward and return the snapshot matrix ``(Nx, n_steps+1)``."""
        u = u0.copy()
        snaps = [u.copy()]
        for n in range(n_steps):
            if verbose:
                print(f"step {n + 1}/{n_steps}")
            u = self.step(u, tol, max_iter, verbose)
            snaps.append(u.copy())
        return np.stack(snaps, axis=1)
