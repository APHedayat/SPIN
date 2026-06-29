r"""Full-order model for the 1D Fisher-KPP reaction-diffusion equation.

The equation

.. math::
    \partial_t u = D\,\partial_{xx} u + \beta\, u (1 - u)

is discretized on a uniform periodic grid with a second-order centered diffusion
term. Time is integrated implicitly with backward Euler; each step is a Newton
solve. A localized pulse grows through the reaction term into travelling fronts
that collide and saturate toward :math:`u \approx 1`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FisherKPPSolver:
    """Implicit FOM solver for periodic 1D Fisher-KPP.

    Parameters
    ----------
    Nx : int
        Number of spatial grid points.
    L : float
        Length of the periodic domain.
    D : float
        Diffusivity.
    alpha : float
        Reaction rate :math:`\\beta`.
    dt : float
        Time step.
    """

    def __init__(self, Nx: int = 256, L: float = 1.0, D: float = 1e-4,
                 alpha: float = 20.0, dt: float = 1e-3):
        self.Nx = Nx
        self.L = L
        self.D = D
        self.alpha = alpha
        self.dt = dt
        self.dx = L / Nx
        self.x = np.linspace(0, L, Nx, endpoint=False)

    def initial_condition(self, sigma: float = 0.08, amplitude: float = 0.2) -> np.ndarray:
        """Small Gaussian pulse that ignites the fronts."""
        return amplitude * np.exp(-0.5 * ((self.x - self.L / 2) / sigma) ** 2)

    def _residual(self, u_new, u_old):
        up = np.roll(u_new, -1)
        um = np.roll(u_new, +1)
        diff = (up - 2.0 * u_new + um) / (self.dx ** 2)
        react = self.alpha * u_new * (1.0 - u_new)
        return u_new - u_old - self.dt * (self.D * diff + react)

    def _jacobian(self, u):
        N = self.Nx
        dx, dt, D, alpha = self.dx, self.dt, self.D, self.alpha
        diag0 = np.ones(N) + 2.0 * dt * D / dx ** 2 - dt * alpha * (1.0 - 2.0 * u)
        off = -dt * D / dx ** 2 * np.ones(N)
        offsets = [0, 1, -1]
        diags = [diag0, np.roll(off, 1), np.roll(off, -1)]
        return sp.diags(diags, offsets, shape=(N, N), format="csr")

    def step(self, u_old, tol: float = 1e-8, max_iter: int = 20, verbose: bool = False):
        """Advance one backward-Euler step (Newton iteration)."""
        u = u_old.copy()
        for _ in range(max_iter):
            res = self._residual(u, u_old)
            if np.linalg.norm(res) < tol:
                break
            J = self._jacobian(u)
            u = u + spla.spsolve(J, -res)
        return u

    def simulate(self, u0, n_steps: int = 500, tol: float = 1e-8, max_iter: int = 20,
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
