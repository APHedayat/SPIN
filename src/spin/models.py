"""Full-order models (FOMs).

Three dynamical systems used in the paper:

* :class:`SpiralModel`    -- a closed-form affine spiral in R^3 (exact discrete flow).
* :class:`BurgersSolver`  -- 1D viscous Burgers, upwind + backward Euler.
* :class:`FisherKPPSolver` -- 1D Fisher-KPP reaction-diffusion, centered + backward Euler.

The two PDE solvers expose the same small interface used by the adaptive ROM:
``step(u_old, ...)`` advances one implicit time step (Newton), and
``simulate(u0, n_steps, ...)`` returns the list of snapshots.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm


# ======================================================================
# Spiral (closed-form, exact discrete flow)
# ======================================================================
class SpiralModel:
    r"""Closed-form spiral :math:`q(t) = [\cos t,\ \sin t,\ \alpha t]^\top`.

    The trajectory solves the affine linear ODE :math:`\dot q = A q + b`, whose
    exact one-step discrete flow is the affine map :math:`q^{k+1} = G q^k + c`
    with :math:`G = \exp(A\,\Delta t)`. Because the flow is exact, the spiral
    experiment has **no time-integration error**: every FOM/ROM discrepancy is
    caused purely by the reduced representation and its updates.
    """

    def __init__(self, alpha=0.4, dt=0.35, t0=4.0):
        self.alpha = alpha
        self.dt = dt
        self.t0 = t0
        A = np.array([[0.0, -1.0, 0.0],
                      [1.0,  0.0, 0.0],
                      [0.0,  0.0, 0.0]])
        b = np.array([0.0, 0.0, alpha])
        self.A, self.b = A, b
        self.G = expm(A * dt)
        # c = integral of exp(A s) b ds over [0, dt]; for this A it is [0, 0, alpha*dt]
        self.c = np.array([0.0, 0.0, alpha * dt])

    def exact_state(self, t):
        """Exact continuous spiral state at time ``t``."""
        return np.array([np.cos(t), np.sin(t), self.alpha * t])

    def trajectory(self, n_steps):
        """Exact FOM trajectory ``X`` (3, n_steps+1) and times, starting at ``t0``."""
        times = self.t0 + self.dt * np.arange(n_steps + 1)
        X = np.row_stack([np.cos(times), np.sin(times), self.alpha * times])
        return times, X

    def step(self, q):
        """One exact discrete step :math:`q^{k+1} = G q^k + c`."""
        return self.G @ q + self.c


# ======================================================================
# Viscous Burgers (upwind convection, centered diffusion, backward Euler)
# ======================================================================
class BurgersSolver:
    r"""1D viscous Burgers: :math:`\partial_t u + u\,\partial_x u = \nu\,\partial_{xx} u`.

    Periodic domain, first-order upwind convection, centered diffusion,
    backward Euler in time with a Newton solve per step.
    """

    def __init__(self, Nx=256, L=1.0, nu=1e-2, dt=1e-3):
        self.Nx = Nx
        self.L = L
        self.nu = nu
        self.dt = dt
        self.dx = L / Nx
        self.x = np.linspace(0, L, Nx, endpoint=False)

    def initial_condition(self, sigma=0.1):
        """Gaussian pulse centered in the domain."""
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

    def step(self, u_old, tol=1e-8, max_iter=10, verbose=False):
        """One backward-Euler step (Newton iteration)."""
        u = u_old.copy()
        for _ in range(max_iter):
            res = self._residual(u, u_old)
            if np.linalg.norm(res) < tol:
                break
            J = self._jacobian(u)
            u += spla.spsolve(J, -res)
        return u

    def simulate(self, u0, n_steps=200, tol=1e-8, max_iter=10, verbose=False):
        """Advance ``n_steps`` and return the list of snapshots (length n_steps+1)."""
        u = u0.copy()
        snaps = [u.copy()]
        for _ in range(n_steps):
            u = self.step(u, tol, max_iter, verbose)
            snaps.append(u.copy())
        return snaps


# ======================================================================
# Fisher-KPP reaction-diffusion (centered diffusion, backward Euler)
# ======================================================================
class FisherKPPSolver:
    r"""1D Fisher-KPP: :math:`\partial_t u = D\,\partial_{xx} u + \beta\, u (1 - u)`.

    Periodic domain, centered diffusion, backward Euler with a Newton solve.
    A localized pulse grows through the reaction term into travelling fronts
    that collide and saturate toward :math:`u \approx 1`.
    """

    def __init__(self, Nx=256, L=1.0, D=1e-4, alpha=20.0, dt=1e-3):
        self.Nx = Nx
        self.L = L
        self.D = D
        self.alpha = alpha   # reaction rate beta
        self.dt = dt
        self.dx = L / Nx
        self.x = np.linspace(0, L, Nx, endpoint=False)

    def initial_condition(self, sigma=0.08, amplitude=0.2):
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

    def step(self, u_old, tol=1e-8, max_iter=20, verbose=False):
        """One backward-Euler step (Newton iteration)."""
        u = u_old.copy()
        for _ in range(max_iter):
            res = self._residual(u, u_old)
            if np.linalg.norm(res) < tol:
                break
            J = self._jacobian(u)
            u += spla.spsolve(J, -res)
        return u

    def simulate(self, u0, n_steps=200, tol=1e-8, max_iter=20, verbose=False):
        """Advance ``n_steps`` and return the list of snapshots (length n_steps+1)."""
        u = u0.copy()
        snaps = [u.copy()]
        for _ in range(n_steps):
            u = self.step(u, tol, max_iter, verbose)
            snaps.append(u.copy())
        return snaps
