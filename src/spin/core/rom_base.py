"""Generic hyper-reduced LSPG ROM base class.

:class:`LSPGROMBase` implements the problem-agnostic pieces of a Least-Squares
Petrov-Galerkin (LSPG) ROM with QDEIM hyper-reduction: storing the basis and
sampling indices, caching the hyper-reduction pseudoinverse, running a Newton
iteration in reduced coordinates, and marching the (static) ROM in time.

A subclass only injects the two problem-specific hooks
:meth:`residual_sample` and :meth:`jacobian_sample`. Everything else --
the Newton solve, the time stepping, and (in :mod:`spin.core.adaptive_rom`)
the in-span / out-of-span adaptation loop -- is shared and equation-agnostic.

Minimal subclass::

    class MyLSPGROM(LSPGROMBase):
        def residual_sample(self, a, a_old):
            u = self.Phi @ a
            # return the sampled residual  Phi_p @ (a - a_old) + dt * R_F
            ...
        def jacobian_sample(self, a):
            # return rows of d(residual)/da at the sample points (m x r)
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LSPGROMBase(ABC):
    """Abstract base class for static LSPG + QDEIM reduced-order models.

    Parameters
    ----------
    Phi : (N, r) ndarray
        Reduced basis (usually POD).
    p_inds : sequence of int, length ``m``
        Sampling indices (e.g. from :func:`spin.core.sampling.qdeim`).
    dt : float
        ROM time step (should match the FOM time step).

    Notes
    -----
    The hyper-reduction pseudoinverse ``M`` is the pseudo-inverse of the basis
    restricted to the sampling indices. It is refreshed by
    :meth:`update_sampling` whenever ``Phi`` or ``p_inds`` changes in place.

    Subclasses may set the class attribute ``_use_pinv`` to choose whether the
    inner LSPG solve uses a pseudoinverse (default) or a direct solve; on the
    square ``m = r`` QDEIM system the two agree up to round-off.
    """

    #: whether the inner LSPG solve uses ``pinv`` (True) or ``solve`` (False)
    _use_pinv = True

    def __init__(self, Phi: np.ndarray, p_inds, dt: float):
        self.Phi = np.asarray(Phi, dtype=float)
        self.p_inds = np.asarray(p_inds, dtype=int)
        self.dt = dt
        self.N, self.r = self.Phi.shape
        self.update_sampling()

    def update_sampling(self) -> None:
        """Rebuild the cached sampling data from the current basis.

        Call after modifying :attr:`Phi` or :attr:`p_inds` in place (for example
        after an online basis adaptation). Caches the sample-point neighbour
        indices and the hyper-reduction pseudoinverse ``M``.
        """
        self.p_inds = np.asarray(self.p_inds, dtype=int)
        self.ns = self.p_inds.size
        self._ip = (self.p_inds + 1) % self.N      # periodic right neighbour
        self._im = (self.p_inds - 1) % self.N      # periodic left  neighbour
        self._Phi_p = self.Phi[self.p_inds, :]
        self._Phi_up = self.Phi[self._ip, :]
        self._Phi_down = self.Phi[self._im, :]
        self.M = np.linalg.pinv(self._Phi_p)

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def residual_sample(self, a: np.ndarray, a_old: np.ndarray) -> np.ndarray:
        """Sampled LSPG residual at reduced coordinates ``a``.

        Should return the residual ``Phi_p @ (a - a_old) + dt * R_F`` evaluated at
        the sampling indices, where ``R_F`` is the discrete spatial-operator
        contribution (its sign folded in by the subclass).

        Returns
        -------
        R_samp : (m,) ndarray
        """

    @abstractmethod
    def jacobian_sample(self, a: np.ndarray) -> np.ndarray:
        """Sampled LSPG test operator ``W = d(residual)/da`` at coordinates ``a``.

        Returns
        -------
        W_samp : (m, r) ndarray
        """

    # ------------------------------------------------------------------ solver
    def step(self, a_old: np.ndarray, tol: float = 1e-8, max_iter: int = 10,
             verbose: bool = False) -> np.ndarray:
        """One implicit LSPG step (Gauss-Newton on the sampled residual)."""
        a = a_old.copy()
        for _ in range(max_iter):
            R_samp = self.residual_sample(a, a_old)
            if np.linalg.norm(R_samp) < tol:
                break
            W_samp = self.jacobian_sample(a)
            LHS = self.M @ W_samp
            RHS = -self.M @ R_samp
            if self._use_pinv:
                a = a + np.linalg.pinv(LHS) @ RHS
            else:
                a = a + np.linalg.solve(LHS, RHS)
            if verbose:
                print(f"    |R_samp| = {np.linalg.norm(R_samp):.3e}")
        return a

    def simulate(self, a0: np.ndarray, n_steps: int, tol: float = 1e-8,
                 max_iter: int = 10, verbose: bool = False) -> np.ndarray:
        """March the static ROM forward and return the full-state trajectory.

        Returns
        -------
        soln : (N, n_steps+1) ndarray
            Reconstructed full-order states over time (columns are time steps).
        """
        a = a0.copy()
        soln = [self.Phi @ a]
        for n in range(n_steps):
            if verbose:
                print(f"step {n + 1}/{n_steps}")
            a = self.step(a_old=a, tol=tol, max_iter=max_iter)
            soln.append(self.Phi @ a)
        return np.array(soln).T
