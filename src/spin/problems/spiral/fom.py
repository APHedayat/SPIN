r"""Full-order model for the closed-form spiral.

The trajectory :math:`q(t) = [\cos t,\ \sin t,\ \alpha t]` solves the affine
linear ODE :math:`\dot q = A q + b`, whose exact one-step discrete flow is the
affine map :math:`q^{k+1} = G q^k + c` with :math:`G = \exp(A\,\Delta t)`. Because
the flow is exact, the spiral experiment has **no time-integration error**: every
FOM/ROM discrepancy is caused purely by the reduced representation and its
updates.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class SpiralModel:
    """Closed-form affine spiral with an exact discrete flow.

    Parameters
    ----------
    alpha : float
        Vertical rate (``q3 = alpha * t``).
    dt : float
        Time step of the exact discrete flow.
    t0 : float
        Start time.
    """

    def __init__(self, alpha: float = 0.4, dt: float = 0.35, t0: float = 4.0):
        self.alpha = alpha
        self.dt = dt
        self.t0 = t0
        A = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        b = np.array([0.0, 0.0, alpha])
        self.A, self.b = A, b
        self.G = expm(A * dt)
        # for this A, the affine offset reduces to [0, 0, alpha*dt]
        self.c = np.array([0.0, 0.0, alpha * dt])

    def exact_state(self, t):
        """Exact continuous spiral state at time ``t``."""
        return np.array([np.cos(t), np.sin(t), self.alpha * t])

    def trajectory(self, n_steps):
        """Exact FOM trajectory ``X`` (3, n_steps+1) and the time grid (from ``t0``)."""
        times = self.t0 + self.dt * np.arange(n_steps + 1)
        X = np.row_stack([np.cos(times), np.sin(times), self.alpha * times])
        return times, X

    def step(self, q):
        r"""One exact discrete step :math:`q^{k+1} = G q^k + c`."""
        return self.G @ q + self.c
