"""SPIN: Spectral Preconditioning via IN-span learning.

A small reference implementation of *in-span learning* for adaptive reduced-order
models. A ROM can learn not only from external out-of-span corrections, but also
from the trajectory it has already produced itself: streaming its own in-span
predictions through an iSVD-with-forgetting reweights and rotates the basis
inside the current subspace, preparing it to absorb the next correction more
effectively. The resulting model is called **SPIN**.

Public API
----------
Models (full-order solvers):
    :class:`spin.models.SpiralModel`
    :class:`spin.models.BurgersSolver`
    :class:`spin.models.FisherKPPSolver`

Adaptive ROMs (static / baseline adaptive / SPIN, via ``mode=...``):
    :class:`spin.rom.BurgersROM`
    :class:`spin.rom.FisherKPPROM`

Offline linear algebra:
    :func:`spin.linalg.compute_pod_basis`, :func:`spin.linalg.qdeim`

Core update:
    :func:`spin.isvd.isvd`

Diagnostics:
    :mod:`spin.diagnostics`

Spiral experiment:
    :func:`spin.spiral.build_spiral_experiment`
"""

from . import diagnostics, isvd, linalg, models, plotting, rom, spiral
from .isvd import isvd as isvd_update
from .linalg import compute_pod_basis, qdeim
from .models import BurgersSolver, FisherKPPSolver, SpiralModel
from .rom import AdaptiveLSPGROM, BurgersROM, FisherKPPROM
from .spiral import build_spiral_experiment

__version__ = "0.1.0"

__all__ = [
    "models", "rom", "isvd", "linalg", "diagnostics", "plotting", "spiral",
    "SpiralModel", "BurgersSolver", "FisherKPPSolver",
    "AdaptiveLSPGROM", "BurgersROM", "FisherKPPROM",
    "compute_pod_basis", "qdeim", "isvd_update", "build_spiral_experiment",
]
