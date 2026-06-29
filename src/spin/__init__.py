"""SPIN: Spectral Preconditioning via IN-span learning.

A small, modular reference implementation of *in-span learning* for adaptive
reduced-order models. A ROM can learn not only from external out-of-span
corrections, but also from the trajectory it has already produced itself:
streaming its own in-span predictions through an iSVD-with-forgetting reweights
and rotates the basis inside the current subspace, preparing it to absorb the
next correction more effectively. The resulting model is called **SPIN**.

The package is organized like the paper:

* :mod:`spin.core` -- problem-agnostic building blocks:
    POD (:func:`compute_pod_basis`), QDEIM (:func:`qdeim`), the iSVD update
    (:func:`isvd`), and the base ROM classes :class:`LSPGROMBase` (static) and
    :class:`SpinROMBase` (static / baseline / SPIN adaptation).

* :mod:`spin.problems` -- reference equations (FOM + ROMs):
    :mod:`~spin.problems.spiral`, :mod:`~spin.problems.burgers`,
    :mod:`~spin.problems.fisher_kpp`.

* :mod:`spin.diagnostics` -- the metrics used in the paper's figures.
* :mod:`spin.utils` -- plotting helpers and the shared palette.

To bring SPIN to a new equation, subclass :class:`SpinROMBase` and implement the
two hyper-reduced hooks ``residual_sample`` / ``jacobian_sample`` (see
:mod:`spin.problems.burgers` for a template).
"""

from . import core, diagnostics, problems, utils
from .core import (
    MODES, LSPGROMBase, SpinROMBase, compute_pod_basis, isvd, qdeim,
)
from .problems.spiral import SpiralModel, build_spiral_experiment

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "core", "problems", "diagnostics", "utils",
    # core API
    "compute_pod_basis", "qdeim", "isvd", "LSPGROMBase", "SpinROMBase", "MODES",
    # spiral convenience
    "SpiralModel", "build_spiral_experiment",
]
