"""Problem-agnostic building blocks for SPIN ROMs.

* :func:`compute_pod_basis` -- offline POD basis (thin SVD).
* :func:`qdeim` -- QDEIM hyper-reduction sampling.
* :func:`isvd` -- incremental SVD with forgetting (the in-span / out-of-span update).
* :class:`LSPGROMBase` -- generic static LSPG + QDEIM ROM.
* :class:`SpinROMBase` -- generic adaptive ROM (static / baseline / SPIN modes).
"""

from .adaptive_rom import MODES, SpinROMBase
from .basis_adaptation import isvd
from .pod import compute_pod_basis
from .rom_base import LSPGROMBase
from .sampling import qdeim

__all__ = [
    "compute_pod_basis", "qdeim", "isvd",
    "LSPGROMBase", "SpinROMBase", "MODES",
]
