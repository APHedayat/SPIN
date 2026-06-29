"""1D Fisher-KPP reaction-diffusion: FOM and static / baseline / SPIN LSPG ROMs."""

from .fom import FisherKPPSolver
from .rom import FisherKPPLSPGROM, FisherKPPSpinROM

__all__ = ["FisherKPPSolver", "FisherKPPLSPGROM", "FisherKPPSpinROM"]
