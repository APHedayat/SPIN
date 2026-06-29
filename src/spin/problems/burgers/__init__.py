"""1D viscous Burgers: FOM and static / baseline / SPIN LSPG ROMs."""

from .fom import BurgersSolver
from .rom import BurgersLSPGROM, BurgersSpinROM

__all__ = ["BurgersSolver", "BurgersLSPGROM", "BurgersSpinROM"]
