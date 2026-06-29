"""Closed-form spiral: the toy example that exposes the in-span mechanism."""

from .experiment import build_spiral_experiment
from .fom import SpiralModel

__all__ = ["SpiralModel", "build_spiral_experiment"]
