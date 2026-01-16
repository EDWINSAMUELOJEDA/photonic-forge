"""Cornerstone SOI Process Design Kit (PDK) definitions.

This module provides layer stack definitions and design rules for the
Cornerstone Silicon-on-Insulator (SOI) platform.
"""

from dataclasses import dataclass
from typing import Tuple

# Layer Type: (layer_number, datatype)
Layer = Tuple[int, int]


@dataclass(frozen=True)
class SOILayerStack:
    """Standard layers for Cornerstone SOI process."""
    
    # Waveguides
    WG_CORE: Layer = (1, 0)      # Silicon core (220nm)
    WG_CLAD: Layer = (2, 0)      # Oxide cladding
    
    # Etch steps
    ETCH_SHALLOW: Layer = (3, 0) # 70nm etch (grating couplers)
    ETCH_DEEP: Layer = (1, 0)    # Full etch (same as core)
    
    # Metallization
    HEATER: Layer = (10, 0)      # TiN heater
    METAL1: Layer = (11, 0)      # Aluminum interconnects
    
    # Floorplan
    FLOORPLAN: Layer = (99, 0)   # Chip boundary


# Instance of the standard stack
LAYERS = SOILayerStack()


@dataclass(frozen=True)
class DesignRules:
    """Geometric design rules."""
    
    min_feature_size: float = 0.20e-6  # 200 nm
    min_spacing: float = 0.20e-6       # 200 nm
    grid_resolution: float = 1e-9      # 1 nm grid


RULES = DesignRules()


def check_width(width: float) -> bool:
    """Check if a width meets the minimum feature size."""
    return width >= RULES.min_feature_size


__all__ = ["LAYERS", "RULES", "check_width", "SOILayerStack", "DesignRules"]
