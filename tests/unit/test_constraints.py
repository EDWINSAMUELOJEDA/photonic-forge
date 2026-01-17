"""Tests for fabrication constraints."""

import numpy as np
import pytest
from photonic_forge.optimize.constraints import MinFeatureConstraint, MinRadiusConstraint


def test_min_feature_constraint():
    """Test minimum gap/width checking."""
    # 10nm resolution
    res = 10e-9
    
    # CASE 1: Gap too small
    # 50nm gap (5 pixels) < 100nm limit
    constraint = MinFeatureConstraint(min_size=100e-9)
    
    # 20x20 grid
    grid = np.zeros((20, 20))
    # Two blocks separated by 5 pixels
    grid[:, :5] = 1.0
    grid[:, 10:] = 1.0
    
    satisfied, violation = constraint.check(grid, res)
    assert not satisfied
    assert violation > 0

    # CASE 2: Gap large enough
    # 110nm gap (11 pixels) > 100nm limit
    grid2 = np.zeros((30, 30))
    grid2[:, :5] = 1.0
    grid2[:, 16:] = 1.0 # Gap from 5 to 16 is 11 pixels
    
    satisfied, violation = constraint.check(grid2, res)
    # Note: binary opening can be tricky on edges, but for clear separation it should work
    assert satisfied
    assert violation == 0


def test_min_radius_constraint():
    """Test minimum bend radius checking."""
    res = 10e-9
    # Limit: 200nm radius (20 pixels)
    constraint = MinRadiusConstraint(min_radius=200e-9)
    
    # CASE 1: Sharp corner (Radius ~ 0)
    grid = np.zeros((40, 40))
    grid[10:30, 10:30] = 1.0 # Square block corner
    
    satisfied, violation = constraint.check(grid, res)
    assert not satisfied
    assert violation > 0
    
    # CASE 2: Large circle (Radius >> Limit)
    # Circle radius 300nm (30 pixels)
    y, x = np.ogrid[-20:21, -20:21]
    disk = x**2 + y**2 <= 30**2
    
    satisfied, violation = constraint.check(disk.astype(float), res)
    assert satisfied
    assert violation == 0
