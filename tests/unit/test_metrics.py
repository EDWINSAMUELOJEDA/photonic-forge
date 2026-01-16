"""Unit tests for photonic metrics module."""

import pytest
import numpy as np

from photonic_forge.solvers.metrics import (
    insertion_loss,
    return_loss,
    crosstalk,
    group_delay,
    transmission_efficiency,
    bandwidth_3db,
)


class TestInsertionLoss:
    """Tests for insertion loss calculation."""
    
    def test_perfect_transmission(self):
        """S21 = 1 → IL = 0 dB."""
        s21 = np.array([1.0 + 0j])
        il = insertion_loss(s21)
        assert il[0] == pytest.approx(0.0, abs=1e-6)
    
    def test_half_power(self):
        """S21 = 1/√2 → IL = 3 dB."""
        s21 = np.array([1 / np.sqrt(2) + 0j])
        il = insertion_loss(s21)
        assert il[0] == pytest.approx(3.01, abs=0.1)
    
    def test_tenth_power(self):
        """S21 = 0.1 → IL = 20 dB."""
        s21 = np.array([0.1 + 0j])
        il = insertion_loss(s21)
        assert il[0] == pytest.approx(20.0, abs=0.1)


class TestReturnLoss:
    """Tests for return loss calculation."""
    
    def test_no_reflection(self):
        """S11 ≈ 0 → RL → infinity (large positive)."""
        s11 = np.array([1e-6 + 0j])
        rl = return_loss(s11)
        assert rl[0] > 100  # Very high return loss
    
    def test_full_reflection(self):
        """S11 = 1 → RL = 0 dB."""
        s11 = np.array([1.0 + 0j])
        rl = return_loss(s11)
        assert rl[0] == pytest.approx(0.0, abs=1e-6)


class TestCrosstalk:
    """Tests for crosstalk calculation."""
    
    def test_no_coupling(self):
        """S_coupled ≈ 0 → CT → -infinity (very negative)."""
        s_coupled = np.array([1e-6 + 0j])
        ct = crosstalk(s_coupled)
        assert ct[0] < -100  # Very low crosstalk
    
    def test_full_coupling(self):
        """S_coupled = 1 → CT = 0 dB."""
        s_coupled = np.array([1.0 + 0j])
        ct = crosstalk(s_coupled)
        assert ct[0] == pytest.approx(0.0, abs=1e-6)


class TestGroupDelay:
    """Tests for group delay calculation."""
    
    def test_constant_phase(self):
        """Constant phase → zero group delay."""
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 100)
        s21 = np.ones_like(wavelengths) * (1 + 0j)  # Constant phase
        
        tau = group_delay(s21, wavelengths)
        assert np.allclose(tau, 0, atol=1e-20)
    
    def test_linear_phase(self):
        """Linear phase → constant (non-zero) group delay."""
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 100)
        # Linear phase: φ = k * λ
        phase = 10 * wavelengths / 1e-6
        s21 = np.exp(1j * phase)
        
        tau = group_delay(s21, wavelengths)
        # Should be approximately constant
        assert np.std(tau) / np.mean(np.abs(tau)) < 0.1


class TestTransmissionEfficiency:
    """Tests for transmission efficiency."""
    
    def test_perfect_transmission(self):
        """S21 = 1 → η = 1."""
        s21 = np.array([1.0 + 0j])
        eta = transmission_efficiency(s21)
        assert eta[0] == pytest.approx(1.0, rel=1e-10)
    
    def test_half_amplitude(self):
        """S21 = 0.5 → η = 0.25."""
        s21 = np.array([0.5 + 0j])
        eta = transmission_efficiency(s21)
        assert eta[0] == pytest.approx(0.25, rel=1e-10)


class TestBandwidth3dB:
    """Tests for 3dB bandwidth calculation."""
    
    def test_flat_response(self):
        """Flat response → full bandwidth."""
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 100)
        s21 = np.ones_like(wavelengths) + 0j
        
        bw = bandwidth_3db(s21, wavelengths)
        assert bw == pytest.approx(0.1e-6, rel=0.1)
    
    def test_narrow_peak(self):
        """Narrow Gaussian peak → small bandwidth."""
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 100)
        center = 1.55e-6
        width = 0.01e-6
        s21 = np.exp(-((wavelengths - center) / width) ** 2) + 0j
        
        bw = bandwidth_3db(s21, wavelengths)
        # 3dB bandwidth of Gaussian ≈ 2 * sqrt(ln2) * σ ≈ 1.67 * σ
        expected = 2 * np.sqrt(np.log(2)) * width
        assert bw == pytest.approx(expected, rel=0.2)
