"""Unit tests for physical constants module."""

import pytest
import math

from photonic_forge.core.constants import (
    C,
    H,
    HBAR,
    EPSILON_0,
    MU_0,
    WAVELENGTH_C_BAND_CENTER,
    WAVELENGTH_O_BAND_CENTER,
    SOI_WAVEGUIDE_HEIGHT,
    SOI_WAVEGUIDE_WIDTH,
    wavelength_to_frequency,
    frequency_to_wavelength,
    wavelength_to_wavenumber,
)


class TestFundamentalConstants:
    """Tests for fundamental physical constants."""

    def test_speed_of_light(self):
        """Speed of light should be ~3e8 m/s."""
        assert C == pytest.approx(299_792_458.0, rel=1e-9)

    def test_planck_constant(self):
        """Planck constant should be ~6.6e-34 J·s."""
        assert H == pytest.approx(6.62607015e-34, rel=1e-9)

    def test_reduced_planck_constant(self):
        """ħ = h / 2π."""
        assert HBAR == pytest.approx(H / (2 * math.pi), rel=1e-12)

    def test_vacuum_permittivity(self):
        """ε₀ should be ~8.85e-12 F/m."""
        assert EPSILON_0 == pytest.approx(8.8541878128e-12, rel=1e-9)

    def test_vacuum_permeability(self):
        """μ₀ should be ~1.26e-6 H/m."""
        assert MU_0 == pytest.approx(1.25663706212e-6, rel=1e-9)

    def test_speed_of_light_relation(self):
        """c = 1 / sqrt(ε₀ * μ₀)."""
        c_computed = 1 / math.sqrt(EPSILON_0 * MU_0)
        assert c_computed == pytest.approx(C, rel=1e-6)


class TestWavelengths:
    """Tests for standard wavelengths."""

    def test_c_band_center(self):
        """C-band center should be 1550 nm."""
        assert WAVELENGTH_C_BAND_CENTER == pytest.approx(1.55e-6, rel=1e-6)

    def test_o_band_center(self):
        """O-band center should be 1310 nm."""
        assert WAVELENGTH_O_BAND_CENTER == pytest.approx(1.31e-6, rel=1e-6)


class TestDimensions:
    """Tests for standard SOI dimensions."""

    def test_soi_waveguide_height(self):
        """Standard SOI height is 220 nm."""
        assert SOI_WAVEGUIDE_HEIGHT == pytest.approx(220e-9, rel=1e-6)

    def test_soi_waveguide_width(self):
        """Standard single-mode width is 500 nm."""
        assert SOI_WAVEGUIDE_WIDTH == pytest.approx(500e-9, rel=1e-6)


class TestConversionFunctions:
    """Tests for wavelength/frequency conversion."""

    def test_wavelength_to_frequency(self):
        """Convert 1550 nm to frequency."""
        freq = wavelength_to_frequency(1.55e-6)
        expected = C / 1.55e-6  # ~193 THz
        assert freq == pytest.approx(expected, rel=1e-9)

    def test_frequency_to_wavelength(self):
        """Convert frequency back to wavelength."""
        freq = 193.4e12  # ~193 THz
        wavelength = frequency_to_wavelength(freq)
        assert wavelength == pytest.approx(C / freq, rel=1e-9)

    def test_roundtrip_conversion(self):
        """Wavelength -> frequency -> wavelength should be identity."""
        original = 1.55e-6
        freq = wavelength_to_frequency(original)
        result = frequency_to_wavelength(freq)
        assert result == pytest.approx(original, rel=1e-12)

    def test_wavenumber(self):
        """k = 2π / λ."""
        wavelength = 1.55e-6
        k = wavelength_to_wavenumber(wavelength)
        expected = 2 * math.pi / wavelength
        assert k == pytest.approx(expected, rel=1e-12)
