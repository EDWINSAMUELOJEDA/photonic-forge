"""Unit tests for materials module."""

import numpy as np
import pytest

from photonic_forge.core.materials import (
    AIR,
    ALUMINUM,
    MATERIAL_REGISTRY,
    SILICON,
    SILICON_DIOXIDE,
    SILICON_NITRIDE,
    VACUUM,
    Material,
    get_material,
    get_permittivity,
    permittivity_to_n,
)


class TestMaterialDataclass:
    """Tests for Material dataclass."""

    def test_material_creation(self):
        """Create a custom material."""
        mat = Material(name="Custom", n=2.5, k=0.1, wavelength=1.55e-6)
        assert mat.name == "Custom"
        assert mat.n == 2.5
        assert mat.k == 0.1
        assert mat.wavelength == 1.55e-6

    def test_material_defaults(self):
        """Material should have sensible defaults."""
        mat = Material(name="Simple", n=1.5)
        assert mat.k == 0.0
        assert mat.wavelength == 1.55e-6

    def test_complex_refractive_index(self):
        """complex_n should return n + ik."""
        mat = Material(name="Test", n=3.0, k=0.5)
        assert mat.complex_n == complex(3.0, 0.5)

    def test_permittivity_lossless(self):
        """Permittivity of lossless material is n^2."""
        mat = Material(name="Test", n=2.0, k=0.0)
        assert mat.permittivity == complex(4.0, 0.0)

    def test_permittivity_lossy(self):
        """Permittivity of lossy material is (n + ik)^2."""
        mat = Material(name="Test", n=3.0, k=0.5)
        # (3 + 0.5i)^2 = 9 + 3i - 0.25 = 8.75 + 3i
        expected = complex(3.0, 0.5) ** 2
        assert mat.permittivity == pytest.approx(expected, rel=1e-10)

    def test_epsilon_real(self):
        """Real part of permittivity: n^2 - k^2."""
        mat = Material(name="Test", n=3.0, k=1.0)
        assert mat.epsilon_real == pytest.approx(8.0, rel=1e-10)  # 9 - 1

    def test_epsilon_imag(self):
        """Imaginary part of permittivity: 2*n*k."""
        mat = Material(name="Test", n=3.0, k=1.0)
        assert mat.epsilon_imag == pytest.approx(6.0, rel=1e-10)  # 2*3*1


class TestBuiltinMaterials:
    """Tests for built-in material definitions."""

    def test_silicon_refractive_index(self):
        """Silicon n ≈ 3.476 at 1550nm."""
        assert SILICON.n == pytest.approx(3.476, rel=1e-3)
        assert SILICON.k == 0.0

    def test_silicon_dioxide_refractive_index(self):
        """SiO2 n ≈ 1.444 at 1550nm."""
        assert SILICON_DIOXIDE.n == pytest.approx(1.444, rel=1e-3)
        assert SILICON_DIOXIDE.k == 0.0

    def test_silicon_nitride_refractive_index(self):
        """Si3N4 n ≈ 2.0 at 1550nm."""
        assert SILICON_NITRIDE.n == pytest.approx(2.0, rel=0.1)

    def test_air_refractive_index(self):
        """Air n = 1.0."""
        assert AIR.n == 1.0
        assert AIR.k == 0.0

    def test_vacuum_refractive_index(self):
        """Vacuum n = 1.0."""
        assert VACUUM.n == 1.0

    def test_aluminum_is_lossy(self):
        """Aluminum has significant k (absorption)."""
        assert ALUMINUM.k > 1.0


class TestMaterialRegistry:
    """Tests for material lookup."""

    def test_get_silicon_by_name(self):
        """Look up silicon."""
        mat = get_material("silicon")
        assert mat == SILICON

    def test_get_silicon_by_alias(self):
        """Look up silicon by alias 'si'."""
        mat = get_material("si")
        assert mat == SILICON

    def test_get_material_case_insensitive(self):
        """Lookup should be case-insensitive."""
        assert get_material("SILICON") == SILICON
        assert get_material("Silicon") == SILICON

    def test_get_unknown_material_raises(self):
        """Unknown material should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown material"):
            get_material("unobtainium")


class TestPermittivityFunctions:
    """Tests for permittivity utility functions."""

    def test_get_permittivity_lossless(self):
        """get_permittivity for lossless material."""
        eps = get_permittivity(n=2.0)
        assert eps == complex(4.0, 0.0)

    def test_get_permittivity_lossy(self):
        """get_permittivity for lossy material."""
        eps = get_permittivity(n=3.0, k=1.0)
        expected = complex(3.0, 1.0) ** 2
        assert eps == pytest.approx(expected, rel=1e-10)

    def test_permittivity_to_n_lossless(self):
        """Convert permittivity back to n."""
        n, k = permittivity_to_n(complex(4.0, 0.0))
        assert n == pytest.approx(2.0, rel=1e-10)
        assert k == pytest.approx(0.0, abs=1e-10)

    def test_permittivity_roundtrip(self):
        """n -> permittivity -> n should be identity."""
        n_orig, k_orig = 3.476, 0.0
        eps = get_permittivity(n_orig, k_orig)
        n_back, k_back = permittivity_to_n(eps)
        assert n_back == pytest.approx(n_orig, rel=1e-10)
        assert k_back == pytest.approx(k_orig, abs=1e-10)
