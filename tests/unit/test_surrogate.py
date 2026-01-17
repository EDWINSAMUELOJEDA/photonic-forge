"""Tests for neural surrogate models."""

import numpy as np
import pytest


class TestSurrogateWithoutTorch:
    """Tests that work without PyTorch installed."""

    def test_has_torch_flag_exists(self):
        """HAS_TORCH flag should always be available."""
        from photonic_forge.neural_models import HAS_TORCH
        assert isinstance(HAS_TORCH, bool)


@pytest.fixture
def mock_permittivity():
    """Create mock permittivity array."""
    return np.random.rand(64, 128).astype(np.float32)


# Only run these tests if PyTorch is installed
pytest_plugins = []

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestSurrogateModel:
    """Tests for SurrogateModel class."""

    def test_model_creation(self):
        """Model should be creatable with default config."""
        from photonic_forge.neural_models import SurrogateModel, SurrogateConfig

        config = SurrogateConfig(hidden_dim=64, output_dim=4)
        model = SurrogateModel(config)

        assert model is not None
        assert model.config.hidden_dim == 64

    def test_forward_pass(self, mock_permittivity):
        """Forward pass should produce correct output shape."""
        from photonic_forge.neural_models import SurrogateModel, SurrogateConfig

        config = SurrogateConfig(output_dim=4)
        model = SurrogateModel(config)

        batch = torch.from_numpy(mock_permittivity).unsqueeze(0).unsqueeze(0)
        output = model(batch)

        assert output.shape == (1, 4)

    def test_predict_numpy(self, mock_permittivity):
        """predict_numpy should work with numpy input."""
        from photonic_forge.neural_models import SurrogateModel

        model = SurrogateModel()
        predictions = model.predict_numpy(mock_permittivity)

        assert predictions.shape == (1, 4)
        assert isinstance(predictions, np.ndarray)

    def test_predict_metrics(self, mock_permittivity):
        """predict_metrics should return dictionary."""
        from photonic_forge.neural_models import SurrogateModel, predict_metrics

        model = SurrogateModel()
        metrics = predict_metrics(model, mock_permittivity)

        assert isinstance(metrics, dict)
        assert "transmission" in metrics
        assert isinstance(metrics["transmission"], float)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPermittivityEncoder:
    """Tests for CNN encoder."""

    def test_encoder_output_shape(self, mock_permittivity):
        """Encoder should produce correct feature dimension."""
        from photonic_forge.neural_models import PermittivityEncoder

        encoder = PermittivityEncoder(in_channels=1, feature_dim=128)
        batch = torch.from_numpy(mock_permittivity).unsqueeze(0).unsqueeze(0)

        features = encoder(batch)

        assert features.shape == (1, 128)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestSaveLoad:
    """Tests for model serialization."""

    def test_save_and_load(self, tmp_path):
        """Model should be saveable and loadable."""
        from photonic_forge.neural_models import (
            SurrogateModel,
            SurrogateConfig,
            save_surrogate,
            load_surrogate,
        )

        config = SurrogateConfig(hidden_dim=32, output_dim=2)
        model = SurrogateModel(config)

        path = tmp_path / "model.pt"
        save_surrogate(model, str(path))

        loaded = load_surrogate(str(path))

        assert loaded.config.hidden_dim == 32
        assert loaded.config.output_dim == 2
