"""Neural surrogate models for fast simulation prediction.

Provides PyTorch-based neural networks that predict photonic metrics
from permittivity arrays, enabling fast design exploration without
running full FDTD simulations.

REQUIRES: Install with `pip install -e ".[ml]"` for PyTorch support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# Check for optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

if TYPE_CHECKING:
    import torch


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model.

    Attributes:
        input_channels: Number of input channels (1 for permittivity).
        hidden_dim: Hidden dimension for MLP head.
        output_dim: Number of output metrics to predict.
        dropout: Dropout rate for regularization.
    """
    input_channels: int = 1
    hidden_dim: int = 128
    output_dim: int = 4  # e.g., transmission, insertion_loss, bandwidth, crosstalk
    dropout: float = 0.1


def _check_torch():
    """Raise error if PyTorch is not installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for neural models. "
            "Install with: pip install -e \".[ml]\""
        )


class PermittivityEncoder(nn.Module if HAS_TORCH else object):
    """CNN encoder for permittivity arrays.

    Extracts features from 2D permittivity grids using convolutional layers.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        """Initialize encoder.

        Args:
            in_channels: Number of input channels.
            feature_dim: Output feature dimension.
        """
        _check_torch()
        super().__init__()

        # Convolutional layers with increasing filter counts
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Project to feature dimension
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        return x


class SurrogateModel(nn.Module if HAS_TORCH else object):
    """Neural surrogate for photonic simulation.

    Predicts S-parameter metrics from permittivity arrays.
    """

    def __init__(self, config: SurrogateConfig | None = None):
        """Initialize surrogate model.

        Args:
            config: Model configuration.
        """
        _check_torch()
        super().__init__()

        self.config = config or SurrogateConfig()

        # Encoder
        self.encoder = PermittivityEncoder(
            in_channels=self.config.input_channels,
            feature_dim=self.config.hidden_dim,
        )

        # MLP head for metric prediction
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim),
        )

    def forward(self, permittivity: torch.Tensor) -> torch.Tensor:
        """Predict metrics from permittivity.

        Args:
            permittivity: Tensor of shape (batch, 1, H, W) or (batch, H, W).

        Returns:
            Predicted metrics of shape (batch, output_dim).
        """
        # Add channel dim if needed
        if permittivity.dim() == 3:
            permittivity = permittivity.unsqueeze(1)

        features = self.encoder(permittivity)
        metrics = self.head(features)

        return metrics

    def predict_numpy(self, permittivity: np.ndarray) -> np.ndarray:
        """Predict metrics from numpy array.

        Args:
            permittivity: Array of shape (H, W) or (batch, H, W).

        Returns:
            Predicted metrics as numpy array.
        """
        _check_torch()
        self.eval()

        # Handle single sample
        if permittivity.ndim == 2:
            permittivity = permittivity[np.newaxis, ...]

        with torch.no_grad():
            x = torch.from_numpy(permittivity).float()
            pred = self.forward(x)
            return pred.numpy()


def predict_metrics(
    model: SurrogateModel,
    permittivity: np.ndarray,
) -> dict[str, float]:
    """Predict photonic metrics from permittivity.

    Args:
        model: Trained surrogate model.
        permittivity: 2D permittivity array.

    Returns:
        Dictionary of metric names to predicted values.
    """
    predictions = model.predict_numpy(permittivity)

    # Default metric names
    metric_names = [
        "transmission",
        "insertion_loss_db",
        "bandwidth_3db",
        "return_loss_db",
    ]

    return {
        name: float(predictions[0, i])
        for i, name in enumerate(metric_names[:predictions.shape[1]])
    }


def load_surrogate(path: str) -> SurrogateModel:
    """Load a trained surrogate model from file.

    Args:
        path: Path to saved model (.pt file).

    Returns:
        Loaded SurrogateModel.
    """
    _check_torch()
    state = torch.load(path, map_location="cpu")

    config = SurrogateConfig(**state.get("config", {}))
    model = SurrogateModel(config)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model


def save_surrogate(model: SurrogateModel, path: str) -> None:
    """Save a trained surrogate model.

    Args:
        model: Model to save.
        path: Destination path.
    """
    _check_torch()
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_channels": model.config.input_channels,
            "hidden_dim": model.config.hidden_dim,
            "output_dim": model.config.output_dim,
            "dropout": model.config.dropout,
        },
    }, path)


__all__ = [
    "HAS_TORCH",
    "SurrogateConfig",
    "PermittivityEncoder",
    "SurrogateModel",
    "predict_metrics",
    "load_surrogate",
    "save_surrogate",
]
