"""Data loading utilities for neural model training.

Loads simulation data from Data Moat logs for training surrogate models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Check for optional PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    Dataset = object
    DataLoader = None

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset


def _check_torch():
    """Raise error if PyTorch is not installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for neural models. "
            "Install with: pip install -e \".[ml]\""
        )


@dataclass
class TrainingSample:
    """Single training sample.

    Attributes:
        permittivity: 2D permittivity array.
        metrics: Target metric values.
        design_id: Optional design identifier.
    """
    permittivity: np.ndarray
    metrics: np.ndarray
    design_id: str | None = None


class SimulationDataset(Dataset if HAS_TORCH else object):
    """PyTorch Dataset for simulation data.

    Loads permittivity arrays and corresponding metrics from
    Data Moat simulation logs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        metric_names: list[str] | None = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing simulation logs.
            metric_names: List of metric names to extract.
        """
        _check_torch()

        self.data_dir = Path(data_dir)
        self.metric_names = metric_names or [
            "transmission",
            "insertion_loss_db",
            "bandwidth_3db",
            "return_loss_db",
        ]

        self.samples: list[TrainingSample] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load simulation data from directory."""
        # Look for JSON log files
        for log_file in self.data_dir.glob("*.json"):
            try:
                with open(log_file) as f:
                    data = json.load(f)

                # Skip if missing required fields
                if "geometry_hash" not in data:
                    continue

                # Load permittivity (assumes saved alongside)
                eps_file = log_file.with_suffix(".npy")
                if not eps_file.exists():
                    continue

                permittivity = np.load(eps_file)

                # Extract metrics
                metrics = []
                predicted = data.get("predicted_metrics", {})
                for name in self.metric_names:
                    value = predicted.get(name, 0.0)
                    metrics.append(float(value))

                self.samples.append(TrainingSample(
                    permittivity=permittivity,
                    metrics=np.array(metrics, dtype=np.float32),
                    design_id=data.get("design_id"),
                ))

            except (json.JSONDecodeError, ValueError, IOError):
                continue

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get single sample.

        Returns:
            (permittivity_tensor, metrics_tensor)
        """
        sample = self.samples[idx]

        eps = torch.from_numpy(sample.permittivity).float()
        metrics = torch.from_numpy(sample.metrics).float()

        # Add channel dimension if needed
        if eps.dim() == 2:
            eps = eps.unsqueeze(0)

        return eps, metrics


def create_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> tuple:
    """Create train/val/test data loaders.

    Args:
        data_dir: Directory with simulation data.
        batch_size: Batch size.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    _check_torch()

    dataset = SimulationDataset(data_dir)
    n = len(dataset)

    if n == 0:
        raise ValueError(f"No valid samples found in {data_dir}")

    # Split indices
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Create subset datasets
    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def generate_synthetic_data(
    output_dir: str | Path,
    n_samples: int = 100,
    grid_size: tuple[int, int] = (64, 128),
) -> None:
    """Generate synthetic training data for testing.

    Creates random permittivity arrays with placeholder metrics.
    Useful for testing the training pipeline.

    Args:
        output_dir: Directory to save synthetic data.
        n_samples: Number of samples to generate.
        grid_size: Size of permittivity grids (H, W).
    """
    import uuid

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eps_min, eps_max = 2.1, 12.1  # SiO2 to Si

    for i in range(n_samples):
        design_id = str(uuid.uuid4())

        # Random permittivity (binary-ish)
        density = np.random.rand(*grid_size)
        threshold = np.random.uniform(0.3, 0.7)
        binary = (density > threshold).astype(np.float32)
        permittivity = eps_min + binary * (eps_max - eps_min)

        # Synthetic metrics (correlated with fill fraction)
        fill = np.mean(binary)
        metrics = {
            "transmission": 0.8 - 0.3 * fill + 0.1 * np.random.randn(),
            "insertion_loss_db": 0.5 + fill * 2 + 0.2 * np.random.randn(),
            "bandwidth_3db": 50e-9 + fill * 20e-9 + 5e-9 * np.random.randn(),
            "return_loss_db": -20 + fill * 5 + np.random.randn(),
        }

        # Save
        base_name = f"sim_{i:04d}"
        np.save(output_path / f"{base_name}.npy", permittivity)

        log_data = {
            "design_id": design_id,
            "geometry_hash": f"synthetic_{i:04d}",
            "predicted_metrics": metrics,
        }
        with open(output_path / f"{base_name}.json", "w") as f:
            json.dump(log_data, f, indent=2)

    print(f"Generated {n_samples} synthetic samples in {output_path}")


__all__ = [
    "TrainingSample",
    "SimulationDataset",
    "create_data_loaders",
    "generate_synthetic_data",
]
