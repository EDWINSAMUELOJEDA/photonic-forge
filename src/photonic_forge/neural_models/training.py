"""Training utilities for neural surrogate models.

Provides training loop, early stopping, and checkpoint management.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Check for optional PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam, Optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    Adam = None
    Optimizer = None

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .surrogate import SurrogateModel


def _check_torch():
    """Raise error if PyTorch is not installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for neural models. "
            "Install with: pip install -e \".[ml]\""
        )


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        epochs: Maximum training epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        early_stop_patience: Epochs without improvement before stopping.
        lr_patience: Epochs without improvement before reducing LR.
        lr_factor: Factor to reduce LR by.
        checkpoint_dir: Directory for saving checkpoints.
        verbose: Print training progress.
    """
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    checkpoint_dir: str | None = None
    verbose: bool = True


@dataclass
class TrainingResult:
    """Result from training.

    Attributes:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        best_epoch: Epoch with best validation loss.
        best_val_loss: Best validation loss achieved.
    """
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


def train_surrogate(
    model: SurrogateModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a surrogate model.

    Args:
        model: SurrogateModel to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.

    Returns:
        TrainingResult with loss history.
    """
    _check_torch()

    config = config or TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_patience,
        factor=config.lr_factor,
    )

    # Tracking
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    # Checkpoint directory
    if config.checkpoint_dir:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch_eps, batch_metrics in train_loader:
            batch_eps = batch_eps.to(device)
            batch_metrics = batch_metrics.to(device)

            optimizer.zero_grad()
            predictions = model(batch_eps)
            loss = criterion(predictions, batch_metrics)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_eps.size(0)
            n_train += batch_eps.size(0)

        train_loss /= n_train
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch_eps, batch_metrics in val_loader:
                batch_eps = batch_eps.to(device)
                batch_metrics = batch_metrics.to(device)

                predictions = model(batch_eps)
                loss = criterion(predictions, batch_metrics)

                val_loss += loss.item() * batch_eps.size(0)
                n_val += batch_eps.size(0)

        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            if config.checkpoint_dir:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, ckpt_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Logging
        if config.verbose and epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, lr={lr:.2e}")

        # Early stopping
        if patience_counter >= config.early_stop_patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if config.checkpoint_dir:
        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

    if config.verbose:
        print(f"\nTraining complete. Best val_loss={best_val_loss:.6f} at epoch {best_epoch}")

    return TrainingResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )


def evaluate_model(
    model: SurrogateModel,
    test_loader: DataLoader,
) -> dict[str, float]:
    """Evaluate model on test data.

    Args:
        model: Trained model.
        test_loader: Test data loader.

    Returns:
        Dictionary of evaluation metrics.
    """
    _check_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_eps, batch_metrics in test_loader:
            batch_eps = batch_eps.to(device)
            predictions = model(batch_eps)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_metrics.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    # R² per output
    ss_res = np.sum((targets - preds) ** 2, axis=0)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
    r2_per_output = 1 - ss_res / (ss_tot + 1e-8)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2_mean": float(np.mean(r2_per_output)),
        "r2_per_output": r2_per_output.tolist(),
    }


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "train_surrogate",
    "evaluate_model",
]
