"""Neural models module for PhotonicForge.

Provides neural surrogate models for fast simulation prediction.
Requires PyTorch - install with `pip install -e ".[ml]"`.
"""

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Conditional imports
if HAS_TORCH:
    from photonic_forge.neural_models.surrogate import (
        SurrogateConfig,
        PermittivityEncoder,
        SurrogateModel,
        predict_metrics,
        load_surrogate,
        save_surrogate,
    )
    from photonic_forge.neural_models.dataloader import (
        TrainingSample,
        SimulationDataset,
        create_data_loaders,
        generate_synthetic_data,
    )
    from photonic_forge.neural_models.training import (
        TrainingConfig,
        TrainingResult,
        train_surrogate,
        evaluate_model,
    )

    __all__ = [
        "HAS_TORCH",
        # Surrogate
        "SurrogateConfig",
        "PermittivityEncoder",
        "SurrogateModel",
        "predict_metrics",
        "load_surrogate",
        "save_surrogate",
        # Data
        "TrainingSample",
        "SimulationDataset",
        "create_data_loaders",
        "generate_synthetic_data",
        # Training
        "TrainingConfig",
        "TrainingResult",
        "train_surrogate",
        "evaluate_model",
    ]
else:
    __all__ = ["HAS_TORCH"]
