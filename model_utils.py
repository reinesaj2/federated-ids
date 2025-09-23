"""
Model persistence utilities for federated learning.
Provides unified save/load functionality for client and global models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn


def save_model(
    model: nn.Module, path: Union[str, Path], metadata: Optional[Dict] = None
) -> None:
    """
    Save a PyTorch model with optional metadata.

    Args:
        model: The PyTorch model to save
        path: File path where to save the model (.pth extension)
        metadata: Optional metadata dict to include in saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }

    if metadata:
        save_dict["metadata"] = metadata

    torch.save(save_dict, path)


def load_model(path: Union[str, Path], model_class, **model_kwargs) -> nn.Module:
    """
    Load a PyTorch model from saved state.

    Args:
        path: File path to the saved model (.pth file)
        model_class: The model class to instantiate
        **model_kwargs: Keyword arguments to pass to model constructor

    Returns:
        Loaded PyTorch model
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    # Create model instance
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_client_model(
    model: nn.Module,
    run_dir: Union[str, Path],
    client_id: int,
    round_num: int,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Save a client model with standardized naming.

    Args:
        model: The client model to save
        run_dir: Run directory path
        client_id: Client identifier
        round_num: FL round number
        metadata: Optional metadata to include

    Returns:
        Path to saved model file
    """
    run_dir = Path(run_dir)
    filename = f"client_{client_id}_model_round_{round_num}.pth"
    path = run_dir / filename

    client_metadata = {
        "client_id": client_id,
        "round": round_num,
        "model_type": "client",
    }
    if metadata:
        client_metadata.update(metadata)

    save_model(model, path, client_metadata)
    return path


def save_global_model(
    model: nn.Module,
    run_dir: Union[str, Path],
    round_num: int,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Save a global model with standardized naming.

    Args:
        model: The global model to save
        run_dir: Run directory path
        round_num: FL round number
        metadata: Optional metadata to include

    Returns:
        Path to saved model file
    """
    run_dir = Path(run_dir)
    filename = f"global_model_round_{round_num}.pth"
    path = run_dir / filename

    global_metadata = {
        "round": round_num,
        "model_type": "global",
    }
    if metadata:
        global_metadata.update(metadata)

    save_model(model, path, global_metadata)
    return path


def save_final_model(
    model: nn.Module, run_dir: Union[str, Path], metadata: Optional[Dict] = None
) -> Path:
    """
    Save the final trained model.

    Args:
        model: The final model to save
        run_dir: Run directory path
        metadata: Optional metadata to include

    Returns:
        Path to saved model file
    """
    run_dir = Path(run_dir)
    filename = "final_global_model.pth"
    path = run_dir / filename

    final_metadata = {
        "model_type": "final_global",
    }
    if metadata:
        final_metadata.update(metadata)

    save_model(model, path, final_metadata)
    return path


def list_model_artifacts(run_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Discover and categorize model artifacts in a run directory.

    Args:
        run_dir: Run directory path to search

    Returns:
        Dictionary with categorized model paths:
        - 'client_models': List of client model paths
        - 'global_models': List of global model paths
        - 'final_model': List containing final model path (if exists)
    """
    run_dir = Path(run_dir)
    artifacts = {"client_models": [], "global_models": [], "final_model": []}

    if not run_dir.exists():
        return artifacts

    # Find all .pth files
    for pth_file in run_dir.glob("*.pth"):
        filename = pth_file.name

        if filename.startswith("client_") and "_model_round_" in filename:
            artifacts["client_models"].append(pth_file)
        elif filename.startswith("global_model_round_"):
            artifacts["global_models"].append(pth_file)
        elif filename == "final_global_model.pth":
            artifacts["final_model"].append(pth_file)

    # Sort by round number where applicable
    artifacts["client_models"].sort()
    artifacts["global_models"].sort()

    return artifacts


def get_model_metadata(path: Union[str, Path]) -> Optional[Dict]:
    """
    Extract metadata from a saved model file.

    Args:
        path: Path to the saved model file

    Returns:
        Metadata dictionary if present, None otherwise
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint.get("metadata")
    except Exception:
        return None
