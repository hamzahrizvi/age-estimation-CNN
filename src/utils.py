"""Shared utility functions."""

from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history(history: tf.keras.callbacks.History, output_path: str | Path) -> Path:
    """Save Keras training history to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: [float(v) for v in values] for key, values in history.history.items()}
    output_path.write_text(json.dumps(serializable, indent=2))
    return output_path
