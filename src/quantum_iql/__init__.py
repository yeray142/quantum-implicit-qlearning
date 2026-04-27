"""Quantum Implicit Q-Learning (quantum-iql)."""

import os as _os
import sys as _sys

from .buffer import Batch, ReplayBuffer, load_minari_dataset
from .config import IQLConfig, NetworkConfig, load_config
from .networks import ActorNetwork, CriticNetwork, ValueNetwork, build_mlp
from .trainer import IQLTrainer

# QuantumValueNetwork lives in scripts/ (issue #7).
# Add scripts/ to sys.path so submodules can import it by bare name.
_scripts_dir = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "scripts")
)
if _scripts_dir not in _sys.path:
    _sys.path.insert(0, _scripts_dir)

# Hybrid quantum-classical pipeline (issue #9)
from .quantum_config import (  # noqa: E402
    LayerwiseScheduleEntry,
    QuantumIQLConfig,
    QuantumNetConfig,
    load_quantum_config,
)
from .quantum_trainer import QuantumIQLTrainer  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    # Classical
    "IQLConfig",
    "NetworkConfig",
    "load_config",
    "Batch",
    "ReplayBuffer",
    "load_minari_dataset",
    "ActorNetwork",
    "CriticNetwork",
    "ValueNetwork",
    "build_mlp",
    "IQLTrainer",
    # Hybrid Q-IQL
    "QuantumIQLConfig",
    "QuantumNetConfig",
    "LayerwiseScheduleEntry",
    "load_quantum_config",
    "QuantumIQLTrainer",
]
