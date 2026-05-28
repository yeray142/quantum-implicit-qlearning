"""Quantum Implicit Q-Learning (quantum-iql)."""

from .buffer import Batch, ReplayBuffer, load_minari_dataset
from .config import IQLConfig, NetworkConfig, load_config
from .networks import ActorNetwork, CriticNetwork, ValueNetwork, build_mlp

# Hybrid quantum-classical pipeline (issue #9)
from .quantum_config import (  # noqa: E402
    LayerwiseScheduleEntry,
    QuantumIQLConfig,
    QuantumNetConfig,
    load_quantum_config,
    make_layerwise_schedule,
)
from .quantum_trainer import QuantumIQLTrainer  # noqa: E402
from .quantum_value_network import QuantumValueNetwork
from .trainer import IQLTrainer

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
    "QuantumValueNetwork",
    "QuantumIQLConfig",
    "QuantumNetConfig",
    "LayerwiseScheduleEntry",
    "make_layerwise_schedule",
    "load_quantum_config",
    "QuantumIQLTrainer",
]
