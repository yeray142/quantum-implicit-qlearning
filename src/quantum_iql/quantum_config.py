"""Configuration for the hybrid Quantum-IQL trainer (issue #9).

Extends the base IQLConfig with quantum-specific fields and a mode switch
that enables clean ablation studies between classical and quantum components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import OmegaConf

from .config import IQLConfig

# ---------------------------------------------------------------------------
# Layerwise warm-up schedule entry
# ---------------------------------------------------------------------------

@dataclass
class LayerwiseScheduleEntry:
    """One segment of the layerwise DRU warm-up schedule (Skolik et al., 2021).

    Attributes:
        start_step:   Training step at which this segment begins.
        active_layers: Number of DRU layers to activate from this point.
    """
    start_step: int = 0
    active_layers: int = 1


# ---------------------------------------------------------------------------
# Quantum-specific configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantumNetConfig:
    """Hyperparameters for the QuantumValueNetwork.

    Attributes:
        n_qubits:        Number of qubits in the DRU circuit.
        n_layers:        Total number of DRU re-uploading layers (L ≤ floor(log2(n_qubits))).
        device_name:     PennyLane device string, e.g. ``"lightning.qubit"`` or ``"default.qubit"``.
        running_stats:   Whether to maintain running μ/σ for arctan encoding.
        layerwise_schedule: List of (start_step, active_layers) pairs for progressive warm-up.
            Leave empty to start all layers active from step 0.
    """
    n_qubits: int = 8
    n_layers: int = 3
    device_name: str = "lightning.qubit"
    running_stats: bool = True
    layerwise_schedule: list[LayerwiseScheduleEntry] = field(
        default_factory=lambda: [
            LayerwiseScheduleEntry(start_step=0,      active_layers=1),
            LayerwiseScheduleEntry(start_step=100_000, active_layers=2),
            LayerwiseScheduleEntry(start_step=300_000, active_layers=3),
        ]
    )


# ---------------------------------------------------------------------------
# Full hybrid config
# ---------------------------------------------------------------------------

@dataclass
class QuantumIQLConfig(IQLConfig):
    """Full configuration for the hybrid Q-IQL trainer.

    Inherits all classical IQL fields and adds:

    Attributes:
        mode:          ``"classical"`` or ``"quantum"``.  Controls which
                       value-network implementation is used.  All other
                       hyperparameters remain shared so a single config file
                       drives both ablation arms.
        quantum_value: Parameters for QuantumValueNetwork (used when
                       ``mode="quantum"``).
        lr_quantum:    Learning rate for the quantum circuit parameters.
                       Defaults to ``lr_v``; set explicitly to tune separately.
        quantum_grad_clip: Max-norm gradient clipping applied to quantum
                       parameters only (0.0 = disabled).
        log_quantum_metrics: Whether to log quantum-specific diagnostics
                       (grad norms, circuit parameter evolution, shot noise
                       proxy) to W&B.  Adds a small overhead per log step.
        stats_update_interval: How often (in steps) to refresh the running
                       μ/σ used for arctan encoding.  0 = never update after
                       init (useful for eval / frozen normalisation).
    """
    mode: str = "classical"            # "classical" | "quantum"
    quantum_value: QuantumNetConfig = field(default_factory=QuantumNetConfig)

    # Optimisation
    lr_quantum: float = 3e-4
    quantum_grad_clip: float = 1.0     # max-norm clipping for quantum params

    # Diagnostics
    log_quantum_metrics: bool = True
    stats_update_interval: int = 1_000  # steps between running-stats refresh


def load_quantum_config(
    path: str,
    overrides: list[str] | None = None,
) -> QuantumIQLConfig:
    """Load a QuantumIQLConfig from YAML, then apply CLI overrides.

    Args:
        path:      Path to a YAML config file.
        overrides: List of ``"key=value"`` or ``"nested.key=value"`` strings.

    Returns:
        A fully resolved :class:`QuantumIQLConfig` instance.

    Example::

        cfg = load_quantum_config(
            "configs/quantum_iql_hopper.yaml",
            ["mode=quantum", "quantum_value.n_qubits=4", "seed=42"],
        )
    """
    base = OmegaConf.structured(QuantumIQLConfig)
    from_file = OmegaConf.load(path)
    merged = OmegaConf.merge(base, from_file)

    if overrides:
        from omegaconf import OmegaConf as _OC
        override_cfg = _OC.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)

    cfg: QuantumIQLConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    return cfg
