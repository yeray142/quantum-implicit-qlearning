# Quantum Implicit Q-Learning

Research project for **QC810: Quantum Machine Learning** — MSc in Quantum Computing, University of Southern Denmark.

This repository implements **Implicit Q-Learning (IQL)** (Kostrikov et al., 2021) as a classical baseline, with a modular architecture designed for progressive quantum extensions using PennyLane.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [User Installation](#user-installation)
  - [Developer Installation](#developer-installation)
- [Configuration](#configuration)
- [Scripts](#scripts)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Testing](#testing)
- [Quantum Extension Guide](#quantum-extension-guide)
- [References](#references)

---

## Overview

IQL is an offline reinforcement learning algorithm that avoids querying out-of-distribution actions during training. It uses:

- **Expectile regression** to train a value network `V(s)` without ever sampling from the policy
- **Bellman backups** with a frozen EMA target `V̄(s)` to train the Q-network `Q(s, a)`
- **Advantage-weighted behaviour cloning** to extract a policy `π(a|s)`

The three losses are:

```
L_V(ψ)  = E[Lτ(Q(s,a) − V(s))]          where Lτ(u) = |τ − 1(u<0)| · u²
L_Q(θ)  = E[(r + γ·V̄(s') − Q(s,a))²]
L_π(φ)  = −E[exp(β·(Q(s,a)−V(s))) · log π(a|s)]
```
---

## Project Structure

```
quantum-implicit-qlearning/
├── configs/
│   └── iql_hopper.yaml          # Reference config for hopper-medium-v0
├── scripts/
│   ├── train_iql.py             # Training entry point
│   └── eval_iql.py              # Standalone evaluation / checkpoint inspection
├── src/
│   └── quantum_iql/
│       ├── __init__.py          # Public API re-exports
│       ├── config.py            # IQLConfig + NetworkConfig dataclasses
│       ├── buffer.py            # Offline replay buffer + Minari loader
│       ├── networks.py          # build_mlp, ValueNetwork, CriticNetwork, ActorNetwork
│       ├── losses.py            # expectile_loss, value_loss, critic_loss, actor_loss
│       ├── trainer.py           # IQLTrainer (training loop, eval, checkpointing)
│       └── utils.py             # soft_update, set_seed, get_device, make_env
├── tests/
│   ├── test_config.py
│   ├── test_buffer.py
│   ├── test_networks.py
│   ├── test_losses.py
│   ├── test_trainer.py
│   └── test_utils.py
├── checkpoints/                 # Saved during training (auto-created)
├── .github/workflows/ci.yml     # GitHub Actions: lint + type-check + tests
└── pyproject.toml
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended for full training runs (CPU works for smoke tests)
- MuJoCo physics engine (installed automatically via `gymnasium[mujoco]`)

---

### User Installation

Install the package and its runtime dependencies:

```bash
git clone https://github.com/yeray142/quantum-implicit-qlearning.git
cd quantum-implicit-qlearning
pip install -e .
```

Core dependencies installed automatically:

| Package | Purpose |
|---|---|
| `torch>=2.3` | Neural networks and autograd |
| `gymnasium[mujoco]>=0.29` | MuJoCo environments for evaluation |
| `minari>=0.5` | Offline dataset download and loading |
| `wandb>=0.17` | Experiment tracking and logging |
| `pennylane>=0.38` | Quantum circuit integration (future) |
| `omegaconf>=2.3` | Typed YAML config with CLI overrides |

---

### Developer Installation

Install with development extras (testing, linting, type-checking):

```bash
git clone https://github.com/yeray142/quantum-implicit-qlearning.git
cd quantum-implicit-qlearning
pip install -e ".[dev]"
```

Additional dev dependencies:

| Package | Purpose |
|---|---|
| `pytest>=8.0` | Test runner |
| `pytest-cov>=5.0` | Coverage reports |
| `ruff>=0.5` | Linter and formatter |
| `mypy>=1.10` | Static type checking |

Verify the installation:

```bash
python -c "import quantum_iql; print(quantum_iql.__version__)"
# 0.1.0
```

---

## Configuration

All hyperparameters are defined in YAML files under `configs/` and loaded into the typed `IQLConfig` dataclass. The reference config is `configs/iql_hopper.yaml`.

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `dataset_id` | `mujoco/hopper/medium-v0` | Minari dataset identifier |
| `env_id` | `Hopper-v4` | Gymnasium environment for evaluation |
| `tau` | `0.7` | Expectile level — use `0.9+` for AntMaze tasks |
| `beta` | `3.0` | Advantage inverse temperature |
| `gamma` | `0.99` | Discount factor |
| `polyak` | `0.005` | EMA rate for the V-target network |
| `lr_v / lr_q / lr_actor` | `3e-4` | Per-network learning rates |
| `batch_size` | `256` | Transitions per gradient step |
| `num_steps` | `1_000_000` | Total gradient steps |
| `use_twin_critic` | `true` | Twin Q-heads to reduce overestimation |
| `advantage_clip` | `100.0` | Max value for `exp(β·A)` in actor loss |
| `warmup_steps` | `0` | Steps before actor updates begin |
| `device` | `auto` | `"auto"` selects CUDA if available |

### Overriding config values at the command line

Any config key can be overridden at runtime using `--overrides key=value`:

```bash
python scripts/train_iql.py --config configs/iql_hopper.yaml \
    --overrides tau=0.9 beta=5.0 seed=42
```

Nested keys use dot notation:

```bash
python scripts/train_iql.py --config configs/iql_hopper.yaml \
    --overrides value_net.hidden_dims=[512,512]
```

---

## Scripts

### Training

```bash
python scripts/train_iql.py --config CONFIG [--overrides KEY=VALUE ...]
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--config` | Yes | Path to YAML config file |
| `--overrides` | No | Space-separated `key=value` overrides |

**Examples:**

```bash
# Full training run on hopper-medium-v0 (downloads dataset on first run)
python scripts/train_iql.py --config configs/iql_hopper.yaml

# Smoke test — 5 steps, W&B offline (no internet required)
python scripts/train_iql.py --config configs/iql_hopper.yaml \
    --overrides num_steps=5 wandb_offline=true

# Custom hyperparameters with a different seed
python scripts/train_iql.py --config configs/iql_hopper.yaml \
    --overrides tau=0.9 seed=1 num_steps=500000

# CPU-only run
python scripts/train_iql.py --config configs/iql_hopper.yaml \
    --overrides device=cpu num_steps=10000
```

**What happens during training:**
- The Minari dataset is downloaded automatically on the first run (~140 MB for hopper)
- Losses are logged to W&B every `log_interval` steps
- The agent is evaluated every `eval_interval` steps; mean return is printed to stdout
- Checkpoints are saved to `checkpoints/checkpoint_XXXXXXXX.pt` at each eval interval
- A final checkpoint is saved to `checkpoints/checkpoint_final.pt`

**W&B metrics logged:**

| Metric | Description |
|---|---|
| `loss/value` | Expectile regression loss for V |
| `loss/critic` | Bellman MSE loss for Q |
| `loss/actor` | Advantage-weighted BC loss for π |
| `advantage_mean` | Mean advantage A(s,a) = Q−V (should stay near 0) |
| `advantage_std` | Std of advantage (monitor for instability) |
| `exp_adv_mean` | Mean of exp(β·A) weighting the actor loss |
| `eval/mean_return` | Mean episode return over eval episodes |
| `eval/std_return` | Std of episode returns |
| `train/steps_per_sec` | Training throughput |

---

### Evaluation

```bash
python scripts/eval_iql.py --config CONFIG [--checkpoint PATH] [--episodes N] [--seed N] [--render]
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--config` | Required | Path to YAML config file |
| `--checkpoint` | None | Path to a `.pt` checkpoint file. If omitted, evaluates the untrained (random-init) policy |
| `--episodes` | `10` | Number of evaluation episodes |
| `--seed` | config value | Override the random seed for evaluation |
| `--render` | off | Render episodes visually (requires a display) |

**Examples:**

```bash
# Baseline: evaluate the untrained (random-init) policy
python scripts/eval_iql.py --config configs/iql_hopper.yaml --episodes 10

# Evaluate a specific checkpoint
python scripts/eval_iql.py --config configs/iql_hopper.yaml \
    --checkpoint checkpoints/checkpoint_00010000.pt --episodes 20

# Evaluate the final checkpoint with rendering
python scripts/eval_iql.py --config configs/iql_hopper.yaml \
    --checkpoint checkpoints/checkpoint_final.pt --render

# Tight confidence interval (100 episodes)
python scripts/eval_iql.py --config configs/iql_hopper.yaml \
    --checkpoint checkpoints/checkpoint_final.pt --episodes 100
```

**Expected output:**

```
Loaded checkpoint from 'checkpoints/checkpoint_final.pt' (step=1000000)

Evaluation over 10 episodes on 'Hopper-v4':
  mean return :  2847.32
  std  return :   312.41
  min  return :  2201.15
  max  return :  3214.78
```

**Interpreting results:**

| Mean return | Interpretation |
|---|---|
| ~20–200 | Random policy (untrained) |
| ~1000–1500 | Early training (~100k steps) |
| ~2500–3000 | Converged policy (~1M steps) |
| >3000 | Well-tuned / expert-level |

---

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ --cov=quantum_iql --cov-report=term-missing
```

Run a specific test file:

```bash
pytest tests/test_losses.py -v
pytest tests/test_networks.py -v
pytest tests/test_trainer.py -v
```

**Test coverage by module:**

| Module | Coverage | Notes |
|---|---|---|
| `config.py` | ~98% | YAML loading, overrides, defaults |
| `losses.py` | ~97% | Expectile correctness, gradient flow |
| `networks.py` | ~99% | Shape checks, twin critic, log-prob |
| `buffer.py` | ~80% | Synthetic episodes; `load_minari_dataset` excluded (requires download) |
| `trainer.py` | ~86% | Smoke tests; `train()` loop excluded (requires W&B) |
| `utils.py` | ~90% | Seed, device, soft/hard update |

All tests run without internet access, GPU, or a Minari dataset download.

**Linting and type-checking:**

```bash
ruff check src/ tests/       # linter
mypy src/quantum_iql         # type checker
```

---

## Quantum Extension Guide

The codebase is structured so that quantum variants require minimal changes.
The single integration point is `build_mlp` in `src/quantum_iql/networks.py` and the `_build_networks` method in `IQLTrainer`.

### Replacing a network with a quantum variant

```python
from quantum_iql.trainer import IQLTrainer
from quantum_iql.networks import CriticNetwork, ActorNetwork

class QuantumIQLTrainer(IQLTrainer):
    def _build_networks(self) -> None:
        # Use a quantum value network, keep classical Q and actor
        self.value_net = QuantumValueNetwork(...).to(self.device)
        self.critic_net = CriticNetwork(...).to(self.device)
        self.actor_net = ActorNetwork(...).to(self.device)
```

Everything else — update logic, target networks, logging, checkpointing — is inherited unchanged.

---

## References

- Kostrikov, I., Nair, A., & Levine, S. (2021). **Offline Reinforcement Learning with Implicit Q-Learning.** *ICLR 2022*. https://arxiv.org/abs/2110.06169
- Official IQL implementation: https://github.com/ikostrikov/implicit_q_learning
- Minari dataset API: https://minari.farama.org
- PennyLane quantum ML framework: https://pennylane.ai