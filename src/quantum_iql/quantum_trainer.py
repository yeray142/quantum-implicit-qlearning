"""Hybrid Quantum-IQL trainer (issue #9).

Integrates the QuantumValueNetwork from issue #7 into the classical IQL
training loop from issue #2, producing the first end-to-end hybrid Q-IQL
pipeline.

Design decisions
----------------
* ``QuantumIQLTrainer`` subclasses ``IQLTrainer`` and overrides only
  ``_build_networks`` and ``_build_optimizers``.  Every loss computation,
  target-network update, evaluation loop, and checkpoint save is reused
  unchanged — this is the "drop-in replacement" guarantee from the issue spec.

* The ``mode`` field in ``QuantumIQLConfig`` selects classical vs. quantum at
  construction time, so the same script / config file drives ablation studies
  with a single CLI override (``mode=quantum`` / ``mode=classical``).

* Gradient flow: PennyLane's ``diff_method="parameter-shift"`` converts the
  QNode into a differentiable PyTorch function.  The quantum parameters
  (``theta``, ``w``) and the classical affine head (``a``, ``b``) all live in
  ``value_optimizer`` / ``quantum_optimizer``, so a single ``optimizer.step()``
  updates both.  No manual parameter-shift bookkeeping is needed in the
  training loop.

* Quantum-specific metrics logged per ``log_interval``:
    - ``quantum/grad_norm_theta`` — L2 norm of ∂L/∂θ after backward()
    - ``quantum/grad_norm_w``     — L2 norm of ∂L/∂w
    - ``quantum/param_theta_mean`` / ``…_std`` — circuit parameter statistics
    - ``quantum/param_w_mean``    / ``…_std``
    - ``quantum/value_output_std``  — std(V(s)) over a held-out probe batch;
      measures output spread across states, NOT shot noise (which requires
      repeated analytic evaluations of the same input)

* Layerwise warm-up: ``_apply_layerwise_schedule`` checks the current step
  against the schedule in ``QuantumIQLConfig.quantum_value.layerwise_schedule``
  and calls ``set_active_layers`` when a threshold is crossed.

* Running-stats refresh: ``_refresh_running_stats`` computes a new μ/σ from
  ``stats_update_interval`` random buffer samples and pushes them to the
  QuantumValueNetwork.  This keeps the arctan encoding in a good dynamic range
  as the buffer statistics are learned.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.optim as optim

# Quantum components (from issue #7 — lives in scripts/)
from quantum_value_network import QuantumValueNetwork

import wandb

from .buffer import Batch, ReplayBuffer
from .losses import value_loss
from .networks import ActorNetwork, CriticNetwork, ValueNetwork

# Hybrid config (this issue)
from .quantum_config import QuantumIQLConfig
from .trainer import IQLTrainer
from .utils import hard_update

# ---------------------------------------------------------------------------
# Helper: gradient-norm utility
# ---------------------------------------------------------------------------

def _grad_norm(param: torch.nn.Parameter) -> float:
    """Return the L2 norm of a parameter's gradient, or 0.0 if None."""
    if param.grad is None:
        return 0.0
    return param.grad.detach().norm(2).item()


# ---------------------------------------------------------------------------
# Hybrid trainer
# ---------------------------------------------------------------------------

class QuantumIQLTrainer(IQLTrainer):
    """End-to-end hybrid Q-IQL trainer.

    Inherits from :class:`~quantum_iql.trainer.IQLTrainer`.  When
    ``config.mode == "quantum"`` the value network V_ψ is replaced by a
    :class:`~scripts.quantum_value_network.QuantumValueNetwork`; all other
    components remain classical.

    When ``config.mode == "classical"`` the trainer behaves identically to the
    base ``IQLTrainer``, making it a transparent drop-in for ablation runs.

    Args:
        config:  A :class:`~quantum_iql.quantum_config.QuantumIQLConfig`.
        buffer:  Pre-populated offline replay buffer.
        env:     Gymnasium environment for evaluation.
    """

    def __init__(
        self,
        config: QuantumIQLConfig,
        buffer: ReplayBuffer,
        env: Any,
    ) -> None:
        # Store quantum config before super().__init__ calls _build_networks
        self._qcfg = config
        super().__init__(config, buffer, env)

        # Layerwise schedule: track which segment is currently active
        self._schedule_idx: int = 0

        # Shot-noise proxy: a small held-out batch re-used every log step
        _probe = buffer.sample(min(64, buffer._size))
        self._probe_obs: torch.Tensor = _probe.observations.to(self.device)

        # Step at which running stats were last refreshed
        self._last_stats_refresh: int = 0

    # ------------------------------------------------------------------
    # Network construction (overrides IQLTrainer._build_networks)
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Instantiate networks based on config.mode.

        - ``"quantum"``: V_ψ → QuantumValueNetwork; Q and π stay classical.
        - ``"classical"``: all three networks are classical MLPs (parent logic).
        """
        obs_dim = self.buffer.obs_dim
        act_dim = self.buffer.act_dim
        cfg: QuantumIQLConfig = self._qcfg

        # ── Critic and Actor are always classical ──────────────────────────
        qcfg = cfg.critic_net
        acfg = cfg.actor_net

        self.critic_net = CriticNetwork(
            obs_dim,
            act_dim,
            hidden_dims=qcfg.hidden_dims,
            activation=qcfg.activation,
            layer_norm=qcfg.layer_norm,
            use_twin=cfg.use_twin_critic,
        ).to(self.device)

        self.actor_net = ActorNetwork(
            obs_dim,
            act_dim,
            hidden_dims=acfg.hidden_dims,
            activation=acfg.activation,
            layer_norm=acfg.layer_norm,
        ).to(self.device)

        # ── Value network: quantum or classical ────────────────────────────
        if cfg.mode == "quantum":
            qv = cfg.quantum_value
            self.value_net: ValueNetwork | QuantumValueNetwork = QuantumValueNetwork(
                n_qubits=qv.n_qubits,
                n_layers=qv.n_layers,
                obs_dim=obs_dim,
                device_name=qv.device_name,
                running_stats=qv.running_stats,
            ).to(self.device)
            self._is_quantum = True
            print(
                f"[QuantumIQLTrainer] mode=quantum  →  {self.value_net!r}\n"
                f"  PennyLane device : {qv.device_name}\n"
                f"  diff_method      : parameter-shift\n"
                f"  obs_dim          : {obs_dim}  →  {qv.n_qubits} qubits "
                f"({'pad' if obs_dim < qv.n_qubits else 'truncate' if obs_dim > qv.n_qubits else 'exact'})"
            )
        else:
            vcfg = cfg.value_net
            self.value_net = ValueNetwork(
                obs_dim,
                hidden_dims=vcfg.hidden_dims,
                activation=vcfg.activation,
                layer_norm=vcfg.layer_norm,
            ).to(self.device)
            self._is_quantum = False
            print(f"[QuantumIQLTrainer] mode=classical  →  ValueNetwork{tuple(cfg.value_net.hidden_dims)}")

    def _build_targets(self) -> None:
        """EMA value-target V̄.

        When mode=quantum, V̄ is also a QuantumValueNetwork with identical
        architecture so that soft_update can copy matching parameter shapes.
        V̄ is initialised as a hard copy of V and never optimised directly.

        When mode=classical, V̄ is a classical MLP (same as parent behaviour).
        """
        cfg: QuantumIQLConfig = self._qcfg

        if self._is_quantum:
            qv = cfg.quantum_value
            self.value_target = QuantumValueNetwork(
                n_qubits=qv.n_qubits,
                n_layers=qv.n_layers,
                obs_dim=self.buffer.obs_dim,
                device_name=qv.device_name,
                running_stats=qv.running_stats,
            ).to(self.device)
            hard_update(self.value_target, self.value_net)
        else:
            vcfg = cfg.value_net
            self.value_target = ValueNetwork(
                self.buffer.obs_dim,
                hidden_dims=vcfg.hidden_dims,
                activation=vcfg.activation,
                layer_norm=vcfg.layer_norm,
            ).to(self.device)
            hard_update(self.value_target, self.value_net)

        for p in self.value_target.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Optimizers (overrides IQLTrainer._build_optimizers)
    # ------------------------------------------------------------------

    def _build_optimizers(self) -> None:
        cfg: QuantumIQLConfig = self._qcfg

        if self._is_quantum:
            # Separate optimizer for quantum parameters so we can apply
            # independent lr and gradient clipping without disturbing critic/actor.
            lr_q = cfg.lr_quantum if cfg.lr_quantum else cfg.lr_v
            self.value_optimizer = optim.Adam(
                self.value_net.parameters(), lr=lr_q
            )
        else:
            self.value_optimizer = optim.Adam(
                self.value_net.parameters(), lr=cfg.lr_v
            )

        self.critic_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=cfg.lr_q
        )
        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=cfg.lr_actor
        )

    # ------------------------------------------------------------------
    # Value update (adds quantum grad clipping)
    # ------------------------------------------------------------------

    def update_value(self, batch: Batch) -> dict[str, float]:
        """Gradient step on V_ψ (quantum or classical).

        Adds optional gradient clipping for quantum parameters and collects
        quantum-specific diagnostic metrics when ``log_quantum_metrics=True``.

        Returns:
            ``{"loss/value": float}`` plus, when mode=quantum:
            ``{"quantum/grad_norm_theta": float, "quantum/grad_norm_w": float}``
        """
        cfg: QuantumIQLConfig = self._qcfg
        self.value_optimizer.zero_grad()

        # ── forward + backward (parameter-shift handled by PennyLane) ─────
        # QuantumValueNetwork.forward now returns (B, 1) float32 directly,
        # matching the ValueNetwork contract — no wrapper needed.
        loss = value_loss(self.value_net, self.critic_net, batch, cfg.tau)
        loss.backward()

        metrics: dict[str, float] = {"loss/value": loss.item()}

        # ── quantum diagnostics & clipping ────────────────────────────────
        if self._is_quantum and cfg.log_quantum_metrics:
            qnet: QuantumValueNetwork = self.value_net  # type: ignore[assignment]
            metrics["quantum/grad_norm_theta"] = _grad_norm(qnet.theta)
            metrics["quantum/grad_norm_w"]     = _grad_norm(qnet.w)

        if self._is_quantum and cfg.quantum_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(), cfg.quantum_grad_clip
            )

        self.value_optimizer.step()
        return metrics

    # ------------------------------------------------------------------
    # Quantum-specific diagnostics
    # ------------------------------------------------------------------

    def _quantum_param_metrics(self) -> dict[str, float]:
        """Circuit parameter statistics for W&B (logged at log_interval)."""
        if not self._is_quantum:
            return {}
        qnet: QuantumValueNetwork = self.value_net  # type: ignore[assignment]
        with torch.no_grad():
            theta = qnet.theta.detach()
            w = qnet.w.detach()
        return {
            "quantum/param_theta_mean": theta.mean().item(),
            "quantum/param_theta_std":  theta.std().item(),
            "quantum/param_w_mean":     w.mean().item(),
            "quantum/param_w_std":      w.std().item(),
        }

    def _value_output_std(self) -> dict[str, float]:
        """Compute std(V(s)) over the fixed probe batch.

        This measures how spread the value outputs are across different states,
        which is a useful training-stability diagnostic.

        Note: this is NOT shot noise. True shot noise requires running the same
        input multiple times with finite shots and measuring the variance of
        those repeated estimates. Since we use analytic simulators
        (``default.qubit`` / ``lightning.qubit``), there is no shot noise.
        To measure shot noise you would need a finite-shot device and compare
        V(s_probe, shots=N) repeated K times.
        """
        if not self._is_quantum:
            return {}
        self.value_net.eval()
        with torch.no_grad():
            v_probe = self.value_net(self._probe_obs)
        self.value_net.train()
        return {"quantum/value_output_std": v_probe.std().item()}

    # ------------------------------------------------------------------
    # Layerwise warm-up
    # ------------------------------------------------------------------

    def _apply_layerwise_schedule(self, step: int) -> None:
        """Update active DRU layers according to the warm-up schedule."""
        if not self._is_quantum:
            return
        cfg: QuantumIQLConfig = self._qcfg
        schedule = cfg.quantum_value.layerwise_schedule
        if not schedule:
            return

        # Find the last schedule entry whose start_step ≤ current step
        active = None
        for entry in schedule:
            if step >= entry.start_step:
                active = entry
        if active is None:
            return

        qnet: QuantumValueNetwork = self.value_net  # type: ignore[assignment]
        if qnet.active_layers != active.active_layers:
            qnet.set_active_layers(active.active_layers)
            print(
                f"  [layerwise-warmup] step={step:,}  →  "
                f"active_layers={active.active_layers}"
            )

    # ------------------------------------------------------------------
    # Running-stats refresh
    # ------------------------------------------------------------------

    def _refresh_running_stats(self, step: int) -> None:
        """Recompute μ/σ from a random buffer sample and push to the circuit."""
        if not self._is_quantum:
            return
        cfg: QuantumIQLConfig = self._qcfg
        qnet: QuantumValueNetwork = self.value_net  # type: ignore[assignment]
        if not qnet._running_stats:
            return
        if cfg.stats_update_interval <= 0:
            return
        if step - self._last_stats_refresh < cfg.stats_update_interval:
            return

        sample = self.buffer.sample(min(4096, self.buffer._size))
        obs = sample.observations  # CPU tensor
        mu = obs.mean(dim=0).to(self.device)
        sigma = obs.std(dim=0).clamp(min=1e-4).to(self.device)
        qnet.update_running_stats(mu, sigma)
        self._last_stats_refresh = step

    # ------------------------------------------------------------------
    # Full training step (overrides IQLTrainer.train_step)
    # ------------------------------------------------------------------

    def train_step(self) -> dict[str, float]:
        """One complete hybrid Q-IQL step.

        Order of operations:
        1. Sample batch from buffer.
        2. Layerwise warm-up check.
        3. Running-stats refresh (quantum only).
        4. Update V (quantum or classical).
        5. Update Q (classical, unchanged).
        6. Update π (classical, unchanged, after warmup_steps).
        7. Soft-update V̄.

        Returns:
            Merged metrics dict from all active updates.
        """
        cfg: QuantumIQLConfig = self._qcfg
        step = self._step + 1  # 1-indexed for schedule comparisons

        # ── 1. Sample ─────────────────────────────────────────────────────
        batch = self.buffer.sample(cfg.batch_size)
        batch = Batch(
            observations=batch.observations.to(self.device),
            actions=batch.actions.to(self.device),
            rewards=batch.rewards.to(self.device),
            next_observations=batch.next_observations.to(self.device),
            dones=batch.dones.to(self.device),
        )

        # ── 2–3. Quantum bookkeeping ───────────────────────────────────────
        self._apply_layerwise_schedule(step)
        self._refresh_running_stats(step)

        # ── 4–6. Parameter updates ─────────────────────────────────────────
        metrics: dict[str, float] = {}
        metrics.update(self.update_value(batch))
        metrics.update(self.update_critic(batch))

        if self._step >= cfg.warmup_steps:
            metrics.update(self.update_actor(batch))

        # ── 7. EMA target ─────────────────────────────────────────────────
        self.update_targets()
        self._step += 1
        return metrics

    # ------------------------------------------------------------------
    # Main training loop (overrides IQLTrainer.train to add quantum logs)
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full hybrid Q-IQL training loop.

        Extends the base loop by injecting quantum diagnostics (param stats,
        shot-noise proxy) into the W&B log at every ``log_interval``.
        """
        cfg: QuantumIQLConfig = self._qcfg
        mode_tag = "quantum" if self._is_quantum else "classical"
        print(
            f"Training hybrid Q-IQL [{mode_tag}] on {cfg.dataset_id} "
            f"for {cfg.num_steps:,} steps  (device={self.device})"
        )
        t0 = time.time()

        for step in range(1, cfg.num_steps + 1):
            metrics = self.train_step()

            if step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                metrics["train/step"] = step
                metrics["train/steps_per_sec"] = step / elapsed
                metrics["train/mode"] = 0.0 if not self._is_quantum else 1.0

                # Quantum-specific diagnostics
                if self._is_quantum and cfg.log_quantum_metrics:
                    metrics.update(self._quantum_param_metrics())
                    metrics.update(self._value_output_std())
                    if self._is_quantum:
                        qnet: QuantumValueNetwork = self.value_net  # type: ignore
                        metrics["quantum/active_layers"] = float(qnet.active_layers)

                wandb.log(metrics, step=step)

            if step % cfg.eval_interval == 0:
                eval_metrics = self.evaluate()
                wandb.log(eval_metrics, step=step)
                mean_ret = eval_metrics["eval/mean_return"]
                print(f"  step {step:>8,}  |  eval return = {mean_ret:.1f}")
                self.save_checkpoint(f"checkpoint_{step:08d}.pt")

        print(f"Training complete in {time.time() - t0:.1f}s")
        self.save_checkpoint("checkpoint_final.pt")

    # ------------------------------------------------------------------
    # Checkpoint (extended to persist quantum circuit parameters)
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        """Save all network weights including quantum circuit parameters.

        Args:
            filename: Filename (not path) for the checkpoint.
        """
        import os
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", filename)

        payload: dict = {
            "step": self._step,
            "mode": "quantum" if self._is_quantum else "classical",
            "value_net": self.value_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "actor_net": self.actor_net.state_dict(),
        }

        if self._is_quantum:
            qnet: QuantumValueNetwork = self.value_net  # type: ignore
            payload["quantum_meta"] = {
                "n_qubits": qnet.n_qubits,
                "n_layers": qnet.n_layers,
                "obs_dim": qnet.obs_dim,
                "active_layers": qnet.active_layers,
            }

        torch.save(payload, path)
        print(f"  checkpoint saved → {path}")
