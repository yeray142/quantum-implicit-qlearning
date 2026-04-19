"""IQL training loop: V, Q, and actor updates with W&B logging.

Subclassing guide
-----------------
To create a quantum variant, subclass ``IQLTrainer`` and override only
``_build_networks``. All update logic, logging, and evaluation remain
identical::

    class QuantumIQLTrainer(IQLTrainer):
        def _build_networks(self) -> None:
            # replace one or more networks with quantum equivalents
            self.value_net = QuantumValueNetwork(...)
            self.critic_net = CriticNetwork(...)   # keep classical if desired
            self.actor_net = ActorNetwork(...)
            # move to device
            for net in (self.value_net, self.critic_net, self.actor_net):
                net.to(self.device)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import wandb

from quantum_iql.buffer import Batch, ReplayBuffer
from quantum_iql.config import IQLConfig
from quantum_iql.losses import actor_loss, critic_loss, value_loss
from quantum_iql.networks import ActorNetwork, CriticNetwork, ValueNetwork
from quantum_iql.utils import get_device, hard_update, soft_update


class IQLTrainer:
    """Orchestrates all IQL updates, target network maintenance, and logging.

    Args:
        config:  Fully resolved :class:`~quantum_iql.config.IQLConfig`.
        buffer:  Pre-populated offline replay buffer.
        env:     Gymnasium environment used for evaluation rollouts.
    """

    def __init__(
        self,
        config: IQLConfig,
        buffer: ReplayBuffer,
        env: Any,
    ) -> None:
        self.cfg = config
        self.buffer = buffer
        self.env = env
        self.device = get_device(config.device)
        self._step = 0

        self._build_networks()
        self._build_targets()
        self._build_optimizers()

    # Network construction — override this in quantum subclasses
    def _build_networks(self) -> None:
        """Instantiate V, Q, and actor networks from config.

        Override in a quantum subclass to swap any network for a quantum
        variant.  After overriding, make sure every network is moved to
        ``self.device``.
        """
        obs_dim = self.buffer.obs_dim
        act_dim = self.buffer.act_dim
        vcfg = self.cfg.value_net
        qcfg = self.cfg.critic_net
        acfg = self.cfg.actor_net

        self.value_net = ValueNetwork(
            obs_dim,
            hidden_dims=vcfg.hidden_dims,
            activation=vcfg.activation,
            layer_norm=vcfg.layer_norm,
        ).to(self.device)

        self.critic_net = CriticNetwork(
            obs_dim,
            act_dim,
            hidden_dims=qcfg.hidden_dims,
            activation=qcfg.activation,
            layer_norm=qcfg.layer_norm,
            use_twin=self.cfg.use_twin_critic,
        ).to(self.device)

        self.actor_net = ActorNetwork(
            obs_dim,
            act_dim,
            hidden_dims=acfg.hidden_dims,
            activation=acfg.activation,
            layer_norm=acfg.layer_norm,
        ).to(self.device)

    def _build_targets(self) -> None:
        """Create the EMA copy of the value network (V̄).

        IQL only needs a frozen V̄ — it is used as the TD target inside
        ``critic_loss``.  Q has no separate target network because V̄
        already plays that role in the Bellman backup.
        """
        self.value_target = ValueNetwork(
            self.buffer.obs_dim,
            hidden_dims=self.cfg.value_net.hidden_dims,
            activation=self.cfg.value_net.activation,
            layer_norm=self.cfg.value_net.layer_norm,
        ).to(self.device)
        hard_update(self.value_target, self.value_net)

        # V̄ is never optimised directly — freeze it to be safe
        for p in self.value_target.parameters():
            p.requires_grad_(False)

    def _build_optimizers(self) -> None:
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.cfg.lr_v
        )
        self.critic_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=self.cfg.lr_q
        )
        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=self.cfg.lr_actor
        )

    # Per-component update steps
    def update_value(self, batch: Batch) -> dict[str, float]:
        """Single gradient step on V_ψ.

        Returns:
            ``{"loss/value": float}``
        """
        self.value_optimizer.zero_grad()
        loss = value_loss(self.value_net, self.critic_net, batch, self.cfg.tau)
        loss.backward()
        self.value_optimizer.step()
        return {"loss/value": loss.item()}

    def update_critic(self, batch: Batch) -> dict[str, float]:
        """Single gradient step on Q_θ.

        Returns:
            ``{"loss/critic": float}``
        """
        self.critic_optimizer.zero_grad()
        loss = critic_loss(self.critic_net, self.value_target, batch, self.cfg.gamma)
        loss.backward()
        self.critic_optimizer.step()
        return {"loss/critic": loss.item()}

    def update_actor(self, batch: Batch) -> dict[str, float]:
        """Single gradient step on π_φ.

        Returns:
            ``{"loss/actor": float, "advantage_mean": float,
               "advantage_std": float, "exp_adv_mean": float}``
        """
        self.actor_optimizer.zero_grad()
        loss, adv_metrics = actor_loss(
            self.actor_net,
            self.critic_net,
            self.value_net,
            batch,
            self.cfg.beta,
            self.cfg.advantage_clip,
        )
        loss.backward()
        self.actor_optimizer.step()
        return {"loss/actor": loss.item(), **adv_metrics}

    def update_targets(self) -> None:
        """Polyak update: V̄ ← (1 - ρ)·V̄ + ρ·V."""
        soft_update(self.value_target, self.value_net, self.cfg.polyak)

    # Full training step
    def train_step(self) -> dict[str, float]:
        """One complete IQL step: sample → V → Q → π (if past warmup) → V̄.

        Returns:
            Merged metrics dict from all active updates.
        """
        batch = self.buffer.sample(self.cfg.batch_size)

        # Move batch to training device (buffer lives on CPU)
        batch = Batch(
            observations=batch.observations.to(self.device),
            actions=batch.actions.to(self.device),
            rewards=batch.rewards.to(self.device),
            next_observations=batch.next_observations.to(self.device),
            dones=batch.dones.to(self.device),
        )

        metrics: dict[str, float] = {}
        metrics.update(self.update_value(batch))
        metrics.update(self.update_critic(batch))

        if self._step >= self.cfg.warmup_steps:
            metrics.update(self.update_actor(batch))

        self.update_targets()
        self._step += 1
        return metrics

    # Evaluation
    def evaluate(self) -> dict[str, float]:
        """Run ``eval_episodes`` deterministic rollouts and report returns.

        Returns:
            ``{"eval/mean_return": float, "eval/std_return": float,
               "eval/min_return": float,  "eval/max_return": float}``
        """
        self.actor_net.eval()
        returns: list[float] = []

        for ep_idx in range(self.cfg.eval_episodes):
            obs, _ = self.env.reset(seed=self.cfg.seed + ep_idx)
            ep_return = 0.0
            done = False

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor_net.get_action(obs_t, deterministic=True)
                action_np = action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
                ep_return += float(reward)
                done = terminated or truncated

            returns.append(ep_return)

        self.actor_net.train()
        returns_arr = np.array(returns)
        return {
            "eval/mean_return": float(returns_arr.mean()),
            "eval/std_return": float(returns_arr.std()),
            "eval/min_return": float(returns_arr.min()),
            "eval/max_return": float(returns_arr.max()),
        }

    # Main training loop
    def train(self) -> None:
        """Run the full IQL training loop for ``num_steps`` gradient steps.

        Logs metrics to W&B at ``log_interval`` and runs evaluation at
        ``eval_interval``.
        """
        print(
            f"Training IQL on {self.cfg.dataset_id} for "
            f"{self.cfg.num_steps:,} steps  (device={self.device})"
        )
        t0 = time.time()

        for step in range(1, self.cfg.num_steps + 1):
            metrics = self.train_step()

            if step % self.cfg.log_interval == 0:
                elapsed = time.time() - t0
                metrics["train/step"] = step
                metrics["train/steps_per_sec"] = step / elapsed
                wandb.log(metrics, step=step)

            if step % self.cfg.eval_interval == 0:
                eval_metrics = self.evaluate()
                wandb.log(eval_metrics, step=step)
                mean_ret = eval_metrics["eval/mean_return"]
                print(f"  step {step:>8,}  |  eval return = {mean_ret:.1f}")
                self.save_checkpoint(f"checkpoint_{step:08d}.pt")

        print(f"Training complete in {time.time() - t0:.1f}s")
        self.save_checkpoint("checkpoint_final.pt")

    def save_checkpoint(self, filename: str) -> None:
        """Save network weights to ``checkpoints/<filename>``.

        Args:
            filename: Filename (not full path) for the checkpoint file.
        """
        import os
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", filename)
        torch.save(
            {
                "step": self._step,
                "value_net": self.value_net.state_dict(),
                "critic_net": self.critic_net.state_dict(),
                "actor_net": self.actor_net.state_dict(),
            },
            path,
        )
        print(f"  checkpoint saved → {path}")
