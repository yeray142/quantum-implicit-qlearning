"""Phase 0 — Build the reference pool.

Rolls out the trained classical actor with quantum V/Q on default.qubit
for multiple seeds. Collects ~10k transitions with V_sim/Q_sim values.
Also samples from the D4RL hopper/medium-v0 dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from quantum_iql import (
    ActorNetwork,
    CriticNetwork,
    QuantumValueNetwork,
    load_minari_dataset,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def rollout_episode(
    env,
    actor: ActorNetwork,
    value_net: QuantumValueNetwork,
    critic: CriticNetwork,
    device: torch.device,
    deterministic: bool = True,
) -> dict:
    """Roll out one episode, collecting transitions and V/Q values.

    Returns a dict with lists of per-step data.
    """
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    results = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "dones": [],
        "timesteps": [],
        "v_sim": [],
        "q_sim": [],
    }

    t = 0
    while True:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            a = actor.get_action(obs_t, deterministic=deterministic)
            a_np = a.cpu().numpy()[0]
            q_val = critic(obs_t, a)
            if isinstance(q_val, tuple):
                q_val = q_val[0]
            q_val = q_val.cpu().item()
            v_val = value_net(obs_t).cpu().item()

        results["observations"].append(obs.copy())
        results["actions"].append(a_np)
        results["timesteps"].append(t)
        results["v_sim"].append(v_val)
        results["q_sim"].append(q_val)

        obs_next, reward, term, trunc, _ = env.step(a_np)
        obs_next = np.array(obs_next, dtype=np.float32)
        done = float(term or trunc)

        results["rewards"].append(float(reward))
        results["next_observations"].append(obs_next)
        results["dones"].append(done)

        obs = obs_next
        t += 1

        if term or trunc:
            break

    return results


def concat_episodes(episode_results: list[dict]) -> dict:
    """Concatenate per-episode results into a single dict of arrays."""
    keys = [
        "observations", "actions", "rewards",
        "next_observations", "dones", "timesteps",
        "v_sim", "q_sim", "episode_ids", "seeds",
    ]
    out = {}
    for k in keys:
        out[k] = np.concatenate([ep[k] for ep in episode_results], axis=0)
    return out


def build_pool(
    checkpoint_path: str | Path,
    env_id: str,
    seeds: list[int],
    episodes_per_seed: int,
    d4rl_sample_count: int,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> None:
    """Main pool-building entry point."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    log.info(f"Loaded checkpoint from {checkpoint_path}, step={ckpt.get('step', '?')}")

    # Load config if available
    cfg = ckpt.get("config", {})
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    # Read network dimensions — prefer quantum_meta from checkpoint (trained config),
    # fall back to YAML config fields
    quantum_meta = ckpt.get("quantum_meta", {})
    n_qubits = quantum_meta.get("n_qubits", cfg.get("quantum_value", {}).get("n_qubits", 8))
    n_layers = quantum_meta.get("n_layers", cfg.get("quantum_value", {}).get("n_layers", 3))
    obs_dim = quantum_meta.get("obs_dim", cfg.get("env", {}).get("obs_dim", 11))
    act_dim = cfg.get("env", {}).get("act_dim", 3)

    log.info(f"Building networks: n_qubits={n_qubits}, n_layers={n_layers}, obs_dim={obs_dim}")

    # Build networks
    multi_qubit_readout = ckpt["value_net"]["a"].shape[0] > 1
    value_net = QuantumValueNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        obs_dim=obs_dim,
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
        use_pre_encoder=True,
        multi_qubit_readout=multi_qubit_readout,
    ).to(device)

    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim).to(device)
    critic = CriticNetwork(obs_dim=obs_dim, act_dim=act_dim, use_twin=True).to(device)

    value_net.load_state_dict(ckpt["value_net"], strict=False)
    actor.load_state_dict(ckpt["actor_net"])
    critic.load_state_dict(ckpt["critic_net"])

    value_net.eval()
    actor.eval()
    critic.eval()

    # Load mu/sigma if present in checkpoint
    if "mu" in ckpt and "sigma" in ckpt:
        value_net.update_running_stats(
            torch.as_tensor(ckpt["mu"], device=device),
            torch.as_tensor(ckpt["sigma"], device=device),
        )

    log.info("Networks loaded and set to eval mode")

    # Roll out episodes per seed
    all_episode_results: list[dict] = []
    episode_id = 0

    for seed in seeds:
        log.info(f"Rolling out seed {seed} ({episodes_per_seed} episodes)")
        env = gym.make(env_id)
        env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for ep in range(episodes_per_seed):
            results = rollout_episode(env, actor, value_net, critic, device, deterministic=True)
            n_steps = len(results["timesteps"])
            log.info(f"  Episode {ep}: {n_steps} steps, return={sum(results['rewards']):.1f}")

            ep_id_arr = np.full(n_steps, episode_id, dtype=np.int32)
            seed_arr = np.full(n_steps, seed, dtype=np.int32)
            results["episode_ids"] = ep_id_arr
            results["seeds"] = seed_arr

            all_episode_results.append(results)
            episode_id += 1

    # Concatenate all rollout data
    pool = concat_episodes(all_episode_results)
    pool["observations"] = pool["observations"].astype(np.float32)
    pool["actions"] = pool["actions"].astype(np.float32)
    pool["rewards"] = pool["rewards"].astype(np.float32)
    pool["next_observations"] = pool["next_observations"].astype(np.float32)
    pool["dones"] = pool["dones"].astype(np.float32)
    pool["timesteps"] = np.array(pool["timesteps"], dtype=np.int32)
    pool["v_sim"] = np.array(pool["v_sim"], dtype=np.float32)
    pool["q_sim"] = np.array(pool["q_sim"], dtype=np.float32)
    pool["episode_ids"] = pool["episode_ids"].astype(np.int32)
    pool["seeds"] = pool["seeds"].astype(np.int32)

    n_rollout = len(pool["observations"])
    log.info(f"Rollout pool: {n_rollout} transitions")

    # Add D4RL samples
    if d4rl_sample_count > 0:
        log.info(f"Loading D4RL dataset for extra samples ({d4rl_sample_count})")
        d4rl_buffer = load_minari_dataset("mujoco/hopper/medium-v0", device="cpu")
        n_d4rl = min(d4rl_sample_count, len(d4rl_buffer))
        d4rl_batch = d4rl_buffer.sample(n_d4rl)
        d4rl_obs = d4rl_batch.observations.numpy()
        d4rl_act = d4rl_batch.actions.numpy()

        # Compute V/Q for D4RL samples using the loaded networks
        with torch.no_grad():
            d4rl_obs_t = torch.from_numpy(d4rl_obs).to(device)
            d4rl_act_t = torch.from_numpy(d4rl_act).to(device)
            d4rl_v = value_net(d4rl_obs_t).cpu().numpy()
            d4rl_q = critic(d4rl_obs_t, d4rl_act_t)
            if isinstance(d4rl_q, tuple):
                d4rl_q = d4rl_q[0]
            d4rl_q = d4rl_q.cpu().numpy().flatten()

        # Assign dummy values for non-rollout fields
        d4rl_timesteps = np.zeros(n_d4rl, dtype=np.int32)
        d4rl_ep_ids = np.arange(n_d4rl, dtype=np.int32) + episode_id
        d4rl_seeds = np.full(n_d4rl, -1, dtype=np.int32)
        d4rl_rewards = d4rl_batch.rewards.numpy().flatten()
        d4rl_dones = d4rl_batch.dones.numpy().flatten()
        d4rl_next = d4rl_batch.next_observations.numpy()

        # Concatenate
        pool["observations"] = np.concatenate([pool["observations"], d4rl_obs])
        pool["actions"] = np.concatenate([pool["actions"], d4rl_act])
        pool["rewards"] = np.concatenate([pool["rewards"], d4rl_rewards])
        pool["next_observations"] = np.concatenate([pool["next_observations"], d4rl_next])
        pool["dones"] = np.concatenate([pool["dones"], d4rl_dones])
        pool["timesteps"] = np.concatenate([pool["timesteps"], d4rl_timesteps])
        pool["v_sim"] = np.concatenate([pool["v_sim"], d4rl_v])
        pool["q_sim"] = np.concatenate([pool["q_sim"], d4rl_q])
        pool["episode_ids"] = np.concatenate([pool["episode_ids"], d4rl_ep_ids])
        pool["seeds"] = np.concatenate([pool["seeds"], d4rl_seeds])

        log.info(f"Total pool after D4RL: {len(pool['observations'])} transitions")

    # Pool provenance
    config_hash = ""
    if config_path and Path(config_path).exists():
        with open(config_path, "rb") as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    pool["pool_config"] = {
        "checkpoint": str(checkpoint_path),
        "env_id": env_id,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "d4rl_sample_count": d4rl_sample_count,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "config_hash": config_hash,
    }

    # Save
    log.info(f"Saving pool to {output_path}")
    np.savez(output_path, **pool)
    log.info(f"Pool saved: {len(pool['observations'])} transitions")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Build reference pool")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_pool(
        checkpoint_path=args.checkpoint,
        env_id=cfg.get("env_id", "Hopper-v4"),
        seeds=cfg["pool"]["seeds"],
        episodes_per_seed=cfg["pool"]["episodes_per_seed"],
        d4rl_sample_count=cfg["pool"].get("d4rl_sample_count", 500),
        output_path=cfg["pool"]["output_path"],
        config_path=args.config,
    )


if __name__ == "__main__":
    main()