#!/usr/bin/env python3
"""
Qualitative evaluation: record rollout videos for every checkpoint.

Discovers all checkpoint_final.pt files under experiments/checkpoints/,
rebuilds the correct network architecture for each mode (classical,
quantum, classical-small), runs N_EPISODES deterministic episodes, and
saves one MP4 per checkpoint to scripts/eval_videos/.

Usage
-----
    python scripts/eval_videos.py                    # all checkpoints, 3 episodes
    python scripts/eval_videos.py --episodes 5       # 5 episodes per checkpoint
    python scripts/eval_videos.py --mode quantum     # only quantum checkpoints
    python scripts/eval_videos.py --seed 2           # only seed 2 checkpoints
    python scripts/eval_videos.py --deterministic 0  # stochastic actions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import gymnasium as gym

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))       # quantum_value_network.py
sys.path.insert(0, str(_PROJECT_ROOT / "src"))  # quantum_iql.*

from quantum_iql.networks import ActorNetwork, CriticNetwork, ValueNetwork

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINTS_ROOT = _PROJECT_ROOT / "experiments" / "checkpoints"
VIDEO_DIR        = _SCRIPTS_DIR / "eval_videos"
ENV_ID           = "Hopper-v4"
OBS_DIM          = 11
ACT_DIM          = 3
FPS              = 30
MAX_STEPS        = 1_000   # MuJoCo Hopper default episode length


# ── Architecture inference ────────────────────────────────────────────────────

def _infer_value_hidden_dims(state_dict: dict) -> list[int]:
    """Read hidden-layer widths from a classical ValueNetwork state dict."""
    weight_keys = sorted(
        [k for k in state_dict if k.endswith(".weight") and k.startswith("net.")],
        key=lambda k: int(k.split(".")[1]),
    )
    # All layers except the last (output) layer contribute a hidden dim.
    return [state_dict[k].shape[0] for k in weight_keys[:-1]]


def _build_classical_value(state_dict: dict) -> ValueNetwork:
    hidden_dims = _infer_value_hidden_dims(state_dict)
    net = ValueNetwork(OBS_DIM, hidden_dims=hidden_dims, activation="relu")
    net.load_state_dict(state_dict)
    return net


def _build_quantum_value(state_dict: dict, meta: dict):
    from quantum_value_network import QuantumValueNetwork
    net = QuantumValueNetwork(
        obs_dim=meta["obs_dim"],
        n_qubits=meta["n_qubits"],
        n_layers=meta["n_layers"],
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
    )
    net.load_state_dict(state_dict)
    net.set_active_layers(meta["active_layers"])
    return net


def _build_actor(state_dict: dict) -> ActorNetwork:
    actor = ActorNetwork(OBS_DIM, ACT_DIM, hidden_dims=[256, 256], activation="relu")
    actor.load_state_dict(state_dict)
    return actor


# ── Single-episode rollout ────────────────────────────────────────────────────

def rollout(
    env: gym.Env,
    actor: ActorNetwork,
    seed: int,
    deterministic: bool = True,
) -> tuple[list[np.ndarray], float]:
    """Run one episode; return (frames, total_return)."""
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_return = 0.0
    device = next(actor.parameters()).device

    for _ in range(MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor.get_action(obs_t, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_return += float(reward)

        if terminated or truncated:
            break

    return frames, total_return


# ── Main evaluation loop ──────────────────────────────────────────────────────

def _best_checkpoint(ckpt_path: Path, min_step: int = 50_000) -> Path:
    """Return ckpt_path unless its step is suspiciously low (overwritten by a
    later short run), in which case fall back to the highest-numbered checkpoint
    in the same directory."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("step", 0) >= min_step:
            return ckpt_path
    except Exception:
        return ckpt_path

    # Find highest-numbered checkpoint in the same folder
    siblings = sorted(ckpt_path.parent.glob("checkpoint_[0-9]*.pt"))
    if not siblings:
        return ckpt_path
    best = siblings[-1]   # sorted lexicographically → highest number last
    print(f"    WARNING: checkpoint_final.pt step < {min_step} "
          f"(overwritten). Using {best.name} instead.")
    return best


def evaluate_checkpoint(
    ckpt_path: Path,
    n_episodes: int,
    deterministic: bool,
    device: torch.device,
) -> dict:
    ckpt_path  = _best_checkpoint(ckpt_path)
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode       = ckpt["mode"]          # "classical" or "quantum"
    step       = ckpt.get("step", "?")
    is_quantum = ckpt.get("quantum_meta") is not None

    # Infer human-readable mode name from path for classical-small
    path_parts = ckpt_path.parts
    dir_mode   = next((p for p in path_parts if p in
                       ("classical", "quantum", "classical-small")), mode)

    # Build value network
    if is_quantum:
        value_net = _build_quantum_value(ckpt["value_net"], ckpt["quantum_meta"])
    else:
        value_net = _build_classical_value(ckpt["value_net"])
    value_net.to(device).eval()

    # Build actor (always classical [256, 256])
    actor = _build_actor(ckpt["actor_net"])
    actor.to(device).eval()

    # Record episodes
    env = gym.make(ENV_ID, render_mode="rgb_array")
    all_frames: list[np.ndarray] = []
    returns: list[float] = []

    for ep in range(n_episodes):
        ep_seed = ep * 100 + 42
        frames, ep_return = rollout(env, actor, seed=ep_seed, deterministic=deterministic)

        # Add a short black separator between episodes (0.5 s)
        if all_frames and frames:
            sep = np.zeros_like(frames[0])
            all_frames.extend([sep] * int(FPS * 0.5))

        all_frames.extend(frames)
        returns.append(ep_return)
        print(f"    ep {ep + 1}/{n_episodes}  return={ep_return:.1f}  frames={len(frames)}")

    env.close()

    return {
        "mode":       dir_mode,
        "step":       step,
        "is_quantum": is_quantum,
        "frames":     all_frames,
        "returns":    returns,
        "mean_return": float(np.mean(returns)),
        "std_return":  float(np.std(returns)),
    }


def save_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=FPS)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record evaluation videos for all checkpoints.")
    p.add_argument("--episodes",     type=int,  default=3,
                   help="Episodes to record per checkpoint (default: 3).")
    p.add_argument("--mode",         type=str,  default=None,
                   choices=["classical", "quantum", "classical-small"],
                   help="Only evaluate checkpoints of this mode.")
    p.add_argument("--seed",         type=int,  default=None,
                   help="Only evaluate checkpoints of this seed index.")
    p.add_argument("--deterministic", type=int, default=1,
                   help="1 = deterministic actions (default), 0 = stochastic.")
    p.add_argument("--device",       type=str,  default="cpu",
                   help="Torch device (default: cpu — quantum circuit runs on CPU).")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    det    = bool(args.deterministic)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # ── Discover checkpoints ─────────────────────────────────────────────────
    ckpt_paths = sorted(CHECKPOINTS_ROOT.glob("**/checkpoint_final.pt"))

    if args.mode is not None:
        ckpt_paths = [p for p in ckpt_paths if args.mode in p.parts]
    if args.seed is not None:
        ckpt_paths = [p for p in ckpt_paths if f"seed_{args.seed}" in p.parts]

    if not ckpt_paths:
        print("No checkpoints found matching the filters.")
        return

    print(f"\nEvaluating {len(ckpt_paths)} checkpoint(s) → {VIDEO_DIR}\n")

    summary: list[dict] = []

    for ckpt_path in ckpt_paths:
        # Extract mode and seed from path: .../mode/seed_N/checkpoint_final.pt
        seed_dir  = ckpt_path.parent.name          # e.g. "seed_0"
        mode_dir  = ckpt_path.parent.parent.name   # e.g. "classical-small"
        seed_idx  = seed_dir.replace("seed_", "")
        video_name = f"{mode_dir}_seed{seed_idx}.mp4"
        video_path = VIDEO_DIR / video_name

        print(f"  [{mode_dir} / seed {seed_idx}]  step={_peek_step(ckpt_path)}")

        result = evaluate_checkpoint(ckpt_path, args.episodes, det, device)

        if result["frames"]:
            save_video(result["frames"], video_path)
            print(f"    saved → {video_path}  "
                  f"({len(result['frames'])} frames, "
                  f"{len(result['frames'])/FPS:.1f}s)")
        else:
            print("    WARNING: no frames captured — check render_mode support.")

        summary.append({
            "mode":  mode_dir,
            "seed":  seed_idx,
            "mean":  result["mean_return"],
            "std":   result["std_return"],
            "video": video_name,
        })
        print()

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  {'Mode':<18} {'Seed':>5}  {'Mean return':>12}  {'Std':>7}  Video")
    print(f"  {'-'*61}")
    for row in sorted(summary, key=lambda r: (r["mode"], r["seed"])):
        print(f"  {row['mode']:<18} {row['seed']:>5}  "
              f"{row['mean']:>12.1f}  {row['std']:>7.1f}  {row['video']}")
    print("=" * 65)
    print(f"\nVideos saved to: {VIDEO_DIR}\n")


def _peek_step(path: Path) -> int | str:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("step", "?")
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
