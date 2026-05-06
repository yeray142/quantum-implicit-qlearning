#!/usr/bin/env python3
"""
Analyze W&B experiment results for Quantum-IQL hopper-medium ablation study.

Fetches run histories from W&B, computes P(A<0) from local checkpoints, and
prints a structured summary suitable for inclusion in the Task 10 report.

Usage
-----
  # Full analysis (requires W&B credentials and local checkpoints):
  python scripts/analyze_wandb_results.py

  # Skip checkpoint-based P(A<0) computation:
  python scripts/analyze_wandb_results.py --no-checkpoints

  # Analyze a single group:
  python scripts/analyze_wandb_results.py --group hopper-medium

  # Output JSON for machine consumption:
  python scripts/analyze_wandb_results.py --json > results.json

Environment
-----------
  Requires: wandb, torch, numpy
  Optional: quantum_iql (for P(A<0)); falls back gracefully if unavailable.
  Checkpoints must be at: experiments/checkpoints/{env}/{dataset}/{mode}/seed_{s}/

W&B project: quantum-iql  (entity inferred from W&B credentials)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WANDB_PROJECT = "quantum-iql"
WANDB_GROUP   = "hopper-medium"

# Maps W&B run name prefix → (env, dataset, mode_dir)
# mode_dir is the sub-directory under experiments/checkpoints/{env}/{dataset}/
RUN_PREFIXES = {
    "classical-hopper-medium":          ("hopper", "medium", "classical"),
    "classical-deep-hopper-medium":     ("hopper", "medium", "classical-deep"),
    "classical-small-hopper-medium":    ("hopper", "medium", "classical-small"),
    "constant-v-hopper-medium":         ("hopper", "medium", "constant-v"),
    "quantum-hopper-medium":            ("hopper", "medium", "quantum"),
    "quantum-no-warmup-hopper-medium":  ("hopper", "medium", "quantum-no-warmup"),
    "quantum-fixed-hopper-medium":      ("hopper", "medium", "quantum-fixed"),
    "quantum-fixed-warmup-hopper-medium": ("hopper", "medium", "quantum-fixed-warmup"),
}

CHECKPOINT_BASE = _PROJECT_ROOT / "experiments" / "checkpoints"

# Gradient explosion threshold
GRAD_EXPLOSION_THRESHOLD = 10.0

# P(A<0) calibration window [lo, hi]
CALIB_LO = 0.60
CALIB_HI = 0.80

# D4RL dataset IDs per env-dataset pair
MINARI_DATASET_IDS = {
    ("hopper", "medium"): "mujoco/hopper/medium-v0",
}


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def fetch_runs(project: str, group: str):
    """Return list of W&B Run objects for the given project+group."""
    import wandb
    api = wandb.Api()
    return list(api.runs(project, filters={"group": group}))


def run_stats(run) -> dict:
    """Extract key statistics from a single W&B run object."""
    h = run.history(
        samples=500,
        keys=[
            "eval/mean_return",
            "loss/value",
            "advantage_mean",
            "advantage_std",
            "quantum/grad_norm_theta",
            "quantum/active_layers",
        ],
    )
    final_return = run.summary.get("eval/mean_return", float("nan"))
    max_return   = h["eval/mean_return"].max() if "eval/mean_return" in h else float("nan")

    grad_col = "quantum/grad_norm_theta"
    has_grad = grad_col in h.columns and h[grad_col].notna().any()
    if has_grad:
        explosion = bool((h[grad_col] > GRAD_EXPLOSION_THRESHOLD).any())
        exp_steps  = h.loc[h[grad_col] > GRAD_EXPLOSION_THRESHOLD, "_step"].tolist()
        first_exp  = int(min(exp_steps)) if exp_steps else None
        max_grad   = float(h[grad_col].max())
    else:
        explosion  = False
        first_exp  = None
        max_grad   = float("nan")

    vloss_final = float(h["loss/value"].iloc[-1]) if "loss/value" in h else float("nan")
    adv_final   = float(h["advantage_mean"].iloc[-1]) if "advantage_mean" in h else float("nan")

    return {
        "name":         run.name,
        "final_return": final_return,
        "max_return":   max_return,
        "vloss_final":  vloss_final,
        "adv_final":    adv_final,
        "explosion":    explosion,
        "first_exp":    first_exp,
        "max_grad":     max_grad,
    }


# ---------------------------------------------------------------------------
# Checkpoint-based P(A<0)
# ---------------------------------------------------------------------------

def _load_buffer_once(env: str, dataset: str):
    """Load the offline dataset into a ReplayBuffer (cached across calls)."""
    key = (env, dataset)
    if key not in _load_buffer_once._cache:
        from quantum_iql.buffer import load_minari_dataset
        dataset_id = MINARI_DATASET_IDS[key]
        buf = load_minari_dataset(dataset_id)
        _load_buffer_once._cache[key] = buf
    return _load_buffer_once._cache[key]
_load_buffer_once._cache: dict = {}


def compute_p_neg_adv(mode_dir: str, seed: int, env: str, dataset: str) -> float | None:
    """
    Compute P(A<0) = P(Q(s,a) - V(s) < 0) over 10k dataset samples.

    Returns None if the checkpoint or required modules are unavailable.
    """
    ckpt_dir  = CHECKPOINT_BASE / env / dataset / mode_dir / f"seed_{seed}"
    ckpt_path = ckpt_dir / "checkpoint_final.pt"

    if not ckpt_path.exists():
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Fall back to the explicit 100k file if the final was overwritten by a short run.
        if ckpt.get("step", 0) < 50_000:
            alt = ckpt_dir / "checkpoint_00100000.pt"
            if alt.exists():
                ckpt = torch.load(alt, map_location="cpu")
            else:
                return None

        from quantum_iql.networks import CriticNetwork

        is_quantum = "quantum" in mode_dir
        if is_quantum:
            from quantum_value_network import QuantumValueNetwork
            vnet = QuantumValueNetwork(n_qubits=8, n_layers=3, obs_dim=11)
        else:
            from quantum_iql.networks import ValueNetwork
            vnet = ValueNetwork(11, hidden_dims=[256, 256])

        vnet.load_state_dict(ckpt["value_net"])
        cnet = CriticNetwork(11, 3, hidden_dims=[256, 256])
        cnet.load_state_dict(ckpt["critic_net"])

        buf = _load_buffer_once(env, dataset)
        rng = np.random.default_rng(42)
        idx  = rng.choice(buf._size, size=10_000, replace=False)
        obs  = torch.FloatTensor(buf._observations[idx])
        acts = torch.FloatTensor(buf._actions[idx])

        vnet.eval(); cnet.eval()
        with torch.no_grad():
            v        = vnet(obs).squeeze()
            q1, q2   = cnet(obs, acts)
            adv      = torch.min(q1, q2).squeeze() - v
            p_neg    = (adv < 0).float().mean().item()

        return p_neg

    except Exception as exc:  # pragma: no cover
        print(f"  [warn] P(A<0) failed for {mode_dir}/seed_{seed}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Group-level aggregation
# ---------------------------------------------------------------------------

def aggregate(stats_list: list[dict]) -> dict:
    finals = [s["final_return"] for s in stats_list if not np.isnan(s["final_return"])]
    p_negs = [s["p_neg"] for s in stats_list if s.get("p_neg") is not None]
    n_exp  = sum(1 for s in stats_list if s["explosion"])
    n      = len(stats_list)
    calib  = sum(1 for p in p_negs if CALIB_LO <= p <= CALIB_HI)
    return {
        "n":            n,
        "mean_return":  float(np.mean(finals))  if finals else float("nan"),
        "std_return":   float(np.std(finals))   if finals else float("nan"),
        "cv_pct":       float(np.std(finals) / np.mean(finals) * 100) if finals else float("nan"),
        "n_explosions": n_exp,
        "p_neg_range":  [float(min(p_negs)), float(max(p_negs))] if p_negs else None,
        "calib_count":  calib,
        "calib_total":  len(p_negs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group",          default=WANDB_GROUP)
    p.add_argument("--project",        default=WANDB_PROJECT)
    p.add_argument("--no-checkpoints", action="store_true",
                   help="Skip checkpoint loading (no P(A<0) computation).")
    p.add_argument("--json",           action="store_true",
                   help="Output raw JSON instead of formatted table.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Fetching runs from W&B: {args.project} / {args.group} …", file=sys.stderr)
    runs = fetch_runs(args.project, args.group)
    run_map = {r.name: r for r in runs}
    print(f"  Found {len(runs)} runs.", file=sys.stderr)

    # Group runs by prefix
    groups: dict[str, list] = {}
    for name, run in run_map.items():
        for prefix, (env, dataset, mode_dir) in RUN_PREFIXES.items():
            if name.startswith(prefix + "-s"):
                seed_str = name[len(prefix) + 2:]  # after "-s"
                if seed_str.isdigit():
                    groups.setdefault(prefix, []).append((int(seed_str), run, env, dataset, mode_dir))
                    break

    results: dict[str, list] = {}
    for prefix, entries in sorted(groups.items()):
        entries.sort(key=lambda x: x[0])
        print(f"\nAnalyzing {prefix} ({len(entries)} seeds) …", file=sys.stderr)
        seed_stats = []
        for seed, run, env, dataset, mode_dir in entries:
            print(f"  seed {seed}: fetching W&B history …", file=sys.stderr)
            s = run_stats(run)
            s["seed"] = seed

            if not args.no_checkpoints:
                print(f"  seed {seed}: computing P(A<0) from checkpoint …", file=sys.stderr)
                s["p_neg"] = compute_p_neg_adv(mode_dir, seed, env, dataset)
            else:
                s["p_neg"] = None

            seed_stats.append(s)
        results[prefix] = seed_stats

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    # ---- formatted output ----
    print("\n" + "=" * 90)
    print(f"QUANTUM-IQL EXPERIMENT SUMMARY  |  group={args.group}")
    print("=" * 90)

    for prefix, seed_stats in results.items():
        agg = aggregate(seed_stats)
        print(f"\n{'─'*70}")
        print(f"  {prefix}  (n={agg['n']} seeds)")
        print(f"  Mean return : {agg['mean_return']:.1f} ± {agg['std_return']:.1f}  "
              f"(CV={agg['cv_pct']:.1f}%)")
        print(f"  Explosions  : {agg['n_explosions']}/{agg['n']}")
        if agg["p_neg_range"]:
            lo, hi = agg["p_neg_range"]
            print(f"  P(A<0) range: {lo:.3f}–{hi:.3f}  "
                  f"(calibrated [{CALIB_LO}–{CALIB_HI}]: "
                  f"{agg['calib_count']}/{agg['calib_total']})")

        header = f"  {'Seed':>4}  {'Final':>8}  {'Max':>8}  {'Vloss':>8}  "
        header += f"{'Explode':>7}  {'P(A<0)':>7}  {'1st_exp':>8}"
        print(header)
        for s in seed_stats:
            p_str = f"{s['p_neg']:.3f}" if s.get("p_neg") is not None else "  n/a "
            exp_str = f"{s['first_exp']:>8}" if s["first_exp"] is not None else "    none"
            print(f"  {s['seed']:>4}  {s['final_return']:>8.1f}  {s['max_return']:>8.1f}  "
                  f"{s['vloss_final']:>8.2f}  {'YES':>7}  {p_str:>7}  {exp_str}"
                  if s["explosion"] else
                  f"  {s['seed']:>4}  {s['final_return']:>8.1f}  {s['max_return']:>8.1f}  "
                  f"{s['vloss_final']:>8.2f}  {'no':>7}  {p_str:>7}  {exp_str}")

    print("\n" + "=" * 90)
    print("Calibration criterion: P(A<0) ∈ [0.60, 0.80]")
    print("Explosion criterion  : max(grad_norm_theta) > 10")
    print("=" * 90)


if __name__ == "__main__":
    main()