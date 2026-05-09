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
    "quantum-fixed-c-hopper-medium":    ("hopper", "medium", "quantum-fixed-c"),
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
    h = run.history(samples=500)
    final_return = run.summary.get("eval/mean_return", float("nan"))

    def _col_max(df, col):
        if col in df.columns and df[col].notna().any():
            return float(df[col].max())
        return float("nan")

    def _col_last(df, col):
        if col in df.columns and df[col].notna().any():
            return float(df[col].dropna().iloc[-1])
        return float("nan")

    max_return   = _col_max(h, "eval/mean_return")
    vloss_final  = _col_last(h, "loss/value")
    adv_final    = _col_last(h, "advantage_mean")
    adv_std_final = _col_last(h, "advantage_std")

    # Gradient statistics
    grad_col = "quantum/grad_norm_theta"
    has_grad = grad_col in h.columns and h[grad_col].notna().any()
    if has_grad:
        explosion = bool((h[grad_col] > GRAD_EXPLOSION_THRESHOLD).any())
        exp_steps  = h.loc[h[grad_col] > GRAD_EXPLOSION_THRESHOLD, "_step"].tolist()
        first_exp  = int(min(exp_steps)) if exp_steps else None
        max_grad   = float(h[grad_col].max())

        # Early gradient (steps 0-10000)
        early_mask = h["_step"] <= 10000
        early_grads = h.loc[early_mask, grad_col]
        mean_early = float(early_grads.mean()) if early_grads.notna().any() else float("nan")

        # Late gradient (steps 30000-100000)
        late_mask = h["_step"] >= 30000
        late_grads = h.loc[late_mask, grad_col]
        mean_late = float(late_grads.mean()) if late_grads.notna().any() else float("nan")
    else:
        explosion  = False
        first_exp   = None
        max_grad    = float("nan")
        mean_early  = float("nan")
        mean_late   = float("nan")

    return {
        "name":         run.name,
        "final_return": final_return,
        "max_return":   max_return,
        "vloss_final":  vloss_final,
        "adv_final":    adv_final,
        "adv_std_final": adv_std_final,
        "explosion":    explosion,
        "first_exp":    first_exp,
        "max_grad":     max_grad,
        "mean_early_grad": mean_early,
        "mean_late_grad":  mean_late,
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


def _value_net_shape_from_ckpt(state_dict: dict) -> list | None:
    """Infer ValueNetwork hidden_dims from a checkpoint state dict.

    build_mlp creates even-indexed Linear layers (0, 2, 4, ...) in an
    nn.Sequential.  We walk through them in order and collect each Linear's
    output dim until we hit the value head (output dim == 1).

    Example: net.0.weight [8, 11],  net.2.weight [8, 8],  net.4.weight [1, 8]
             → hidden_dims = [8, 8]
    """
    import re

    linear_keys = sorted(
        [k for k in state_dict.keys() if k.endswith(".weight")],
        key=lambda k: int(re.search(r"\d+", k).group()),
    )
    hidden = []
    for key in linear_keys:
        out_dim = state_dict[key].shape[0]
        if out_dim == 1:
            break
        hidden.append(out_dim)
    return hidden if hidden else None


def _critic_net_shape_from_ckpt(
    state_dict: dict, obs_dim: int = 11
) -> tuple[list, int] | None:
    """Infer (hidden_dims, action_dim) from a CriticNetwork checkpoint.

    For CriticNetwork(obs_dim, act_dim, hidden_dims):
      - q1.0: Linear(obs_dim + act_dim,  hidden_dims[0])  → weight [hd0, obs+act]
      - q1.2: Linear(hidden_dims[0],      hidden_dims[1])  → weight [hd1, hd0]
      - ...

    The first layer's in_dim is (obs_dim + act_dim), so
    act_dim = first_in_dim - obs_dim.
    """
    import re

    linear_keys = sorted(
        [k for k in state_dict.keys() if k.endswith(".weight")],
        key=lambda k: int(re.search(r"\d+", k).group()),
    )
    hidden = []
    action_dim = None
    for key in linear_keys:
        w = state_dict[key]
        out_dim, in_dim = w.shape
        if out_dim == 1:
            break
        if action_dim is None:
            # First layer: in_dim = obs_dim + act_dim  →  act_dim = in_dim - obs_dim
            action_dim = in_dim - obs_dim
        hidden.append(out_dim)
    if action_dim is None or action_dim <= 0:
        return None
    return hidden, action_dim


def compute_p_neg_adv(mode_dir: str, seed: int, env: str, dataset: str) -> float | None:
    """
    Compute P(A<0) = P(Q(s,a) - V(s) < 0) over 50k dataset samples.

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
        is_constant_v = "constant" in mode_dir

        if is_constant_v:
            vnet = None
        elif is_quantum:
            from quantum_iql import QuantumValueNetwork
            vnet = QuantumValueNetwork(n_qubits=8, n_layers=3, obs_dim=11)
        else:
            from quantum_iql.networks import ValueNetwork
            hidden_dims = _value_net_shape_from_ckpt(ckpt["value_net"]) or [256, 256]
            vnet = ValueNetwork(11, hidden_dims=hidden_dims)

        critic_info = _critic_net_shape_from_ckpt(ckpt["critic_net"], obs_dim=11)
        if critic_info is not None:
            c_hidden, action_dim = critic_info
        else:
            c_hidden, action_dim = [256, 256], 3
        cnet = CriticNetwork(11, action_dim, hidden_dims=c_hidden)

        if vnet is not None:
            vnet.load_state_dict(ckpt["value_net"], strict=False)
        cnet.load_state_dict(ckpt["critic_net"])

        buf = _load_buffer_once(env, dataset)
        rng = np.random.default_rng(42)
        idx  = rng.choice(buf._size, size=50_000, replace=False)
        obs  = torch.FloatTensor(buf._observations[idx])
        acts = torch.FloatTensor(buf._actions[idx])

        if vnet is not None:
            vnet.eval()
        cnet.eval()
        with torch.no_grad():
            if is_constant_v:
                v_value = float(ckpt["v_constant"])
            else:
                v = vnet(obs).squeeze()
            q1, q2   = cnet(obs, acts)
            if is_constant_v:
                adv = torch.min(q1, q2).squeeze() - v_value
            else:
                adv = torch.min(q1, q2).squeeze() - v
            p_neg    = (adv < 0).float().mean().item()

        return p_neg

    except Exception as exc:  # pragma: no cover
        print(f"  [warn] P(A<0) failed for {mode_dir}/seed_{seed}: {exc}", file=sys.stderr)
        return None


def compute_ckpt_metrics(mode_dir: str, seed: int, env: str, dataset: str) -> dict | None:
    """
    Compute all checkpoint-based metrics:
    - b_final, a_final (affine head parameters for quantum models)
    - E[A] (expected advantage)
    - E[e^{beta*A}] (for constant-V comparison, beta=5)
    - L(V)_final (value loss)

    Returns None if checkpoint unavailable.
    """
    ckpt_dir  = CHECKPOINT_BASE / env / dataset / mode_dir / f"seed_{seed}"
    ckpt_path = ckpt_dir / "checkpoint_final.pt"

    if not ckpt_path.exists():
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("step", 0) < 50_000:
            alt = ckpt_dir / "checkpoint_00100000.pt"
            if alt.exists():
                ckpt = torch.load(alt, map_location="cpu")
            else:
                return None

        is_quantum = "quantum" in mode_dir
        is_constant_v = "constant" in mode_dir

        # Extract affine head parameters
        b_final = None
        a_final = None
        if is_quantum and "value_net" in ckpt:
            sd = ckpt["value_net"]
            # Quantum models store a and b directly
            b_final = float(sd["b"].item()) if "b" in sd else None
            a_final = float(sd["a"].item()) if "a" in sd else None
        elif not is_constant_v and not is_quantum and "value_net" in ckpt:
            sd = ckpt["value_net"]
            # Classical models have net.4 as output layer with shape [1, hidden]
            import re
            linear_keys = sorted(
                [k for k in sd.keys() if k.endswith(".weight")],
                key=lambda k: int(re.search(r"\d+", k).group()),
            )
            for key in linear_keys:
                if sd[key].shape[0] == 1:
                    b_key = key.replace(".weight", ".bias")
                    if b_key in sd:
                        b_final = float(sd[b_key].item())
                    # a_final for classical is the output weight - not a scalar
                    # For classical, V(s) = net(s) directly, no affine scaling
                    a_final = None
                    break

        vnet = None
        cnet = None

        # Load networks for advantage computation
        if is_constant_v:
            # For constant-V, use the stored V value
            v_constant = ckpt["v_constant"]
            v_final = float(v_constant.item()) if hasattr(v_constant, 'item') else float(v_constant)
            # Load critic to compute advantages with constant V
            from quantum_iql.networks import CriticNetwork
            critic_info = _critic_net_shape_from_ckpt(ckpt["critic_net"], obs_dim=11)
            if critic_info is not None:
                c_hidden, action_dim = critic_info
            else:
                c_hidden, action_dim = [256, 256], 3
            cnet = CriticNetwork(11, action_dim, hidden_dims=c_hidden)
            cnet.load_state_dict(ckpt["critic_net"])
        else:
            from quantum_iql.networks import CriticNetwork
            if is_quantum:
                from quantum_iql import QuantumValueNetwork
                vnet = QuantumValueNetwork(n_qubits=8, n_layers=3, obs_dim=11)
            else:
                from quantum_iql.networks import ValueNetwork
                hidden_dims = _value_net_shape_from_ckpt(ckpt["value_net"]) or [256, 256]
                vnet = ValueNetwork(11, hidden_dims=hidden_dims)

            critic_info = _critic_net_shape_from_ckpt(ckpt["critic_net"], obs_dim=11)
            if critic_info is not None:
                c_hidden, action_dim = critic_info
            else:
                c_hidden, action_dim = [256, 256], 3
            cnet = CriticNetwork(11, action_dim, hidden_dims=c_hidden)

            vnet.load_state_dict(ckpt["value_net"], strict=False)
            cnet.load_state_dict(ckpt["critic_net"])

        # Compute advantage statistics on dataset
        buf = _load_buffer_once(env, dataset)
        rng = np.random.default_rng(42)
        idx = rng.choice(buf._size, size=50_000, replace=False)
        obs = torch.FloatTensor(buf._observations[idx])
        acts = torch.FloatTensor(buf._actions[idx])

        if cnet is not None:
            cnet.eval()
        if vnet is not None:
            vnet.eval()

        with torch.no_grad():
            if is_constant_v:
                # Constant V: V(s) = v_final for all states
                q1, q2 = cnet(obs, acts)
                adv = torch.min(q1, q2).squeeze() - v_final
            else:
                v = vnet(obs).squeeze()
                q1, q2 = cnet(obs, acts)
                adv = torch.min(q1, q2).squeeze() - v

            E_A = float(adv.mean().item())
            # Clamp advantages to avoid overflow in exp(beta*A)
            clamped_adv = torch.clamp(adv, min=-10, max=10)
            E_exp_beta_A = float(torch.exp(clamped_adv * 5).mean().item())  # beta = 5

        # Value loss (final) - only use checkpoint value if W&B value is missing
        vloss_final = float("nan")
        ckpt_vloss = ckpt.get("value_loss", float("nan"))
        if ckpt_vloss is not None and not np.isnan(ckpt_vloss):
            vloss_final = ckpt_vloss
        # else leave vloss_final as NaN - will use W&B value from run_stats

        return {
            "b_final": b_final,
            "a_final": a_final,
            "E_A": E_A,
            "E_exp_beta_A": E_exp_beta_A,
            # Only add vloss_final if it's valid; run_stats already has the W&B value
            **({"vloss_final": vloss_final} if not np.isnan(vloss_final) else {}),
        }

    except Exception as exc:
        print(f"  [warn] checkpoint metrics failed for {mode_dir}/seed_{seed}: {exc}", file=sys.stderr)
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
    p.add_argument("--models", nargs="+",
                   help="Only analyze specific models by their W&B run name prefix. "
                        "Examples: --models quantum-fixed-hopper-medium classical-hopper-medium")
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
        # Filter by --models if specified
        if args.models and prefix not in args.models:
            print(f"  Skipping {prefix} (not in --models)", file=sys.stderr)
            continue

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
                print(f"  seed {seed}: computing checkpoint metrics …", file=sys.stderr)
                ckpt_metrics = compute_ckpt_metrics(mode_dir, seed, env, dataset)
                if ckpt_metrics:
                    s.update(ckpt_metrics)
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