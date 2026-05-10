#!/usr/bin/env python3
"""
Comprehensive data collection for Task 11 report.
Gathers all metrics, computes P(A<0), gradient explosions, etc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

import wandb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import importlib.util as _ilu  # noqa: E402

from quantum_iql.buffer import load_minari_dataset  # noqa: E402
from quantum_iql.networks import CriticNetwork, ValueNetwork  # noqa: E402

WANDB_PROJECT = "quantum-iql"
WANDB_GROUP   = "hopper-medium"

GRAD_EXPLOSION_THRESHOLD = 10.0
CALIB_LO, CALIB_HI = 0.60, 0.80

# Load QVN class
_spec = _ilu.spec_from_file_location("qvn", _PROJECT_ROOT / "scripts" / "quantum_value_network.py")
_qvn_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_qvn_mod)
QVN = _qvn_mod.QuantumValueNetwork

# Buffer
buf = load_minari_dataset("mujoco/hopper/medium-v0")
rng = np.random.default_rng(42)
idx = rng.choice(buf._size, size=20_000, replace=False)
OBS = torch.FloatTensor(buf._observations[idx])
ACTS = torch.FloatTensor(buf._actions[idx])

# Modes and seeds
MODES = {
    "classical":          list(range(3)),
    "classical-deep":     list(range(8)),
    "classical-small":    list(range(3)),
    "constant-v":         list(range(8)),
    "quantum":           list(range(3)),
    "quantum-no-warmup":  list(range(8)),
    "quantum-fixed":      list(range(8)),
    "quantum-fixed-warmup": list(range(8)),
    "quantum-fixed-c":    list(range(8)),
}

def compute_p_neg(mode_dir: str, seed: int) -> dict | None:
    ckpt_dir = Path(f"experiments/checkpoints/hopper/medium/{mode_dir}/seed_{seed}")
    for fname in ["checkpoint_00100000.pt", "checkpoint_final.pt"]:
        fpath = ckpt_dir / fname
        if fpath.exists():
            ckpt = torch.load(fpath, map_location="cpu")
            if ckpt.get("step", 0) >= 95_000:
                break
    else:
        return None

    vsd = ckpt.get("value_net")
    is_quantum = "quantum" in mode_dir

    if is_quantum:
        vnet = QVN(n_qubits=8, n_layers=3, obs_dim=11)
        filtered_sd = {k: v for k, v in vsd.items() if "pre_encode" not in k}
        vnet.load_state_dict(filtered_sd, strict=False)
    else:
        if mode_dir == "classical-small":
            vnet = ValueNetwork(11, hidden_dims=[11])
        elif mode_dir == "classical-deep":
            vnet = ValueNetwork(11, hidden_dims=[8, 8, 8])
        elif mode_dir == "constant-v":
            # Constant-V doesn't compute P(A<0) meaningfully
            return None
        else:
            vnet = ValueNetwork(11, hidden_dims=[256, 256])
        vnet.load_state_dict(vsd)

    cnet = CriticNetwork(11, 3, hidden_dims=[256, 256])
    cnet.load_state_dict(ckpt["critic_net"])

    vnet.eval(); cnet.eval()
    with torch.no_grad():
        v = vnet(OBS).squeeze()
        q1, q2 = cnet(OBS, ACTS)
        q = torch.min(q1, q2).squeeze()
        adv = q - v
        p_neg = float((adv < 0).float().mean().item())
        e_a = float(adv.mean().item())

    result = {"p_neg": p_neg, "e_a": e_a}
    if is_quantum:
        result["a"] = float(vsd["a"].item())
        result["b"] = float(vsd["b"].item())
    return result


def get_wandb_data(run_name: str) -> dict | None:
    api = wandb.Api()
    # Filter by partial name match using the runs list directly
    matching = [r for r in api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP}) if r.name == run_name]
    if not matching:
        return None
    run = matching[0]
    h = run.history(samples=2000, keys=[
        "eval/mean_return", "loss/value", "advantage_mean",
        "quantum/grad_norm_theta", "quantum/active_layers",
    ])
    summary = run.summary

    final_return = summary.get("eval/mean_return", float("nan"))
    max_return = float(h["eval/mean_return"].max()) if "eval/mean_return" in h else float("nan")

    col = "quantum/grad_norm_theta"
    if col in h.columns and h[col].notna().any():
        max_grad = float(h[col].max())
        explosion = max_grad > GRAD_EXPLOSION_THRESHOLD
        exp_steps = h.loc[h[col] > GRAD_EXPLOSION_THRESHOLD, "_step"].tolist()
        first_exp = int(min(exp_steps)) if exp_steps else None
    else:
        max_grad = float("nan")
        explosion = False
        first_exp = None

    vloss_final = float(h["loss/value"].iloc[-1]) if "loss/value" in h else float("nan")
    adv_final = float(h["advantage_mean"].iloc[-1]) if "advantage_mean" in h else float("nan")

    return {
        "final_return": float(final_return),
        "max_return": max_return,
        "max_grad_norm": max_grad,
        "explosion": explosion,
        "first_exp_step": first_exp,
        "vloss_final": vloss_final,
        "adv_final": adv_final,
    }


def welch_t(a, b):
    t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2)
    return {"t": float(t), "p": float(p), "d": float(d)}


def fisher_exact(n_ok_a, n_a, n_ok_b, n_b):
    from scipy.stats import fisher_exact as _fe
    table = [[n_ok_a, n_a - n_ok_a], [n_ok_b, n_b - n_ok_b]]
    OR, p = _fe(table)
    return {"OR": float(OR), "p": float(p)}


def collect_all():
    api = wandb.Api()
    all_runs = list(api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP}))
    run_map = {r.name: r for r in all_runs}

    results = {}
    prefix_map = {
        "classical": "classical-hopper-medium",
        "classical-deep": "classical-deep-hopper-medium",
        "classical-small": "classical-small-hopper-medium",
        "constant-v": "constant-v-hopper-medium",
        "quantum": "quantum-hopper-medium",
        "quantum-no-warmup": "quantum-no-warmup-hopper-medium",
        "quantum-fixed": "quantum-fixed-hopper-medium",
        "quantum-fixed-warmup": "quantum-fixed-warmup-hopper-medium",
        "quantum-fixed-c": "quantum-fixed-c-hopper-medium",
    }

    for mode, seeds in MODES.items():
        mode_results = []
        prefix = prefix_map[mode]

        for seed in seeds:
            run_name = f"{prefix}-s{seed}"
            wandb_data = get_wandb_data(run_name) if run_name in run_map else None
            p_neg_data = compute_p_neg(mode, seed)

            record = {"seed": seed}
            if wandb_data:
                record.update(wandb_data)
            if p_neg_data:
                record.update(p_neg_data)

            mode_results.append(record)

        results[mode] = mode_results

    return results


def aggregate(results: dict) -> dict:
    agg = {}
    for mode, mode_res in results.items():
        returns = [r["final_return"] for r in mode_res if not np.isnan(r.get("final_return", float("nan")))]
        p_negs = [r["p_neg"] for r in mode_res if r.get("p_neg") is not None]
        e_as = [r["e_a"] for r in mode_res if r.get("e_a") is not None]
        n = len(mode_res)

        n_exp = sum(1 for r in mode_res if r.get("explosion", False))
        n_dir_ok = sum(1 for p in p_negs if p is not None and p > 0.5) if p_negs else 0
        n_on_tgt = sum(1 for p in p_negs if p is not None and CALIB_LO <= p <= CALIB_HI) if p_negs else 0
        n_ovr = sum(1 for p in p_negs if p is not None and p > 0.8) if p_negs else 0

        agg[mode] = {
            "n": n,
            "returns": returns,
            "mean_ret": float(np.mean(returns)) if returns else float("nan"),
            "std_ret": float(np.std(returns)) if returns else float("nan"),
            "cv_pct": float(np.std(returns) / np.mean(returns) * 100) if returns and np.mean(returns) > 0 else float("nan"),
            "p_negs": p_negs,
            "e_as": e_as,
            "n_exp": n_exp,
            "n_dir_ok": n_dir_ok,
            "n_on_tgt": n_on_tgt,
            "n_ovr": n_ovr,
            "mean_p_neg": float(np.mean(p_negs)) if p_negs else float("nan"),
            "mean_e_a": float(np.mean(e_as)) if e_as else float("nan"),
            "p_neg_range": ([float(min(p_negs)), float(max(p_negs))] if p_negs else None),
        }
    return agg


def main():
    print("Collecting all data from W&B and checkpoints …", file=sys.stderr)
    results = collect_all()
    agg = aggregate(results)

    # Save results
    output = {"per_seed": results, "aggregated": agg}
    with open("task11_data.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Saved to task11_data.json", file=sys.stderr)

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Mode':<25}  {'N':>3}  {'Mean Ret':>9}  {'Std':>7}  {'CV%':>5}  "
          f"{'P(A<0) mean':>10}  {'DirOK':>6}  {'OnTgt':>6}  {'Exp':>4}  {'E[A] mean':>8}")
    print("=" * 110)

    for mode in MODES.keys():
        if mode not in agg:
            continue
        a = agg[mode]
        n = a["n"]
        p_str = f"{a['mean_p_neg']:.3f}" if not np.isnan(a["mean_p_neg"]) else "   n/a"
        e_str = f"{a['mean_e_a']:.2f}" if not np.isnan(a["mean_e_a"]) else "   n/a"
        n_dir = a["n_dir_ok"]
        n_p = len(a["p_negs"]) if a["p_negs"] else 0
        n_tgt = a["n_on_tgt"]
        n_exp = a["n_exp"]
        print(f"{mode:<25}  {n:>3}  {a['mean_ret']:>9.1f}  {a['std_ret']:>7.1f}  "
              f"{a['cv_pct']:>5.1f}  {p_str:>10}  {n_dir}/{n_p:>5}  "
              f"{n_tgt}/{n_p:>5}  {n_exp:>4}  {e_str:>8}")

    print("=" * 110)

    # Statistical tests
    print("\nKEY STATISTICAL TESTS")
    print("=" * 80)

    # Pool quantum-fixed + quantum-fixed-warmup + quantum-fixed-c (Fix-A family) vs others
    fix_a_modes = ["quantum-fixed", "quantum-fixed-warmup", "quantum-fixed-c"]
    default_modes = ["quantum", "quantum-no-warmup", "constant-v"]

    # Pool 1: All Fix-A (n=24) vs Classical (n=3)
    fix_a_returns = []
    for m in fix_a_modes:
        fix_a_returns.extend(agg[m]["returns"])
    classical_returns = agg["classical"]["returns"]

    print(f"\nFix-A pool (n={len(fix_a_returns)}) vs Classical (n={len(classical_returns)}):")
    res = welch_t(fix_a_returns, classical_returns)
    print(f"  Return: t={res['t']:.2f} p={res['p']:.4f} d={res['d']:.2f}")

    n_ok_fix_a = sum(agg[m]["n_dir_ok"] for m in fix_a_modes)
    n_fix_a = sum(len(agg[m]["p_negs"]) for m in fix_a_modes)
    n_ok_class = agg["classical"]["n_dir_ok"]
    n_class = len(agg["classical"]["p_negs"])
    fres = fisher_exact(n_ok_fix_a, n_fix_a, n_ok_class, n_class)
    print(f"  Dir-OK: {n_ok_fix_a}/{n_fix_a} vs {n_ok_class}/{n_class}  OR={fres['OR']:.2f} p={fres['p']:.4f}")

    # Pool 2: quantum-fixed-c (Fix-C) vs quantum-fixed (Fix A+B)
    print(f"\nquantum-fixed-c vs quantum-fixed:")
    res = welch_t(agg["quantum-fixed-c"]["returns"], agg["quantum-fixed"]["returns"])
    print(f"  Return: t={res['t']:.2f} p={res['p']:.4f} d={res['d']:.2f}")

    n_ok_c = agg["quantum-fixed-c"]["n_dir_ok"]
    n_c = len(agg["quantum-fixed-c"]["p_negs"])
    n_ok_f = agg["quantum-fixed"]["n_dir_ok"]
    n_f = len(agg["quantum-fixed"]["p_negs"])
    if n_c > 0 and n_f > 0:
        fres = fisher_exact(n_ok_c, n_c, n_ok_f, n_f)
        print(f"  Dir-OK: {n_ok_c}/{n_c} vs {n_ok_f}/{n_f}  OR={fres['OR']:.2f} p={fres['p']:.4f}")

    # Pool 3: All quantum (Fix + default) vs Classical
    all_quantum_returns = []
    for m in ["quantum", "quantum-no-warmup", "quantum-fixed", "quantum-fixed-warmup", "quantum-fixed-c"]:
        all_quantum_returns.extend(agg[m]["returns"])
    print(f"\nAll quantum (n={len(all_quantum_returns)}) vs Classical (n={len(classical_returns)}):")
    res = welch_t(all_quantum_returns, classical_returns)
    print(f"  Return: t={res['t']:.2f} p={res['p']:.4f} d={res['d']:.2f}")

    # Pool 4: quantum-fixed vs quantum-no-warmup
    print(f"\nquantum-fixed vs quantum-no-warmup:")
    res = welch_t(agg["quantum-fixed"]["returns"], agg["quantum-no-warmup"]["returns"])
    print(f"  Return: t={res['t']:.2f} p={res['p']:.4f} d={res['d']:.2f}")

    n_ok_f = agg["quantum-fixed"]["n_dir_ok"]
    n_f = len(agg["quantum-fixed"]["p_negs"])
    n_ok_nw = agg["quantum-no-warmup"]["n_dir_ok"]
    n_nw = len(agg["quantum-no-warmup"]["p_negs"])
    if n_f > 0 and n_nw > 0:
        fres = fisher_exact(n_ok_f, n_f, n_ok_nw, n_nw)
        print(f"  Dir-OK: {n_ok_f}/{n_f} vs {n_ok_nw}/{n_nw}  OR={fres['OR']:.2f} p={fres['p']:.4f}")

    # Pool 5: quantum-fixed-c vs constant-v
    print(f"\nquantum-fixed-c vs constant-v:")
    res = welch_t(agg["quantum-fixed-c"]["returns"], agg["constant-v"]["returns"])
    print(f"  Return: t={res['t']:.2f} p={res['p']:.4f} d={res['d']:.2f}")

    # Pool 6: quantum-fixed-c vs quantum-no-warmup
    print(f"\nquantum-fixed-c vs quantum-no-warmup:")
    res = welch_t(agg["quantum-fixed-c"]["returns"], agg["quantum-no-warmup"]["returns"])
    print(f"  Return: t={res['t']:.2f} p={res['p']:.4f} d={res['d']:.2f}")

    print("\n" + "=" * 80)
    return results, agg


if __name__ == "__main__":
    results, agg = main()