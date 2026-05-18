"""Cost calibration: measure empirical HQC for the DRU circuit on helios_emulator.

Run with:
  python src/eval/calibrate_cost.py --checkpoint checkpoints/checkpoint_seed6_best.pt \
      --n_calibrate 15

DO NOT run this before spending any HQC budget — it is meant to be executed once
when the Helios emulator is accessible to verify the 13.0 HQC/call estimate.
"""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timezone

import torch
import yaml

from quantum_iql import QuantumValueNetwork, load_minari_dataset
from src.eval.utils import build_dru_hugr


def main():
    parser = argparse.ArgumentParser(description="Calibrate HQC cost for DRU circuit on Helios")
    parser.add_argument("--config", type=str, default="configs/eval_hardware.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_calibrate", type=int, default=15,
                        help="Number of emulator calls to average")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    n_qubits = cfg.get("n_qubits", 8)
    n_layers = cfg.get("n_layers", 3)
    shots = cfg.get("shots", 100)

    value_net = QuantumValueNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        obs_dim=11,
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
        use_pre_encoder=True,
    ).to(device)
    value_net.load_state_dict(ckpt["value_net"], strict=False)
    if "mu" in ckpt and "sigma" in ckpt:
        value_net.update_running_stats(
            torch.as_tensor(ckpt["mu"]),
            torch.as_tensor(ckpt["sigma"]),
        )
    value_net.eval()
    mu = value_net.mu
    sigma = value_net.sigma

    # Load random states
    buffer = load_minari_dataset("mujoco/hopper/medium-v0", device="cpu")
    batch = buffer.sample(args.n_calibrate)
    states = batch.observations.numpy()

    try:
        import qnexus as qnx
    except ImportError:
        raise RuntimeError("qnexus not installed — cannot calibrate HQC costs")

    try:
        qnx.login()
        project = qnx.projects.get(name="Test")
        qnx.context.set_active_project(project)
    except Exception as e:
        raise RuntimeError(f"qnexus login failed: {e}") from e

    cost_estimates = []
    for i, state in enumerate(states):
        compiled_hugr = build_dru_hugr(value_net, mu, sigma, state, shots=shots)
        ref = qnx.hugr.upload(compiled_hugr, name=f"calibrate_{i}")
        pred = qnx.hugr.cost_confidence(
            programs=[ref],
            n_shots=[shots],
            system_name="Helios-1",
        )
        hqc_est = pred[0][0]
        cost_estimates.append(hqc_est)
        print(f"  Call {i+1}/{args.n_calibrate}: HQC={hqc_est:.2f}")

    mean_cost = statistics.mean(cost_estimates)
    stdev_cost = statistics.stdev(cost_estimates) if len(cost_estimates) > 1 else 0.0
    expected = 13.0
    delta = mean_cost - expected

    print(f"\nCalibration results ({args.n_calibrate} calls, shots={shots})")
    print(f"  Mean HQC:   {mean_cost:.2f}")
    print(f"  Std HQC:    {stdev_cost:.2f}")
    print(f"  Expected:  {expected:.2f}")
    print(f"  Delta:     {delta:+.2f}")
    print(f"  Per-call breakdown: {[f'{c:.1f}' for c in cost_estimates]}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()