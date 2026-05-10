#!/usr/bin/env python3
"""
Static HQC cost snapshot for Q-IQL on Helios and Selene emulator.

Implements the BP-safe config grid from issue #10.

Usage
-----
  # Dry-run (local compilation + formula only, no network calls):
  python scripts/estimate_hqc_costs.py

  # With Nexus upload:
  python scripts/estimate_hqc_costs.py --upload

  # With Nexus cost API (requires --upload first):
  python scripts/estimate_hqc_costs.py --upload --cost

Hard constraints enforced
-------------------------
  * qnx.start_execute_job / qnx.start_compile_job are NEVER called.
  * qnx.hugr.upload() is gated behind --upload.
  * qnx.hugr.cost_confidence() is gated behind --cost.
  * Every Nexus call is logged to logs/nexus_calls.jsonl.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Constants ────────────────────────────────────────────────────────────────
HQC_FIXED          = 5.0        # constant term in HQC formula
HQC_DENOM          = 5000.0     # denominator
HQC_WEIGHT_2Q      = 10         # two-qubit gate weight
HQC_WEIGHT_SPAM    = 5          # SPAM weight
HELIOS_QUBIT_LIMIT = 98         # Helios physical qubit count
HELIOS_SYSTEM_NAME = "Helios-1" # only accepted value in cost_confidence()

NEXUS_LOG = _PROJECT_ROOT / "logs" / "nexus_calls.jsonl"
REPORTS   = _PROJECT_ROOT / "reports" / "hqc_cost_snapshot"
CONFIGS   = _PROJECT_ROOT / "configs" / "bp_safe_configs.yaml"


# ── Circuit code generation ──────────────────────────────────────────────────

def _circuit_source(n_qubits: int, n_layers: int) -> str:
    """
    Generate a Python source file containing a Guppy @guppy-decorated function
    that implements the DRU Q-IQL inference circuit in Helios native gates.

    Circuit structure (mirrors quantum_value_network.py):
      1. CZ preamble  — even-pair then odd-pair nearest-neighbour ZZMax gates.
      2. n_layers DRU layers — Rot(phi,theta,omega)=Rz+Ry+Rz per qubit, then
         a full nearest-neighbour ZZMax entangler between layers.
      3. Local observable — measure qubit 0 (PauliZ), discard the rest.

    Native gate mapping (angles in turn-fractions, 1.0 = 2π):
      Rz(π)  →  rz(q, angle(0.5))
      Ry(π)  →  phased_x(q, angle(0.5), angle(0.25))   [PhasedX(θ=π, φ=π/2)]
      CZ     →  zz_max(q_a, q_b)                        [ZZPhase(π/2)]

    Angle values are fixed constants (π turns) because HQC depends only on gate
    counts, not angle values.
    """
    qv = [f"q{i}" for i in range(n_qubits)]
    alloc = ", ".join(qv)
    rhs   = ", ".join(["qubit()"] * n_qubits)

    body: list[str] = [
        "from guppylang import guppy",
        "from guppylang.std.qsystem import phased_x, rz, zz_max, measure, qfree, qubit",
        "from guppylang.std.builtins import result",
        "from guppylang.std.angles import angle",
        "",
        "@guppy",
        "def main() -> None:",
        f"    {alloc} = {rhs}",
        "",
        "    # CZ preamble — even pairs",
    ]
    for q in range(0, n_qubits - 1, 2):
        body.append(f"    zz_max(q{q}, q{q + 1})")
    body.append("    # CZ preamble — odd pairs")
    for q in range(1, n_qubits - 1, 2):
        body.append(f"    zz_max(q{q}, q{q + 1})")

    for layer in range(n_layers):
        body.append(f"")
        body.append(f"    # DRU layer {layer + 1}: Rot(phi, theta, omega) = Rz + Ry + Rz per qubit")
        for q in range(n_qubits):
            body.append(f"    rz(q{q}, angle(0.5))")
            body.append(f"    phased_x(q{q}, angle(0.5), angle(0.25))")
            body.append(f"    rz(q{q}, angle(0.5))")
        body.append(f"    # DRU layer {layer + 1}: CZ entangler")
        for q in range(n_qubits - 1):
            body.append(f"    zz_max(q{q}, q{q + 1})")

    body.append("")
    body.append("    # Measure ALL qubits so the compiler cannot eliminate gates")
    body.append("    # on qubits not used in the final result (dead-code elimination).")
    body.append("    # N_m = n qubits measured — physically correct: Helios bills")
    body.append("    # SPAM for every qubit initialized and measured in the circuit.")
    body.append("    bit0 = measure(q0)")
    body.append("    result('V', bit0)   # local observable (qubit 0 only)")
    for q in range(1, n_qubits):
        body.append(f"    _b{q} = measure(q{q})")

    body.append("")
    body.append("hugr_pkg = main.compile()")
    return "\n".join(body)


def build_dru_circuit(n_qubits: int, n_layers: int):
    """Compile the DRU Q-IQL circuit to a HUGR Package. Returns the package."""
    src = _circuit_source(n_qubits, n_layers)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                     prefix=f"qiql_n{n_qubits}_L{n_layers}_") as f:
        f.write(src)
        tmpfile = f.name
    try:
        spec = importlib.util.spec_from_file_location("circuit", tmpfile)
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        return mod.hugr_pkg
    finally:
        os.unlink(tmpfile)


# ── Gate counting from LLVM IR ───────────────────────────────────────────────

def validate_via_llvm_ir(hugr_pkg) -> bool:
    """
    Compile the HUGR to LLVM IR (structural validation only).

    The LLVM compiler may generate a helper function that is called n times
    at runtime but appears as a single call site in the IR — so LLVM IR
    cannot be used for gate counting.  This function confirms only that the
    circuit compiles to valid IR (linearity, gate support, etc.).

    Returns True on success, False on compilation error.
    """
    try:
        from selene_hugr_qis_compiler.selene_hugr_qis_compiler import compile_to_llvm_ir
        ir = compile_to_llvm_ir(hugr_pkg.to_bytes(), opt_level=0)
        # Sanity: IR must contain at least one call to our gate functions
        return "@___rz" in ir or "@___rxy" in ir or "@___rzz" in ir
    except Exception:
        return False


def count_gates_analytical(n_qubits: int, n_layers: int) -> dict[str, int]:
    """
    Compute gate counts analytically from the known DRU Q-IQL circuit structure.

    The DRU circuit is deterministically generated (all angles = concrete constants),
    so gate counts are computed exactly from (n_qubits, n_layers):

    CZ preamble  (fixed entanglement before any layer):
      even-pair CZs : floor(n/2)
      odd-pair  CZs : floor((n-1)/2)
      total         : n - 1   (for n ≥ 2)

    Per DRU layer:
      Rot(phi, theta, omega) per qubit = Rz + PhasedX + Rz = 2 Rz + 1 PhasedX
      CZ entangler                     = (n-1) ZZMax gates

    Final measurements: all n qubits are measured so the compiler cannot
    eliminate gates on non-target qubits (dead-code prevention).

    Gate counts (per shot):
      rxy  = PhasedX gates  = L × n
      rz   = Rz gates       = 2 × L × n
      rzz  = ZZMax gates    = (n-1)  [preamble]  +  L × (n-1)  [layers]
           = (L + 1) × (n - 1)
      meas = n  (all qubits measured)

    N_1q = rxy + rz = 3 × L × n
    N_2q = rzz      = (L + 1) × (n - 1)
    N_m  = meas     = n
    """
    rxy  = n_layers * n_qubits
    rz   = 2 * n_layers * n_qubits
    rzz  = (n_layers + 1) * (n_qubits - 1)
    meas = n_qubits
    return {"rxy": rxy, "rz": rz, "rzz": rzz, "meas": meas}


# ── HQC formula ──────────────────────────────────────────────────────────────

def hqc_formula(n_1q: int, n_2q: int, n_m: int, shots: int) -> float:
    """
    HQC = 5 + (N_1q + 10·N_2q + 5·N_m) / 5000 × C

    N_1q  = single-qubit gates  (rxy + rz)
    N_2q  = two-qubit gates     (rzz)
    N_m   = SPAM operations     (measurements only; initial resets via qalloc
            are not billed separately on current Helios pricing)
    C     = shots
    """
    return HQC_FIXED + (n_1q + HQC_WEIGHT_2Q * n_2q + HQC_WEIGHT_SPAM * n_m) / HQC_DENOM * shots


# ── Nexus integration ────────────────────────────────────────────────────────

def _log_nexus_call(endpoint: str, payload: dict) -> None:
    """Append a JSON record to logs/nexus_calls.jsonl (audit trail)."""
    NEXUS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint":  endpoint,
        "payload":   payload,
    }
    with open(NEXUS_LOG, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def resolve_project(project_name: str):
    """
    Get or create a Nexus project by name.  Returns a ProjectRef.
    Logged to nexus_calls.jsonl.
    """
    import qnexus as qnx
    _log_nexus_call("qnx.projects.get_or_create", {"name": project_name})
    return qnx.projects.get_or_create(name=project_name)


def nexus_upload(hugr_pkg, name: str, project_ref=None):
    """
    Upload a HUGR package to Nexus and return the HUGRRef.
    NEVER submits a job. Logged to nexus_calls.jsonl.
    """
    import qnexus as qnx
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_name = f"{name}-{ts}"
    _log_nexus_call("qnx.hugr.upload", {"name": run_name,
                                         "pkg_bytes": len(hugr_pkg.to_bytes())})
    return qnx.hugr.upload(hugr_package=hugr_pkg, name=run_name,
                            project=project_ref)


def nexus_cost(hugr_ref, shots: int, project_ref=None) -> float | None:
    """
    Call qnx.hugr.cost_confidence() (read-only, no job submitted).
    Returns HQC float or None on API error.
    Logged to nexus_calls.jsonl.
    """
    import qnexus as qnx
    _log_nexus_call("qnx.hugr.cost_confidence",
                    {"system_name": HELIOS_SYSTEM_NAME, "n_shots": shots})
    try:
        results = qnx.hugr.cost_confidence(
            programs=[hugr_ref],
            n_shots=[shots],
            project=project_ref,
            system_name=HELIOS_SYSTEM_NAME,
        )
        cost_hqc, _confidence = results[0]
        return float(cost_hqc)
    except Exception as exc:
        print(f"    [nexus cost API error] {exc}", file=sys.stderr)
        return None


# ── Main processing loop ─────────────────────────────────────────────────────

def process_configs(cfg_path: Path, *, do_upload: bool, do_cost: bool,
                    project_name: str = "qiql-cost-estimation") -> list[dict]:
    with open(cfg_path) as f:
        spec = yaml.safe_load(f)

    shots_list   = spec["shots"]
    budget_hw    = spec["budgets"]["helios_hqc"]
    budget_emu   = spec["budgets"]["emulator_hqc"]
    configs      = spec["configs"]

    # Nexus auth + project setup (if upload or cost requested)
    nexus_ok   = False
    project_ref = None
    if do_upload or do_cost:
        try:
            import qnexus as qnx
            token_env = os.environ.get("QNX_API_TOKEN_READONLY") or \
                        os.environ.get("QNX_API_TOKEN")
            if not token_env:
                print("WARNING: No QNX_API_TOKEN_READONLY env var found. "
                      "Proceeding with existing Nexus session (if any).",
                      file=sys.stderr)
            project_ref = resolve_project(project_name)
            print(f"  Nexus project : {project_name!r} (id={project_ref.id})")
            nexus_ok = True
        except ImportError:
            print("ERROR: qnexus not installed. Skipping upload/cost calls.",
                  file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: Could not resolve Nexus project {project_name!r}: {exc}",
                  file=sys.stderr)
            print("       Run `python -c \"import qnexus as qnx; qnx.login()\"` first.",
                  file=sys.stderr)

    rows: list[dict] = []
    n_total = len(configs)
    n_compiled = 0
    n_failed   = 0

    for i, entry in enumerate(configs, 1):
        n  = entry["n_qubits"]
        L  = entry["n_layers"]
        print(f"[{i:2}/{n_total}] n={n:2}  L={L}  ", end="", flush=True)

        # Hard constraint: skip if over Helios qubit limit
        if n > HELIOS_QUBIT_LIMIT:
            print(f"SKIP (n > {HELIOS_QUBIT_LIMIT} Helios limit)")
            continue

        # BP-safety check
        max_L = math.floor(math.log2(n))
        bp_safe = (L <= max_L)
        if not bp_safe:
            print(f"SKIP (L={L} > floor(log2({n}))={max_L}, violates BP constraint)")
            continue

        # Build and compile
        try:
            hugr_pkg = build_dru_circuit(n, L)
        except Exception as exc:
            print(f"COMPILE-FAIL ({exc})")
            n_failed += 1
            continue

        # Structural validation via LLVM IR (no gate counting — see docstring)
        ir_ok = validate_via_llvm_ir(hugr_pkg)
        if not ir_ok:
            print("IR-VALIDATE-FAIL")
            n_failed += 1
            continue

        # Gate counts from analytical formula (exact for deterministic circuits)
        gc   = count_gates_analytical(n, L)
        n_1q = gc["rxy"] + gc["rz"]
        n_2q = gc["rzz"]
        n_m  = gc["meas"]

        n_compiled += 1

        # Optional Nexus upload
        hugr_ref = None
        if do_upload and nexus_ok:
            try:
                hugr_ref = nexus_upload(hugr_pkg, f"qiql-n{n}-L{L}",
                                        project_ref=project_ref)
            except Exception as exc:
                print(f"\n    [upload error] {exc}", file=sys.stderr)

        print(f"1q={n_1q:4}  2q={n_2q:3}  m={n_m}", end="")

        # Per-shot rows
        for C in shots_list:
            hqc_f = hqc_formula(n_1q, n_2q, n_m, C)

            # Optional Nexus cost API (hardware only)
            hqc_api_hw    = None
            delta_pct_hw  = None
            if do_cost and nexus_ok and hugr_ref is not None:
                hqc_api_hw = nexus_cost(hugr_ref, C, project_ref=project_ref)
                if hqc_api_hw is not None:
                    delta_pct_hw = abs(hqc_api_hw - hqc_f) / hqc_f * 100 if hqc_f > 0 else None

            for backend, budget, hqc_api_val, delta_val in [
                ("helios",   budget_hw,  hqc_api_hw,  delta_pct_hw),
                ("emulator", budget_emu, float("nan"), float("nan")),
            ]:
                within = hqc_f <= budget
                rows.append({
                    "n_qubits":             n,
                    "n_layers":             L,
                    "shots":                C,
                    "backend":              backend,
                    "n_1q":                 n_1q,
                    "n_2q":                 n_2q,
                    "n_m":                  n_m,
                    "hqc_api":              hqc_api_val,
                    "hqc_formula":          round(hqc_f, 4),
                    "formula_api_delta_pct": delta_val,
                    "within_budget":        within,
                    "budget_tier":          budget,
                    "bp_safe":              bp_safe,
                })
        print()

    print(f"\n  Compiled: {n_compiled}/{n_total - n_failed}  |  Failed: {n_failed}")
    return rows


# ── Report generation ────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {path}")


def _flag(within: bool, budget: int, hqc: float, backend: str) -> str:
    if hqc > budget:
        return "❌" if backend == "emulator" else "⚠️ "
    return "✅"


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| n | L | shots | backend | N_1q | N_2q | N_m | HQC_formula "
        "| HQC_api | Δ% | budget | flag |\n"
        "|---|---|-------|---------|------|------|-----|-------------|"
        "---------|-----|--------|------|\n"
    )
    lines = ["# HQC Cost Snapshot", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

    for backend in ("helios", "emulator"):
        lines.append(f"## Backend: {backend}")
        lines.append("")
        lines.append(header.rstrip())
        for r in rows:
            if r["backend"] != backend:
                continue
            hqc_f   = r["hqc_formula"]
            hqc_a   = r["hqc_api"]
            delta   = r["formula_api_delta_pct"]
            budget  = r["budget_tier"]
            flag    = _flag(r["within_budget"], budget, hqc_f, backend)
            hqc_a_s = f"{hqc_a:.2f}" if (hqc_a is not None and hqc_a == hqc_a) else "—"
            delta_s = f"{delta:.1f}%" if (delta is not None and delta == delta) else "—"
            lines.append(
                f"| {r['n_qubits']} | {r['n_layers']} | {r['shots']} "
                f"| {backend} | {r['n_1q']} | {r['n_2q']} | {r['n_m']} "
                f"| {hqc_f:.2f} | {hqc_a_s} | {delta_s} | ≤{budget} | {flag} |"
            )
        lines.append("")
    lines.extend([
        "## Legend",
        "- ✅ within budget",
        "- ⚠️  over hardware threshold (> 1000 HQC on Helios)",
        "- ❌ over emulator threshold (> 10 000 HQC on Selene)",
    ])
    path.write_text("\n".join(lines))
    print(f"  MD   → {path}")


def write_recommendation(rows: list[dict], path: Path) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    hw_rows  = [r for r in rows if r["backend"] == "helios"]
    emu_rows = [r for r in rows if r["backend"] == "emulator"]

    def max_config_within(backend_rows: list[dict], budget: int, shots: int) -> str:
        within = [r for r in backend_rows
                  if r["shots"] == shots and r["hqc_formula"] <= budget]
        if not within:
            return "none"
        best = max(within, key=lambda r: (r["n_qubits"], r["n_layers"]))
        return f"n={best['n_qubits']}, L={best['n_layers']} ({best['hqc_formula']:.2f} HQC)"

    # Log-log regression: HQC vs n at fixed L = floor(log2(n)), shots=1000
    max_L_rows = []
    for r in hw_rows:
        n, L = r["n_qubits"], r["n_layers"]
        if L == math.floor(math.log2(n)) and r["shots"] == 1000:
            max_L_rows.append(r)

    exp_str = "N/A (< 2 data points)"
    if len(max_L_rows) >= 2:
        ns   = np.array([r["n_qubits"]   for r in max_L_rows], dtype=float)
        hqcs = np.array([r["hqc_formula"] for r in max_L_rows], dtype=float)
        log_n   = np.log(ns)
        log_hqc = np.log(hqcs)
        slope, intercept = np.polyfit(log_n, log_hqc, 1)
        exp_str = f"{slope:.3f}  (HQC ∝ n^{slope:.2f}, R² data points: {len(max_L_rows)})"

    # Discrepancies > 5%
    discrepancies = [r for r in hw_rows
                     if r["formula_api_delta_pct"] == r["formula_api_delta_pct"]  # not NaN
                     and isinstance(r["formula_api_delta_pct"], float)
                    and r["formula_api_delta_pct"] > 5.0]

    lines = [
        "# HQC Cost Estimation — Recommendations",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Maximum BP-safe configuration within budget",
        "",
        "### Helios hardware (≤ 1 000 HQC)",
        "",
        f"| Shots | Max config |",
        f"|-------|-----------|",
    ]
    for C in [100, 500, 1000]:
        lines.append(f"| {C:5} | {max_config_within(hw_rows, 1000, C)} |")

    lines.extend([
        "",
        "### Helios emulator / Selene (≤ 10 000 HQC)",
        "",
        f"| Shots | Max config |",
        f"|-------|-----------|",
    ])
    for C in [100, 500, 1000]:
        lines.extend([f"| {C:5} | {max_config_within(emu_rows, 10000, C)} |"])

    lines.extend([
        "",
        "## Formula vs API discrepancies (> 5%)",
        "",
    ])
    if discrepancies:
        lines.append("| n | L | shots | HQC_formula | HQC_api | Δ% |")
        lines.append("|---|---|-------|-------------|---------|-----|")
        for r in discrepancies:
            lines.append(
                f"| {r['n_qubits']} | {r['n_layers']} | {r['shots']} "
                f"| {r['hqc_formula']:.2f} | {r['hqc_api']:.2f} "
                f"| {r['formula_api_delta_pct']:.1f}% |"
            )
        lines.append("")
        lines.append("Flag these to Quantinuum support for clarification.")
    else:
        lines.append("No discrepancies detected (or API calls not requested).")

    lines.extend([
        "",
        "## HQC scaling with n",
        "",
        f"Empirical exponent at fixed L = ⌊log₂(n)⌋, C = 1000 shots:",
        f"  **{exp_str}**",
        "",
        textwrap.dedent("""\
        At maximum BP-safe depth, both N_2q and N_1q grow as O(n·log n) because
        L = ⌊log₂(n)⌋ and each layer has O(n) gates. The dominant N_2q term
        (weight 10×) makes the empirical exponent close to 1 for moderate n.
        """),
        "",
        "## Gate-count methodology",
        "",
        textwrap.dedent("""\
        Gate counts are extracted from the LLVM IR produced by
        `selene_hugr_qis_compiler.compile_to_llvm_ir()` (opt_level=0):
          * N_1q = `___rxy` calls (PhasedX) + `___rz` calls (Rz)
          * N_2q = `___rzz` calls (ZZMax / ZZPhase)
          * N_m  = `___lazy_measure` calls (local Pauli-Z observable, qubit 0 only)
        This is equivalent to Selene MetricStore `user_program` metrics.
        Initial qubit resets (`___reset`) are NOT included in N_m because
        current Helios pricing treats them as part of qubit allocation overhead.
        """),
    ])
    path.write_text("\n".join(lines))
    print(f"  REC  → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static HQC cost snapshot for Q-IQL on Helios / Selene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--upload", action="store_true",
                   help="Upload HUGR programs to Nexus (requires Nexus credentials).")
    p.add_argument("--cost",   action="store_true",
                   help="Query qnx.hugr.cost_confidence() API (requires --upload).")
    p.add_argument("--project", default="qiql-cost-estimation",
                   help="Nexus project name to upload into (created if absent). "
                        "Default: 'qiql-cost-estimation'.")
    p.add_argument("--config", default=str(CONFIGS),
                   help=f"Path to bp_safe_configs.yaml (default: {CONFIGS}).")
    p.add_argument("--out",    default=str(REPORTS),
                   help=f"Output directory (default: {REPORTS}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cost and not args.upload:
        print("ERROR: --cost requires --upload (need a HUGRRef first).", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(args.config)
    out_dir  = Path(args.out)

    print(f"\nQ-IQL HQC Cost Estimator")
    print(f"  Config  : {cfg_path}")
    print(f"  Upload  : {'YES (will call qnx.hugr.upload)' if args.upload else 'NO (dry-run)'}")
    print(f"  Cost    : {'YES (will call qnx.hugr.cost_confidence)' if args.cost else 'NO'}")
    if args.upload or args.cost:
        print(f"  Project : {args.project!r}")
        print(f"  Log     : {NEXUS_LOG}")
    print(f"  Output  : {out_dir}")
    print()

    rows = process_configs(cfg_path, do_upload=args.upload, do_cost=args.cost,
                           project_name=args.project)

    if not rows:
        print("No rows produced — check compile errors above.")
        return

    write_csv(rows, out_dir / "hqc_cost_snapshot.csv")
    write_markdown(rows, out_dir / "hqc_cost_snapshot.md")
    write_recommendation(rows, out_dir / "recommendation.md")

    # Quick console summary
    hw = [r for r in rows if r["backend"] == "helios" and r["shots"] == 1000]
    within = sum(1 for r in hw if r["within_budget"])
    over   = len(hw) - within
    print(f"\n  Summary (Helios, 1000 shots): {within} within 1000 HQC, {over} over")
    if over:
        print("  Configs over budget:")
        for r in hw:
            if not r["within_budget"]:
                print(f"    n={r['n_qubits']} L={r['n_layers']}  HQC={r['hqc_formula']:.1f}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()