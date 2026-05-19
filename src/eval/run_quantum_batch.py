"""Backend-agnostic V(s) evaluation dispatcher.

Supports three backends:
  - "default.qubit" : local PennyLane noiseless simulation (free)
  - "helios_emulator": selene_sim DepolarizingErrorModel emulation (free)
  - "helios"         : real Helios via qnexus (HQC-cost tracked)

Each call processes one state per invocation (1 state = 1 HQC call ≈ 13 HQC at shots=100).
"""

from __future__ import annotations

import json
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantum_iql import QuantumValueNetwork
from src.eval.hqc_tracker import HQCTracker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── Helios error model ───────────────────────────────────────────────────────
HELIOS_ERROR_MODEL = {
    "p_1q": 2.5e-5,   # Helios single-qubit fidelity 99.9975%
    "p_2q": 7.9e-4,   # Helios two-qubit fidelity 99.921%
    # p_init/p_meas are conservative SPAM estimates (not published)
    "p_init": 1e-3,
    "p_meas": 1e-3,
}


# ─── qnexus mock for dry-run ─────────────────────────────────────────────────
class _MockRefHugr:
    """Minimal mock of a qnexus HUGR reference."""
    def __init__(self, compiled_hugr, name: str):
        assert compiled_hugr is not None, "qnexus mock received None HUGR"
        self._hugr = compiled_hugr
        self.name = name


class _MockJob:
    """Minimal mock of a qnexus job."""
    def __init__(self, job_id: str, cost: float = 13.0, expvals: np.ndarray | None = None):
        self._job_id = job_id
        self._cost = cost
        self._expvals = expvals

    def cost(self) -> float:
        return self._cost

    def set_expvals(self, expvals: np.ndarray) -> None:
        self._expvals = expvals


class _MockResults:
    """Minimal mock of qnexus job results with synthetic counts."""
    def __init__(self, expvals: np.ndarray, shots: int, n_qubits: int):
        self._expvals = expvals
        self._shots = shots
        self._n_qubits = n_qubits

    def download_result(self) -> _MockDownloadResult:
        return _MockDownloadResult(self._expvals, self._shots, self._n_qubits)


class _MockDownloadResult:
    """Minimal mock of downloaded result with collated counts."""
    def __init__(self, expvals: np.ndarray, shots: int, n_qubits: int):
        self._expvals = expvals
        self._shots = shots
        self._n_qubits = n_qubits

    def collated_counts(self) -> dict:
        """Reconstruct counts from expectation values.

        Given ⟨Zᵢ⟩ = (n0 - n1)/shots, we set n0 = (1+⟨Zᵢ⟩)/2 * shots, n1 = shots - n0.
        This produces the correct expvals when run through expvals_from_counts.
        """
        counts: dict[tuple, int] = {}
        for q in range(self._n_qubits):
            p1 = (1.0 - self._expvals[q]) / 2.0
            n0 = int(round(p1 * self._shots))
            n1 = self._shots - n0
            key = tuple(('Z%d' % q, str(b)) for b in [0, 1])
            # For each qubit, we need separate entries for 0 and 1 outcomes
            # Use the convention: a single measured value per qubit per shot
            pass  # simplified: generate a canonical all-zeros then flip bits
        # Build a canonical all-zeros outcome then flip bits per qubit
        all_zero_key = tuple(('Z%d' % q, '0') for q in range(self._n_qubits))
        all_one_key = tuple(('Z%d' % q, '1') for q in range(self._n_qubits))
        counts[all_zero_key] = 0
        counts[all_one_key] = 0
        for q in range(self._n_qubits):
            # For qubit q, probability of measuring 1 is (1 - ⟨Zq⟩) / 2
            p1 = (1.0 - self._expvals[q]) / 2.0
            expected_one_counts = int(round(p1 * self._shots))
            expected_zero_counts = self._shots - expected_one_counts
            # We need to model this as individual qubit measurements, not joint
            # Simulate: for each shot, qubit q measured either 0 or 1
            # Use the mean-field approximation: counts per qubit independently
            for outcome, n_count in [('0', expected_zero_counts), ('1', expected_one_counts)]:
                key = tuple(('Z%d' % i, '0' if i != q else outcome) for i in range(self._n_qubits))
                counts[key] = n_count
        # This is an approximation - the actual collated_counts is more complex
        # Let's just generate counts from a Bernoulli distribution per qubit
        rng = np.random.default_rng(42)
        counts = {}
        shot_patterns = []
        for _ in range(self._shots):
            pattern = []
            for q in range(self._n_qubits):
                p1 = (1.0 - self._expvals[q]) / 2.0
                bit = 1 if rng.random() < p1 else 0
                pattern.append(bit)
            shot_patterns.append(tuple(pattern))
        # Collate: for each unique pattern, count occurrences
        from collections import Counter
        pattern_counts = Counter(shot_patterns)
        for pattern, n in pattern_counts.items():
            key = tuple(('Z%d' % q, str(pattern[q])) for q in range(self._n_qubits))
            counts[key] = n
        return counts


class QNexusMock:
    """In-process qnexus mock that exercises the real HUGR through default.qubit.

    Args:
        value_net: QuantumValueNetwork (used to re-run circuit for expvals)
        mu, sigma: normalization stats
        shots: number of shots for default.qubit reference run
    """
    def __init__(
        self,
        value_net,
        mu,
        sigma,
        shots: int = 100,
        seed: int = 42,
        states: np.ndarray | None = None,
    ):
        self._value_net = value_net
        self._mu = mu
        self._sigma = sigma
        self._shots = shots
        self._rng = np.random.default_rng(seed)
        self._job_counter = 0
        self._states = states if states is not None else np.array([])
        self._expval_cache: dict[int, np.ndarray] = {}

    def _run_default_qubit_one_state(self, state: np.ndarray) -> tuple[float, np.ndarray]:
        """Get V and expvals via default.qubit (real circuit, no noise)."""
        value_net = self._value_net
        mu, sigma = self._mu, self._sigma
        shots = self._shots

        with torch.no_grad():
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(value_net.mu.device)
            out = value_net(s_t)
            v = float(out.float().cpu().item())
            xs_enc = torch.arctan((s_t - mu) / (sigma + 1e-8))
            if value_net.obs_dim < value_net.n_qubits:
                pad = torch.zeros(1, value_net.n_qubits - value_net.obs_dim, device=xs_enc.device)
                xs_enc = torch.cat([xs_enc, pad], dim=-1)
            elif value_net.obs_dim > value_net.n_qubits:
                xs_enc = value_net.pre_encode(xs_enc)
            if value_net._diff_method == "backprop":
                raw_expvals_list = value_net._circuit(
                    value_net.theta, value_net.w, xs_enc, value_net._active_layers
                )
            else:
                raw_expvals_list = value_net._circuit(
                    value_net.theta.cpu(), value_net.w.cpu(), xs_enc.cpu(), value_net._active_layers
                ).to(xs_enc.device)
            raw_expvals = torch.stack(raw_expvals_list, dim=-1).float().cpu().numpy()[0]
        return v, raw_expvals

    # ── mocked qnexus API ────────────────────────────────────────────────────
    def login(self) -> None:
        pass

    @property
    def projects(self) -> _MockProjects:
        return _MockProjects()

    @property
    def context(self) -> _MockContext:
        return _MockContext()

    @property
    def hugr(self) -> _MockHugr:
        return _MockHugr(self)

    def start_execute_job(self, **kwargs) -> _MockJob:
        self._job_counter += 1
        cost = kwargs.get("max_cost", [13.0])[0]
        # Get the state index from job name (e.g., "helios_s0_3" -> state index 3)
        job_name = kwargs.get("name", f"mock_job_{self._job_counter}")
        state_idx = int(job_name.split("_")[-1])
        state = self._states[state_idx] if state_idx < len(self._states) else self._states[0]
        _, expvals = self._run_default_qubit_one_state(state)
        job = _MockJob(job_id=f"mock_job_{self._job_counter}", cost=cost)
        job.set_expvals(expvals)
        return job

    @property
    def jobs(self) -> _MockJobs:
        return _MockJobs(self)


class _MockProjects:
    def get(self, name: str) -> _MockProject:
        return _MockProject(name)


class _MockProject:
    def __init__(self, name: str):
        self.name = name


class _MockContext:
    def set_active_project(self, project) -> None:
        pass


class _MockHugr:
    def __init__(self, mock: QNexusMock):
        self._mock = mock

    def upload(self, compiled_hugr, name: str = "") -> _MockRefHugr:
        assert compiled_hugr is not None, f"qnexus.hugr.upload received None HUGR (name={name})"
        return _MockRefHugr(compiled_hugr, name)

    def cost_confidence(self, programs, n_shots, system_name) -> list:
        # Return (estimated_hqc, confidence) tuple per program
        return [[13.0, 95.0]] * len(programs)


class _MockJobs:
    def __init__(self, mock: QNexusMock):
        self._mock = mock

    def wait_for(self, job: _MockJob) -> None:
        pass

    def results(self, job: _MockJob) -> list[_MockResults]:
        # Use the expvals stored in the job (set by start_execute_job)
        expvals = job._expvals if job._expvals is not None else np.zeros(self._mock._value_net.n_qubits)
        return [_MockResults(expvals, self._mock._shots, self._mock._value_net.n_qubits)]

    def get(self, job: _MockJob) -> _MockJob:
        return job


@contextmanager
def mock_qnexus(value_net, mu, sigma, shots: int = 100, seed: int = 42, states: np.ndarray | None = None):
    """Context manager that mocks qnexus module for dry-run testing.

    Patches `qnexus` in `sys.modules` to use QNexusMock so the real circuit
    code path is exercised end-to-end without contacting real Helios hardware.

    Usage:
        states = np.array([...])
        with mock_qnexus(value_net, mu, sigma, shots=100, states=states) as mock:
            run_on_helios(...)  # qnexus calls go to mock
    """
    import sys

    mock = QNexusMock(value_net, mu, sigma, shots=shots, seed=seed, states=states)

    class _PatchedQnx:
        """Virtual qnexus module that routes all calls through the mock."""
        def __init__(self, _mock: QNexusMock):
            object.__setattr__(self, '_mock', _mock)

        def login(self) -> None:
            self._mock.login()

        @property
        def projects(self) -> _MockProjects:
            return self._mock.projects

        @property
        def context(self) -> _MockContext:
            return self._mock.context

        @property
        def hugr(self) -> _MockHugr:
            return self._mock.hugr

        def start_execute_job(self, **kwargs):
            return self._mock.start_execute_job(**kwargs)

        @property
        def jobs(self) -> _MockJobs:
            return self._mock.jobs

        class QuantinuumConfig:
            def __init__(self, device_name: str = "Helios-1"):
                self.device_name = device_name

    class _PatchedQnx:
        """Virtual qnexus module that routes all calls through the mock."""
        def __init__(self, _mock: QNexusMock):
            object.__setattr__(self, '_mock', _mock)

        def login(self) -> None:
            self._mock.login()

        @property
        def projects(self) -> _MockProjects:
            return self._mock.projects

        @property
        def context(self) -> _MockContext:
            return self._mock.context

        @property
        def hugr(self) -> _MockHugr:
            return self._mock.hugr

        def start_execute_job(self, **kwargs):
            return self._mock.start_execute_job(**kwargs)

        @property
        def jobs(self) -> _MockJobs:
            return self._mock.jobs

        class QuantinuumConfig:
            def __init__(self, device_name: str = "Helios-1"):
                self.device_name = device_name

    patched = _PatchedQnx(mock)
    original_qnx_ref = sys.modules.get('qnexus')
    try:
        sys.modules['qnexus'] = patched
        yield patched
    finally:
        if original_qnx_ref is not None:
            sys.modules['qnexus'] = original_qnx_ref
        else:
            sys.modules.pop('qnexus', None)


@dataclass
class BatchResult:
    states_indices: list[int]
    v_values: list[float]
    v_raw_expval: list[np.ndarray]  # per-qubit ⟨Zᵢ⟩ per state
    hqc_consumed: float
    shots: int
    active_layers: int
    backend_info: dict


def _arctan_encode(s: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.arctan((s - mu) / (sigma + 1e-8))


def _encode_state(
    state: np.ndarray,
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Arctan-encode a single state to xs tensor for the circuit."""
    s_t = torch.from_numpy(state.astype(np.float32))
    if s_t.ndim == 1:
        s_t = s_t.unsqueeze(0)
    xs = _arctan_encode(s_t, mu, sigma)

    obs_dim = value_net.obs_dim
    n_qubits = value_net.n_qubits

    if obs_dim < n_qubits:
        pad = torch.zeros(1, n_qubits - obs_dim)
        xs = torch.cat([xs, pad], dim=-1)
    elif obs_dim > n_qubits:
        if value_net.use_pre_encoder:
            xs = value_net.pre_encode(xs)
        else:
            xs = xs[:, :n_qubits]

    return xs  # (1, n_qubits)


# ─── Core execution ──────────────────────────────────────────────────────────

def _execute_v_circuit(
    value_net,
    mu,
    sigma,
    state,
    runner_kind,
    shots,
    **runner_kwargs,
):
    """Build HUGR, run via the given runner, reconstruct V via canonical path.

    Args:
        value_net: QuantumValueNetwork with loaded weights
        mu, sigma: running stats for arctan encoding
        state: (obs_dim,) single state vector
        runner_kind: "selene" | "qnexus"
        shots: number of measurement shots
        **runner_kwargs: passed to _run_via_selene or _run_via_qnexus

    Returns:
        (v, expvals) where v is V(s) and expvals is np.ndarray(n_qubits,)
    """
    from src.eval.utils import build_dru_hugr, compute_v_from_expvals, expvals_from_counts

    compiled_hugr = build_dru_hugr(value_net, mu, sigma, state, shots)

    if runner_kind == "selene":
        counts = _run_via_selene(
            compiled_hugr, value_net.n_qubits, shots,
            error_model=runner_kwargs.get("error_model"),
        )
    elif runner_kind == "qnexus":
        counts = _run_via_qnexus(
            compiled_hugr, value_net.n_qubits, shots,
            **runner_kwargs,
        )
    else:
        raise ValueError(f"Unknown runner_kind: {runner_kind}")

    expvals = expvals_from_counts(counts, value_net.n_qubits, shots)
    v = compute_v_from_expvals(expvals, value_net)
    return v, expvals


def _run_via_selene(
    compiled_hugr,
    n_qubits: int,
    shots: int,
    error_model=None,
):
    """Run HUGR via selene_sim Quest simulator.

    Args:
        compiled_hugr: result of build_dru_hugr()
        n_qubits: number of qubits
        shots: number of measurement shots
        error_model: selene_sim DepolarizingErrorModel or None for noiseless

    Returns:
        collated_counts dict
    """
    try:
        from hugr.qsystem.result import QsysResult
        from selene_sim import Quest
        from selene_sim import build as build_runner
    except ImportError as exc:
        raise RuntimeError("selene-sim not installed") from exc

    runner = build_runner(compiled_hugr)
    kwargs = dict(simulator=Quest(), n_qubits=n_qubits, n_shots=shots)
    if error_model is not None:
        kwargs["error_model"] = error_model

    raw_result = runner.run_shots(**kwargs)
    result_obj = QsysResult(raw_result)
    return result_obj.collated_counts()


def _run_via_qnexus(
    compiled_hugr,
    n_qubits: int,
    shots: int,
    project_name: str = "Test",
    system_name: str = "Helios-1",
):
    """Run HUGR on real Helios hardware via qnexus.

    Args:
        compiled_hugr: result of build_dru_hugr()
        n_qubits: number of qubits
        shots: number of measurement shots
        project_name: qnexus project name
        system_name: Quantinuum system name (e.g. "Helios-1")

    Returns:
        (collated_counts, actual_cost) tuple
    """
    try:
        import qnexus as qnx
    except ImportError as exc:
        raise RuntimeError("qnexus not installed") from exc

    qnx.login()
    project = qnx.projects.get(name=project_name)
    qnx.context.set_active_project(project)

    ref_hugr = qnx.hugr.upload(compiled_hugr, name=f"batch_exec_n{n_qubits}")

    # Pre-flight cost estimate
    prediction = qnx.hugr.cost_confidence(
        programs=[ref_hugr],
        n_shots=[shots],
        system_name=system_name,
    )
    cost_estimate = prediction[0][0]

    job = qnx.start_execute_job(
        programs=[ref_hugr],
        name="batch_exec",
        n_shots=[shots],
        n_qubits=n_qubits,
        max_cost=[cost_estimate],
        backend_config=qnx.QuantinuumConfig(device_name=system_name),
    )
    qnx.jobs.wait_for(job)
    results = qnx.jobs.results(job)
    counts = results[0].download_result().collated_counts()

    # Actual cost from job metadata
    actual_cost = cost_estimate  # qnexus may not expose actual; use estimate as fallback
    try:
        job_meta = qnx.jobs.get(job)
        if hasattr(job_meta, "cost") and job_meta.cost is not None:
            actual_cost = float(job_meta.cost)
    except Exception:
        pass

    return counts, actual_cost


# ─── Backend implementations ─────────────────────────────────────────────────

def run_on_default_qubit(
    states: np.ndarray,
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    shots: int = 100,
) -> tuple[list[float], list[np.ndarray]]:
    """Run V(s) on local PennyLane default.qubit (noiseless).

    Returns (v_values, raw_expvals) where raw_expvals is list of (n_qubits,) arrays.
    """
    value_net.eval()
    v_results = []
    raw_results = []

    with torch.no_grad():
        for state in states:
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(value_net.mu.device)
            # Canonical forward pass — no direct _circuit access
            out = value_net(s_t)  # uses .forward() which wraps _circuit + readout
            # out is (B,) = V(s) after the linear head; raw expvals not separately returned
            v = float(out.float().cpu().item())
            # For v_raw_expval: reconstruct via _arctan_encode + _circuit to get ⟨Z_i⟩
            xs_enc = torch.arctan((s_t - mu) / (sigma + 1e-8))
            if value_net.obs_dim < value_net.n_qubits:
                pad = torch.zeros(1, value_net.n_qubits - value_net.obs_dim, device=xs_enc.device)
                xs_enc = torch.cat([xs_enc, pad], dim=-1)
            elif value_net.obs_dim > value_net.n_qubits:
                xs_enc = value_net.pre_encode(xs_enc)
            if value_net._diff_method == "backprop":
                raw_expvals_list = value_net._circuit(value_net.theta, value_net.w, xs_enc, value_net._active_layers)
            else:
                raw_expvals_list = value_net._circuit(
                    value_net.theta.cpu(), value_net.w.cpu(), xs_enc.cpu(), value_net._active_layers
                ).to(xs_enc.device)
            raw_expvals = torch.stack(raw_expvals_list, dim=-1).float().cpu().numpy()[0]  # (n_qubits,)
            raw_results.append(raw_expvals)
            v_results.append(v)

    return v_results, raw_results


def run_on_helios_emulator(
    states: np.ndarray,
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    shots: int = 100,
    error_model=None,
) -> tuple[list[float], list[np.ndarray]]:
    """Run V(s) on helios_emulator via selene_sim DepolarizingErrorModel.

    Error rates (from Helios published specs):
      - p_1q = 2.5e-5  (single-qubit fidelity 99.9975%)
      - p_2q = 7.9e-4  (two-qubit fidelity 99.921%)
      - p_init, p_meas are conservative SPAM estimates (not published)

    Args:
        states: (N, obs_dim) array
        value_net: QuantumValueNetwork with loaded weights
        mu, sigma: running stats for arctan encoding
        shots: number of shots per circuit evaluation
        error_model: selene_sim DepolarizingErrorModel or None to use HELIOS_ERROR_MODEL
    """
    try:
        from selene_sim import DepolarizingErrorModel
    except ImportError:
        log.warning("selene-sim not available, falling back to default.qubit")
        return run_on_default_qubit(states, value_net, mu, sigma, shots)

    if error_model is None:
        em = HELIOS_ERROR_MODEL
        error_model = DepolarizingErrorModel(
            p_1q=em["p_1q"],
            p_2q=em["p_2q"],
            p_init=em["p_init"],
            p_meas=em["p_meas"],
        )

    v_results = []
    raw_results = []

    for state in states:
        v, expvals = _execute_v_circuit(
            value_net, mu, sigma, state,
            runner_kind="selene",
            shots=shots,
            error_model=error_model,
        )
        v_results.append(v)
        raw_results.append(expvals)

    return v_results, raw_results


def run_on_helios(
    states: np.ndarray,
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tracker: HQCTracker,
    shots: int = 100,
    batch_id: int = 0,
    results_dir: str | Path = "eval_pools/phase2_results",
    active_layers: int = 3,
    project_name: str = "Test",
    system_name: str = "Helios-1",
    states_indices: list[int] | None = None,
) -> BatchResult:
    """Run V(s) on real Helios hardware via qnexus.

    HQC is tracked via tracker. Raises BudgetExceededError if budget would be exceeded.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    hqc_per_call = 13.0

    v_results = []
    raw_results = []
    hqc_used = 0.0
    if states_indices is None:
        states_indices = list(range(len(states)))

    for i, state in enumerate(states):
        if not tracker.check(hqc_per_call):
            tracker.hard_abort()

        # Build HUGR for this state
        from src.eval.utils import build_dru_hugr, compute_v_from_expvals, expvals_from_counts
        compiled_hugr = build_dru_hugr(value_net, mu, sigma, state, shots)
        assert compiled_hugr is not None, (
            f"build_dru_hugr returned None for state {i}. "
            "Check guppylang installation and circuit compilation."
        )
        # Re-run via qnexus properly
        try:
            import qnexus as qnx
        except ImportError as exc:
            raise RuntimeError("qnexus not installed") from exc

        qnx.login()
        project = qnx.projects.get(name=project_name)
        qnx.context.set_active_project(project)

        ref_hugr = qnx.hugr.upload(compiled_hugr, name=f"helios_s{batch_id}_{i}")

        prediction = qnx.hugr.cost_confidence(
            programs=[ref_hugr],
            n_shots=[shots],
            system_name=system_name,
        )
        cost_estimate = prediction[0][0]

        if not tracker.check(cost_estimate):
            tracker.hard_abort()

        job = qnx.start_execute_job(
            programs=[ref_hugr],
            name=f"helios_s{batch_id}_{i}",
            n_shots=[shots],
            n_qubits=value_net.n_qubits,
            max_cost=[cost_estimate],
            backend_config=qnx.QuantinuumConfig(device_name=system_name),
        )
        qnx.jobs.wait_for(job)
        results = qnx.jobs.results(job)
        counts = results[0].download_result().collated_counts()

        actual_cost = cost_estimate
        try:
            job_meta = qnx.jobs.get(job)
            if hasattr(job_meta, "cost") and job_meta.cost is not None:
                actual_cost = float(job_meta.cost)
        except Exception:
            pass

        expvals = expvals_from_counts(counts, value_net.n_qubits, shots)
        v = compute_v_from_expvals(expvals, value_net)

        v_results.append(v)
        raw_results.append(expvals)
        tracker.record(actual_cost, batch_id=batch_id * 1000 + i)
        hqc_used += actual_cost

        batch_data = {
            "phase": "quantum_batch",
            "batch_id": batch_id * 1000 + i,
            "states_indices": [states_indices[i]],  # actual pool index, not loop index
            "v_values": [v],
            "v_raw_expval": [expvals.tolist()],
            "hqc_consumed": actual_cost,
            "shots": shots,
            "n_qubits": value_net.n_qubits,
            "n_layers": value_net.n_layers,
            "active_layers": active_layers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend_info": {"device": "helios", "simulator": "qnexus"},
        }
        with open(results_dir / f"batch_{batch_id * 1000 + i:06d}.json", "w") as f:
            json.dump(batch_data, f, indent=2)

    return BatchResult(
        states_indices=states_indices,
        v_values=v_results,
        v_raw_expval=raw_results,
        hqc_consumed=hqc_used,
        shots=shots,
        active_layers=active_layers,
        backend_info={"device": "helios", "simulator": "qnexus"},
    )


def run_quantum_batch(
    states: np.ndarray,
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    backend: str,
    shots: int = 100,
    tracker: HQCTracker | None = None,
    batch_id: int = 0,
    states_indices: list[int] | None = None,
    results_dir: str | Path = "eval_pools/phase1_results",
    active_layers: int = 3,
    dry_run: bool = False,
    project_name: str = "Test",
    system_name_hardware: str = "Helios-1",
    system_name_emulator: str = "Helios-1E",
) -> BatchResult:
    """Dispatch V(s) evaluation to the specified backend.

    Args:
        states: (N, obs_dim) array of states
        value_net: QuantumValueNetwork with loaded weights
        mu, sigma: normalization stats for arctan encoding
        backend: "default.qubit" | "helios_emulator" | "helios"
        shots: number of shots per circuit evaluation
        tracker: HQCTracker for helios backend
        batch_id: batch identifier for checkpointing
        states_indices: pool indices for the states (for provenance)
        results_dir: where to write per-batch JSON results
        active_layers: number of active DRU layers
        dry_run: route helios/helios_emulator to default.qubit, track HQC as if real
        project_name: qnexus project name
        system_name_hardware: Quantinuum system for helios backend
        system_name_emulator: Quantinuum system for helios_emulator pricing

    Returns:
        BatchResult with V values and metadata
    """
    n_states = len(states)
    if states_indices is None:
        states_indices = list(range(n_states))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Running {n_states} states on backend={backend}"
             + (" (DRY RUN)" if dry_run else ""))

    hqc_per_call = 13.0

    # Dry run: route to default.qubit but track HQC as if real Helios
    if dry_run:
        if backend == "helios":
            # Route to default.qubit for execution, but exercise tracker logic
            log.info(f"  dry_run: routed to default.qubit, tracking HQC as if helios")
            if tracker is not None:
                for i in range(n_states):
                    if not tracker.check(hqc_per_call):
                        tracker.hard_abort()
                    tracker.record(hqc_per_call, batch_id=batch_id * 1000 + i)
            backend = "default.qubit"
        elif backend == "helios_emulator":
            original_backend = backend
            backend = "default.qubit"
            log.info(f"  dry_run: routed to default.qubit (original={original_backend}), "
                     f"tracking HQC as if {original_backend}")
            if tracker is not None:
                if not tracker.check(hqc_per_call * n_states):
                    tracker.hard_abort()
                tracker.record(hqc_per_call * n_states, batch_id=batch_id)

    if backend == "default.qubit":
        v_vals, raw_vals = run_on_default_qubit(states, value_net, mu, sigma, shots=shots)
        hqc_used = 0.0
        backend_info = {"device": "default.qubit", "simulator": "pennylane"}

    elif backend == "helios_emulator":
        v_vals, raw_vals = run_on_helios_emulator(states, value_net, mu, sigma, shots=shots)
        hqc_used = 0.0
        backend_info = {"device": "helios_emulator", "simulator": "selene_sim"}

    elif backend == "helios":
        result = run_on_helios(
            states, value_net, mu, sigma, tracker,
            shots=shots, batch_id=batch_id,
            states_indices=states_indices,
            results_dir=results_dir, active_layers=active_layers,
            project_name=project_name,
            system_name=system_name_hardware,
        )
        return result

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Persist batch results
    # raw_vals is list[np.ndarray]; serialize as list[list[float]]
    raw_vals_serializable = [arr.tolist() for arr in raw_vals]

    batch_data = {
        "phase": "quantum_batch",
        "batch_id": batch_id,
        "states_indices": states_indices,
        "v_values": v_vals,
        "v_raw_expval": raw_vals_serializable,
        "hqc_consumed": hqc_used,
        "shots": shots,
        "n_qubits": value_net.n_qubits,
        "n_layers": value_net.n_layers,
        "active_layers": active_layers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend_info": backend_info,
    }

    checkpoint_path = results_dir / f"batch_{batch_id:04d}.json"
    with open(checkpoint_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    log.info(f"Batch {batch_id}: saved to {checkpoint_path}, hqc={hqc_used:.1f}")

    return BatchResult(
        states_indices=states_indices,
        v_values=v_vals,
        v_raw_expval=raw_vals,
        hqc_consumed=hqc_used,
        shots=shots,
        active_layers=active_layers,
        backend_info=backend_info,
    )


def load_batch_results(results_dir: str | Path) -> list[dict]:
    """Load all batch JSONs from a results directory.

    Excludes batch_0099 (the noiseless default.qubit reference run for Phase 1),
    which contains all 250 states and would cause duplicate counting.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    batches = []
    for path in sorted(results_dir.glob("batch_*.json")):
        with open(path) as f:
            d = json.load(f)
        # Skip the noiseless reference batch (batch_id=99 in Phase 1)
        if d.get("batch_id") == 99 and d.get("backend_info", {}).get("device") == "default.qubit":
            continue
        batches.append(d)
    return batches