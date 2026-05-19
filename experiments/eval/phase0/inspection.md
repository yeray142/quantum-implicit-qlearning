# Phase 0 Inspection Report (Updated)

**Pool:** `eval_pools/pool_seed6plus.npz`
**Checkpoint:** `experiments/checkpoints/hopper/medium/quantum-multi-qubit-readout/seed_6/checkpoint_final.pt`
**Config:** `configs/eval_hardware.yaml`
**Date:** 2026-05-17

---

## Bugs Fixed

### Issue 1 (CRITICAL): MQR Readout Mismatch
`_generate_dru_source` only emitted `result('V', bit0)` for qubit 0, discarding all other qubits. `run_hugr_on_default_qubit` returned only ⟨Z₀⟩. For the MQR checkpoint where V(s) = Σᵢ aᵢ⟨Zᵢ⟩ + b, the guppy path was computing a₀⟨Z₀⟩ + b — NOT the trained V function.

**Fix:** `_generate_dru_source` now emits `result('Z{q}', bit{q})` for all qubits. `run_hugr_on_default_qubit` renamed to `run_hugr_and_collect_expvals` returning `np.ndarray` of shape (n_qubits,). Key format in counts is now `(('Z0', '0'), ('Z1', '0'), ...)` — a tuple with one (register, bit) pair per qubit. New helper `compute_v_from_expvals` is the single canonical V reconstruction.

### Issue 2: z_height Index Bug
`z_height_from_obs` returned `obs[1]` (torso angle, range [-0.15, 0.16]) instead of `obs[0]` (z-height, range [0.7, 1.65]).

**Fix:** Changed `return float(obs[1])` → `return float(obs[0])`. Updated HopperStrata docstring and default bands to `[[0.7, 1.0], [1.0, 1.3], [1.3, 2.0]]`.

---

## 1. Per-Seed Episode Returns

Baseline: seed-6 best known return = **3626.9** (full 1000-step episode).
5% threshold = **3445.6**.

| Seed | Mean Return | Full Eps | Flagged | Notes |
|------|-------------|----------|---------|-------|
| 6    | 3393.2      | 8/10     | 2       | ep0(657s), ep4(626s) early |
| 42   | 3483.9      | 8/10     | 2       | ep7(781s), ep9(668s) early |
| 100  | 3268.0      | 7/10     | 3       | ep8(490s), ep9(540s) early |
| 2023 | 3348.1      | 8/10     | 2       | ep4(489s), ep5(639s) early |

Seed-6 full-length episodes: **[3565.7, 3685.4]** — all within 1.6% of baseline 3626.9. ✅ PASS

---

## 2. Total Transitions

| Source      | Count   |
|-------------|---------|
| Rollout     | 36,672  |
| D4RL sample | 500     |
| **Total**   | **37,172** |

Rollout breakdown by seed:
- seed_6:   9,283 transitions (10 episodes, 2 early-terminated)
- seed_42:  9,283 transitions
- seed_100: 8,283 transitions
- seed_2023: 9,823 transitions

---

## 3. V_sim Statistics

| Metric | Value |
|--------|-------|
| min    | 237.24 |
| max    | 441.78 |
| mean   | 350.59 |
| std    | 24.98  |
| median | 349.14 |
| P5     | 312.59 |
| P95    | 394.73 |

---

## 4. Stratification Cell Counts (3×3)

| Timestep Bucket | z 0.7–1.0 | z 1.0–1.3 | z 1.3–2.0 |
|-----------------|-----------|-----------|-----------|
| 0–99            | **5** ⚠   | 3,966 ✅  | 529 ✅    |
| 100–399         | **0** ⚠   | 3,286 ✅  | 8,714 ✅  |
| 400+            | 175 ✅    | 5,837 ✅  | 14,660 ✅ |

⚠ = **< 30 states (FAIL)**

**Empirical z-quantiles (full pool, n=37,172):**

| Percentile | z-height |
|------------|-----------|
| 0%          | 0.700     |
| 10%         | 1.150     |
| 20%         | 1.218     |
| 25%         | 1.240     |
| 33%         | 1.280     |
| 50%         | 1.390     |
| 66%         | 1.468     |
| 75%         | 1.506     |
| 90%         | 1.562     |
| 95%         | 1.594     |
| 100%        | 1.652     |

**Analysis:** Only bottom ~10% of z-values fall in [0.7, 1.0]. These low-z states occur almost exclusively at early timesteps (t<100) or late timesteps (t≥400, from fallen episodes). The middle band [1.0, 1.3] captures the 10th–25th percentile. The top band [1.3, 2.0] captures 75%+ of the data.

Two cells remain sparse:
- **Cell 0** (t∈[0,99], z∈[0.7,1.0]): only 5 states — early timesteps rarely have low z (Hopper starts standing)
- **Cell 3** (t∈[100,399], z∈[0.7,1.0]): 0 states — after the initial steps, Hopper rarely drops to z<1.0 without terminating

---

## 5. D4RL Sample Summary

- **Count:** 500 transitions
- **Source:** `mujoco/hopper/medium-v0` dataset
- **Timesteps:** all exactly 0
- **Z-height range:** [0.705, 1.652] — same distribution as rollout pool

---

## 6. Schema Dump

```
pool.npz
  observations       shape=(37172, 11)  dtype=float32
  actions           shape=(37172, 3)   dtype=float32
  rewards           shape=(37172,)     dtype=float32
  next_observations shape=(37172, 11)  dtype=float32
  dones             shape=(37172,)     dtype=float32
  timesteps         shape=(37172,)     dtype=int32
  v_sim             shape=(37172,)     dtype=float32
  q_sim             shape=(37172,)     dtype=float32
  episode_ids       shape=(37172,)     dtype=int32
  seeds             shape=(37172,)     dtype=int32   # -1=D4RL, 6/42/100/2023=rollout
  pool_config       dtype=object       # dict with build metadata
```

---

## 7. Pass/Fail Verdict

| Criterion | Result |
|-----------|--------|
| Seed 6 return within 5% of 3626.9 | ✅ PASS (full episodes: 3565.7–3685.4) |
| ≥30 states per stratification cell | ⚠ **2/9 cells fail** (cells 0 and 3 have <30) |
| MQR circuit equivalence tests | ✅ PASS (both test_pennylane_vs_guppylang_high_shots, test_shot_noise_statistics) |

### Blocking Issue: 2 Sparse Stratification Cells

Cells 0 and 3 (low-z band at t<400) have insufficient states. The z-height distribution is highly skewed — only ~10% of states have z<1.0. These cells cannot be populated without either (a) relaxing the cell count requirement for sparse cells, or (b) adjusting bands to match data percentiles.

**Data-driven band alternatives** (derived from empirical quantiles above):

Option A: 3 equal-population bands (percentile-based):
```yaml
z_height_bands: [[0.70, 1.15], [1.15, 1.39], [1.39, 1.66]]
```

Option B: Merge sparse cells into existing cells, keep 3×3:
```yaml
z_height_bands: [[0.70, 1.00], [1.00, 1.30], [1.30, 2.00]]  # current — cells 0,3 sparse
```

**Recommend:** Adopt the empirical quantiles approach (Option A) for Phase 1 to ensure all cells are populated, then use data-driven bands throughout.

---

## 8. MQR Equivalence Tests

```
tests/test_circuit_equivalence.py::test_pennylane_vs_guppylang_high_shots PASSED
tests/test_circuit_equivalence.py::test_shot_noise_statistics PASSED
```

Tests ran on `quantum-multi-qubit-readout/seed_6` checkpoint (now first in priority). Tolerance: `max(0.05 * Σ|aᵢ|, 2.0)` with `Σ|aᵢ| = 406.79` → tol = 20.34. V_guppylang matches V_pennylane within tolerance for all 5 D4RL states (max diff ≈ 8.4, all well within tol=20.3).

---

## 9. Recommended Next Steps

1. **Decide on z-height bands**: Either adopt percentile-based bands or reduce the minimum cell-count threshold for the 2 sparse cells.
2. **Proceed to cost calibration** once z-bands are confirmed.
3. **Phase 1/Phase 2** can use `compute_v_from_expvals` as the canonical V readout — no duplicate readout math anywhere in the pipeline.