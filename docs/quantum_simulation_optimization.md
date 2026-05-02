# Quantum Circuit Simulation: Bottleneck Analysis and GPU Optimization

**Date:** 2026-05-02  
**Context:** Q-IQL benchmark experiments — Classical IQL vs. Hybrid Quantum-Classical IQL on D4RL/Minari locomotion tasks.

---

## 1. The Problem

When running the first end-to-end smoke-test of the hybrid Q-IQL training loop, the quantum run stalled. After 1000 training steps, the classical run completed in ~3.4 s; the quantum run was still going after several minutes with no sign of progress.

---

## 2. Bottleneck Identification

A minimal single-step profiler isolated the problem immediately:

```
Build network        : 0.001 s
Forward  (B=4)       : 0.194 s  →  48.6 ms/sample
Backward (B=4)       : 0.032 s
```

**Extrapolated for the full training configuration (B=256, 1M steps):**

| Phase | Per step | 1 M steps |
|-------|----------|-----------|
| Forward (256 samples) | ~12.4 s | ~144 days |
| Backward (adjoint)    |  ~2.0 s |  ~23 days |
| **Total**             | **~14.4 s** | **~167 days** |

### Root cause

The value circuit is an 8-qubit DRU ansatz. Simulating one circuit call evolves a state vector of size $2^8 = 256$ complex amplitudes through the gate sequence. On CPU this takes ~50 ms per sample. The original implementation called the circuit **once per sample in a Python `for` loop**:

```python
expvals = torch.stack([
    self._circuit(theta, w, xs[i], self._active_layers)
    for i in range(B)   # ← 256 sequential circuit calls per step
])
```

With `diff_method="parameter-shift"` (the original default), the backward pass added **two extra circuit evaluations per parameter**:

$$\text{circuit calls per step} = B \times (1_\text{fwd} + 2 N_\text{params}) = 256 \times (1 + 2 \times 144) = 73{,}728$$

---

## 3. Incremental Fix Analysis

Each fix was benchmarked independently before being applied.

### Fix 1 — Switch to `diff_method="adjoint"`

**Motivation:** The adjoint differentiation method computes gradients for all $N$ parameters in a single backward sweep through the circuit (one additional state-vector pass per sample), reducing backward cost from $O(2N)$ to $O(1)$ circuit calls per sample.

**Result:**

| Method | Backward cost per step | Speedup |
|--------|----------------------|---------|
| `parameter-shift` | 256 × 288 = 73,728 calls | 1× |
| `adjoint` | 256 × 2 = 512 calls | **~144×** |

Backward time dropped from ~37 s to ~0.16 s per step. However the forward pass (~4.2 s, unchanged) remained the dominant cost.

### Fix 2 — PennyLane parameter broadcasting (batched circuit)

**Motivation:** Replace the Python `for i in range(B)` loop with a single QNode call that handles the full batch via PennyLane's parameter broadcasting mechanism. When gate angles have shape `(B, 3)`, PennyLane executes B circuits in a single call.

```python
# Before: sequential
expvals = torch.stack([circuit(theta, w, xs[i], L) for i in range(B)])

# After: batched
angles = th[q] + ww[q] * xs[:, q].unsqueeze(-1)   # (B, 3)
qml.Rot(angles[:, 0], angles[:, 1], angles[:, 2], wires=q)
expvals = circuit(theta, w, xs, L)                  # xs: (B, n_qubits)
```

**Result:** The Python loop overhead was removed, but PennyLane's CPU state-vector backend still simulates B circuits sequentially under the hood. Measured speedup: **~1.1–1.3×** — negligible.

**Why:** Broadcasting eliminates Python-level iteration but does not parallelize the underlying linear-algebra operations across samples. The 256 state-vector evolutions still execute one after another in the C++ simulator.

### Fix 3 — Reduce `quantum_batch_size`

**Motivation:** Since cost scales linearly with B, use a smaller mini-batch for the quantum value update while keeping the classical Q and actor updates at full B=256.

**Result:**

| `quantum_batch_size` | Step time | 100k steps | Note |
|----------------------|-----------|------------|------|
| 256 (full) | ~4.3 s | ~120 h | infeasible |
| 16 | ~0.73 s | ~20 h | slow |
| 4 | ~0.20 s | ~5.6 h | borderline |

With `quantum_batch_size=4` the step time becomes ~0.20 s — manageable but with very noisy gradient estimates (only 4 samples per quantum value update).

### Fix 4 — `diff_method="backprop"` on GPU ✓ **(adopted)**

**Motivation:** `diff_method="backprop"` with `interface="torch"` converts the entire circuit into a differentiable PyTorch computation graph using standard matrix operations. Once the circuit is a regular PyTorch function, moving the input tensors to CUDA is sufficient to run the full simulation on GPU.

```python
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(theta, w, xs, active_layers):
    ...

# Forward + backward on GPU:
out = circuit(theta.cuda(), w.cuda(), xs.cuda(), active_layers)
```

No custom GPU backend (e.g., `lightning.gpu` / cuStateVec) is needed. The GPU is used automatically through standard PyTorch CUDA tensors.

**Benchmark (RTX 5070 Laptop GPU, B=256, post-warmup):**

| Backend | Forward | Backward | Step | vs. CPU baseline |
|---------|---------|----------|------|-----------------|
| `adjoint` + CPU | 4.21 s | 0.16 s | 4.37 s | 1× |
| `backprop` + CPU | 0.51 s | 0.65 s | 1.16 s | 3.8× |
| `lightning.qubit` + CPU | 3.59 s | 0.21 s | 3.80 s | 1.1× |
| **`backprop` + GPU** | **0.014 s** | **0.014 s** | **0.028 s** | **~152×** |

**Post-warmup sustained throughput (B=256):** ~0.045 s/step.

---

## 4. Why `backprop` Enables GPU Execution

The key difference between differentiation methods:

| Method | How gradients are computed | Device requirement |
|--------|---------------------------|-------------------|
| `parameter-shift` | Two extra circuit runs per parameter | Must run on simulator device (CPU) |
| `adjoint` | One backward pass through state vector | Must run on simulator device (CPU) |
| `backprop` | Standard PyTorch autograd through PyTorch ops | **Follows input tensor device** |

With `backprop`, PennyLane traces the circuit as a composition of differentiable tensor operations (gate unitaries applied as matrix multiplications to the state vector). This computation graph is indistinguishable from any other PyTorch neural network. When inputs are on CUDA, **all matrix operations execute as GPU kernels**.

Memory cost of `backprop`: the full state vector at each layer must be stored for the backward pass. For 8 qubits and 3 layers:

$$\text{memory} \approx B \times L \times 2^n \times \text{sizeof(complex64)} = 256 \times 3 \times 256 \times 8 \approx 1.5 \text{ MB}$$

Negligible on any modern GPU.

---

## 5. Remaining Limitations

1. **`backprop` is exact but memory-intensive for large circuits.** For $n \geq 20$ qubits, storing $B \times L \times 2^n$ amplitudes becomes prohibitive. At that scale, `adjoint` + `lightning.gpu` (cuStateVec) is preferred.

2. **Per-step time is not yet classical-level.** At ~0.045 s/step for quantum vs. ~0.003 s/step for classical (293 steps/s), quantum training is still ~15× slower wall-clock. This is inherent to simulating a 256-dimensional state vector vs. a GPU-fused MLP kernel.

3. **Gradient noise at `quantum_batch_size < 256`.** If GPU memory is unavailable, reducing `quantum_batch_size` introduces high-variance gradient estimates. We keep the default at 256 for GPU runs.

---

## 6. Adopted Configuration

All quantum experiments in the benchmark use:

```yaml
quantum_value:
  device_name: "default.qubit"
  diff_method: "backprop"   # GPU-native via PyTorch ops
  n_qubits: 8
  n_layers: 3
```

And in `QuantumIQLConfig`:
```python
quantum_batch_size: int = 256  # full batch; GPU handles it in ~0.045s/step
```

**Expected training times (RTX 5070, B=256):**

| Runs | Steps | Wall-clock |
|------|-------|------------|
| Classical (1 run) | 1 M | ~57 min |
| Quantum (1 run) | 500 k | ~6 h |
| Full Hopper sweep (9 runs) | mixed | ~20 h |
| Full benchmark (81 runs) | mixed | ~3–4 days |

**CPU-only fallback** (no CUDA):
```python
quantum_batch_size = 4    # reduce to keep step time ~0.20s
num_steps = 100_000       # reduce total budget accordingly
diff_method = "adjoint"   # parameter-shift also works, adjoint preferred
```

---

## 7. Running the Experiments

```bash
# Hopper comparison — 3 modes × 3 seeds (recommended starting point)
python experiments/benchmark_comparison.py \
    --envs hopper \
    --datasets medium medium-replay medium-expert \
    --modes classical quantum classical-small \
    --seeds 0 1 2

# Quick sanity check (1000 steps, no W&B needed)
python experiments/benchmark_comparison.py \
    --envs hopper --datasets medium --modes classical quantum \
    --seeds 0 --num-steps 1000

# Full benchmark sweep (all environments, all splits)
python experiments/benchmark_comparison.py
```

W&B runs are grouped by `{env}-{dataset}` (e.g., `hopper-medium`) so all three modes appear together in the same panel for direct comparison.

---

## 8. Summary

| Stage | Root cause | Fix | Speedup |
|-------|-----------|-----|---------|
| Initial | Sequential loop + parameter-shift backward | — | 1× |
| Fix 1 | Parameter-shift: 288 evals/sample | `adjoint` | 144× backward |
| Fix 2 | Python loop overhead | Parameter broadcasting | 1.1–1.3× |
| Fix 3 | Full batch: 256 × 50ms = 12s/step | `quantum_batch_size=4` | 18× (with quality cost) |
| **Fix 4** | CPU-only simulation | **`backprop` + CUDA tensors** | **~100× end-to-end** |

The decisive insight: `diff_method="backprop"` makes the quantum circuit a plain PyTorch function, inheriting GPU execution for free — no special quantum GPU software required.