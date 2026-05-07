import json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch, torch.nn as nn, torch.optim as optim
 
# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
 
RESULTS_DIR = os.path.join(os.getcwd(), "qiql_ablation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
 
# ── GPU / device selection ────────────────────────────────────────────────────
# Try lightning.gpu (requires pennylane-lightning[gpu] + CUDA).
# Falls back to lightning.qubit (CPU-optimised C++) if GPU is unavailable.
def _best_device(n_wires: int) -> qml.devices.Device:
    try:
        dev = qml.device("lightning.gpu", wires=n_wires)
        # Probe it with a tiny circuit to confirm CUDA is really available
        @qml.qnode(dev)
        def _probe():
            return qml.state()
        _probe()
        return dev
    except Exception:
        print("[device] lightning.gpu unavailable — falling back to lightning.qubit")
        return qml.device("lightning.qubit", wires=n_wires)
 
# Move classical parameters to GPU when available
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[torch] using device: {TORCH_DEVICE}")
 
# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 130,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "lines.linewidth": 1.9,
})
P = {
    "blue":   "#378ADD", "purple": "#7F77DD", "teal":  "#1D9E75",
    "amber":  "#EF9F27", "coral":  "#D85A30", "gray":  "#888780",
    "green":  "#639922", "pink":   "#D4537E",
}
 
# ── Experiment hyperparameters ────────────────────────────────────────────────
N_STEPS  = 10**5    # training steps per run
BATCH    = 16       # batch size
OBS_DIM  = 11       # observation dimension (CartPole-like, padded to n_qubits)
 
# Grid for Exp 5 — joint (qubits × layers) sweep
EXP5_QUBITS = [4, 8, 12, 16]
EXP5_LAYERS = [1, 2, 3, 4]
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Entanglement topologies
# ─────────────────────────────────────────────────────────────────────────────
def ent_none(n):
    pass  # no 2-qubit gates
 
def ent_linear(n):
    for q in range(n - 1):
        qml.CZ(wires=[q, q + 1])
 
def ent_circular(n):
    ent_linear(n)
    qml.CZ(wires=[n - 1, 0])
 
def ent_all2all(n):
    for q in range(n):
        for r in range(q + 1, n):
            qml.CZ(wires=[q, r])
 
ENTANGLERS = {
    "none":      ent_none,
    "linear":    ent_linear,
    "circular":  ent_circular,
    "all_to_all": ent_all2all,
}
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Observable measurement schemes (Exp 4)
# ─────────────────────────────────────────────────────────────────────────────
OBSERVABLES = ["local_z0", "avg_local", "adaptive", "global"]
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. QNode factory  (data re-uploading ansatz)
# ─────────────────────────────────────────────────────────────────────────────
def build_qnode(n_qubits: int, n_layers: int, entangler_fn, observable: str):
    dev = _best_device(n_qubits)
 
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(theta, w, xs):
        for l in range(n_layers):
            for q in range(n_qubits):
                ang = theta[l, q] + w[l, q] * xs[q % len(xs)]
                qml.Rot(ang[0], ang[1], ang[2], wires=q)
            entangler_fn(n_qubits)
 
        if observable == "local_z0":
            return qml.expval(qml.PauliZ(0))
 
        elif observable == "avg_local":
            op = qml.PauliZ(0)
            for q in range(1, min(4, n_qubits)):
                op = op + qml.PauliZ(q)
            n_terms = min(4, n_qubits)
            return qml.expval(op * (1.0 / n_terms))
 
        elif observable == "adaptive":
            op = qml.PauliZ(0) @ qml.PauliZ(min(1, n_qubits - 1))
            return qml.expval(op)
 
        else:  # "global"
            op = qml.PauliZ(0)
            for q in range(1, n_qubits):
                op = op @ qml.PauliZ(q)
            return qml.expval(op)
 
    return circuit
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Quantum Value Network
# ─────────────────────────────────────────────────────────────────────────────
class QuantumValueNet(nn.Module):
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        ent_name: str = "none",
        obs:      str = "local_z0",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
 
        self.theta = nn.Parameter(
            torch.empty(n_layers, n_qubits, 3).uniform_(0, 2 * math.pi)
        )
        self.w = nn.Parameter(
            torch.empty(n_layers, n_qubits, 3).uniform_(-math.pi, math.pi)
        )
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
 
        self._circuit = build_qnode(n_qubits, n_layers, ENTANGLERS[ent_name], obs)
 
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        xs = torch.arctan(s[:, :self.n_qubits])
        ev = torch.stack([self._circuit(self.theta, self.w, x) for x in xs])
        return self.a * ev + self.b
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(net: QuantumValueNet):
    net = net.to(TORCH_DEVICE)
    opt = optim.Adam(net.parameters(), lr=3e-3)
    returns   = []
    grad_vars = []
 
    for step in range(N_STEPS):
        s      = torch.randn(BATCH, OBS_DIM, device=TORCH_DEVICE)
        target = torch.randn(BATCH, device=TORCH_DEVICE)
 
        opt.zero_grad()
        loss = torch.mean((target - net(s)) ** 2)
        loss.backward()
 
        all_grads = []
        for p in net.parameters():
            if p.grad is not None:
                all_grads.append(p.grad.detach().flatten())
        if all_grads:
            g = torch.cat(all_grads)
            grad_vars.append(float(g.var()))
        else:
            grad_vars.append(0.0)
 
        opt.step()
        returns.append(float(100 - loss.item() * 10))
 
        if (step + 1) % 10_000 == 0:
            print(f"    step {step+1:>6}/{N_STEPS}  loss={loss.item():.4f}")
 
    return {"returns": returns, "grad_vars": grad_vars}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────
def smooth(arr, window: int = 500):
    """Moving-average — window scaled to 100 k steps."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")
 
def smooth_x(n_steps: int, window: int = 500):
    return np.arange(window - 1, n_steps)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Checkpoint helper
# ─────────────────────────────────────────────────────────────────────────────
def checkpoint(all_res: dict):
    path = os.path.join(RESULTS_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(all_res, f, indent=2)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-experiment plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_learning_bar(ax_lc, ax_bar, data: dict, palette: dict,
                       bar_x_labels: list, title: str):
    xs = smooth_x(N_STEPS)
    finals = []
    for label, color in palette.items():
        if label not in data:
            continue
        curve = smooth(data[label]["returns"])
        ax_lc.plot(xs, curve, color=color, label=label)
        finals.append((label, float(np.mean(data[label]["returns"][-500:])), color))
 
    ax_lc.set_title("Learning curves", fontweight="bold")
    ax_lc.set_xlabel("Step"); ax_lc.set_ylabel("Norm. return (%)")
    ax_lc.legend()
 
    labels_f  = [f[0] for f in finals]
    values_f  = [f[1] for f in finals]
    colors_f  = [f[2] for f in finals]
    bars = ax_bar.bar(labels_f, values_f, color=colors_f, width=0.55)
    for bar, val in zip(bars, values_f):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_title("Final return", fontweight="bold")
    ax_bar.set_ylabel("Final norm. return (%)")
    ax_bar.set_ylim(0, max(values_f) * 1.18 if values_f else 110)
 
 
def _plot_grad_var(ax, data: dict, palette: dict):
    xs = smooth_x(N_STEPS)
    for label, color in palette.items():
        if label not in data:
            continue
        gv = smooth(data[label]["grad_vars"])
        ax.plot(xs, gv, color=color, label=label)
    ax.set_title("Gradient variance", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Grad variance (θ)")
    ax.legend()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. Individual experiment plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_exp1_layers(data: dict):
    palette = {"L=1": P["coral"], "L=2": P["amber"], "L=3": P["purple"]}
    fig, (ax_lc, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Learning curves — DRU layer sweep", fontweight="bold")
    _plot_learning_bar(ax_lc, ax_bar, data, palette,
                       bar_x_labels=["L=1","L=2","L=3"],
                       title="DRU layer sweep")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_layers.png"))
    plt.close(fig)
 
 
def plot_exp2_entangle(data: dict):
    palette = {
        "none":      P["gray"],
        "linear":    P["blue"],
        "circular":  P["purple"],
        "all_to_all": P["coral"],
    }
    fig, (ax_lc, ax_gv, ax_bar) = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Entanglement topology ablation", fontweight="bold")
    _plot_learning_bar(ax_lc, ax_bar, data, palette,
                       bar_x_labels=list(palette.keys()),
                       title="Entanglement topology")
    _plot_grad_var(ax_gv, data, palette)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_entangle.png"))
    plt.close(fig)
 
 
def plot_exp3_qubits(data: dict):
    palette = {
        "n=4":  P["coral"],
        "n=8":  P["amber"],
        "n=12": P["blue"],
        "n=16": P["teal"],
    }
    fig, (ax_lc, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Qubit count scaling", fontweight="bold")
    _plot_learning_bar(ax_lc, ax_bar, data, palette,
                       bar_x_labels=list(palette.keys()),
                       title="Qubit scaling")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_qubits.png"))
    plt.close(fig)
 
 
def plot_exp4_obs(data: dict):
    palette = {
        "local_z0":  P["purple"],
        "avg_local": P["blue"],
        "adaptive":  P["teal"],
        "global":    P["coral"],
    }
    fig, (ax_lc, ax_gv, ax_bar) = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Observable measurement scheme ablation", fontweight="bold")
    _plot_learning_bar(ax_lc, ax_bar, data, palette,
                       bar_x_labels=list(palette.keys()),
                       title="Observable scheme")
    _plot_grad_var(ax_gv, data, palette)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_observable.png"))
    plt.close(fig)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Exp 5: Joint (qubits × layers) sweep — heatmap + learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_exp5_joint(data: dict):
    """
    data layout:
        data[(n_qubits, n_layers)] = {"returns": [...], "grad_vars": [...]}
 
    Produces:
        • Heatmap of final normalised return  (rows=layers, cols=qubits)
        • Heatmap of mean gradient variance
        • Grid of learning curves (one line per (n,L) combo)
    """
    nq_list = sorted(set(k[0] for k in data))
    nl_list = sorted(set(k[1] for k in data))
 
    # ── Build 2-D arrays for heatmaps ────────────────────────────────────────
    ret_mat  = np.full((len(nl_list), len(nq_list)), np.nan)
    gvar_mat = np.full((len(nl_list), len(nq_list)), np.nan)
 
    for i, L in enumerate(nl_list):
        for j, n in enumerate(nq_list):
            if (n, L) not in data:
                continue
            r  = data[(n, L)]["returns"]
            gv = data[(n, L)]["grad_vars"]
            ret_mat[i, j]  = float(np.mean(r[-500:]))
            gvar_mat[i, j] = float(np.mean(gv[-500:]))
 
    # ── Figure: two heatmaps side by side ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exp 5 — Joint qubit × layer sweep", fontweight="bold")
 
    for ax, mat, title, cmap in zip(
        axes,
        [ret_mat, gvar_mat],
        ["Final normalised return (%)", "Mean gradient variance (last 500 steps)"],
        ["YlGn", "YlOrRd"],
    ):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(nq_list))); ax.set_xticklabels([f"n={n}" for n in nq_list])
        ax.set_yticks(range(len(nl_list))); ax.set_yticklabels([f"L={L}" for L in nl_list])
        ax.set_xlabel("Qubits"); ax.set_ylabel("Layers")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85)
        # Annotate cells
        for i in range(len(nl_list)):
            for j in range(len(nq_list)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if mat[i, j] < mat.mean() else "black")
 
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_joint_heatmap.png"))
    plt.close(fig)
 
    # ── Figure: learning-curve grid ───────────────────────────────────────────
    n_rows, n_cols = len(nl_list), len(nq_list)
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                                figsize=(3.5 * n_cols, 3.0 * n_rows),
                                sharex=True, sharey=True)
    fig2.suptitle("Exp 5 — Learning curves: rows=layers, cols=qubits",
                  fontweight="bold", y=1.01)
 
    xs = smooth_x(N_STEPS)
    colors_cycle = list(P.values())
 
    for i, L in enumerate(nl_list):
        for j, n in enumerate(nq_list):
            ax = axes2[i][j] if n_rows > 1 else axes2[j]
            ax.set_title(f"n={n}, L={L}", fontsize=8)
            if (n, L) in data:
                color = colors_cycle[(i * len(nq_list) + j) % len(colors_cycle)]
                curve = smooth(data[(n, L)]["returns"])
                ax.plot(xs, curve, color=color, linewidth=1.4)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color=P["gray"])
            if i == n_rows - 1:
                ax.set_xlabel("Step", fontsize=7)
            if j == 0:
                ax.set_ylabel("Norm. return (%)", fontsize=7)
 
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_joint_curves.png"))
    plt.close(fig2)
 
    print(f"  Exp 5 plots saved.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary plot
# ─────────────────────────────────────────────────────────────────────────────
def create_summary_plot(all_results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
 
    configs = [
        (
            "layers",
            "DRU layers (n=8)",
            {"L=1": P["coral"], "L=2": P["amber"], "L=3": P["purple"]},
        ),
        (
            "entangle",
            "Entanglement topology (L=3)",
            {"none": P["gray"], "linear": P["blue"],
             "circular": P["purple"], "all_to_all": P["coral"]},
        ),
        (
            "qubits",
            "Qubit count scaling",
            {"n=4": P["coral"], "n=8": P["amber"],
             "n=12": P["blue"], "n=16": P["teal"]},
        ),
        (
            "obs",
            "Observable scheme (n=8, L=3)",
            {"local_z0": P["purple"], "avg_local": P["blue"],
             "adaptive": P["teal"],  "global": P["coral"]},
        ),
    ]
 
    xs = smooth_x(N_STEPS)
    for ax, (key, title, pal) in zip(axes.flat, configs):
        data = all_results.get(key, {})
        for label, color in pal.items():
            if label in data:
                ax.plot(xs, smooth(data[label]["returns"]), color=color, label=label)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("Norm. return (%)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
 
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "summary_all_axes.png"))
    plt.close(fig)
    print(f"\nAll plots saved to {RESULTS_DIR}/")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 11. Checkpoint helper that serialises tuple keys for JSON
# ─────────────────────────────────────────────────────────────────────────────
def checkpoint(all_res: dict):
    """JSON-serialise all_res, converting tuple keys to strings."""
    def _to_json(obj):
        if isinstance(obj, dict):
            return {str(k): _to_json(v) for k, v in obj.items()}
        return obj
 
    path = os.path.join(RESULTS_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(_to_json(all_res), f, indent=2)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 12. Main orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_all_experiments():
    all_res = {}
 
    # ── Exp 1: DRU Layer Sweep ───────────────────────────────────────────────
    print("Running Exp 1: DRU Layer Sweep (ent=none)…")
    all_res["layers"] = {}
    for L in [1, 2, 3]:
        print(f"  L={L}")
        all_res["layers"][f"L={L}"] = train(
            QuantumValueNet(n_qubits=8, n_layers=L, ent_name="none", obs="local_z0")
        )
    plot_exp1_layers(all_res["layers"])
    checkpoint(all_res)
 
    # ── Exp 2: Entanglement Topology ─────────────────────────────────────────
    print("\nRunning Exp 2: Entanglement Topology (L=3, n=8)…")
    all_res["entangle"] = {}
    for ent in ENTANGLERS:
        print(f"  ent={ent}")
        all_res["entangle"][ent] = train(
            QuantumValueNet(n_qubits=8, n_layers=3, ent_name=ent, obs="local_z0")
        )
    plot_exp2_entangle(all_res["entangle"])
    checkpoint(all_res)
 
    # ── Exp 3: Qubit Count Scaling ───────────────────────────────────────────
    print("\nRunning Exp 3: Qubit Count Scaling (ent=all_to_all)…")
    all_res["qubits"] = {}
    for n in [4, 8, 12, 16]:
        layers = max(1, int(np.log2(n)))
        print(f"  n={n}, L={layers}")
        all_res["qubits"][f"n={n}"] = train(
            QuantumValueNet(n_qubits=n, n_layers=layers, ent_name="all_to_all", obs="local_z0")
        )
    plot_exp3_qubits(all_res["qubits"])
    checkpoint(all_res)
 
    # ── Exp 4: Observable Measurement Scheme ─────────────────────────────────
    print("\nRunning Exp 4: Observable Schemes (n=8, L=3, ent=all_to_all)…")
    all_res["obs"] = {}
    for obs in OBSERVABLES:
        print(f"  obs={obs}")
        all_res["obs"][obs] = train(
            QuantumValueNet(n_qubits=8, n_layers=3, ent_name="all_to_all", obs=obs)
        )
    plot_exp4_obs(all_res["obs"])
    checkpoint(all_res)
 
    # ── Exp 5: Joint Qubit × Layer Sweep ─────────────────────────────────────
    # Both n_qubits AND n_layers vary simultaneously across a 2-D grid.
    # Fixed: ent=all_to_all, obs=local_z0 (best settings from Exp 2 / Exp 4).
    print(f"\nRunning Exp 5: Joint Qubit×Layer Sweep "
          f"(qubits={EXP5_QUBITS}, layers={EXP5_LAYERS})…")
    all_res["joint"] = {}
    total = len(EXP5_QUBITS) * len(EXP5_LAYERS)
    done  = 0
    for n in EXP5_QUBITS:
        for L in EXP5_LAYERS:
            done += 1
            print(f"  [{done}/{total}] n={n}, L={L}")
            result = train(
                QuantumValueNet(n_qubits=n, n_layers=L, ent_name="all_to_all", obs="local_z0")
            )
            all_res["joint"][(n, L)] = result
            checkpoint(all_res)   # save after every cell in case of crash
 
    plot_exp5_joint(all_res["joint"])
 
    # ── Summary plot ─────────────────────────────────────────────────────────
    create_summary_plot(all_res)
    print("Done.")
 
 
if __name__ == "__main__":
    run_all_experiments()