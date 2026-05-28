import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
 
n_qubits = 8
n_layers = 3
 
fig, ax = plt.subplots(figsize=(18, 6))
ax.set_xlim(-0.5, 16)
ax.set_ylim(-0.5, n_qubits - 0.5)
ax.axis("off")
ax.invert_yaxis()
 
wire_color = "#222222"
gate_color = "white"
gate_edge  = "#222222"
 
x_wire_end = {}
 
def cz(x, q1, q2):
    ax.plot(x, q1, "o", color=wire_color, markersize=7, zorder=3)
    ax.plot(x, q2, "o", color=wire_color, markersize=7, zorder=3)
    ax.plot([x, x], [q1, q2], color=wire_color, lw=1.5, zorder=2)
 
def rot_gate(x, q):
    fancy = mpatches.FancyBboxPatch(
        (x - 0.38, q - 0.28), 0.76, 0.56,
        boxstyle="round,pad=0.05",
        facecolor=gate_color, edgecolor=gate_edge,
        lw=1.5, zorder=4
    )
    ax.add_patch(fancy)
    ax.text(x, q, "Rot", ha="center", va="center", fontsize=10, zorder=5)
 
def measure_gate(x, q):
    fancy = mpatches.FancyBboxPatch(
        (x - 0.38, q - 0.28), 0.76, 0.56,
        boxstyle="round,pad=0.05",
        facecolor=gate_color, edgecolor=gate_edge, lw=2, zorder=4
    )
    ax.add_patch(fancy)
    arc_x = [x + 0.22 * math.cos(t + math.pi) for t in [i*math.pi/20 for i in range(21)]]
    arc_y = [q + 0.18 * math.sin(t + math.pi) + 0.1  for t in [i*math.pi/20 for i in range(21)]]
    ax.plot(arc_x, arc_y, color=wire_color, lw=1.2, zorder=5)
    ax.annotate("", xy=(x + 0.18, q - 0.12),
                xytext=(x - 0.05, q + 0.1),
                arrowprops=dict(arrowstyle="-|>", color=wire_color, lw=1.2),
                zorder=5)
 

x = 1.0
x_start = 0.0
 

x_preamble_label = x + 0.5
x_preamble_even = x;       x += 0.9
x_preamble_odd  = x;       x += 1.1
 

layer_x = []
cz_x    = []
for layer in range(n_layers):
    layer_x.append(x);    x += 1.3
    if layer < n_layers - 1:
        cz_x.append(x);   x += 1.0
 
x_measure = x + 0.2
x_end_all  = x_measure + 0.5 
x_end_rest = layer_x[-1] + 0.38 + 0.1 
 

x_end_q1_7 = cz_x[-1] + 0.1 if cz_x else layer_x[-2] + 0.38 + 0.1
 

for q in range(n_qubits):
    end = x_end_all if q == 0 else x_end_q1_7
    ax.plot([x_start, end], [q, q], color=wire_color, lw=1.2, zorder=0)
 

for q in range(n_qubits):
    ax.text(-0.3, q, str(q), va="center", ha="right", fontsize=12)
 

ax.text(x_preamble_label, -0.42, "preamble", ha="center", fontsize=9, color="#555555")
for q in range(0, n_qubits - 1, 2):
    cz(x_preamble_even, q, q + 1)
for q in range(1, n_qubits - 1, 2):
    cz(x_preamble_odd, q, q + 1)
 

for layer in range(n_layers):
    ax.text(layer_x[layer] + 0.5, -0.42, f"layer {layer+1}",
            ha="center", fontsize=9, color="#555555")
    ax.axvline(layer_x[layer] - 0.55, color="#cccccc",
               lw=0.8, ls="--", ymin=0.02, ymax=0.98, zorder=0)
 
    last_layer = (layer == n_layers - 1)
    qubits_this_layer = [0] if last_layer else range(n_qubits)
    for q in qubits_this_layer:
        rot_gate(layer_x[layer], q)
 
    if not last_layer:
        for q in range(n_qubits - 1):
            cz(cz_x[layer], q, q + 1)
 

ax.axvline(x_measure - 0.5, color="#cccccc",
           lw=0.8, ls="--", ymin=0.02, ymax=0.98, zorder=0)
measure_gate(x_measure, 0)
 
ax.set_title(
    r"DRU Circuit  —  $n=8$ qubits, $L=3$ layers"
    "\n"
    r"$L_j^{(i)} = \mathrm{Rot}\!\left(\boldsymbol{\theta}_j^{(i)} + \mathbf{w}_j^{(i)} \odot \mathbf{x}_s\right)$"
    r"     $\langle \sigma_z \rangle$ on qubit 0 only",
    fontsize=12, pad=14
)
 
plt.tight_layout()
plt.savefig("dru_circuit.png", dpi=300, bbox_inches="tight")
print("Saved dru_circuit.png")
 