import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n_qubits = 4
n_layers = 3

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def dru_circuit():
    for q in range(n_qubits - 1):
        qml.CZ(wires=[q, q + 1])
    for layer in range(1, n_layers + 1):
        for q in range(n_qubits):
            qml.RZ(0.0, wires=q)
        if layer < n_layers:
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
    return qml.expval(qml.PauliZ(0))

fig, ax = qml.draw_mpl(
    dru_circuit,
    wire_order=list(range(n_qubits)),
    show_all_wires=True,
    decimals=None,
)()

fig.set_size_inches(14, 4)

ax.set_title(
    r"DRU Circuit  —  $n=4$ qubits, $L=3$ layers"
    "\n"
    r"$L_j^{(i)} = R_Z\!\left(\boldsymbol{\theta}_j^{(i)} + \mathbf{w}_j^{(i)} \odot \mathbf{x}_s\right)$"
    r"     $\langle \sigma_z \rangle$ on qubit 0 only",
    fontsize=11,
    pad=14,
)

out = "dru_circuit.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved to {out}")