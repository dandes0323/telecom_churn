import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def draw_neural_network(save_path=None):
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    layer_specs = [
        {"name": "Capa de Entrada", "neurons": 5, "show": 5, "activacion": "", "bias": False},
        {"name": "Capa Oculta 1", "neurons": 64, "show": 6, "activacion": "ReLU", "bias": True},
        {"name": "Capa Oculta 2", "neurons": 32, "show": 5, "activacion": "ReLU", "bias": True},
        {"name": "Capa Oculta 3", "neurons": 16, "show": 4, "activacion": "ReLU", "bias": True},
        {"name": "Capa Salida", "neurons": 1, "show": 1, "activacion": "Sigmoid", "bias": False}
    ]

    feature_names = ["antigüedad", "cargo mensual", "contrato", "internet", "teléfono"]

    n_layers = len(layer_specs)
    layer_x = np.linspace(0.05, 0.95, n_layers)

    neuron_positions = []
    bias_positions = []

    max_neurons = max(s["neurons"] for s in layer_specs)

    for li, spec in enumerate(layer_specs):
        x = layer_x[li]
        n_show = spec["show"]
        total = spec["neurons"]

        if total == 1:
            y_vals = [0.5]
        else:
            y_vals = np.linspace(0.15, 0.85, n_show)

        positions = []
        for yi, y in enumerate(y_vals):
            label = ""
            if li == 0 and yi < len(feature_names):
                label = feature_names[yi]
            elif li == n_layers - 1 and yi == 0:
                label = "¿Churn?"
            positions.append({"x": x, "y": y, "label": label, "is_bias": False})

        if spec.get("bias"):
            bias_y = 0.05
            positions.append({"x": x, "y": bias_y, "label": "b", "is_bias": True})
            bias_positions.append((x, bias_y))

        neuron_positions.append(positions)

    edge_color = "#FF8C00"

    for li in range(len(neuron_positions) - 1):
        layer_a = neuron_positions[li]
        layer_b = neuron_positions[li + 1]

        bias_a = [p for p in layer_a if p["is_bias"]]
        non_bias_a = [p for p in layer_a if not p["is_bias"]]
        non_bias_b = [p for p in layer_b if not p["is_bias"]]

        num_connections = min(3, len(non_bias_a), len(non_bias_b))
        if num_connections > 0:
            indices_a = np.linspace(0, len(non_bias_a) - 1, num_connections, dtype=int)
            indices_b = np.linspace(0, len(non_bias_b) - 1, num_connections, dtype=int)
            for src_idx, dst_idx in zip(indices_a, indices_b):
                src = non_bias_a[src_idx]
                dst = non_bias_b[dst_idx]
                ax.plot([src["x"], dst["x"]], [src["y"], dst["y"]],
                        color=edge_color, alpha=0.3, linewidth=0.8)

        if len(bias_a) > 0 and len(non_bias_b) > 0:
            bias = bias_a[0]
            for dst in non_bias_b[:2]:
                ax.plot([bias["x"], dst["x"]], [bias["y"], dst["y"]],
                        color=edge_color, alpha=0.2, linewidth=0.6)

    for li, spec in enumerate(layer_specs):
        for p in neuron_positions[li]:
            if p["is_bias"]:
                circle = plt.Circle((p["x"], p["y"]), 0.015, color="#e74c3c", zorder=5)
                ax.add_patch(circle)
                ax.text(p["x"], p["y"] - 0.025, "bias", color="#e74c3c", fontsize=8,
                        ha="center", va="top", fontweight="bold")
            else:
                circle = plt.Circle((p["x"], p["y"]), 0.025, color="#4fc3f7", zorder=5)
                ax.add_patch(circle)
                if p["label"]:
                    ax.text(p["x"], p["y"] - 0.04, p["label"], color="white", fontsize=7,
                            ha="center", va="top")

        if spec["show"] < spec["neurons"]:
            last_y = neuron_positions[li][spec["show"] - 1]["y"] if not any(
                pp["is_bias"] for pp in neuron_positions[li][:spec["show"]]
            ) else neuron_positions[li][spec["show"] - 2]["y"]
            extra_text = f"...+{spec['neurons'] - spec['show']}"
            ax.text(layer_x[li], last_y - 0.06, extra_text, color="#FF8C00",
                    fontsize=9, ha="center", fontweight="bold")

        ax.text(layer_x[li], 0.95, spec["name"], color="white", fontsize=11,
                ha="center", fontweight="bold")
        ax.text(layer_x[li], 0.92, f"{spec['neurons']} neuronas", color="#4fc3f7",
                fontsize=9, ha="center")
        if spec["activacion"]:
            ax.text(layer_x[li], 0.89, f"Activación: {spec['activacion']}",
                    color="#90EE90", fontsize=8, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.title("Red Neuronal - Predicción de Churn Telecom", color="white", fontsize=18,
              fontweight="bold", pad=20)

    legend_elements = [
        mpatches.Patch(color="#4fc3f7", label="Neuronas"),
        mpatches.Patch(color="#FF8C00", label="Conexiones (pesos)"),
        mpatches.Patch(color="#e74c3c", label="Sesgo (bias)")
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=10, labelcolor="white")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nn_diagram.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()

    return fig
