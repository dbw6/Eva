from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "llama_7b": "7B",
    "llama_13b": "13B",
    "llama_30b": "30B",
    "llama_65b": "65B",
    "llama_2_7b": "7B",
    "llama_2_13b": "13B",
    "llama3_8b": "8B",
}

MODEL_GROUPS = [
    ("LLaMA-1", ["llama_7b", "llama_13b", "llama_30b", "llama_65b"]),
    ("LLaMA-2", ["llama_2_7b", "llama_2_13b"]),
    ("LLaMA-3", ["llama3_8b"]),
]

FIG9_METHODS = [
    "systolic_array",
    "ant",
    "figna",
    "figlut_4",
    "vqarray_4_decode",
    "vqarray_3_decode",
    "vqarray_2_decode",
]

FIG9_METHOD_LABELS = {
    "systolic_array": "SA-A8W8",
    "ant": "ANT-A8W8",
    "figna": "FIGNA-A16W4",
    "figlut_4": "FIGLUT-A16W4",
    "vqarray_4_decode": "EVA-A16W4",
    "vqarray_3_decode": "EVA-A16W3",
    "vqarray_2_decode": "EVA-A16W2",
}

ENERGY_METHOD_SHORT = {
    "systolic_array": "SA",
    "ant": "ANT",
    "figna": "FIGNA",
    "figlut_4": "FIGLUT",
    "vqarray_4_decode": "EVA-W4",
    "vqarray_3_decode": "EVA-W3",
    "vqarray_2_decode": "EVA-W2",
}

METHOD_COLORS = {
    "systolic_array": "#5aa6a5",
    "ant": "#8ec1bc",
    "figna": "#b8d7e8",
    "figlut_4": "#d8dee9",
    "vqarray_4_decode": "#f4b266",
    "vqarray_3_decode": "#f08a7e",
    "vqarray_2_decode": "#de5a4f",
}


def load_aggregated_sequence1(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    dataframe["model"] = dataframe["model"].replace("", np.nan).ffill()
    return dataframe


def plot_fig9_from_aggregated_csv(csv_path: str | Path, output_path: str | Path | None = None):
    dataframe = load_aggregated_sequence1(csv_path)
    figure_dataframe = dataframe[dataframe["method"].isin(FIG9_METHODS)].copy()

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[8, 2.2], height_ratios=[1, 1], hspace=0.55, wspace=0.15)
    latency_ax = fig.add_subplot(grid[0, 0])
    speedup_ax = fig.add_subplot(grid[0, 1])
    energy_ax = fig.add_subplot(grid[1, 0])
    efficiency_ax = fig.add_subplot(grid[1, 1])

    _plot_latency_panel(figure_dataframe, latency_ax, speedup_ax)
    _plot_energy_panel(figure_dataframe, energy_ax, efficiency_ax)

    fig.suptitle("Figure 9 Reproduction From Eva Outputs", fontsize=14, y=0.99)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def _plot_latency_panel(dataframe: pd.DataFrame, latency_ax, speedup_ax) -> None:
    bar_width = 0.11
    x_positions: list[float] = []
    model_boundaries: list[tuple[float, float, str]] = []
    cursor = 0.0

    for group_label, models in MODEL_GROUPS:
        group_start = cursor
        for model in models:
            x_positions.append(cursor)
            cursor += 1.0
        model_boundaries.append((group_start, cursor - 1.0, group_label))
        cursor += 0.8

    ordered_models = [model for _, models in MODEL_GROUPS for model in models]
    for idx, method in enumerate(FIG9_METHODS):
        values = [
            float(
                dataframe[(dataframe["model"] == model) & (dataframe["method"] == method)]["total_time_s"].iloc[0]
            )
            for model in ordered_models
        ]
        offset = (idx - (len(FIG9_METHODS) - 1) / 2) * bar_width
        latency_ax.bar(
            np.array(x_positions)[: len(ordered_models)] + offset,
            values,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.3,
            label=FIG9_METHOD_LABELS[method],
        )

    latency_ax.set_ylabel("Latency (s)")
    latency_ax.set_title("(a)")
    latency_ax.set_xticks(np.array(x_positions)[: len(ordered_models)])
    latency_ax.set_xticklabels([MODEL_LABELS[model] for model in ordered_models])
    latency_ax.grid(axis="y", alpha=0.25)
    latency_ax.set_ylim(0, max(dataframe["total_time_s"]) * 1.2)
    latency_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)

    for start, end, label in model_boundaries:
        latency_ax.text((start + end) / 2, -0.009, label, ha="center", va="top", fontsize=10)

    baseline = dataframe[dataframe["method"] == "systolic_array"].set_index("model")["total_time_s"]
    geomean_speedup = []
    for method in FIG9_METHODS:
        method_times = dataframe[dataframe["method"] == method].set_index("model")["total_time_s"]
        ratios = baseline.loc[method_times.index] / method_times
        geomean_speedup.append(float(np.exp(np.log(ratios).mean())))

    speedup_ax.bar(
        range(len(FIG9_METHODS)),
        geomean_speedup,
        color=[METHOD_COLORS[method] for method in FIG9_METHODS],
        edgecolor="black",
        linewidth=0.3,
    )
    speedup_ax.set_xticks(range(len(FIG9_METHODS)))
    speedup_ax.set_xticklabels([ENERGY_METHOD_SHORT[method] for method in FIG9_METHODS], rotation=90)
    speedup_ax.set_ylabel("Norm. Speedup")
    speedup_ax.set_title("Speedup\nGEOMEAN")
    speedup_ax.grid(axis="y", alpha=0.25)


def _plot_energy_panel(dataframe: pd.DataFrame, energy_ax, efficiency_ax) -> None:
    ordered_models = [model for _, models in MODEL_GROUPS for model in models]
    per_model_offsets = np.arange(len(ordered_models))
    method_width = 0.11

    for idx, method in enumerate(FIG9_METHODS):
        method_df = dataframe[dataframe["method"] == method].set_index("model").loc[ordered_models]
        offset = (idx - (len(FIG9_METHODS) - 1) / 2) * method_width
        x = per_model_offsets + offset
        energy_ax.bar(x, method_df["dram_energy_j"], width=method_width, color="#4c78a8", label="DRAM" if idx == 0 else "")
        energy_ax.bar(
            x,
            method_df["buffer_energy_j"],
            bottom=method_df["dram_energy_j"],
            width=method_width,
            color="#f2a541",
            label="Buffer" if idx == 0 else "",
        )
        energy_ax.bar(
            x,
            method_df["core_energy_j"],
            bottom=method_df["dram_energy_j"] + method_df["buffer_energy_j"],
            width=method_width,
            color="#c9ccd3",
            label="Core" if idx == 0 else "",
        )

    energy_ax.set_title("(b)")
    energy_ax.set_ylabel("Energy (J)")
    energy_ax.set_xticks(per_model_offsets)
    energy_ax.set_xticklabels([MODEL_LABELS[model] for model in ordered_models])
    energy_ax.grid(axis="y", alpha=0.25)
    energy_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    for start_idx, (group_label, models) in enumerate(MODEL_GROUPS):
        group_start = sum(len(group_models) for _, group_models in MODEL_GROUPS[:start_idx])
        group_end = group_start + len(models) - 1
        energy_ax.text((group_start + group_end) / 2, -0.065, group_label, ha="center", va="top", fontsize=10)

    baseline = dataframe[dataframe["method"] == "systolic_array"].set_index("model")["total_energy_j"]
    geomean_efficiency = []
    for method in FIG9_METHODS:
        method_energy = dataframe[dataframe["method"] == method].set_index("model")["total_energy_j"]
        ratios = baseline.loc[method_energy.index] / method_energy
        geomean_efficiency.append(float(np.exp(np.log(ratios).mean())))

    efficiency_ax.bar(
        range(len(FIG9_METHODS)),
        geomean_efficiency,
        color=[METHOD_COLORS[method] for method in FIG9_METHODS],
        edgecolor="black",
        linewidth=0.3,
    )
    efficiency_ax.set_xticks(range(len(FIG9_METHODS)))
    efficiency_ax.set_xticklabels([ENERGY_METHOD_SHORT[method] for method in FIG9_METHODS], rotation=90)
    efficiency_ax.set_ylabel("Norm. Energy Efficiency")
    efficiency_ax.set_title("Energy Efficiency\nGEOMEAN")
    efficiency_ax.grid(axis="y", alpha=0.25)
