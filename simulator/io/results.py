from pathlib import Path

import pandas as pd


FIG9_MODEL_ORDER = [
    "llama_7b",
    "llama_13b",
    "llama_2_7b",
    "llama_2_13b",
    "llama_30b",
    "llama_65b",
    "llama3_8b",
]

FIG9_METHOD_ORDER = [
    "systolic_array",
    "ant",
    "figna",
    "figlut_4",
    "vqarray_2_decode",
    "vqarray_3_decode",
    "vqarray_4_decode",
]


def write_rows(rows: list[dict], path: Path, columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe.to_csv(path, index=False)
    return path


def aggregate_sequence1(
    cycles_csv: Path,
    energy_csv: Path,
    power_csv: Path,
    output_csv: Path,
) -> Path:
    cycles_df = pd.read_csv(cycles_csv)
    energy_df = pd.read_csv(energy_csv)
    power_df = pd.read_csv(power_csv)

    cycles_seq1 = cycles_df[cycles_df["sequence_length"] == 1].copy()
    energy_seq1 = energy_df[energy_df["sequence_length"] == 1].copy()
    power_seq1 = power_df[power_df["sequence_length"] == 1].copy()

    for dataframe in (cycles_seq1, energy_seq1, power_seq1):
        dataframe.drop(dataframe[dataframe["method"] == "vqarray_2_prefill"].index, inplace=True)

    merged = cycles_seq1[
        ["model", "method", "sequence_length", "total_cycles", "compute_cycles", "total_time_s"]
    ].copy()
    merged = merged.merge(
        energy_seq1[["model", "method", "total_energy_j", "core_energy_j", "buffer_energy_j", "dram_energy_j"]],
        on=["model", "method"],
        how="left",
    )
    merged = merged.merge(
        power_seq1[["model", "method", "total_power_w", "core_power_w", "sram_power_w", "dram_power_w"]],
        on=["model", "method"],
        how="left",
    )

    merged["model"] = pd.Categorical(merged["model"], categories=FIG9_MODEL_ORDER, ordered=True)
    merged["method"] = pd.Categorical(merged["method"], categories=FIG9_METHOD_ORDER, ordered=True)
    merged = merged[
        [
            "model",
            "method",
            "sequence_length",
            "total_cycles",
            "compute_cycles",
            "total_time_s",
            "total_energy_j",
            "core_energy_j",
            "buffer_energy_j",
            "dram_energy_j",
            "total_power_w",
            "core_power_w",
            "sram_power_w",
            "dram_power_w",
        ]
    ].sort_values(["model", "method"]).reset_index(drop=True)

    merged["model"] = merged["model"].astype(str)
    merged["method"] = merged["method"].astype(str)
    previous_model = None
    for idx in merged.index:
        current_model = merged.at[idx, "model"]
        if current_model == previous_model:
            merged.at[idx, "model"] = ""
        else:
            previous_model = current_model

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv
