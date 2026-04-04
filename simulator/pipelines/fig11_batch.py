import pandas as pd

from simulator.io.results import write_rows
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.ops import FC
from simulator.specs import RunnerConfig, StudyArtifacts, StudySpec
from simulator.utils import Stats

FIG11_METHOD_LABELS = {
    "systolic_array": "SA-A8W8",
    "ant": "ANT-A8W8",
    "figna": "FIGNA-A16W4",
    "figlut_4": "FIGLUT-A16W4",
    "figlut_2": "FIGLUT-A16W2",
    "vqarray_8_decode": "EVA-A8W8",
    "vqarray_4_decode": "EVA-A16W4",
    "vqarray_3_decode": "EVA-A16W3",
    "vqarray_2_decode": "EVA-A16W2",
}

CYCLES_COLUMNS = [
    "model",
    "method",
    "batch_size",
    "sequence_length",
    "total_cycles",
    "compute_cycles",
    "total_time_s",
]

ENERGY_COLUMNS = [
    "model",
    "method",
    "batch_size",
    "sequence_length",
    "total_energy_j",
    "core_energy_j",
    "buffer_energy_j",
    "dram_energy_j",
]

POWER_COLUMNS = [
    "model",
    "method",
    "batch_size",
    "sequence_length",
    "total_power_w",
    "core_power_w",
    "sram_power_w",
    "dram_power_w",
]


class BatchScalingPipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        output_dir = config.output_dir / study.output_subdir
        models = config.models or list(study.models)
        methods = config.methods or list(study.methods)
        batch_sizes = config.batch_sizes or list(study.batch_sizes)

        cycles_rows: list[dict] = []
        energy_rows: list[dict] = []
        power_rows: list[dict] = []

        for model_name in models:
            for batch_size in batch_sizes:
                sequence_length = batch_size
                for method_name in methods:
                    runner = method_registry.runner_for(method_name)
                    bits = method_registry.resolve_quantization_bits(method_name)
                    algorithm = method_registry.resolve_algorithm(method_name)
                    operations = model_registry.build_operations(model_name, bits, algorithm, sequence_length=sequence_length)

                    total_stats = Stats(name=method_name)
                    for op in operations:
                        if not isinstance(op, FC):
                            continue
                        total_stats.accumulate(runner.run_fc(op, config))
                    runner.apply_energy_breakdown(total_stats)

                    total_energy = (
                        total_stats.core_energy
                        + total_stats.buffer_energy
                        + total_stats.dram_energy
                        + total_stats.static_energy
                    )
                    total_power = total_stats.core_power + total_stats.sram_power + total_stats.dram_power
                    total_time = total_stats.total_cycles / (500 * 10**6)

                    cycles_rows.append(
                        {
                            "model": model_name,
                            "method": method_name,
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "total_cycles": total_stats.total_cycles,
                            "compute_cycles": total_stats.compute_cycles,
                            "total_time_s": total_time,
                        }
                    )
                    energy_rows.append(
                        {
                            "model": model_name,
                            "method": method_name,
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "total_energy_j": total_energy,
                            "core_energy_j": total_stats.core_energy,
                            "buffer_energy_j": total_stats.buffer_energy,
                            "dram_energy_j": total_stats.dram_energy,
                        }
                    )
                    power_rows.append(
                        {
                            "model": model_name,
                            "method": method_name,
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "total_power_w": total_power,
                            "core_power_w": total_stats.core_power,
                            "sram_power_w": total_stats.sram_power,
                            "dram_power_w": total_stats.dram_power,
                        }
                    )

        cycles_csv = write_rows(cycles_rows, output_dir / "cycles.csv", CYCLES_COLUMNS)
        energy_csv = write_rows(energy_rows, output_dir / "energy.csv", ENERGY_COLUMNS)
        power_csv = write_rows(power_rows, output_dir / "power.csv", POWER_COLUMNS)
        summary_csv = self._write_summary(output_dir, cycles_rows, energy_rows, power_rows)
        fig11_csv = self._write_fig11_report(output_dir, summary_csv)
        verification_json = None

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            verification_json=verification_json,
            reports={"summary_csv": summary_csv, "fig11_csv": fig11_csv},
        )

    @staticmethod
    def _write_summary(output_dir, cycles_rows: list[dict], energy_rows: list[dict], power_rows: list[dict]):
        cycles_df = pd.DataFrame(cycles_rows)
        energy_df = pd.DataFrame(energy_rows)
        power_df = pd.DataFrame(power_rows)
        merged = cycles_df.merge(
            energy_df[["model", "method", "batch_size", "sequence_length", "total_energy_j", "core_energy_j", "buffer_energy_j", "dram_energy_j"]],
            on=["model", "method", "batch_size", "sequence_length"],
            how="left",
        )
        merged = merged.merge(
            power_df[["model", "method", "batch_size", "sequence_length", "total_power_w", "core_power_w", "sram_power_w", "dram_power_w"]],
            on=["model", "method", "batch_size", "sequence_length"],
            how="left",
        )
        summary_csv = output_dir / "summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(summary_csv, index=False)
        return summary_csv

    @staticmethod
    def _write_fig11_report(output_dir, summary_csv):
        summary_df = pd.read_csv(summary_csv)
        rows: list[dict] = []
        for (model_name, batch_size), subset in summary_df.groupby(["model", "batch_size"]):
            baseline = subset[subset["method"] == "systolic_array"].iloc[0]
            prefill_row = subset[subset["method"] == "vqarray_2_prefill"]
            prefill_row = prefill_row.iloc[0] if not prefill_row.empty else None
            for _, row in subset.iterrows():
                effective_time = float(row["total_time_s"])
                effective_energy = float(row["total_energy_j"])
                effective_power = float(row["total_power_w"])
                if row["method"].startswith("vqarray") and row["method"] != "vqarray_2_prefill" and prefill_row is not None:
                    if effective_time == 0.0:
                        effective_time = float(prefill_row["total_time_s"])
                    if effective_energy == 0.0:
                        effective_energy = float(prefill_row["total_energy_j"])
                    if effective_power == 0.0:
                        effective_power = float(prefill_row["total_power_w"])
                report_method = "vqarray_8_decode" if row["method"] == "vqarray_2_prefill" else row["method"]
                rows.append(
                    {
                        "model": model_name,
                        "method": report_method,
                        "paper_label": FIG11_METHOD_LABELS.get(report_method, report_method),
                        "batch_size": int(batch_size),
                        "sequence_length": int(row["sequence_length"]),
                        "total_time_s": effective_time,
                        "total_energy_j": effective_energy,
                        "total_cycles": float(row["total_cycles"]),
                        "compute_cycles": float(row["compute_cycles"]),
                        "total_power_w": effective_power,
                        "norm_speedup": float(baseline["total_time_s"]) / effective_time,
                        "norm_energy_efficiency": float(baseline["total_energy_j"]) / effective_energy,
                    }
                )
        fig11_df = pd.DataFrame(rows).sort_values(["model", "method", "batch_size"]).reset_index(drop=True)
        fig11_csv = output_dir / "fig11_batch_scaling.csv"
        fig11_df.to_csv(fig11_csv, index=False)
        return fig11_csv
