from pathlib import Path

import pandas as pd

from simulator.io.results import write_rows
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.ops import FC
from simulator.specs import RunnerConfig, StudyArtifacts, StudySpec
from simulator.utils import Stats


CYCLES_COLUMNS = [
    "model",
    "method",
    "sequence_length",
    "total_cycles",
    "compute_cycles",
    "total_time_s",
]

ENERGY_COLUMNS = [
    "model",
    "method",
    "sequence_length",
    "total_energy_j",
    "core_energy_j",
    "buffer_energy_j",
    "dram_energy_j",
]

POWER_COLUMNS = [
    "model",
    "method",
    "sequence_length",
    "total_power_w",
    "core_power_w",
    "sram_power_w",
    "dram_power_w",
]


class GptvqValidationPipeline:
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
        sequence_lengths = config.sequence_lengths or list(study.sequence_lengths)

        cycles_rows: list[dict] = []
        energy_rows: list[dict] = []
        power_rows: list[dict] = []

        for model_name in models:
            for sequence_length in sequence_lengths:
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
        summary_csv = self._write_summary(output_dir, cycles_rows, study)
        verification_json = None
        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            verification_json=verification_json,
            reports={"summary_csv": summary_csv},
        )

    @staticmethod
    def _write_summary(output_dir: Path, cycles_rows: list[dict], study: StudySpec) -> Path:
        dataframe = pd.DataFrame(cycles_rows)
        baseline_method = str(study.extra.get("baseline_method", "vqarray_2_decode"))
        algorithm_labels = study.extra.get(
            "algorithm_labels",
            {
                "vqarray_2_decode": "AQLM 2x8",
                "vqarray_3_decode": "AQLM 3x8",
                "vqarray_4_decode": "AQLM 4x8",
                "vqarray_2_gptvq_decode": "GPTVQ-4D",
            },
        )
        configuration_labels = study.extra.get("configuration_labels", {})
        baseline_cycles = float(dataframe[dataframe["method"] == baseline_method]["total_cycles"].iloc[0])
        dataframe["algorithm"] = dataframe["method"].map(algorithm_labels)
        dataframe["configuration"] = dataframe["method"].map(configuration_labels)
        dataframe["normalized_latency"] = dataframe["total_cycles"] / baseline_cycles
        dataframe = dataframe[
            ["model", "method", "algorithm", "configuration", "sequence_length", "total_cycles", "total_time_s", "normalized_latency"]
        ]
        summary_csv = output_dir / "table_ix.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(summary_csv, index=False)
        return summary_csv
