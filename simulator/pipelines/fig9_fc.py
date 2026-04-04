from pathlib import Path

from simulator.io.results import aggregate_sequence1, write_rows
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


class Fig9Pipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        output_dir = config.output_dir / study.output_subdir
        cycles_rows: list[dict] = []
        energy_rows: list[dict] = []
        power_rows: list[dict] = []

        models = config.models or list(study.models)
        methods = config.methods or list(study.methods)
        sequence_lengths = config.sequence_lengths or list(study.sequence_lengths)

        for model_name in models:
            for sequence_length in sequence_lengths:
                for method_name in methods:
                    method_runner = method_registry.runner_for(method_name)
                    method_bits = method_registry.resolve_quantization_bits(method_name)
                    algorithm = method_registry.resolve_algorithm(method_name)
                    operations = model_registry.build_operations(model_name, method_bits, algorithm, sequence_length=sequence_length)

                    total_stats = Stats(name=method_name)
                    for op in operations:
                        if study.ops_mode == "fc_only" and not isinstance(op, FC):
                            continue
                        op_stats = method_runner.run_fc(op, config)
                        total_stats.accumulate(op_stats)

                    method_runner.apply_energy_breakdown(total_stats)

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

        aggregated_csv: Path | None = None
        if study.aggregate_sequence1:
            aggregated_csv = aggregate_sequence1(cycles_csv, energy_csv, power_csv, output_dir / "aggregated_sequence1.csv")

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            aggregated_csv=aggregated_csv,
        )