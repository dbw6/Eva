import json
from pathlib import Path

import pandas as pd

from simulator.io.results import write_rows
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.ops import FC
from simulator.config import load_hardware_config
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

TABLE_VI_COLUMNS = [
    "method",
    "architecture",
    "pe_array",
    "core_area_mm2",
    "sfu_area_mm2",
    "buffer_area_mm2",
    "total_area_mm2",
    "total_core_power_w",
    "on_chip_power_w",
    "total_power_w",
    "throughput_gops",
    "throughput_norm",
    "area_efficiency_gops_per_mm2",
    "area_efficiency_norm",
    "energy_efficiency_gops_per_w",
    "energy_efficiency_norm",
]

FIG10_AREA_COLUMNS = ["component", "area_mm2", "total_area_mm2"]
FIG10_POWER_COLUMNS = ["component", "power_w", "total_power_w"]


class HardwareCharacterizationPipeline:
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
        power_detail_by_method: dict[str, dict[str, float]] = {}

        for model_name in models:
            for sequence_length in sequence_lengths:
                for method_name in methods:
                    runner = method_registry.runner_for(method_name)
                    bits = method_registry.resolve_quantization_bits(method_name)
                    algorithm = method_registry.resolve_algorithm(method_name)
                    operations = model_registry.build_operations(model_name, bits, algorithm, sequence_length=sequence_length)

                    total_stats = Stats(name=method_name)
                    for op in operations:
                        if study.ops_mode == "fc_only" and not isinstance(op, FC):
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
                    power_detail_by_method[method_name] = {
                        "array_power_w": float(getattr(total_stats, "array_power_w", 0.0)),
                        "epilogue_power_w": float(getattr(total_stats, "epilogue_power_w", 0.0)),
                        "sfu_power_w": float(getattr(total_stats, "sfu_power_w", 0.0)),
                    }

        cycles_csv = write_rows(cycles_rows, output_dir / "cycles.csv", CYCLES_COLUMNS)
        energy_csv = write_rows(energy_rows, output_dir / "energy.csv", ENERGY_COLUMNS)
        power_csv = write_rows(power_rows, output_dir / "power.csv", POWER_COLUMNS)

        table_vi_csv = self._write_table_vi(output_dir, cycles_rows, power_rows, model_registry, method_registry, study)
        fig10_area_csv = self._write_fig10_area(output_dir)
        fig10_power_csv = self._write_fig10_power(output_dir, power_rows, power_detail_by_method)

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            verification_json=None,
            reports={
                "table_vi_csv": table_vi_csv,
                "fig10_area_csv": fig10_area_csv,
                "fig10_power_csv": fig10_power_csv,
            },
        )

    def _write_table_vi(
        self,
        output_dir: Path,
        cycles_rows: list[dict],
        power_rows: list[dict],
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
        study: StudySpec,
    ) -> Path:
        hardware = load_hardware_config()
        baseline_method = str(hardware["baseline_method"])
        architectures: dict[str, dict] = hardware["architectures"]

        cycles_df = pd.DataFrame(cycles_rows).set_index("method")
        power_df = pd.DataFrame(power_rows).set_index("method")
        total_ops = self._count_total_ops(
            model_registry,
            model_name=str(hardware["model"]),
            sequence_length=int(hardware["sequence_length"]),
            method_name=baseline_method,
            method_registry=method_registry,
        )

        records: list[dict[str, object]] = []
        for method_name in study.methods:
            method_power = power_df.loc[method_name]
            method_cycles = cycles_df.loc[method_name]
            area_components = architectures[method_name]["area_components_mm2"]
            core_area = float(area_components.get("core", area_components.get("array", 0.0) + area_components.get("epilogue", 0.0)))
            sfu_area = float(area_components.get("sfu", 0.0))
            buffer_area = float(area_components.get("buffer", 0.0))
            total_area = core_area + sfu_area + buffer_area
            on_chip_power = float(method_power["core_power_w"] + method_power["sram_power_w"])
            throughput = total_ops / float(method_cycles["total_time_s"]) / 1e9
            area_efficiency = throughput / total_area
            energy_efficiency = throughput / on_chip_power
            records.append(
                {
                    "method": method_name,
                    "architecture": architectures[method_name]["table_label"],
                    "pe_array": architectures[method_name]["pe_array"],
                    "core_area_mm2": core_area,
                    "sfu_area_mm2": sfu_area,
                    "buffer_area_mm2": buffer_area,
                    "total_area_mm2": total_area,
                    "total_core_power_w": float(method_power["core_power_w"]),
                    "on_chip_power_w": on_chip_power,
                    "total_power_w": float(method_power["total_power_w"]),
                    "throughput_gops": throughput,
                    "area_efficiency_gops_per_mm2": area_efficiency,
                    "energy_efficiency_gops_per_w": energy_efficiency,
                }
            )

        dataframe = pd.DataFrame(records)
        baseline = dataframe[dataframe["method"] == baseline_method].iloc[0]
        dataframe["throughput_norm"] = dataframe["throughput_gops"] / float(baseline["throughput_gops"])
        dataframe["area_efficiency_norm"] = dataframe["area_efficiency_gops_per_mm2"] / float(
            baseline["area_efficiency_gops_per_mm2"]
        )
        dataframe["energy_efficiency_norm"] = dataframe["energy_efficiency_gops_per_w"] / float(
            baseline["energy_efficiency_gops_per_w"]
        )
        dataframe = dataframe[TABLE_VI_COLUMNS]

        table_vi_csv = output_dir / "table_vi.csv"
        table_vi_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(table_vi_csv, index=False)
        return table_vi_csv

    @staticmethod
    def _write_fig10_area(output_dir: Path) -> Path:
        hardware = load_hardware_config()
        eva_components = hardware["architectures"]["vqarray_2_decode"]["area_components_mm2"]
        total_area = sum(float(value) for value in eva_components.values())
        rows = [
            {"component": component, "area_mm2": float(eva_components[component]), "total_area_mm2": total_area}
            for component in hardware["architectures"]["vqarray_2_decode"]["fig10_component_order"]
        ]
        fig10_area_csv = output_dir / "fig10_area_breakdown.csv"
        write_rows(rows, fig10_area_csv, FIG10_AREA_COLUMNS)
        return fig10_area_csv

    @staticmethod
    def _write_fig10_power(output_dir: Path, power_rows: list[dict], power_detail_by_method: dict[str, dict[str, float]]) -> Path:
        power_df = pd.DataFrame(power_rows).set_index("method")
        eva_power = power_df.loc["vqarray_2_decode"]
        details = power_detail_by_method["vqarray_2_decode"]
        rows = [
            {"component": "array", "power_w": details["array_power_w"], "total_power_w": float(eva_power["total_power_w"])},
            {
                "component": "epilogue",
                "power_w": details["epilogue_power_w"],
                "total_power_w": float(eva_power["total_power_w"]),
            },
            {"component": "sfu", "power_w": details["sfu_power_w"], "total_power_w": float(eva_power["total_power_w"])},
            {"component": "buffer", "power_w": float(eva_power["sram_power_w"]), "total_power_w": float(eva_power["total_power_w"])},
            {"component": "dram", "power_w": float(eva_power["dram_power_w"]), "total_power_w": float(eva_power["total_power_w"])},
        ]
        fig10_power_csv = output_dir / "fig10_power_breakdown.csv"
        write_rows(rows, fig10_power_csv, FIG10_POWER_COLUMNS)
        return fig10_power_csv

    @staticmethod
    def _count_total_ops(
        model_registry: ModelRegistry,
        model_name: str,
        sequence_length: int,
        method_name: str,
        method_registry: MethodRegistry,
    ) -> int:
        bits = method_registry.resolve_quantization_bits(method_name)
        algorithm = method_registry.resolve_algorithm(method_name)
        operations = model_registry.build_operations(model_name, bits, algorithm, sequence_length=sequence_length)
        return sum(2 * op.sequence_length * op.input_dim * op.output_dim for op in operations if isinstance(op, FC))
