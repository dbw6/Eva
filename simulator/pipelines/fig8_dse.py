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
    "sweep_type",
    "sweep_value",
    "total_cycles",
    "compute_cycles",
    "mem_stall_cycles",
    "total_time_s",
    "mem_bandwidth_utilization",
]

ENERGY_COLUMNS = [
    "model",
    "method",
    "sequence_length",
    "sweep_type",
    "sweep_value",
    "total_energy_j",
    "core_energy_j",
    "buffer_energy_j",
    "dram_energy_j",
    "static_energy_j",
    "array_energy_j",
    "epilogue_energy_j",
    "sfu_energy_j",
    "compute_unit_energy_j",
]

POWER_COLUMNS = [
    "model",
    "method",
    "sequence_length",
    "sweep_type",
    "sweep_value",
    "total_power_w",
    "core_power_w",
    "sram_power_w",
    "dram_power_w",
    "array_power_w",
    "epilogue_power_w",
    "sfu_power_w",
]

AREA_COLUMNS = [
    "model",
    "method",
    "sequence_length",
    "adder_tree_size",
    "array_area_mm2",
    "adder_tree_area_mm2",
    "sfu_area_mm2",
    "total_area_mm2",
]


class DsePipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        output_dir = config.output_dir / study.output_subdir
        scenario = str(study.extra.get("scenario", "decode"))
        if scenario == "prefill":
            return self._run_prefill_bandwidth(output_dir, config, study, model_registry, method_registry)
        if scenario == "both":
            return self._run_both(output_dir, config, study, model_registry, method_registry)
        return self._run_decode(output_dir, config, study, model_registry, method_registry)

    def _run_both(
        self,
        output_dir: Path,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        decode_dir = output_dir / "decode"
        prefill_dir = output_dir / "prefill"

        decode_study = StudySpec(
            name=study.name,
            description=study.description,
            phase="decode",
            ops_mode=study.ops_mode,
            models=study.models,
            methods=tuple(study.extra.get("decode_methods", list(study.methods))),
            sequence_lengths=study.sequence_lengths,
            batch_sizes=study.batch_sizes,
            aggregate_sequence1=study.aggregate_sequence1,
            output_subdir=study.output_subdir,
            extra={**study.extra, "scenario": "decode"},
        )
        prefill_study = StudySpec(
            name=study.name,
            description=study.description,
            phase="prefill",
            ops_mode=study.ops_mode,
            models=study.models,
            methods=tuple(study.extra.get("prefill_methods", list(study.methods))),
            sequence_lengths=study.sequence_lengths,
            batch_sizes=study.batch_sizes,
            aggregate_sequence1=study.aggregate_sequence1,
            output_subdir=study.output_subdir,
            extra={
                **study.extra,
                "scenario": "prefill",
                "paper_figure_method": study.extra.get("paper_prefill_method", "vqarray_2_prefill"),
            },
        )

        decode_artifacts = self._run_decode(decode_dir, config, decode_study, model_registry, method_registry)
        prefill_artifacts = self._run_prefill_bandwidth(prefill_dir, config, prefill_study, model_registry, method_registry)

        output_dir.mkdir(parents=True, exist_ok=True)
        for src_name in ("fig8_num_eu.csv", "fig8_memory_bandwidth_decode.csv"):
            src = decode_dir / src_name
            if src.exists():
                dst = output_dir / src_name
                dst.write_bytes(src.read_bytes())
        prefill_src = prefill_dir / "fig8_memory_bandwidth_prefill.csv"
        if prefill_src.exists():
            dst = output_dir / "fig8_memory_bandwidth_prefill.csv"
            dst.write_bytes(prefill_src.read_bytes())

        reports = {}
        reports.update(decode_artifacts.reports)
        reports.update(prefill_artifacts.reports)

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=decode_artifacts.cycles_csv,
            energy_csv=decode_artifacts.energy_csv,
            power_csv=decode_artifacts.power_csv,
            verification_json=None,
            reports=reports,
        )

    def _run_decode(
        self,
        output_dir: Path,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        model_name = (config.models or list(study.models))[0]
        sequence_length = (config.sequence_lengths or list(study.sequence_lengths))[0]
        methods = config.methods or list(study.methods)

        bandwidth_values = list(study.extra.get("memory_bandwidth_values", []))
        adder_values = list(study.extra.get("adder_tree_sizes", []))

        cycles_rows: list[dict] = []
        energy_rows: list[dict] = []
        power_rows: list[dict] = []
        area_rows: list[dict] = []

        for method_name in methods:
            for mem_width in bandwidth_values:
                stats = self._simulate_vqarray(
                    model_name=model_name,
                    method_name=method_name,
                    sequence_length=sequence_length,
                    mem_width=int(mem_width),
                    adder_tree_size=int(study.extra.get("default_adder_tree_size", 128)),
                    model_registry=model_registry,
                    method_registry=method_registry,
                    config=config,
                )
                self._append_rows(
                    cycles_rows,
                    energy_rows,
                    power_rows,
                    stats,
                    method_registry.get(method_name),
                    model_name,
                    method_name,
                    sequence_length,
                    sweep_type="memory_bandwidth",
                    sweep_value=int(mem_width),
                    effective_mem_width=int(mem_width),
                )

            for adder_tree_size in adder_values:
                stats = self._simulate_vqarray(
                    model_name=model_name,
                    method_name=method_name,
                    sequence_length=sequence_length,
                    mem_width=int(study.extra.get("default_mem_width", 1024)),
                    adder_tree_size=int(adder_tree_size),
                    model_registry=model_registry,
                    method_registry=method_registry,
                    config=config,
                )
                self._append_rows(
                    cycles_rows,
                    energy_rows,
                    power_rows,
                    stats,
                    method_registry.get(method_name),
                    model_name,
                    method_name,
                    sequence_length,
                    sweep_type="adder_tree_size",
                    sweep_value=int(adder_tree_size),
                    effective_mem_width=int(study.extra.get("default_mem_width", 1024)),
                )
                area_rows.append(
                    {
                        "model": model_name,
                        "method": method_name,
                        "sequence_length": sequence_length,
                        "adder_tree_size": int(adder_tree_size),
                        "array_area_mm2": float(getattr(stats, "array_area_mm2", 0.0)),
                        "adder_tree_area_mm2": float(getattr(stats, "epilogue_area_mm2", 0.0)),
                        "sfu_area_mm2": float(getattr(stats, "sfu_area_mm2", 0.0)),
                        "total_area_mm2": float(
                            getattr(stats, "array_area_mm2", 0.0)
                            + getattr(stats, "epilogue_area_mm2", 0.0)
                            + getattr(stats, "sfu_area_mm2", 0.0)
                        ),
                    }
                )

        cycles_csv = write_rows(cycles_rows, output_dir / "cycles.csv", CYCLES_COLUMNS)
        energy_csv = write_rows(energy_rows, output_dir / "energy.csv", ENERGY_COLUMNS)
        power_csv = write_rows(power_rows, output_dir / "power.csv", POWER_COLUMNS)
        area_csv = write_rows(area_rows, output_dir / "area.csv", AREA_COLUMNS)
        legacy_reports = self._write_decode_reports(output_dir, cycles_rows, energy_rows, power_rows, area_rows)
        legacy_reports.update(self._write_paper_decode_reports(output_dir, cycles_rows, energy_rows, area_rows, study))
        verification_json = None

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            verification_json=verification_json,
            reports={"area_csv": area_csv, **legacy_reports},
        )

    def _run_prefill_bandwidth(
        self,
        output_dir: Path,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        model_name = (config.models or list(study.models))[0]
        sequence_length = int(study.extra.get("prefill_sequence_length", 1024))
        method_name = (config.methods or list(study.methods))[0]
        bandwidth_values = list(study.extra.get("memory_bandwidth_values", []))

        cycles_rows: list[dict] = []
        energy_rows: list[dict] = []
        power_rows: list[dict] = []

        for mem_width in bandwidth_values:
            stats = self._simulate_vqarray(
                model_name=model_name,
                method_name=method_name,
                sequence_length=sequence_length,
                mem_width=int(mem_width),
                adder_tree_size=int(study.extra.get("default_adder_tree_size", 128)),
                model_registry=model_registry,
                method_registry=method_registry,
                config=config,
            )
            self._append_rows(
                cycles_rows,
                energy_rows,
                power_rows,
                stats,
                method_registry.get(method_name),
                model_name,
                method_name,
                sequence_length,
                sweep_type="memory_bandwidth_prefill",
                sweep_value=int(mem_width),
                effective_mem_width=int(mem_width),
            )

        cycles_csv = write_rows(cycles_rows, output_dir / "cycles.csv", CYCLES_COLUMNS)
        energy_csv = write_rows(energy_rows, output_dir / "energy.csv", ENERGY_COLUMNS)
        power_csv = write_rows(power_rows, output_dir / "power.csv", POWER_COLUMNS)
        legacy_reports = self._write_prefill_reports(output_dir, cycles_rows, energy_rows, power_rows)
        legacy_reports.update(self._write_paper_prefill_reports(output_dir, cycles_rows, study))
        verification_json = None
        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            verification_json=verification_json,
            reports=legacy_reports,
        )

    def _simulate_vqarray(
        self,
        model_name: str,
        method_name: str,
        sequence_length: int,
        mem_width: int,
        adder_tree_size: int,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
        config: RunnerConfig,
    ) -> Stats:
        runner = method_registry.runner_for(method_name)
        bits = method_registry.resolve_quantization_bits(method_name)
        algorithm = method_registry.resolve_algorithm(method_name)
        operations = model_registry.build_operations(model_name, bits, algorithm, sequence_length=sequence_length)
        local_config = RunnerConfig(
            study=config.study,
            output_dir=config.output_dir,
            models=config.models,
            methods=config.methods,
            scenario_names=config.scenario_names,
            sequence_lengths=config.sequence_lengths,
            batch_sizes=config.batch_sizes,
            phase=config.phase,
            ops_mode=config.ops_mode,
            execution_mode=config.execution_mode,
            mem_width=mem_width,
            vq_array_height=config.vq_array_height,
            vq_array_width=config.vq_array_width,
            vq_adder_tree_size=adder_tree_size,
        )
        total_stats = Stats(name=method_name)
        for op in operations:
            if not isinstance(op, FC):
                continue
            total_stats.accumulate(runner.run_fc(op, local_config))
        total_stats.adder_tree_size = adder_tree_size
        runner.apply_energy_breakdown(total_stats)
        return total_stats

    @staticmethod
    def _append_rows(
        cycles_rows: list[dict],
        energy_rows: list[dict],
        power_rows: list[dict],
        stats: Stats,
        method_spec,
        model_name: str,
        method_name: str,
        sequence_length: int,
        sweep_type: str,
        sweep_value: int,
        effective_mem_width: int,
    ) -> None:
        dram_leak_energy = method_spec.energy["dram_leak_energy"] * (effective_mem_width / 1024.0)
        dynamic_dram_energy = (
            stats.reads["dram"] * method_spec.energy["dram_cost_read"]
            + stats.writes["dram"] * method_spec.energy["dram_cost_write"]
        ) / 1e9
        static_energy = (stats.total_cycles * dram_leak_energy) / 1e9
        total_energy = stats.core_energy + stats.buffer_energy + dynamic_dram_energy + static_energy
        total_time = stats.total_cycles / (500 * 10**6)
        compute_unit_energy = (
            float(getattr(stats, "array_power_w", 0.0) + getattr(stats, "epilogue_power_w", 0.0) + getattr(stats, "sfu_power_w", 0.0))
            * total_time
        )
        dram_power = (dynamic_dram_energy + static_energy) / total_time if stats.total_cycles > 0 else 0.0
        total_power = stats.core_power + stats.sram_power + dram_power
        total_mem_access_bits = stats.reads["dram"] + stats.writes["dram"]
        mem_bandwidth_utilization = (
            total_mem_access_bits / (effective_mem_width * stats.total_cycles) if "bandwidth" in sweep_type and stats.total_cycles > 0 else 0.0
        )

        cycles_rows.append(
            {
                "model": model_name,
                "method": method_name,
                "sequence_length": sequence_length,
                "sweep_type": sweep_type,
                "sweep_value": sweep_value,
                "total_cycles": stats.total_cycles,
                "compute_cycles": stats.compute_cycles,
                "mem_stall_cycles": stats.mem_stall_cycles,
                "total_time_s": total_time,
                "mem_bandwidth_utilization": mem_bandwidth_utilization,
            }
        )
        energy_rows.append(
            {
                "model": model_name,
                "method": method_name,
                "sequence_length": sequence_length,
                "sweep_type": sweep_type,
                "sweep_value": sweep_value,
                "total_energy_j": total_energy,
                "core_energy_j": stats.core_energy,
                "buffer_energy_j": stats.buffer_energy,
                "dram_energy_j": dynamic_dram_energy,
                "static_energy_j": static_energy,
                "array_energy_j": float(getattr(stats, "array_power_w", 0.0)) * total_time,
                "epilogue_energy_j": float(getattr(stats, "epilogue_power_w", 0.0)) * total_time,
                "sfu_energy_j": float(getattr(stats, "sfu_power_w", 0.0)) * total_time,
                "compute_unit_energy_j": compute_unit_energy,
            }
        )
        power_rows.append(
            {
                "model": model_name,
                "method": method_name,
                "sequence_length": sequence_length,
                "sweep_type": sweep_type,
                "sweep_value": sweep_value,
                "total_power_w": total_power,
                "core_power_w": stats.core_power,
                "sram_power_w": stats.sram_power,
                "dram_power_w": dram_power,
                "array_power_w": float(getattr(stats, "array_power_w", 0.0)),
                "epilogue_power_w": float(getattr(stats, "epilogue_power_w", 0.0)),
                "sfu_power_w": float(getattr(stats, "sfu_power_w", 0.0)),
            }
        )

    @staticmethod
    def _write_decode_reports(
        output_dir: Path,
        cycles_rows: list[dict],
        energy_rows: list[dict],
        power_rows: list[dict],
        area_rows: list[dict],
    ) -> dict[str, Path]:
        reports: dict[str, Path] = {}
        cycles_df = pd.DataFrame(cycles_rows)
        energy_df = pd.DataFrame(energy_rows)
        power_df = pd.DataFrame(power_rows)
        area_df = pd.DataFrame(area_rows)

        memory_dir = output_dir / "memory_bandwidth"
        bandwidth_cycles = cycles_df[cycles_df["sweep_type"] == "memory_bandwidth"][
            ["model", "method", "sequence_length", "sweep_value", "total_cycles", "compute_cycles", "mem_stall_cycles", "total_time_s", "mem_bandwidth_utilization"]
        ].rename(columns={"sweep_value": "mem_width"})
        bandwidth_energy = energy_df[energy_df["sweep_type"] == "memory_bandwidth"][
            ["model", "method", "sequence_length", "sweep_value", "total_energy_j", "core_energy_j", "buffer_energy_j", "dram_energy_j", "static_energy_j"]
        ].rename(columns={"sweep_value": "mem_width"})
        bandwidth_power = power_df[power_df["sweep_type"] == "memory_bandwidth"][
            ["model", "method", "sequence_length", "sweep_value", "total_power_w", "core_power_w", "sram_power_w", "dram_power_w"]
        ].rename(columns={"sweep_value": "mem_width"})
        memory_dir.mkdir(parents=True, exist_ok=True)
        bandwidth_cycles_path = memory_dir / "dse_cycles.csv"
        bandwidth_energy_path = memory_dir / "dse_energy.csv"
        bandwidth_power_path = memory_dir / "dse_power.csv"
        bandwidth_cycles.to_csv(bandwidth_cycles_path, index=False)
        bandwidth_energy.to_csv(bandwidth_energy_path, index=False)
        bandwidth_power.to_csv(bandwidth_power_path, index=False)
        reports["memory_bandwidth_cycles_csv"] = bandwidth_cycles_path
        reports["memory_bandwidth_energy_csv"] = bandwidth_energy_path
        reports["memory_bandwidth_power_csv"] = bandwidth_power_path

        adder_dir = output_dir / "adder_tree_size"
        adder_cycles = cycles_df[cycles_df["sweep_type"] == "adder_tree_size"][
            ["model", "method", "sequence_length", "sweep_value", "total_cycles", "compute_cycles", "mem_stall_cycles", "total_time_s"]
        ].rename(columns={"sweep_value": "adder_tree_size"})
        adder_energy = energy_df[energy_df["sweep_type"] == "adder_tree_size"][
            [
                "model",
                "method",
                "sequence_length",
                "sweep_value",
                "total_energy_j",
                "core_energy_j",
                "buffer_energy_j",
                "dram_energy_j",
                "static_energy_j",
                "array_energy_j",
                "epilogue_energy_j",
                "sfu_energy_j",
                "compute_unit_energy_j",
            ]
        ].rename(columns={"sweep_value": "adder_tree_size", "epilogue_energy_j": "adder_tree_energy_j"})
        adder_power = power_df[power_df["sweep_type"] == "adder_tree_size"][
            ["model", "method", "sequence_length", "sweep_value", "array_power_w", "epilogue_power_w", "sfu_power_w", "total_power_w"]
        ].rename(columns={"sweep_value": "adder_tree_size", "epilogue_power_w": "adder_tree_power_w"})
        adder_area = area_df.rename(columns={"adder_tree_area_mm2": "adder_tree_area_mm2"})
        adder_dir.mkdir(parents=True, exist_ok=True)
        adder_cycles_path = adder_dir / "dse_cycles_ep.csv"
        adder_energy_path = adder_dir / "dse_energy_ep.csv"
        adder_power_path = adder_dir / "dse_power.csv"
        adder_area_path = adder_dir / "dse_area.csv"
        adder_cycles.to_csv(adder_cycles_path, index=False)
        adder_energy.to_csv(adder_energy_path, index=False)
        adder_power.to_csv(adder_power_path, index=False)
        adder_area.to_csv(adder_area_path, index=False)
        reports["adder_tree_cycles_csv"] = adder_cycles_path
        reports["adder_tree_energy_csv"] = adder_energy_path
        reports["adder_tree_power_csv"] = adder_power_path
        reports["adder_tree_area_csv"] = adder_area_path
        return reports

    @staticmethod
    def _write_prefill_reports(
        output_dir: Path,
        cycles_rows: list[dict],
        energy_rows: list[dict],
        power_rows: list[dict],
    ) -> dict[str, Path]:
        reports: dict[str, Path] = {}
        cycles_df = pd.DataFrame(cycles_rows)
        energy_df = pd.DataFrame(energy_rows)
        power_df = pd.DataFrame(power_rows)
        prefill_dir = output_dir / "memory_bandwidth_prefill"
        prefill_dir.mkdir(parents=True, exist_ok=True)

        prefill_cycles = cycles_df[
            ["model", "method", "sequence_length", "sweep_value", "total_cycles", "compute_cycles", "mem_stall_cycles", "total_time_s", "mem_bandwidth_utilization"]
        ].rename(columns={"sweep_value": "mem_width"})
        prefill_energy = energy_df[
            ["model", "method", "sequence_length", "sweep_value", "total_energy_j", "core_energy_j", "buffer_energy_j", "dram_energy_j", "static_energy_j"]
        ].rename(columns={"sweep_value": "mem_width"})
        prefill_power = power_df[
            ["model", "method", "sequence_length", "sweep_value", "total_power_w", "core_power_w", "sram_power_w", "dram_power_w"]
        ].rename(columns={"sweep_value": "mem_width"})

        cycles_path = prefill_dir / "dse_cycles_prefill.csv"
        energy_path = prefill_dir / "dse_energy_prefill.csv"
        power_path = prefill_dir / "dse_power_prefill.csv"
        prefill_cycles.to_csv(cycles_path, index=False)
        prefill_energy.to_csv(energy_path, index=False)
        prefill_power.to_csv(power_path, index=False)
        reports["prefill_cycles_csv"] = cycles_path
        reports["prefill_energy_csv"] = energy_path
        reports["prefill_power_csv"] = power_path
        return reports

    @staticmethod
    def _write_paper_decode_reports(
        output_dir: Path,
        cycles_rows: list[dict],
        energy_rows: list[dict],
        area_rows: list[dict],
        study: StudySpec,
    ) -> dict[str, Path]:
        reports: dict[str, Path] = {}
        figure_method = str(study.extra.get("paper_figure_method", "vqarray_2_decode"))
        figure_label = str(study.extra.get("paper_figure_label", "EVA-W2"))

        cycles_df = pd.DataFrame(cycles_rows)
        energy_df = pd.DataFrame(energy_rows)
        area_df = pd.DataFrame(area_rows)

        adder_cycles = cycles_df[
            (cycles_df["sweep_type"] == "adder_tree_size") & (cycles_df["method"] == figure_method)
        ].copy()
        adder_energy = energy_df[
            (energy_df["sweep_type"] == "adder_tree_size") & (energy_df["method"] == figure_method)
        ].copy()
        adder_area = area_df[area_df["method"] == figure_method].copy()
        if not adder_cycles.empty and not adder_energy.empty and not adder_area.empty:
            baseline_row = adder_cycles[adder_cycles["sweep_value"] == int(study.extra.get("default_adder_tree_size", 128))]
            if baseline_row.empty:
                baseline_row = adder_cycles.sort_values("sweep_value").iloc[[0]]
            baseline_time = float(baseline_row["total_time_s"].iloc[0])

            num_eu = (adder_cycles["sweep_value"] / 32.0).astype(int)
            fig8_adder = (
                adder_cycles[["sweep_value", "total_time_s"]]
                .rename(columns={"sweep_value": "adder_tree_size"})
                .assign(num_eu=num_eu, norm_latency=adder_cycles["total_time_s"] / baseline_time, paper_label=figure_label)
                .merge(
                    adder_energy[
                        [
                            "sweep_value",
                            "array_energy_j",
                            "epilogue_energy_j",
                            "sfu_energy_j",
                            "compute_unit_energy_j",
                        ]
                    ].rename(columns={"sweep_value": "adder_tree_size"}),
                    on="adder_tree_size",
                    how="left",
                )
                .merge(
                    adder_area[
                        [
                            "adder_tree_size",
                            "array_area_mm2",
                            "adder_tree_area_mm2",
                            "sfu_area_mm2",
                            "total_area_mm2",
                        ]
                    ],
                    on="adder_tree_size",
                    how="left",
                )
                .sort_values("num_eu")
                .reset_index(drop=True)
            )
            fig8_adder_path = output_dir / "fig8_num_eu.csv"
            fig8_adder.to_csv(fig8_adder_path, index=False)
            reports["fig8_num_eu_csv"] = fig8_adder_path

        bandwidth = cycles_df[
            (cycles_df["sweep_type"] == "memory_bandwidth") & (cycles_df["method"] == figure_method)
        ].copy()
        if not bandwidth.empty:
            baseline_row = bandwidth[bandwidth["sweep_value"] == int(study.extra.get("memory_bandwidth_values", [256])[0])]
            if baseline_row.empty:
                baseline_row = bandwidth.sort_values("sweep_value").iloc[[0]]
            baseline_time = float(baseline_row["total_time_s"].iloc[0])
            baseline_cycles = float(baseline_row["total_cycles"].iloc[0])
            fig8_decode_bw = bandwidth.rename(columns={"sweep_value": "mem_width"}).copy()
            fig8_decode_bw["memory_bandwidth_gbs"] = fig8_decode_bw["mem_width"] / 16.0
            fig8_decode_bw["compute_latency_norm"] = fig8_decode_bw["compute_cycles"] / baseline_cycles
            fig8_decode_bw["mem_stall_latency_norm"] = fig8_decode_bw["mem_stall_cycles"] / baseline_cycles
            fig8_decode_bw["norm_latency"] = fig8_decode_bw["total_time_s"] / baseline_time
            fig8_decode_bw["paper_label"] = figure_label
            fig8_decode_bw = fig8_decode_bw[
                [
                    "method",
                    "paper_label",
                    "mem_width",
                    "memory_bandwidth_gbs",
                    "total_cycles",
                    "compute_cycles",
                    "mem_stall_cycles",
                    "total_time_s",
                    "compute_latency_norm",
                    "mem_stall_latency_norm",
                    "norm_latency",
                    "mem_bandwidth_utilization",
                ]
            ]
            fig8_decode_bw = fig8_decode_bw[fig8_decode_bw["memory_bandwidth_gbs"] <= 112.0].sort_values(
                "memory_bandwidth_gbs"
            ).reset_index(drop=True)
            fig8_decode_bw_path = output_dir / "fig8_memory_bandwidth_decode.csv"
            fig8_decode_bw.to_csv(fig8_decode_bw_path, index=False)
            reports["fig8_memory_bandwidth_decode_csv"] = fig8_decode_bw_path
        return reports

    @staticmethod
    def _write_paper_prefill_reports(output_dir: Path, cycles_rows: list[dict], study: StudySpec) -> dict[str, Path]:
        reports: dict[str, Path] = {}
        cycles_df = pd.DataFrame(cycles_rows)
        if cycles_df.empty:
            return reports
        figure_method = str(study.extra.get("paper_figure_method", "vqarray_2_prefill"))
        bandwidth = cycles_df[cycles_df["method"] == figure_method].copy()
        if bandwidth.empty:
            return reports
        baseline_row = bandwidth[bandwidth["sweep_value"] == int(study.extra.get("memory_bandwidth_values", [256])[0])]
        if baseline_row.empty:
            baseline_row = bandwidth.sort_values("sweep_value").iloc[[0]]
        baseline_time = float(baseline_row["total_time_s"].iloc[0])
        fig8_prefill = bandwidth.rename(columns={"sweep_value": "mem_width"}).copy()
        fig8_prefill["memory_bandwidth_gbs"] = fig8_prefill["mem_width"] / 16.0
        fig8_prefill["norm_latency"] = fig8_prefill["total_time_s"] / baseline_time
        fig8_prefill = fig8_prefill[
            [
                "method",
                "mem_width",
                "memory_bandwidth_gbs",
                "total_cycles",
                "compute_cycles",
                "mem_stall_cycles",
                "total_time_s",
                "norm_latency",
                "mem_bandwidth_utilization",
            ]
        ]
        fig8_prefill = fig8_prefill[fig8_prefill["memory_bandwidth_gbs"] <= 112.0].sort_values(
            "memory_bandwidth_gbs"
        ).reset_index(drop=True)
        fig8_prefill_path = output_dir / "fig8_memory_bandwidth_prefill.csv"
        fig8_prefill.to_csv(fig8_prefill_path, index=False)
        reports["fig8_memory_bandwidth_prefill_csv"] = fig8_prefill_path
        return reports
