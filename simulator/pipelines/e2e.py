import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from simulator.config import resolve_repo_path
from simulator.datasets import get_dataset_lengths, load_trace_json_files, traces_from_lengths
from simulator.end_to_end_ops import METHOD_COMPONENTS, EndToEndExecutor
from simulator.io.results import write_rows
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.specs import RunnerConfig, StudyArtifacts, StudySpec


MODEL_TO_TOKENIZER = {
    "llama_2_7b": "meta-llama/Llama-2-7b-hf",
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "qwen3_30b_a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

CYCLES_COLUMNS = ["scenario", "stage", "model", "dataset", "method", "phase", "component", "avg_cycles", "avg_time_s"]
ENERGY_COLUMNS = ["scenario", "stage", "model", "dataset", "method", "phase", "component", "avg_energy_j"]
POWER_COLUMNS = ["scenario", "stage", "model", "dataset", "method", "phase", "component", "avg_power_w"]
END_TO_END_METHOD_LABELS = {
    "systolic_array": "SA",
    "ant": "ANT",
    "figna": "FIGNA",
    "figlut_4": "FIGLUT",
    "vqarray_4_decode": "EVA-W4",
    "vqarray_3_decode": "EVA-W3",
    "vqarray_2_decode": "EVA-W2",
    "vqarray_2_gptvq_decode": "GPT-W2*",
    "vqarray_2_decode_kvq": "EVA-KVQ",
}


class EndToEndPipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        execution_mode = str(config.execution_mode or study.extra.get("default_execution_mode", "full"))
        output_dir = config.output_dir / study.output_subdir / execution_mode
        output_dir.mkdir(parents=True, exist_ok=True)

        executor = EndToEndExecutor(model_registry, method_registry, config)
        scenario_summaries: list[pd.DataFrame] = []
        cycles_rows: list[dict] = []
        energy_rows: list[dict] = []
        power_rows: list[dict] = []
        reports: dict[str, Path] = {}
        scenario_results: dict[str, pd.DataFrame] = {}
        scenario_contexts: dict[str, dict[str, object]] = {}

        for scenario in self._select_scenarios(study.extra.get("scenarios", []), config.scenario_names):
            scenario_name = str(scenario["name"])
            scenario_dir = output_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_stage = str(scenario.get("moe_stage", "stage1"))
            scenario_ops_mode = str(scenario.get("ops_mode", study.ops_mode))
            scenario_methods = list(config.methods or self._scenario_value(scenario, "methods", execution_mode, study.methods))
            scenario_models = list(config.models or self._scenario_value(scenario, "models", execution_mode, study.models))
            trace_sources = self._resolve_trace_sources(scenario, scenario_models, execution_mode)

            scenario_rows: list[dict] = []
            input_contexts: list[dict[str, object]] = []
            for model_name in scenario_models:
                for dataset_name, traces in trace_sources[model_name]:
                    input_contexts.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "trace_count": len(traces),
                            "traces": traces,
                        }
                    )
                    rows = executor.simulate_trace(
                        model_name=model_name,
                        method_names=scenario_methods,
                        traces=traces,
                        ops_mode=scenario_ops_mode,
                        moe_stage=scenario_stage,
                    )
                    for row in rows:
                        row["scenario"] = scenario_name
                        row["stage"] = scenario_stage
                        row["model"] = model_name
                        row["dataset"] = dataset_name
                    scenario_rows.extend(rows)

                    trace_csv = scenario_dir / f"trace_results_{model_name}_{dataset_name}.csv"
                    pd.DataFrame(rows).to_csv(trace_csv, index=False)
                    reports[f"{scenario_name}_{model_name}_{dataset_name}_trace_csv"] = trace_csv

            summary_df = summarize_trace_rows(
                pd.DataFrame(scenario_rows),
                scenario_methods,
                scenario_name,
                scenario_stage,
            )
            summary_csv = scenario_dir / "summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            reports[f"{scenario_name}_summary_csv"] = summary_csv
            scenario_summaries.append(summary_df)
            scenario_results[scenario_name] = summary_df
            scenario_contexts[scenario_name] = {
                "scenario": scenario,
                "execution_mode": execution_mode,
                "ops_mode": scenario_ops_mode,
                "stage": scenario_stage,
                "methods": scenario_methods,
                "inputs": input_contexts,
            }

            cycles_rows.extend(summary_to_long_cycles(summary_df))
            energy_rows.extend(summary_to_long_energy(summary_df))
            power_rows.extend(summary_to_long_power(summary_df))

        cycles_csv = write_rows(cycles_rows, output_dir / "cycles.csv", CYCLES_COLUMNS)
        energy_csv = write_rows(energy_rows, output_dir / "energy.csv", ENERGY_COLUMNS)
        power_csv = write_rows(power_rows, output_dir / "power.csv", POWER_COLUMNS)

        summary_df = pd.concat(scenario_summaries, ignore_index=True) if scenario_summaries else pd.DataFrame()
        summary_csv = output_dir / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        reports["summary_csv"] = summary_csv
        reports.update(self._write_paper_reports(output_dir, scenario_results))

        manifest_json = self._write_manifest(output_dir, execution_mode, scenario_contexts)
        reports["manifest_json"] = manifest_json

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=energy_csv,
            power_csv=power_csv,
            verification_json=None,
            reports=reports,
        )

    @staticmethod
    def _resolve_trace_sources(
        scenario: dict,
        model_names: list[str],
        execution_mode: str,
    ) -> dict[str, list[tuple[str, list[dict[str, int | float]]]]]:
        per_model: dict[str, list[tuple[str, list[dict[str, int | float]]]]] = defaultdict(list)
        if "trace_files" in scenario:
            dataset_name = str(scenario.get("dataset", scenario["name"]))
            traces = load_trace_json_files([resolve_repo_path(path) for path in scenario["trace_files"]])
            trace_limit = EndToEndPipeline._scenario_value(scenario, "trace_limit", execution_mode, None)
            if trace_limit is not None:
                traces = traces[: int(trace_limit)]
            for model_name in model_names:
                per_model[model_name].append((dataset_name, traces))
            return per_model

        datasets = list(EndToEndPipeline._scenario_value(scenario, "datasets", execution_mode, []))
        num_samples = int(EndToEndPipeline._scenario_value(scenario, "num_samples", execution_mode, 100))
        seed = int(scenario.get("dataset_seed", 42))
        target_rps = float(scenario.get("target_rps", 1.0))

        for model_name in model_names:
            tokenizer_name = str(scenario.get("tokenizer", MODEL_TO_TOKENIZER[model_name]))
            for dataset_name in datasets:
                lengths = get_dataset_lengths(dataset_name, tokenizer_name, num_samples, seed=seed)
                per_model[model_name].append((dataset_name, traces_from_lengths(lengths, target_rps=target_rps, seed=seed)))
        return per_model

    @staticmethod
    def _scenario_value(scenario: dict, key: str, execution_mode: str, default):
        sample_key = f"sample_{key}"
        if execution_mode == "sample" and sample_key in scenario:
            return scenario[sample_key]
        return scenario.get(key, default)

    @staticmethod
    def _select_scenarios(scenarios: list[dict], selected_names: list[str] | None) -> list[dict]:
        if not selected_names:
            return list(scenarios)
        selected = [scenario for scenario in scenarios if str(scenario.get("name")) in selected_names]
        missing = sorted(set(selected_names) - {str(scenario.get("name")) for scenario in selected})
        if missing:
            raise ValueError(f"Unknown end-to-end scenarios: {', '.join(missing)}")
        return selected

    @staticmethod
    def _write_manifest(
        output_dir: Path,
        execution_mode: str,
        scenario_contexts: dict[str, dict[str, object]],
    ) -> Path:
        payload = {
            "execution_mode": execution_mode,
            "scenarios": {
                scenario_name: {
                    "ops_mode": context["ops_mode"],
                    "stage": context["stage"],
                    "methods": context["methods"],
                    "trace_inputs": [
                        {
                            "model": item["model"],
                            "dataset": item["dataset"],
                            "trace_count": item["trace_count"],
                        }
                        for item in context["inputs"]
                    ],
                }
                for scenario_name, context in scenario_contexts.items()
            },
        }
        manifest_json = output_dir / "manifest.json"
        manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_json

    @staticmethod
    def _write_paper_reports(output_dir: Path, scenario_results: dict[str, pd.DataFrame]) -> dict[str, Path]:
        reports: dict[str, Path] = {}
        combined_frames: list[pd.DataFrame] = []
        report_specs = {
            "fig12_llama2": "fig12_llama2.csv",
            "fig13_moe": "fig13_moe.csv",
        }
        for scenario_name, filename in report_specs.items():
            summary_df = scenario_results.get(scenario_name)
            if summary_df is None or summary_df.empty:
                continue
            export_df = EndToEndPipeline._prepare_paper_summary(summary_df)
            path = output_dir / filename
            export_df.to_csv(path, index=False)
            reports[filename.replace('.csv', '_csv')] = path
            combined_frames.append(export_df.assign(report_name=filename))
        if combined_frames:
            combined_path = output_dir / "paper_e2e_summary.csv"
            pd.concat(combined_frames, ignore_index=True).to_csv(combined_path, index=False)
            reports["paper_e2e_summary_csv"] = combined_path
        return reports

    @staticmethod
    def _prepare_paper_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for (_, model_name, dataset_name), subset in summary_df.groupby(["scenario", "model", "dataset"]):
            baseline = float(subset[subset["method"] == "systolic_array"]["total_time_s"].iloc[0])
            export = subset.copy()
            export["paper_label"] = export["method"].map(END_TO_END_METHOD_LABELS).fillna(export["method"])
            export["speedup_vs_sa"] = baseline / export["total_time_s"]
            rows.append(export)
        combined = pd.concat(rows, ignore_index=True) if rows else summary_df.copy()
        ordered_columns = [
            "scenario",
            "stage",
            "model",
            "dataset",
            "method",
            "paper_label",
            "prefill_fc_s",
            "prefill_attention_s",
            "prefill_sfu_s",
            "prefill_total_s",
            "decode_fc_s",
            "decode_attention_s",
            "decode_sfu_s",
            "decode_total_s",
            "total_time_s",
            "speedup_vs_sa",
        ]
        existing_columns = [column for column in ordered_columns if column in combined.columns]
        return combined[existing_columns].sort_values(["scenario", "model", "dataset", "method"]).reset_index(drop=True)


def summarize_trace_rows(dataframe: pd.DataFrame, methods: list[str], scenario_name: str, scenario_stage: str) -> pd.DataFrame:
    rows: list[dict] = []
    if dataframe.empty:
        return pd.DataFrame(rows)

    group_columns = ["model", "dataset"]
    for (model_name, dataset_name), subset in dataframe.groupby(group_columns):
        for method_name in methods:
            row = {
                "scenario": scenario_name,
                "stage": scenario_stage,
                "model": model_name,
                "dataset": dataset_name,
                "method": method_name,
            }
            for phase in ("prefill", "decode"):
                phase_total = subset[f"{phase}_{method_name}_time_s"].mean()
                row[f"{phase}_total_s"] = float(phase_total)
                row[f"{phase}_total_cycles"] = float(subset[f"{phase}_{method_name}_cycles"].mean())
                row[f"{phase}_total_energy_j"] = float(subset[f"{phase}_{method_name}_energy_j"].mean())
                row[f"{phase}_total_power_w"] = float(subset[f"{phase}_{method_name}_power_w"].mean())
                for component in METHOD_COMPONENTS:
                    row[f"{phase}_{component}_s"] = float(subset[f"{phase}_{method_name}_{component}_time_s"].mean())
                    row[f"{phase}_{component}_cycles"] = float(subset[f"{phase}_{method_name}_{component}_cycles"].mean())
                    row[f"{phase}_{component}_energy_j"] = float(subset[f"{phase}_{method_name}_{component}_energy_j"].mean())
            row["total_time_s"] = row["prefill_total_s"] + row["decode_total_s"]
            rows.append(row)
    return pd.DataFrame(rows)


def summary_to_long_cycles(summary_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in summary_df.iterrows():
        for phase in ("prefill", "decode"):
            rows.append(
                {
                    "scenario": row["scenario"],
                    "stage": row["stage"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "phase": phase,
                    "component": "total",
                    "avg_cycles": row[f"{phase}_total_cycles"],
                    "avg_time_s": row[f"{phase}_total_s"],
                }
            )
            for component in METHOD_COMPONENTS:
                rows.append(
                    {
                        "scenario": row["scenario"],
                        "stage": row["stage"],
                        "model": row["model"],
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "phase": phase,
                        "component": component,
                        "avg_cycles": row[f"{phase}_{component}_cycles"],
                        "avg_time_s": row[f"{phase}_{component}_s"],
                    }
                )
    return rows


def summary_to_long_energy(summary_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in summary_df.iterrows():
        for phase in ("prefill", "decode"):
            rows.append(
                {
                    "scenario": row["scenario"],
                    "stage": row["stage"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "phase": phase,
                    "component": "total",
                    "avg_energy_j": row[f"{phase}_total_energy_j"],
                }
            )
            for component in METHOD_COMPONENTS:
                rows.append(
                    {
                        "scenario": row["scenario"],
                        "stage": row["stage"],
                        "model": row["model"],
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "phase": phase,
                        "component": component,
                        "avg_energy_j": row[f"{phase}_{component}_energy_j"],
                    }
                )
    return rows


def summary_to_long_power(summary_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in summary_df.iterrows():
        for phase in ("prefill", "decode"):
            rows.append(
                {
                    "scenario": row["scenario"],
                    "stage": row["stage"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "phase": phase,
                    "component": "total",
                    "avg_power_w": row[f"{phase}_total_power_w"],
                }
            )
    return rows
