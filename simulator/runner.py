from pathlib import Path

from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.pipelines import (
    AblationPipeline,
    BatchScalingPipeline,
    DatasetStatsPipeline,
    DsePipeline,
    EndToEndPipeline,
    Fig9Pipeline,
    GptvqValidationPipeline,
    HardwareCharacterizationPipeline,
    IndexAnalysisPipeline,
)
from simulator.specs import RunnerConfig, StudyArtifacts
from simulator.studies import StudyRegistry


def run(config: RunnerConfig) -> StudyArtifacts:
    study_registry = StudyRegistry()
    model_registry = ModelRegistry()
    method_registry = MethodRegistry()

    study = study_registry.get(config.study)
    if config.phase is None:
        config.phase = study.phase
    if config.ops_mode is None:
        config.ops_mode = study.ops_mode

    if config.study == "fig9_fc":
        return Fig9Pipeline().run(config, study, model_registry, method_registry)

    if config.study == "fig8_dse":
        return DsePipeline().run(config, study, model_registry, method_registry)

    if config.study == "fig10_hw":
        return HardwareCharacterizationPipeline().run(config, study, model_registry, method_registry)

    if config.study == "fig11_batch":
        return BatchScalingPipeline().run(config, study, model_registry, method_registry)

    if config.study == "e2e":
        return EndToEndPipeline().run(config, study, model_registry, method_registry)

    if config.study == "fig14_index":
        return IndexAnalysisPipeline().run(config, study, model_registry, method_registry)

    if config.study == "table_vii_abl":
        return AblationPipeline().run(config, study, model_registry, method_registry)

    if config.study == "table_viii_data":
        return DatasetStatsPipeline().run(config, study, model_registry, method_registry)

    if config.study == "table_ix_vq":
        return GptvqValidationPipeline().run(config, study, model_registry, method_registry)

    raise ValueError(f"Unsupported study: {config.study}")


def parse_csv_list(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_int_csv_list(raw_value: str | None) -> list[int] | None:
    if not raw_value:
        return None
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def build_runner_config(
    study: str,
    output_dir: str,
    models: str | None = None,
    methods: str | None = None,
    scenario_names: str | None = None,
    sequence_lengths: str | None = None,
    batch_sizes: str | None = None,
    phase: str | None = None,
    ops_mode: str | None = None,
    execution_mode: str | None = None,
    mem_width: int = 1024,
    vq_array_height: int = 32,
    vq_array_width: int = 8,
    vq_adder_tree_size: int | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        study=study,
        output_dir=Path(output_dir),
        models=parse_csv_list(models),
        methods=parse_csv_list(methods),
        scenario_names=parse_csv_list(scenario_names),
        sequence_lengths=parse_int_csv_list(sequence_lengths),
        batch_sizes=parse_int_csv_list(batch_sizes),
        phase=phase,
        ops_mode=ops_mode,
        execution_mode=execution_mode,
        mem_width=mem_width,
        vq_array_height=vq_array_height,
        vq_array_width=vq_array_width,
        vq_adder_tree_size=vq_adder_tree_size,
    )
