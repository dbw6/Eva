from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OperationSpec:
    type: str
    name: str
    input_dim: int | None = None
    output_dim: int | None = None
    dim: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    op_type: str | None = None
    nbits: int = 16
    is_expert: bool = False
    is_shared_expert: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    family: str
    kind: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int = 1
    num_experts_per_tok: int = 1
    shared_expert: bool = False
    shared_expert_intermediate_size: int = 0
    activation: str = "silu"
    ops: tuple[OperationSpec, ...] = ()


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    display_name: str
    quantization_bits: int
    algorithm: str
    bitwidths: dict[str, int]
    array: dict[str, int]
    tiles: dict[str, int]
    buffers: dict[str, int]
    extra: dict[str, int | float | bool | str]
    energy: dict[str, float]


@dataclass(frozen=True)
class StudySpec:
    name: str
    description: str
    phase: str
    ops_mode: str
    models: tuple[str, ...]
    methods: tuple[str, ...]
    sequence_lengths: tuple[int, ...]
    batch_sizes: tuple[int, ...] = (1,)
    aggregate_sequence1: bool = False
    output_subdir: str = ""
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    study: str
    output_dir: Path
    models: list[str] | None = None
    methods: list[str] | None = None
    scenario_names: list[str] | None = None
    sequence_lengths: list[int] | None = None
    batch_sizes: list[int] | None = None
    phase: str | None = None
    ops_mode: str | None = None
    execution_mode: str | None = None
    mem_width: int = 1024
    vq_array_height: int = 32
    vq_array_width: int = 8
    vq_adder_tree_size: int | None = None


@dataclass(frozen=True)
class StudyArtifacts:
    output_dir: Path
    cycles_csv: Path
    energy_csv: Path
    power_csv: Path
    aggregated_csv: Path | None = None
    verification_json: Path | None = None
    reports: dict[str, Path] = field(default_factory=dict)
