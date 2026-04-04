from pathlib import Path

from simulator.config import CONFIG_ROOT, load_yaml_directory
from simulator.ops import Attention, FC, SFU
from simulator.specs import ModelSpec, OperationSpec


class ModelRegistry:
    def __init__(self, config_root: Path | None = None) -> None:
        model_root = (config_root or CONFIG_ROOT) / "models"
        raw_models = load_yaml_directory(model_root, "models")
        self._models = {name: self._build_spec(name, payload) for name, payload in raw_models.items()}

    def names(self) -> list[str]:
        return sorted(self._models)

    def get(self, model_name: str) -> ModelSpec:
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")
        return self._models[model_name]

    def build_operations(
        self,
        model_name: str,
        num_bits: int,
        algorithm: str,
        sequence_length: int = 1,
    ) -> list[object]:
        model = self.get(model_name)
        operations: list[object] = []
        for op in model.ops:
            if op.type == "fc":
                operations.append(
                    FC(
                        op.name,
                        op.output_dim or 0,
                        op.input_dim or 0,
                        sequence_length,
                        num_bits=num_bits,
                        algorithm=algorithm,
                        is_expert=op.is_expert,
                        num_experts=model.num_experts,
                        top_k=model.num_experts_per_tok,
                        is_shared_expert=op.is_shared_expert,
                    )
                )
            elif op.type == "attention":
                operations.append(
                    Attention(
                        op.name,
                        op.dim or 0,
                        sequence_length,
                        num_bits,
                        num_kv_heads=op.num_kv_heads or model.num_kv_heads,
                        num_attention_heads=op.num_attention_heads or model.num_attention_heads,
                        head_dim=op.head_dim or model.head_dim,
                    )
                )
            elif op.type == "sfu":
                sfu = SFU(op.name, sequence_length, op.dim or 0, op.op_type or "rmsnorm", nbits=op.nbits)
                sfu.is_expert = op.is_expert
                sfu.num_experts = model.num_experts
                sfu.top_k = model.num_experts_per_tok
                operations.append(sfu)
            else:
                raise ValueError(f"Unsupported op type: {op.type}")
        return operations

    def _build_spec(self, name: str, payload: dict) -> ModelSpec:
        ops = tuple(OperationSpec(**entry) for entry in payload.get("ops", []))
        return ModelSpec(
            name=name,
            display_name=payload["display_name"],
            family=payload["family"],
            kind=payload["kind"],
            num_layers=payload["num_layers"],
            hidden_size=payload["hidden_size"],
            num_attention_heads=payload["num_attention_heads"],
            num_kv_heads=payload["num_kv_heads"],
            head_dim=payload["head_dim"],
            num_experts=payload.get("num_experts", 1),
            num_experts_per_tok=payload.get("num_experts_per_tok", 1),
            shared_expert=payload.get("shared_expert", False),
            shared_expert_intermediate_size=payload.get("shared_expert_intermediate_size", 0),
            activation=payload.get("activation", "silu"),
            ops=ops,
        )
