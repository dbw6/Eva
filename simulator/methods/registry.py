from pathlib import Path

from simulator.config import CONFIG_ROOT, load_yaml_directory
from simulator.methods.dense import DenseArrayMethod
from simulator.methods.figlut import FiglutMethod
from simulator.methods.vqarray import VqArrayMethod
from simulator.specs import MethodSpec


class MethodRegistry:
    def __init__(self, config_root: Path | None = None) -> None:
        method_root = (config_root or CONFIG_ROOT) / "methods"
        raw_methods = load_yaml_directory(method_root, "methods")
        self._specs = {name: self._build_spec(name, payload) for name, payload in raw_methods.items()}
        self._runners = {name: self._build_runner(spec) for name, spec in self._specs.items()}

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, method_name: str) -> MethodSpec:
        if method_name not in self._specs:
            raise ValueError(f"Unknown method: {method_name}")
        return self._specs[method_name]

    def runner_for(self, method_name: str):
        if method_name not in self._runners:
            raise ValueError(f"Unknown method: {method_name}")
        return self._runners[method_name]

    def resolve_quantization_bits(self, method_name: str) -> int:
        return self.get(method_name).quantization_bits

    def resolve_algorithm(self, method_name: str) -> str:
        return self.get(method_name).algorithm

    def _build_runner(self, spec: MethodSpec):
        if spec.family == "dense_array":
            return DenseArrayMethod(spec)
        if spec.family == "figlut":
            return FiglutMethod(spec)
        if spec.family in {"vqarray_decode", "vqarray_prefill", "vqarray_gptvq_decode"}:
            return VqArrayMethod(spec)
        raise ValueError(f"Unsupported method family: {spec.family}")

    def _build_spec(self, name: str, payload: dict) -> MethodSpec:
        return MethodSpec(
            name=name,
            family=payload["family"],
            display_name=payload["display_name"],
            quantization_bits=payload["quantization_bits"],
            algorithm=payload["algorithm"],
            bitwidths=payload.get("bitwidths", {}),
            array=payload.get("array", {}),
            tiles=payload.get("tiles", {}),
            buffers=payload.get("buffers", {}),
            extra=payload.get("extra", {}),
            energy=payload.get("energy", {}),
        )
