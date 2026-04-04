import hashlib
import math
from dataclasses import dataclass

import numpy as np

from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.ops import Attention, FC, SFU
from simulator.specs import RunnerConfig
from simulator.utils import Stats, ceil_a_by_b


FREQ_HZ = 500 * 10**6
METHOD_COMPONENTS = ("fc", "attention", "sfu")


@dataclass(frozen=True)
class RoutingBucket:
    tokens_per_expert: int
    expert_count: int


class Stage1Router:
    def buckets(self, sequence_length: int, top_k: int, num_experts: int, seed_key: str) -> list[RoutingBucket]:
        total_tasks = sequence_length * top_k
        if total_tasks == 0:
            return []
        if total_tasks < num_experts:
            return [RoutingBucket(tokens_per_expert=1, expert_count=total_tasks)]
        return [RoutingBucket(tokens_per_expert=int(math.ceil(total_tasks / num_experts)), expert_count=num_experts)]


class Stage2Router:
    def __init__(self, skewness: float = 0.2) -> None:
        self.skewness = skewness

    def buckets(self, sequence_length: int, top_k: int, num_experts: int, seed_key: str) -> list[RoutingBucket]:
        total_tasks = sequence_length * top_k
        if total_tasks == 0:
            return []

        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        base_logits = np.zeros(num_experts)
        if self.skewness > 0:
            base_logits = rng.normal(0.0, self.skewness * 3.0, num_experts)
        routing_scores = base_logits + rng.gumbel(size=(sequence_length, num_experts))
        top_k_indices = np.argpartition(routing_scores, -top_k, axis=1)[:, -top_k:]
        unique_experts, counts = np.unique(top_k_indices, return_counts=True)
        load_per_expert = np.zeros(num_experts, dtype=int)
        load_per_expert[unique_experts] = counts
        unique_loads, load_counts = np.unique(load_per_expert, return_counts=True)

        buckets = []
        for tokens_per_expert, expert_count in sorted(zip(unique_loads.tolist(), load_counts.tolist()), reverse=True):
            if tokens_per_expert <= 0:
                continue
            buckets.append(RoutingBucket(tokens_per_expert=tokens_per_expert, expert_count=expert_count))
        return buckets


class EndToEndExecutor:
    def __init__(
        self,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
        config: RunnerConfig,
    ) -> None:
        self.model_registry = model_registry
        self.method_registry = method_registry
        self.config = config
        self.stage1_router = Stage1Router()
        self.stage2_router = Stage2Router()

    def build_network(self, model_name: str, method_name: str, sequence_length: int) -> list[object]:
        bits = self.method_registry.resolve_quantization_bits(self._base_method_name(method_name))
        algorithm = self.method_registry.resolve_algorithm(self._base_method_name(method_name))
        return self.model_registry.build_operations(model_name, bits, algorithm, sequence_length=sequence_length)

    def simulate_trace(
        self,
        model_name: str,
        method_names: list[str],
        traces: list[dict[str, int | float]],
        ops_mode: str,
        moe_stage: str = "stage1",
    ) -> list[dict[str, int | float | str]]:
        prefill_cache: dict[tuple[str, int], dict[str, dict]] = {}
        decode_cache: dict[tuple[str, int], dict[str, dict]] = {}
        rows: list[dict[str, int | float | str]] = []

        for trace_idx, trace in enumerate(traces):
            input_length = int(trace["input_length"])
            output_length = int(trace["output_length"])
            timestamp = float(trace["timestamp"])
            kv_length = int((input_length + output_length) / 2)

            row: dict[str, int | float | str] = {
                "trace_idx": trace_idx,
                "timestamp": timestamp,
                "input_length": input_length,
                "output_length": output_length,
            }

            for method_name in method_names:
                prefill_key = (method_name, input_length)
                if prefill_key not in prefill_cache:
                    operations = self.build_network(model_name, method_name, input_length)
                    total, used_prefill, breakdown = self.run_network(
                        model_name,
                        method_name,
                        operations,
                        phase="prefill",
                        kv_length=input_length,
                        sequence_length=input_length,
                        ops_mode=ops_mode,
                        trace_index=trace_idx,
                        moe_stage=moe_stage,
                    )
                    self.apply_energy(method_name, total, used_prefill)
                    prefill_cache[prefill_key] = {
                        "total": extract_stats(total),
                        "breakdown": {key: extract_stats(value) for key, value in breakdown.items()},
                    }

                decode_key = (method_name, kv_length)
                if decode_key not in decode_cache:
                    operations = self.build_network(model_name, method_name, 1)
                    total, used_prefill, breakdown = self.run_network(
                        model_name,
                        method_name,
                        operations,
                        phase="decode",
                        kv_length=kv_length,
                        sequence_length=1,
                        ops_mode=ops_mode,
                        trace_index=trace_idx,
                        moe_stage=moe_stage,
                    )
                    self.apply_energy(method_name, total, used_prefill)
                    decode_cache[decode_key] = {
                        "total": extract_stats(total),
                        "breakdown": {key: extract_stats(value) for key, value in breakdown.items()},
                    }

                prefill_total = prefill_cache[prefill_key]["total"]
                prefill_breakdown = prefill_cache[prefill_key]["breakdown"]
                decode_total = decode_cache[decode_key]["total"]
                decode_breakdown = decode_cache[decode_key]["breakdown"]

                write_trace_metrics(row, method_name, "prefill", prefill_total, prefill_breakdown, scale=1)
                write_trace_metrics(row, method_name, "decode", decode_total, decode_breakdown, scale=output_length)

            rows.append(row)
        return rows

    def run_network(
        self,
        model_name: str,
        method_name: str,
        operations: list[object],
        phase: str,
        kv_length: int,
        sequence_length: int,
        ops_mode: str,
        trace_index: int,
        moe_stage: str,
    ) -> tuple[Stats, bool, dict[str, Stats]]:
        total = Stats(name=method_name)
        breakdown = {component: Stats(name=component) for component in METHOD_COMPONENTS}
        used_prefill = False

        for op in operations:
            if ops_mode == "fc_only" and not isinstance(op, FC):
                continue
            op_stats, op_used_prefill = self.run_op(
                model_name=model_name,
                method_name=method_name,
                op=op,
                phase=phase,
                kv_length=kv_length,
                sequence_length=sequence_length,
                trace_index=trace_index,
                moe_stage=moe_stage,
            )
            total.accumulate(op_stats)
            used_prefill = used_prefill or op_used_prefill
            component = "fc" if isinstance(op, FC) else "attention" if isinstance(op, Attention) else "sfu"
            breakdown[component].accumulate(op_stats)

        array_cycles = breakdown["fc"].total_cycles + breakdown["attention"].total_cycles
        sfu_cycles = breakdown["sfu"].total_cycles
        num_layers = max(1, sum(1 for op in operations if isinstance(op, Attention)))
        array_bottleneck = array_cycles > sfu_cycles
        breakdown["sfu"].total_cycles = (sfu_cycles / num_layers) if array_bottleneck else sfu_cycles
        array_scale = 1.0 if array_bottleneck else (1.0 / num_layers)
        breakdown["fc"].total_cycles *= array_scale
        breakdown["attention"].total_cycles *= array_scale
        total.total_cycles = (
            breakdown["fc"].total_cycles + breakdown["attention"].total_cycles + breakdown["sfu"].total_cycles
        )

        for component, stats in breakdown.items():
            if stats.total_cycles > 0:
                self.apply_energy(method_name, stats, used_prefill)
            else:
                stats.core_power = 0.0
                stats.sram_power = 0.0
                stats.dram_power = 0.0
        return total, used_prefill, breakdown

    def run_op(
        self,
        model_name: str,
        method_name: str,
        op: object,
        phase: str,
        kv_length: int,
        sequence_length: int,
        trace_index: int,
        moe_stage: str,
    ) -> tuple[Stats, bool]:
        if isinstance(op, Attention):
            return self.run_attention(method_name, op, kv_length), False

        if isinstance(op, SFU):
            return self.run_sfu(
                model_name=model_name,
                method_name=method_name,
                op=op,
                phase=phase,
                sequence_length=sequence_length,
                trace_index=trace_index,
                moe_stage=moe_stage,
            )

        if not isinstance(op, FC):
            return Stats(), False

        return self.run_fc(
            model_name=model_name,
            method_name=method_name,
            op=op,
            phase=phase,
            sequence_length=sequence_length,
            trace_index=trace_index,
            moe_stage=moe_stage,
        )

    def run_attention(self, method_name: str, op: Attention, kv_length: int) -> Stats:
        method_key = self._base_method_name(method_name)
        m_size = op.sequence_length
        head_dim = op.head_dim or 128
        num_kv_heads = op.num_kv_heads or op.num_attention_heads or 1
        num_attention_heads = op.num_attention_heads or num_kv_heads
        num_groups = max(1, num_attention_heads // num_kv_heads)
        bits = self._attention_bits(method_key)

        qk = FC("attention_qk", kv_length, head_dim, m_size * num_groups, num_bits=bits, algorithm="aqlm")
        av = FC("attention_pv", head_dim, kv_length, m_size * num_groups, num_bits=bits, algorithm="aqlm")
        total = Stats(name=f"{method_name}_attention")

        for attention_fc in (qk, av):
            stats = self._run_attention_fc(method_key, attention_fc)
            total.accumulate(stats.scale(num_kv_heads))
        return total

    def run_sfu(
        self,
        model_name: str,
        method_name: str,
        op: SFU,
        phase: str,
        sequence_length: int,
        trace_index: int,
        moe_stage: str,
    ) -> tuple[Stats, bool]:
        if phase == "prefill" and getattr(op, "is_expert", False):
            routed = Stats(name=f"{method_name}_sfu")
            for bucket in self._routing_buckets(
                model_name=model_name,
                op_name=op.name,
                sequence_length=sequence_length,
                top_k=getattr(op, "top_k", 1),
                num_experts=getattr(op, "num_experts", 1),
                trace_index=trace_index,
                moe_stage=moe_stage,
            ):
                single = self._run_single_sfu(op, bucket.tokens_per_expert)
                routed.accumulate(single.scale(bucket.expert_count))
            return routed, False

        multiplier = getattr(op, "top_k", 1) if getattr(op, "is_expert", False) and phase == "decode" else 1
        return self._run_single_sfu(op, op.sequence_length).scale(multiplier), False

    def run_fc(
        self,
        model_name: str,
        method_name: str,
        op: FC,
        phase: str,
        sequence_length: int,
        trace_index: int,
        moe_stage: str,
    ) -> tuple[Stats, bool]:
        if getattr(op, "is_shared_expert", False):
            return self._run_single_fc(op, method_name, phase, sequence_length)

        if getattr(op, "is_expert", False) and phase == "prefill":
            routed = Stats(name=f"{method_name}_expert")
            used_prefill = False
            for bucket in self._routing_buckets(
                model_name=model_name,
                op_name=op.name,
                sequence_length=sequence_length,
                top_k=getattr(op, "top_k", 1),
                num_experts=getattr(op, "num_experts", 1),
                trace_index=trace_index,
                moe_stage=moe_stage,
            ):
                single, single_prefill = self._run_single_fc(op, method_name, phase, bucket.tokens_per_expert)
                routed.accumulate(single.scale(bucket.expert_count))
                used_prefill = used_prefill or single_prefill
            return routed, used_prefill

        if getattr(op, "is_expert", False) and phase == "decode":
            single, used_prefill = self._run_single_fc(op, method_name, phase, 1)
            return single.scale(getattr(op, "top_k", 1)), used_prefill

        return self._run_single_fc(op, method_name, phase, sequence_length)

    def apply_energy(self, method_name: str, stats: Stats, used_prefill: bool) -> Stats:
        runner_name = self._energy_runner_name(method_name, used_prefill)
        return self.method_registry.runner_for(runner_name).apply_energy_breakdown(stats)

    def _run_single_sfu(self, op: SFU, sequence_length: int) -> Stats:
        stats = Stats(name=op.name)
        sfu_width = self.config.vq_array_width
        passes_per_elem = ceil_a_by_b(op.dim, sfu_width)

        if op.op_type == "softmax":
            stats.compute_cycles += 4 * passes_per_elem * sequence_length
            stats.reads["s_act"] += sequence_length * op.dim * op.nbits
            stats.writes["s_act"] += sequence_length * op.dim * op.nbits
        elif op.op_type == "swiglu":
            stats.compute_cycles += 3 * passes_per_elem * sequence_length
            stats.reads["s_act"] += 2 * sequence_length * op.dim * op.nbits
            stats.writes["s_act"] += sequence_length * op.dim * op.nbits
        elif op.op_type == "rmsnorm":
            stats.compute_cycles += 3 * passes_per_elem * sequence_length
            stats.reads["s_act"] += sequence_length * op.dim * op.nbits
            stats.reads["s_wgt"] += op.dim * op.nbits
            stats.writes["s_act"] += sequence_length * op.dim * op.nbits
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
        return stats

    def _run_single_fc(self, op: FC, method_name: str, phase: str, sequence_length: int) -> tuple[Stats, bool]:
        method_key = self._base_method_name(method_name)
        original_sequence_length = op.sequence_length
        op.sequence_length = sequence_length
        op.activation_tensor.shape[0] = sequence_length
        op.output_tensor.shape[0] = sequence_length
        used_prefill = False

        if phase == "prefill":
            runner_name, used_prefill = self._prefill_runner_name(method_key, sequence_length)
        else:
            runner_name = self._decode_runner_name(method_key)

        stats = self.method_registry.runner_for(runner_name).run_fc(op, self.config)

        op.sequence_length = original_sequence_length
        op.activation_tensor.shape[0] = original_sequence_length
        op.output_tensor.shape[0] = original_sequence_length
        return stats, used_prefill

    def _run_attention_fc(self, method_name: str, op: FC) -> Stats:
        if method_name == "vqarray_2_decode_kvq":
            if op.sequence_length > 32:
                return self.method_registry.runner_for("vqarray_2_prefill").run_fc(op, self.config)
            return self.method_registry.runner_for("vqarray_2_decode").run_fc(op, self.config)

        if method_name.startswith("vqarray"):
            return self.method_registry.runner_for("vqarray_2_prefill").run_fc(op, self.config)
        return self.method_registry.runner_for(method_name).run_fc(op, self.config)

    def _prefill_runner_name(self, method_name: str, sequence_length: int) -> tuple[str, bool]:
        if method_name == "vqarray_2_gptvq_decode":
            return ("vqarray_2_prefill", True) if sequence_length > 4 else (method_name, False)
        if method_name in {"vqarray_2_decode", "vqarray_2_decode_kvq"}:
            return ("vqarray_2_prefill", True) if sequence_length > 32 else ("vqarray_2_decode", False)
        if method_name == "vqarray_3_decode":
            return ("vqarray_2_prefill", True) if sequence_length > 16 else (method_name, False)
        if method_name == "vqarray_4_decode":
            return ("vqarray_2_prefill", True) if sequence_length > 8 else (method_name, False)
        return method_name, False

    def _decode_runner_name(self, method_name: str) -> str:
        if method_name == "vqarray_2_decode_kvq":
            return "vqarray_2_decode"
        return method_name

    def _energy_runner_name(self, method_name: str, used_prefill: bool) -> str:
        method_key = self._base_method_name(method_name)
        if used_prefill and method_key.startswith("vqarray") and method_key != "vqarray_2_prefill":
            return "vqarray_2_prefill"
        if method_key == "vqarray_2_decode_kvq":
            return "vqarray_2_decode"
        return method_key

    def _base_method_name(self, method_name: str) -> str:
        return method_name

    def _attention_bits(self, method_name: str) -> int:
        if method_name.startswith("vqarray"):
            return 2 if method_name == "vqarray_2_gptvq_decode" else int(method_name.split("_")[1])
        if method_name in {"systolic_array", "figna", "ant"}:
            return 8
        return 4

    def _routing_buckets(
        self,
        model_name: str,
        op_name: str,
        sequence_length: int,
        top_k: int,
        num_experts: int,
        trace_index: int,
        moe_stage: str,
    ) -> list[RoutingBucket]:
        seed_key = f"{model_name}:{op_name}:{sequence_length}:{trace_index}"
        if moe_stage == "stage2":
            return self.stage2_router.buckets(sequence_length, top_k, num_experts, seed_key)
        return self.stage1_router.buckets(sequence_length, top_k, num_experts, seed_key)


def extract_stats(stats: Stats) -> dict[str, float]:
    return {
        "total_cycles": float(stats.total_cycles),
        "compute_cycles": float(stats.compute_cycles),
        "total_energy": float(stats.core_energy + stats.buffer_energy + stats.dram_energy + stats.static_energy),
        "total_power": float(stats.core_power + stats.sram_power + stats.dram_power),
        "core_energy": float(stats.core_energy),
        "buffer_energy": float(stats.buffer_energy),
        "dram_energy": float(stats.dram_energy),
        "static_energy": float(stats.static_energy),
        "core_power": float(stats.core_power),
        "sram_power": float(stats.sram_power),
        "dram_power": float(stats.dram_power),
    }


def write_trace_metrics(
    row: dict[str, int | float | str],
    method_name: str,
    phase: str,
    totals: dict[str, float],
    breakdown: dict[str, dict[str, float]],
    scale: int,
) -> None:
    row[f"{phase}_{method_name}_cycles"] = totals["total_cycles"] * scale
    row[f"{phase}_{method_name}_compute_cycles"] = totals["compute_cycles"] * scale
    row[f"{phase}_{method_name}_time_s"] = (totals["total_cycles"] / FREQ_HZ) * scale
    row[f"{phase}_{method_name}_energy_j"] = totals["total_energy"] * scale
    row[f"{phase}_{method_name}_core_energy_j"] = totals["core_energy"] * scale
    row[f"{phase}_{method_name}_buffer_energy_j"] = totals["buffer_energy"] * scale
    row[f"{phase}_{method_name}_dram_energy_j"] = totals["dram_energy"] * scale
    row[f"{phase}_{method_name}_static_energy_j"] = totals["static_energy"] * scale
    row[f"{phase}_{method_name}_power_w"] = totals["total_power"]
    row[f"{phase}_{method_name}_core_power_w"] = totals["core_power"]
    row[f"{phase}_{method_name}_sram_power_w"] = totals["sram_power"]
    row[f"{phase}_{method_name}_dram_power_w"] = totals["dram_power"]

    for component in METHOD_COMPONENTS:
        component_stats = breakdown.get(component, {})
        component_cycles = component_stats.get("total_cycles", 0.0)
        component_energy = component_stats.get("total_energy", 0.0)
        row[f"{phase}_{method_name}_{component}_cycles"] = component_cycles * scale
        row[f"{phase}_{method_name}_{component}_time_s"] = (component_cycles / FREQ_HZ) * scale
        row[f"{phase}_{method_name}_{component}_energy_j"] = component_energy * scale
