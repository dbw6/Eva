from abc import ABC, abstractmethod

from simulator.ops import FC
from simulator.specs import MethodSpec, RunnerConfig
from simulator.utils import Stats, ceil_a_by_b


def get_weight_stationary_cycles(
    m_size: int,
    n_size: int,
    k_size: int,
    array_height: int,
    array_width: int,
    pipeline_overhead_cycles: int = 0,
) -> int:
    compute_cycles = ceil_a_by_b(n_size, array_height) * ceil_a_by_b(k_size, array_width) * m_size
    compute_cycles += array_height + array_width + pipeline_overhead_cycles
    return compute_cycles


def get_input_stationary_cycles(
    m_size: int,
    n_size: int,
    k_size: int,
    array_height: int,
    array_width: int,
) -> int:
    compute_cycles = ceil_a_by_b(m_size, array_height) * ceil_a_by_b(k_size, array_width) * n_size
    compute_cycles += array_height + array_width
    return compute_cycles


class MethodRunner(ABC):
    def __init__(self, spec: MethodSpec) -> None:
        self.spec = spec

    @abstractmethod
    def run_fc(self, op: FC, config: RunnerConfig) -> Stats:
        raise NotImplementedError

    @abstractmethod
    def apply_energy_breakdown(self, stats: Stats) -> Stats:
        raise NotImplementedError
