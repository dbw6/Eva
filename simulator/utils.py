import math
from dataclasses import dataclass, field


def ceil_a_by_b(a: int, b: int) -> int:
    return int(math.ceil(float(a) / b))


@dataclass
class Stats:
    name: str = "eva"
    total_cycles: int = 0
    mem_stall_cycles: int = 0
    compute_cycles: int = 0
    num_ops: int = 0
    preprocess_stall_cycles: int = 0
    static_energy: float = 0.0
    dram_energy: float = 0.0
    buffer_energy: float = 0.0
    core_energy: float = 0.0
    core_power: float = 0.0
    sram_power: float = 0.0
    dram_power: float = 0.0
    mem_namespace: tuple[str, ...] = (
        "dram",
        "s_wgt",
        "s_act",
        "s_codebook",
        "s_cap",
        "s_double_cap",
        "s_output",
        "s_scale",
    )
    reads: dict[str, int] = field(init=False)
    writes: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.reads = {space: 0 for space in self.mem_namespace}
        self.writes = {space: 0 for space in self.mem_namespace}

    def clone(self) -> "Stats":
        copied = Stats(name=self.name)
        copied.total_cycles = self.total_cycles
        copied.mem_stall_cycles = self.mem_stall_cycles
        copied.compute_cycles = self.compute_cycles
        copied.num_ops = self.num_ops
        copied.preprocess_stall_cycles = self.preprocess_stall_cycles
        copied.static_energy = self.static_energy
        copied.dram_energy = self.dram_energy
        copied.buffer_energy = self.buffer_energy
        copied.core_energy = self.core_energy
        copied.core_power = self.core_power
        copied.sram_power = self.sram_power
        copied.dram_power = self.dram_power
        copied.reads = dict(self.reads)
        copied.writes = dict(self.writes)
        return copied

    def scale(self, factor: int) -> "Stats":
        if factor == 1:
            return self.clone()

        scaled = self.clone()
        scaled.total_cycles *= factor
        scaled.mem_stall_cycles *= factor
        scaled.compute_cycles *= factor
        scaled.num_ops *= factor
        scaled.preprocess_stall_cycles *= factor
        scaled.static_energy *= factor
        scaled.dram_energy *= factor
        scaled.buffer_energy *= factor
        scaled.core_energy *= factor
        scaled.reads = {space: value * factor for space, value in self.reads.items()}
        scaled.writes = {space: value * factor for space, value in self.writes.items()}
        return scaled

    def accumulate(self, other: "Stats") -> None:
        self.total_cycles += other.total_cycles
        self.mem_stall_cycles += other.mem_stall_cycles
        self.compute_cycles += other.compute_cycles
        self.num_ops += other.num_ops
        self.preprocess_stall_cycles += other.preprocess_stall_cycles
        self.static_energy += other.static_energy
        self.dram_energy += other.dram_energy
        self.buffer_energy += other.buffer_energy
        self.core_energy += other.core_energy
        for space in self.mem_namespace:
            self.reads[space] += other.reads.get(space, 0)
            self.writes[space] += other.writes.get(space, 0)
