from simulator.methods.base import MethodRunner, get_weight_stationary_cycles
from simulator.ops import FC
from simulator.specs import RunnerConfig
from simulator.utils import Stats, ceil_a_by_b


class DenseArrayMethod(MethodRunner):
    def run_fc(self, op: FC, config: RunnerConfig) -> Stats:
        stats = Stats(name=self.spec.name)
        op.activation_tensor.nbits = self.spec.bitwidths["activation"]
        op.weight.nbits = self.spec.bitwidths["weight"]
        op.output_tensor.nbits = self.spec.bitwidths["output"]

        m_size = op.sequence_length
        k_size = op.input_dim
        n_size = op.output_dim

        array_height = self.spec.array["height"]
        array_width = self.spec.array["width"]
        tile_size_m = self.spec.tiles["m"]
        tile_size_k = self.spec.tiles["k"]
        tile_size_n = self.spec.tiles["n"]
        buffer_size_m = self.spec.buffers["m"]
        buffer_size_k = self.spec.buffers["k"]
        buffer_size_n = self.spec.buffers["n"]
        mem_width = config.mem_width
        pipeline_overhead_cycles = int(self.spec.extra.get("pipeline_overhead_cycles", 0))

        tile_num_m = ceil_a_by_b(m_size, tile_size_m)
        tile_num_k = ceil_a_by_b(k_size, tile_size_k)
        tile_num_n = ceil_a_by_b(n_size, tile_size_n)

        for m_idx in range(tile_num_m):
            for n_idx in range(tile_num_n):
                for k_idx in range(tile_num_k):
                    current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
                    current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
                    current_tile_size_n = min(tile_size_n, n_size - n_idx * tile_size_n)

                    stats.reads["s_act"] += current_tile_size_k * current_tile_size_m * op.activation_tensor.nbits
                    stats.reads["s_wgt"] += current_tile_size_n * current_tile_size_k * op.weight.nbits
                    stats.reads["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
                    stats.writes["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits

        stats.compute_cycles += self._run_linear_dense(
            m_size,
            n_size,
            k_size,
            array_height,
            array_width,
            tile_size_m,
            pipeline_overhead_cycles,
        )

        buffer_tile_num_m = ceil_a_by_b(m_size, buffer_size_m)
        buffer_tile_num_k = ceil_a_by_b(k_size, buffer_size_k)
        buffer_tile_num_n = ceil_a_by_b(n_size, buffer_size_n)

        for m_idx in range(buffer_tile_num_m):
            for n_idx in range(buffer_tile_num_n):
                current_buffer_tile_size_n = min(buffer_size_n, n_size - n_idx * buffer_size_n)
                for k_idx in range(buffer_tile_num_k):
                    current_buffer_tile_size_k = min(buffer_size_k, k_size - k_idx * buffer_size_k)
                    current_buffer_tile_size_m = min(buffer_size_m, m_size - m_idx * buffer_size_m)

                    stats.reads["dram"] += current_buffer_tile_size_k * current_buffer_tile_size_m * op.activation_tensor.nbits
                    stats.writes["s_act"] += current_buffer_tile_size_k * current_buffer_tile_size_m * op.activation_tensor.nbits

                    stats.reads["dram"] += current_buffer_tile_size_n * current_buffer_tile_size_k * op.weight.nbits
                    stats.writes["s_wgt"] += current_buffer_tile_size_n * current_buffer_tile_size_k * op.weight.nbits

                    if k_idx > 0:
                        stats.reads["dram"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits
                        stats.writes["s_output"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits

        stats.reads["s_output"] += m_size * n_size * op.output_tensor.nbits
        stats.writes["dram"] += m_size * n_size * op.output_tensor.nbits

        first_buffer_tile_size_k = min(tile_size_k, k_size)
        first_buffer_tile_size_m = min(tile_size_m, m_size)
        first_buffer_tile_size_n = min(tile_size_n, n_size)
        init_mem_access = first_buffer_tile_size_k * first_buffer_tile_size_m * op.activation_tensor.nbits
        init_mem_access += first_buffer_tile_size_n * first_buffer_tile_size_k * op.weight.nbits
        init_latency = ceil_a_by_b(init_mem_access, mem_width)
        stats.mem_stall_cycles += init_latency

        total_mem_access = stats.reads["dram"] + stats.writes["dram"]
        middle_mem_access = total_mem_access - init_mem_access
        middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
        stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
        return stats

    def apply_energy_breakdown(self, stats: Stats) -> Stats:
        energy = self.spec.energy
        core_energy = stats.total_cycles * energy["core_leak_energy"]
        core_energy += (stats.total_cycles - stats.mem_stall_cycles) * energy["core_dynamic_energy"]

        sram_energy = stats.reads["s_wgt"] * energy["wbuf_read_energy"]
        sram_energy += stats.writes["s_wgt"] * energy["wbuf_write_energy"]
        sram_energy += stats.reads["s_act"] * energy["ibuf_read_energy"]
        sram_energy += stats.writes["s_act"] * energy["ibuf_write_energy"]
        sram_energy += stats.reads["s_output"] * energy["obuf_read_energy"]
        sram_energy += stats.writes["s_output"] * energy["obuf_write_energy"]
        sram_energy += stats.total_cycles * energy["sram_leak_energy"]

        dram_energy = (
            stats.reads["dram"] * energy["dram_cost_read"]
            + stats.writes["dram"] * energy["dram_cost_write"]
            + stats.total_cycles * energy["dram_leak_energy"]
        )

        stats.dram_energy = dram_energy / 1e9
        stats.buffer_energy = sram_energy / 1e9
        stats.core_energy = core_energy / 1e9
        stats.core_power = energy["core_power"]
        stats.sram_power = sram_energy / (stats.total_cycles / (500 * 1e6)) / 1e9
        stats.dram_power = dram_energy / (stats.total_cycles / (500 * 1e6)) / 1e9
        return stats

    def _run_linear_dense(
        self,
        m_size: int,
        n_size: int,
        k_size: int,
        array_height: int,
        array_width: int,
        m_tile_size: int,
        pipeline_overhead_cycles: int,
    ) -> int:
        m_num_tiles = ceil_a_by_b(m_size, m_tile_size)
        m_input_size = ceil_a_by_b(m_size, m_num_tiles)
        n_tile_size = array_height
        k_tile_size = array_width
        compute_cycles = get_weight_stationary_cycles(
            m_input_size,
            n_tile_size,
            k_tile_size,
            array_height,
            array_width,
            pipeline_overhead_cycles=pipeline_overhead_cycles,
        ) * m_num_tiles
        n_repeat = ceil_a_by_b(n_size, n_tile_size)
        k_repeat = ceil_a_by_b(k_size, k_tile_size)
        return compute_cycles * n_repeat * k_repeat
