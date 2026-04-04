from simulator.methods.base import MethodRunner, get_weight_stationary_cycles
from simulator.ops import FC
from simulator.specs import RunnerConfig
from simulator.utils import Stats, ceil_a_by_b


class FiglutMethod(MethodRunner):
    def run_fc(self, op: FC, config: RunnerConfig) -> Stats:
        stats = Stats(name=self.spec.name)

        m_size = op.sequence_length
        k_size = op.input_dim
        n_size = op.output_dim
        op.activation_tensor.nbits = self.spec.bitwidths["activation"]
        op.output_tensor.nbits = self.spec.bitwidths["output"]

        array_height = self.spec.array["height"]
        array_width = self.spec.array["width"]
        bit_planes = op.bitwidth
        mu = int(self.spec.extra["mu"])
        k_value = int(self.spec.extra["k"])
        num_mpu = int(self.spec.extra["num_mpu"])
        parallel_compute = ceil_a_by_b(num_mpu, bit_planes)

        tile_size_m = ceil_a_by_b(self.spec.tiles["m_base"], parallel_compute)
        tile_size_k = array_width * mu * parallel_compute
        tile_size_n = array_height * k_value
        buffer_size_m = ceil_a_by_b(self.spec.tiles["m_base"], parallel_compute)
        buffer_size_k = self.spec.tiles["buffer_k_base"] * parallel_compute
        buffer_size_n = self.spec.tiles["buffer_n"]
        mem_width = config.mem_width

        tile_num_m = ceil_a_by_b(m_size, tile_size_m)
        tile_num_k = ceil_a_by_b(k_size, tile_size_k)
        tile_num_n = ceil_a_by_b(n_size, tile_size_n)
        tile_num_bit_planes = ceil_a_by_b(bit_planes, num_mpu)

        for m_idx in range(tile_num_m):
            for n_idx in range(tile_num_n):
                for k_idx in range(tile_num_k):
                    for bit_plane_idx in range(tile_num_bit_planes):
                        current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
                        current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
                        current_tile_size_n = min(tile_size_n, n_size - n_idx * tile_size_n)
                        current_bit_planes = min(num_mpu, bit_planes - bit_plane_idx * num_mpu)

                        binary_weight_bits = current_tile_size_n * current_tile_size_k * current_bit_planes
                        current_lut_groups = ceil_a_by_b(current_tile_size_k, mu * parallel_compute)

                        stats.reads["s_act"] += current_tile_size_k * current_tile_size_m * op.activation_tensor.nbits
                        stats.reads["s_wgt"] += binary_weight_bits
                        stats.reads["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
                        stats.writes["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
                        stats.reads["s_scale"] += current_tile_size_n * op.activation_tensor.nbits * current_bit_planes

                        lut_gen_cycles = 2
                        current_output_size = ceil_a_by_b(current_tile_size_n, k_value)
                        array_compute_cycles = get_weight_stationary_cycles(
                            current_tile_size_m,
                            current_output_size,
                            current_lut_groups,
                            array_height,
                            array_width,
                        )
                        reduction_cycles = 2
                        stats.compute_cycles += lut_gen_cycles + array_compute_cycles + reduction_cycles

        buffer_tile_num_m = ceil_a_by_b(m_size, buffer_size_m)
        buffer_tile_num_k = ceil_a_by_b(k_size, buffer_size_k)
        buffer_tile_num_n = ceil_a_by_b(n_size, buffer_size_n)

        scaling_factor_bits = n_size * op.activation_tensor.nbits * bit_planes
        stats.reads["dram"] += scaling_factor_bits
        stats.writes["s_scale"] += scaling_factor_bits

        offset_bits = n_size * op.output_tensor.nbits
        stats.reads["dram"] += offset_bits
        stats.writes["s_scale"] += offset_bits

        for m_idx in range(buffer_tile_num_m):
            for n_idx in range(buffer_tile_num_n):
                current_buffer_tile_size_n = min(buffer_size_n, n_size - n_idx * buffer_size_n)
                for k_idx in range(buffer_tile_num_k):
                    for bit_plane_idx in range(tile_num_bit_planes):
                        current_buffer_tile_size_k = min(buffer_size_k, k_size - k_idx * buffer_size_k)
                        current_buffer_tile_size_m = min(buffer_size_m, m_size - m_idx * buffer_size_m)
                        current_bit_planes = min(num_mpu, bit_planes - bit_plane_idx * num_mpu)

                        binary_weight_bits = current_buffer_tile_size_n * current_buffer_tile_size_k * current_bit_planes
                        stats.reads["dram"] += current_buffer_tile_size_k * current_buffer_tile_size_m * op.activation_tensor.nbits
                        stats.writes["s_act"] += current_buffer_tile_size_k * current_buffer_tile_size_m * op.activation_tensor.nbits
                        stats.reads["dram"] += binary_weight_bits
                        stats.writes["s_wgt"] += binary_weight_bits

        stats.reads["dram"] += m_size * n_size * op.output_tensor.nbits
        stats.writes["s_output"] += m_size * n_size * op.output_tensor.nbits
        stats.reads["s_output"] += m_size * n_size * op.output_tensor.nbits
        stats.writes["dram"] += m_size * n_size * op.output_tensor.nbits

        first_buffer_tile_size_k = min(tile_size_k, k_size)
        first_buffer_tile_size_m = min(tile_size_m, m_size)
        first_buffer_tile_size_n = min(tile_size_n, n_size)
        first_binary_weight_bits = first_buffer_tile_size_n * first_buffer_tile_size_k * bit_planes
        init_mem_access = first_buffer_tile_size_k * first_buffer_tile_size_m * op.activation_tensor.nbits
        init_mem_access += first_binary_weight_bits
        init_mem_access += scaling_factor_bits + offset_bits
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
