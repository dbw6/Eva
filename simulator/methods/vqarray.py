from simulator.methods.base import MethodRunner, get_input_stationary_cycles, get_weight_stationary_cycles
from simulator.ops import FC
from simulator.specs import RunnerConfig
from simulator.utils import Stats, ceil_a_by_b


class VqArrayMethod(MethodRunner):
    def run_fc(self, op: FC, config: RunnerConfig) -> Stats:
        if self.spec.family == "vqarray_prefill":
            return self._run_prefill(op, config)
        if self.spec.family == "vqarray_gptvq_decode":
            return self._run_decode(op, config, gptvq=True)
        return self._run_decode(op, config, gptvq=False)

    def apply_energy_breakdown(self, stats: Stats) -> Stats:
        if stats.total_cycles == 0:
            stats.core_power = 0.0
            stats.sram_power = 0.0
            stats.dram_power = 0.0
            stats.array_power_w = 0.0
            stats.epilogue_power_w = 0.0
            stats.sfu_power_w = 0.0
            stats.array_area_mm2 = 0.0
            stats.epilogue_area_mm2 = 0.0
            stats.sfu_area_mm2 = 0.0
            return stats
        energy = self.spec.energy
        core_energy = stats.total_cycles * energy["core_leak_energy"]
        core_energy += (stats.total_cycles - stats.mem_stall_cycles) * energy["core_dynamic_energy"]

        sram_energy = stats.reads["s_wgt"] * energy["wbuf_read_energy"]
        sram_energy += stats.writes["s_wgt"] * energy["wbuf_write_energy"]
        sram_energy += stats.reads["s_act"] * energy["ibuf_read_energy"]
        sram_energy += stats.writes["s_act"] * energy["ibuf_write_energy"]
        sram_energy += stats.reads["s_output"] * energy["obuf_read_energy"]
        sram_energy += stats.writes["s_output"] * energy["obuf_write_energy"]

        if self.spec.family != "vqarray_prefill":
            sram_energy += stats.reads["s_cap"] * energy["cap_buf_read_energy"]
            sram_energy += stats.writes["s_cap"] * energy["cap_buf_write_energy"]
            sram_energy += stats.reads["s_codebook"] * energy["codebook_buf_read_energy"]
            sram_energy += stats.writes["s_codebook"] * energy["codebook_buf_write_energy"]

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
        stats.array_power_w = float(energy.get("array_power", 0.0))
        adder_tree_size = int(getattr(stats, "adder_tree_size", self.spec.tiles.get("adder_tree_size", 128)))
        num_epilogue_units = max(1, adder_tree_size // 32)
        stats.epilogue_power_w = float(energy.get("epilogue_power_per_unit", 0.0)) * num_epilogue_units
        stats.sfu_power_w = float(energy.get("sfu_power", 0.0))
        stats.array_area_mm2 = float(energy.get("array_area_mm2", 0.0))
        stats.epilogue_area_mm2 = float(energy.get("epilogue_area_per_unit_mm2", 0.0)) * num_epilogue_units
        stats.sfu_area_mm2 = float(energy.get("sfu_area_mm2", 0.0))
        return stats

    def _run_decode(self, op: FC, config: RunnerConfig, gptvq: bool) -> Stats:
        stats = Stats(name=self.spec.name)
        m_size = op.sequence_length
        if m_size > 64:
            return stats

        original_num_codebooks = op.num_codebooks
        original_vector_size = op.vector_size
        original_num_entries = op.num_entries
        default_num_codebooks = 1 if gptvq else op.num_codebooks
        default_vector_size = 4 if gptvq else op.vector_size
        default_num_entries = 256 if gptvq else op.num_entries
        op.num_codebooks = int(self.spec.extra.get("decode_num_codebooks", default_num_codebooks))
        op.vector_size = int(self.spec.extra.get("decode_vector_size", default_vector_size))
        op.num_entries = int(self.spec.extra.get("decode_num_entries", default_num_entries))

        op.activation_tensor.nbits = self.spec.bitwidths["activation"]
        op.codebooks.nbits = self.spec.bitwidths["codebook"]
        op.index.nbits = self.spec.bitwidths["index"]
        op.output_tensor.nbits = self.spec.bitwidths["output"]

        k_size = op.input_dim
        n_size = op.output_dim
        array_height = config.vq_array_height
        array_width = config.vq_array_width
        adder_tree_size = config.vq_adder_tree_size or self.spec.tiles["adder_tree_size"]
        tile_size_codebook = self.spec.tiles["tile_size_codebook"]
        buffer_bits = self.spec.tiles["buffer_bits"]
        grouped_decode = bool(self.spec.extra.get("grouped_decode", gptvq))

        tile_size_m = 2 ** int((m_size).bit_length() - 1)
        tile_size_n = self.spec.tiles.get("group_size_n", n_size) if grouped_decode else n_size
        tile_size_k = ceil_a_by_b(adder_tree_size * op.vector_size, m_size * tile_size_codebook)
        buffer_size_m = tile_size_m
        buffer_size_n = ceil_a_by_b(buffer_bits, m_size)
        buffer_size_k = ceil_a_by_b(buffer_bits, m_size)

        tile_num_m = ceil_a_by_b(m_size, tile_size_m)
        tile_num_k = ceil_a_by_b(k_size, tile_size_k)
        tile_num_codebook = ceil_a_by_b(op.num_codebooks, tile_size_codebook)
        tile_num_n = ceil_a_by_b(n_size, tile_size_n)

        first_vector_num = ceil_a_by_b(min(tile_size_k, k_size), op.vector_size)
        first_codebook_size = min(tile_size_codebook, op.num_codebooks)
        first_tile_size_m = min(tile_size_m, m_size)
        stats.compute_cycles += get_input_stationary_cycles(
            array_height,
            op.num_entries * first_codebook_size,
            op.vector_size,
            array_height,
            array_width,
        ) * ceil_a_by_b(first_vector_num * first_tile_size_m, array_height)

        for m_idx in range(tile_num_m):
            current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
            for k_idx in range(tile_num_k):
                current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
                vector_num = ceil_a_by_b(current_tile_size_k, op.vector_size)
                for codebook_idx in range(tile_num_codebook):
                    current_tile_size_codebook = min(tile_size_codebook, op.num_codebooks - codebook_idx * tile_size_codebook)
                    n_range = range(tile_num_n) if grouped_decode else range(1)
                    for n_idx in n_range:
                        current_tile_size_n = min(tile_size_n, n_size - n_idx * tile_size_n)
                        adder_tree_cycles = current_tile_size_n if gptvq else n_size

                        stats.reads["s_act"] += current_tile_size_k * current_tile_size_m * op.activation_tensor.nbits
                        stats.reads["s_codebook"] += (
                            current_tile_size_codebook
                            * op.vector_size
                            * op.num_entries
                            * ceil_a_by_b(current_tile_size_m * vector_num, array_height)
                            * op.codebooks.nbits
                        )
                        stats.writes["s_cap"] += (
                            current_tile_size_codebook
                            * vector_num
                            * op.num_entries
                            * current_tile_size_m
                            * op.activation_tensor.nbits
                        )
                        systolic_array_cycles = get_input_stationary_cycles(
                            array_height,
                            current_tile_size_codebook * op.num_entries,
                            op.vector_size,
                            array_height,
                            array_width,
                        ) * ceil_a_by_b(vector_num * current_tile_size_m, array_height)

                        stats.reads["s_wgt"] += current_tile_size_n * vector_num * op.index.nbits * current_tile_size_codebook
                        stats.reads["s_cap"] += current_tile_size_n * vector_num * op.activation_tensor.nbits * current_tile_size_codebook
                        if codebook_idx > 0 or k_idx > 0:
                            stats.reads["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
                        stats.writes["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits

                        if k_idx != tile_num_k - 1 or codebook_idx != tile_num_codebook - 1:
                            stats.compute_cycles += max(systolic_array_cycles, adder_tree_cycles)
                        else:
                            stats.compute_cycles += adder_tree_cycles + 6

        buffer_tile_num_m = ceil_a_by_b(m_size, buffer_size_m)
        buffer_tile_num_k = ceil_a_by_b(k_size, buffer_size_k)
        buffer_tile_num_n = ceil_a_by_b(n_size, buffer_size_n)

        stats.reads["dram"] += op.num_codebooks * op.vector_size * op.num_entries * op.codebooks.nbits
        stats.writes["s_codebook"] += op.num_codebooks * op.vector_size * op.num_entries * op.codebooks.nbits
        stats.reads["dram"] += n_size * op.activation_tensor.nbits
        stats.writes["s_scale"] += n_size * op.activation_tensor.nbits

        for m_idx in range(buffer_tile_num_m):
            for k_idx in range(buffer_tile_num_k):
                current_buffer_tile_size_k = min(buffer_size_k, k_size - k_idx * buffer_size_k)
                current_buffer_tile_size_m = min(buffer_size_m, m_size - m_idx * buffer_size_m)
                buffer_vector_num = ceil_a_by_b(current_buffer_tile_size_k, op.vector_size)

                stats.reads["dram"] += current_buffer_tile_size_k * current_buffer_tile_size_m * op.activation_tensor.nbits
                stats.writes["s_act"] += current_buffer_tile_size_k * current_buffer_tile_size_m * op.activation_tensor.nbits

                for n_idx in range(buffer_tile_num_n):
                    current_buffer_tile_size_n = min(buffer_size_n, n_size - n_idx * buffer_size_n)
                    for _ in range(op.num_codebooks):
                        stats.reads["dram"] += current_buffer_tile_size_n * buffer_vector_num * op.index.nbits
                        stats.writes["s_wgt"] += current_buffer_tile_size_n * buffer_vector_num * op.index.nbits
                    if k_idx > 0:
                        stats.reads["dram"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits
                        stats.writes["s_output"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits
                    stats.reads["s_output"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits
                    stats.writes["dram"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits

        first_buffer_tile_size_k = min(tile_size_k, k_size)
        first_buffer_tile_size_m = min(tile_size_m, m_size)
        first_buffer_tile_codebook_size = min(tile_size_codebook, op.num_codebooks)
        init_mem_access = op.vector_size * op.num_entries * op.codebooks.nbits * first_buffer_tile_codebook_size
        init_mem_access += first_buffer_tile_size_k * first_buffer_tile_size_m * op.activation_tensor.nbits
        init_latency = ceil_a_by_b(init_mem_access, config.mem_width)
        stats.mem_stall_cycles += init_latency

        total_mem_access = stats.reads["dram"] + stats.writes["dram"]
        middle_mem_access = total_mem_access - init_mem_access
        middle_latency = ceil_a_by_b(middle_mem_access, config.mem_width)
        stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
        stats.adder_tree_size = int(adder_tree_size)

        op.num_codebooks = original_num_codebooks
        op.vector_size = original_vector_size
        op.num_entries = original_num_entries
        return stats

    def _run_prefill(self, op: FC, config: RunnerConfig) -> Stats:
        stats = Stats(name="prefill")
        m_size = op.sequence_length
        k_size = op.input_dim
        n_size = op.output_dim
        op.activation_tensor.nbits = self.spec.bitwidths["activation"]
        op.weight.nbits = self.spec.bitwidths["weight"]
        op.output_tensor.nbits = self.spec.bitwidths["output"]

        array_height = self.spec.array["height"]
        array_width = self.spec.array["width"]
        simd_width = self.spec.tiles["simd_width"]

        tile_size_m = 1024
        tile_size_k = array_width * simd_width
        tile_size_n = array_height
        buffer_size_m = self.spec.buffers["m"]
        buffer_size_k = self.spec.buffers["k"]
        buffer_size_n = self.spec.buffers["n"]

        tile_num_m = ceil_a_by_b(m_size, tile_size_m)
        tile_num_k = ceil_a_by_b(k_size, tile_size_k)
        tile_num_n = ceil_a_by_b(n_size, tile_size_n)

        for m_idx in range(tile_num_m):
            for n_idx in range(tile_num_n):
                for k_idx in range(tile_num_k):
                    current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
                    current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
                    current_tile_size_n = min(tile_size_n, n_size - n_idx * tile_size_n)
                    simd_groups = ceil_a_by_b(current_tile_size_k, simd_width)

                    stats.reads["s_act"] += current_tile_size_k * current_tile_size_m * op.activation_tensor.nbits
                    stats.reads["s_wgt"] += current_tile_size_n * current_tile_size_k * op.weight.nbits
                    stats.reads["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
                    stats.writes["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
                    stats.compute_cycles += get_weight_stationary_cycles(
                        current_tile_size_m,
                        current_tile_size_n,
                        simd_groups,
                        array_height,
                        array_width,
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
        init_latency = ceil_a_by_b(init_mem_access, config.mem_width)
        stats.mem_stall_cycles += init_latency

        total_mem_access = stats.reads["dram"] + stats.writes["dram"]
        middle_mem_access = total_mem_access - init_mem_access
        middle_latency = ceil_a_by_b(middle_mem_access, config.mem_width)
        stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
        return stats
