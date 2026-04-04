from pathlib import Path

import numpy as np
import pandas as pd

from simulator.aqlm_analysis import extract_weight_indices, load_quantized_model
from simulator.io.results import write_rows
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.ops import FC
from simulator.config import REPO_ROOT
from simulator.specs import RunnerConfig, StudyArtifacts, StudySpec
from simulator.utils import Stats, ceil_a_by_b


TABLE_VII_COLUMNS = [
    "entry",
    "workbook_label",
    "total_cycles",
    "compute_cycles",
    "mem_stall_cycles",
    "speedup_vs_os_sa",
    "array_compute_area_raw",
    "epilogue_unit_area_raw",
    "sram_area_raw",
    "total_area_raw",
    "array_area_raw",
    "normalized_array_area",
    "area_source",
]

AREA_COLUMNS = [
    "entry",
    "array_compute_area_raw",
    "epilogue_unit_area_raw",
    "array_area_raw",
    "normalized_array_area",
]

SPEED_COLUMNS = [
    "entry",
    "total_cycles",
    "speedup_vs_os_sa",
]

ABLATION_ENTRY_ORDER = [
    "VQ_conflict",
    "VQ_LLM 0.5",
    "OS_SA",
    "EVA 4x8",
    "EVA 32x8",
    "EVA scale x 4",
]


def _get_systolic_array_cycles_is(m_size: int, n_size: int, k_size: int, array_height: int, array_width: int) -> int:
    compute_cycles = ceil_a_by_b(m_size, array_height) * ceil_a_by_b(k_size, array_width) * n_size
    compute_cycles += array_height + array_width - 1
    return compute_cycles


def _get_systolic_array_cycles_os(m_size: int, n_size: int, k_size: int, array_height: int, array_width: int) -> int:
    compute_cycles = ceil_a_by_b(m_size, array_height) * ceil_a_by_b(n_size, array_width) * k_size
    compute_cycles += m_size + array_width - 1
    return compute_cycles


def _run_linear_dense(m_size: int, n_size: int, k_size: int, array_height: int, array_width: int, k_tile_size: int = 1024) -> int:
    k_num_tile = ceil_a_by_b(k_size, k_tile_size)
    k_input_size = ceil_a_by_b(k_size, k_num_tile)
    compute_cycles = _get_systolic_array_cycles_os(m_size, array_width, k_input_size, array_height, array_width) * k_num_tile
    n_repeat = ceil_a_by_b(n_size, array_width)
    m_repeat = ceil_a_by_b(m_size, m_size)
    return compute_cycles * n_repeat * m_repeat


def _memory_cycles_for_tile(tile_indices: np.ndarray, num_banks: int, entries_per_bank: int) -> int:
    if tile_indices.size == 0:
        return 0
    n_rows, n_cols = tile_indices.shape[0], tile_indices.shape[1]
    num_codebooks = tile_indices.shape[-1]
    tile_mem_cycles = 0
    tile_num_banks = ceil_a_by_b(n_rows, num_banks)
    for bank_tile in range(tile_num_banks):
        current_tile_size_bank = min(num_banks, n_rows - bank_tile * num_banks)
        k_lo = bank_tile * num_banks
        k_hi = k_lo + current_tile_size_bank
        for col_idx in range(n_cols):
            cycles_this_col = 0
            for codebook_idx in range(num_codebooks):
                column_indices = tile_indices[k_lo:k_hi, col_idx, codebook_idx].flatten().astype(np.uint32)
                bank_ids = column_indices // entries_per_bank
                bank_counts = np.bincount(bank_ids, minlength=num_banks)
                cycles_this_col = max(cycles_this_col, int(bank_counts.max()) if bank_counts.size else 0)
            tile_mem_cycles += cycles_this_col
    return tile_mem_cycles


def _memory_cycles_for_tile_with_replication(
    tile_indices: np.ndarray,
    num_banks: int,
    entries_per_bank: int,
    replicated_sets: list[set[int]],
) -> int:
    if tile_indices.size == 0:
        return 0
    n_rows, n_cols = tile_indices.shape[0], tile_indices.shape[1]
    num_codebooks = tile_indices.shape[-1]
    tile_mem_cycles = 0
    tile_num_banks = ceil_a_by_b(n_rows, num_banks)
    for bank_tile in range(tile_num_banks):
        current_tile_size_bank = min(num_banks, n_rows - bank_tile * num_banks)
        k_lo = bank_tile * num_banks
        k_hi = k_lo + current_tile_size_bank
        for col_idx in range(n_cols):
            cycles_this_col = 0
            for codebook_idx in range(num_codebooks):
                column_indices = tile_indices[k_lo:k_hi, col_idx, codebook_idx].flatten().astype(np.uint32)
                if replicated_sets and codebook_idx < len(replicated_sets):
                    rep_set = replicated_sets[codebook_idx]
                    if rep_set:
                        column_indices = column_indices[~np.isin(column_indices, list(rep_set))]
                if column_indices.size > 0:
                    bank_ids = column_indices // entries_per_bank
                    bank_counts = np.bincount(bank_ids, minlength=num_banks)
                    cycles_this_col = max(cycles_this_col, int(bank_counts.max()) if bank_counts.size else 0)
            tile_mem_cycles += cycles_this_col
    return tile_mem_cycles


def _run_naive_vq(op: FC, stats: Stats, indices_full: np.ndarray, mem_width: int) -> Stats:
    stats.name = "vq_conflict"
    op.activation_tensor.nbits = 16
    op.weight.nbits = 16
    op.output_tensor.nbits = 16
    index_nbits = 8

    m_size = op.sequence_length
    k_size = op.input_dim
    n_size = op.output_dim
    array_height = 8
    array_width = 32
    codebook_entries = 256
    vector_size = 8
    num_codebooks = indices_full.shape[-1]
    num_banks = array_width // vector_size
    entries_per_bank = codebook_entries // num_banks

    tile_size_m = array_height
    tile_size_k = 1024
    tile_size_n = array_width
    buffer_size_m = 64
    buffer_size_k = 1024
    buffer_size_n = 128

    tile_num_m = ceil_a_by_b(m_size, tile_size_m)
    tile_num_k = ceil_a_by_b(k_size, tile_size_k)
    tile_num_n = ceil_a_by_b(n_size, tile_size_n)
    k_input_size = ceil_a_by_b(k_size, tile_num_k)
    compute_cycles_per_tile = _get_systolic_array_cycles_os(m_size, tile_size_n, k_input_size, array_height, array_width)
    total_sram_conflict_stall = 0

    for m_idx in range(tile_num_m):
        for n_idx in range(tile_num_n):
            for k_idx in range(tile_num_k):
                current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
                current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
                current_tile_size_n = min(tile_size_n, n_size - n_idx * tile_size_n)
                num_index_groups = ceil_a_by_b(current_tile_size_k, vector_size)

                stats.reads["s_act"] += current_tile_size_k * current_tile_size_m * op.activation_tensor.nbits
                stats.reads["s_wgt"] += num_index_groups * current_tile_size_n * num_codebooks * index_nbits
                stats.reads["s_codebook"] += num_index_groups * current_tile_size_n * num_codebooks * vector_size * op.weight.nbits

                k_lo = k_idx * tile_size_k // vector_size
                n_lo = n_idx * tile_size_n
                k_hi = k_lo + num_index_groups
                n_hi = n_lo + current_tile_size_n
                tile = indices_full[k_lo:k_hi, n_lo:n_hi, :]
                mem_cycles = _memory_cycles_for_tile(tile, num_banks, entries_per_bank)
                total_sram_conflict_stall += max(0, mem_cycles - compute_cycles_per_tile)

            stats.reads["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
            stats.writes["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits

    stats.compute_cycles = _run_linear_dense(m_size, n_size, k_size, array_height, array_width, k_tile_size=1024)

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
                num_index_groups = ceil_a_by_b(current_buffer_tile_size_k, vector_size)
                index_bits = num_index_groups * current_buffer_tile_size_n * num_codebooks * index_nbits
                stats.reads["dram"] += index_bits
                stats.writes["s_wgt"] += index_bits
            if k_idx > 0:
                stats.reads["dram"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits
                stats.writes["s_output"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits

    codebook_bits = num_codebooks * codebook_entries * vector_size * op.weight.nbits
    stats.reads["dram"] += codebook_bits
    stats.writes["s_codebook"] += codebook_bits
    stats.reads["s_output"] += m_size * n_size * op.output_tensor.nbits
    stats.writes["dram"] += m_size * n_size * op.output_tensor.nbits

    first_buffer_tile_size_k = min(tile_size_k, k_size)
    first_buffer_tile_size_m = min(tile_size_m, m_size)
    first_buffer_tile_size_n = min(tile_size_n, n_size)
    init_mem_access = first_buffer_tile_size_k * first_buffer_tile_size_m * op.activation_tensor.nbits
    init_index_groups = ceil_a_by_b(first_buffer_tile_size_k, vector_size)
    init_mem_access += init_index_groups * first_buffer_tile_size_n * num_codebooks * index_nbits
    init_mem_access += codebook_bits
    stats.mem_stall_cycles += ceil_a_by_b(init_mem_access, mem_width)

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.mem_stall_cycles += total_sram_conflict_stall
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def _run_os_systolic_array(op: FC, stats: Stats, mem_width: int) -> Stats:
    stats.name = "os_sa"
    op.activation_tensor.nbits = 16
    op.weight.nbits = 16
    op.output_tensor.nbits = 16

    m_size = op.sequence_length
    k_size = op.input_dim
    n_size = op.output_dim
    array_height = 8
    array_width = 32

    tile_size_m = array_height
    tile_size_k = 1024
    tile_size_n = array_width
    buffer_size_m = 64
    buffer_size_k = 1024
    buffer_size_n = 128

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

    stats.compute_cycles = _run_linear_dense(m_size, n_size, k_size, array_height, array_width, k_tile_size=1024)

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
    stats.mem_stall_cycles += ceil_a_by_b(init_mem_access, mem_width)

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def _run_vq_replicated_core(
    op: FC,
    stats: Stats,
    indices_full: np.ndarray,
    replicated_sets: list[set[int]],
    mem_width: int,
) -> Stats:
    op.activation_tensor.nbits = 16
    op.weight.nbits = 16
    op.output_tensor.nbits = 16
    index_nbits = 8

    m_size = op.sequence_length
    k_size = op.input_dim
    n_size = op.output_dim
    array_height = 8
    array_width = 32
    codebook_entries = 256
    vector_size = 8
    num_codebooks = indices_full.shape[-1]
    num_banks = array_width // vector_size
    entries_per_bank = codebook_entries // num_banks

    tile_size_m = array_height
    tile_size_k = 1024
    tile_size_n = array_width
    buffer_size_m = 64
    buffer_size_k = 1024
    buffer_size_n = 128

    tile_num_m = ceil_a_by_b(m_size, tile_size_m)
    tile_num_k = ceil_a_by_b(k_size, tile_size_k)
    tile_num_n = ceil_a_by_b(n_size, tile_size_n)
    k_input_size = ceil_a_by_b(k_size, tile_num_k)
    compute_cycles_per_tile = _get_systolic_array_cycles_os(m_size, tile_size_n, k_input_size, array_height, array_width)
    total_sram_conflict_stall = 0

    for m_idx in range(tile_num_m):
        for n_idx in range(tile_num_n):
            for k_idx in range(tile_num_k):
                current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
                current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
                current_tile_size_n = min(tile_size_n, n_size - n_idx * tile_size_n)
                num_index_groups = ceil_a_by_b(current_tile_size_k, vector_size)

                stats.reads["s_act"] += current_tile_size_k * current_tile_size_m * op.activation_tensor.nbits
                stats.reads["s_wgt"] += num_index_groups * current_tile_size_n * num_codebooks * index_nbits
                stats.reads["s_codebook"] += num_index_groups * current_tile_size_n * num_codebooks * vector_size * op.weight.nbits

                k_lo = k_idx * tile_size_k // vector_size
                n_lo = n_idx * tile_size_n
                k_hi = k_lo + num_index_groups
                n_hi = n_lo + current_tile_size_n
                tile = indices_full[k_lo:k_hi, n_lo:n_hi, :]
                mem_cycles = _memory_cycles_for_tile_with_replication(tile, num_banks, entries_per_bank, replicated_sets)
                total_sram_conflict_stall += max(0, mem_cycles - compute_cycles_per_tile)

            stats.reads["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits
            stats.writes["s_output"] += current_tile_size_m * current_tile_size_n * op.output_tensor.nbits

    stats.compute_cycles = _run_linear_dense(m_size, n_size, k_size, array_height, array_width, k_tile_size=1024)

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
                num_index_groups = ceil_a_by_b(current_buffer_tile_size_k, vector_size)
                index_bits = num_index_groups * current_buffer_tile_size_n * num_codebooks * index_nbits
                stats.reads["dram"] += index_bits
                stats.writes["s_wgt"] += index_bits
            if k_idx > 0:
                stats.reads["dram"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits
                stats.writes["s_output"] += current_buffer_tile_size_m * current_buffer_tile_size_n * op.output_tensor.nbits

    codebook_bits = num_codebooks * codebook_entries * vector_size * op.weight.nbits
    stats.reads["dram"] += codebook_bits
    stats.writes["s_codebook"] += codebook_bits
    stats.reads["s_output"] += m_size * n_size * op.output_tensor.nbits
    stats.writes["dram"] += m_size * n_size * op.output_tensor.nbits

    first_buffer_tile_size_k = min(tile_size_k, k_size)
    first_buffer_tile_size_m = min(tile_size_m, m_size)
    first_buffer_tile_size_n = min(tile_size_n, n_size)
    init_mem_access = first_buffer_tile_size_k * first_buffer_tile_size_m * op.activation_tensor.nbits
    init_index_groups = ceil_a_by_b(first_buffer_tile_size_k, vector_size)
    init_mem_access += init_index_groups * first_buffer_tile_size_n * num_codebooks * index_nbits
    init_mem_access += codebook_bits
    stats.mem_stall_cycles += ceil_a_by_b(init_mem_access, mem_width)

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.mem_stall_cycles += total_sram_conflict_stall
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def _run_vq_llm_frequency_replicated(
    op: FC,
    stats: Stats,
    indices_full: np.ndarray,
    ratio: float,
    mem_width: int,
) -> Stats:
    stats.name = f"vq_llm_freq_rep_{ratio}"
    num_codebooks = indices_full.shape[-1]
    codebook_entries = 256
    num_replicated = int(codebook_entries * ratio)
    replicated_sets: list[set[int]] = []
    for codebook_idx in range(num_codebooks):
        unique, counts = np.unique(indices_full[:, :, codebook_idx], return_counts=True)
        sorted_indices = unique[np.argsort(-counts)]
        replicated_sets.append(set(sorted_indices[:num_replicated]))
    return _run_vq_replicated_core(op, stats, indices_full, replicated_sets, mem_width)


def _run_eva_decode(op: FC, stats: Stats, mem_width: int, adder_tree_size: int) -> Stats:
    stats.name = f"eva_adder_tree_{adder_tree_size}"
    array_height = 32
    array_width = 8

    m_size = op.sequence_length
    k_size = op.input_dim
    n_size = op.output_dim

    if m_size > 64:
        return stats

    tile_size_m = 2 ** int((m_size).bit_length() - 1)
    tile_size_n = n_size
    tile_size_codebook = 1
    tile_size_k = ceil_a_by_b(adder_tree_size * op.vector_size, m_size * tile_size_codebook)
    buffer_size_m = tile_size_m
    buffer_size_n = ceil_a_by_b(8 * 1024, m_size)
    buffer_size_k = ceil_a_by_b(8 * 1024, m_size)

    tile_num_m = ceil_a_by_b(m_size, tile_size_m)
    tile_num_k = ceil_a_by_b(k_size, tile_size_k)
    tile_num_codebook = ceil_a_by_b(op.num_codebooks, tile_size_codebook)

    first_vector_num = ceil_a_by_b(min(tile_size_k, k_size), op.vector_size)
    first_codebook_size = min(tile_size_codebook, op.num_codebooks)
    first_tile_size_m = min(tile_size_m, m_size)
    stats.compute_cycles += _get_systolic_array_cycles_is(
        array_height,
        op.num_entries * first_codebook_size,
        op.vector_size,
        array_height,
        array_width,
    ) * ceil_a_by_b(first_vector_num * first_tile_size_m, array_height)

    adder_tree_cycles = n_size
    for m_idx in range(tile_num_m):
        current_tile_size_m = min(tile_size_m, m_size - m_idx * tile_size_m)
        for k_idx in range(tile_num_k):
            current_tile_size_k = min(tile_size_k, k_size - k_idx * tile_size_k)
            vector_num = ceil_a_by_b(current_tile_size_k, op.vector_size)
            for codebook_idx in range(tile_num_codebook):
                current_tile_size_codebook = min(tile_size_codebook, op.num_codebooks - codebook_idx * tile_size_codebook)
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
                systolic_array_cycles = _get_systolic_array_cycles_is(
                    array_height,
                    current_tile_size_codebook * op.num_entries,
                    op.vector_size,
                    array_height,
                    array_width,
                ) * ceil_a_by_b(vector_num * current_tile_size_m, array_height)

                stats.reads["s_wgt"] += n_size * vector_num * op.index.nbits * current_tile_size_codebook
                stats.reads["s_cap"] += n_size * vector_num * op.activation_tensor.nbits * current_tile_size_codebook
                if codebook_idx > 0 or k_idx > 0:
                    stats.reads["s_output"] += current_tile_size_m * n_size * op.output_tensor.nbits
                stats.writes["s_output"] += current_tile_size_m * n_size * op.output_tensor.nbits

                if k_idx != tile_num_k - 1 or codebook_idx != tile_num_codebook - 1:
                    stats.compute_cycles += max(systolic_array_cycles, adder_tree_cycles)
                else:
                    stats.compute_cycles += adder_tree_cycles + 7

    stats.reads["dram"] += op.num_codebooks * op.vector_size * op.num_entries * op.codebooks.nbits
    stats.writes["s_codebook"] += op.num_codebooks * op.vector_size * op.num_entries * op.codebooks.nbits
    stats.reads["dram"] += n_size * op.activation_tensor.nbits
    stats.writes["s_scale"] += n_size * op.activation_tensor.nbits

    buffer_tile_num_m = ceil_a_by_b(m_size, buffer_size_m)
    buffer_tile_num_k = ceil_a_by_b(k_size, buffer_size_k)
    buffer_tile_num_n = ceil_a_by_b(n_size, buffer_size_n)
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
    stats.mem_stall_cycles += ceil_a_by_b(init_mem_access, mem_width)

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def _build_fc_from_indices(weight_type: str, sequence_length: int, indices_full: np.ndarray) -> FC:
    num_codebooks = indices_full.shape[-1]
    return FC(
        name=weight_type,
        sequence_length=sequence_length,
        input_dim=indices_full.shape[0] * 8,
        output_dim=indices_full.shape[1],
        num_bits=num_codebooks,
        algorithm="aqlm",
    )


class AblationPipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        del model_registry, method_registry
        output_dir = config.output_dir / study.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        area_csv_path = REPO_ROOT / "simulator" / "data" / "ablation_area_reference.csv"
        reference = pd.read_csv(area_csv_path)
        simulated_speed = self._simulate_speed_rows(config, study)
        dataframe = reference.drop(columns=["total_cycles", "compute_cycles", "mem_stall_cycles", "speedup_vs_os_sa"]).merge(
            simulated_speed,
            on="entry",
            how="left",
        )
        dataframe["workbook_label"] = dataframe["workbook_label"].fillna(dataframe["entry"])
        dataframe["area_source"] = "normalized_array_area"
        dataframe = dataframe[TABLE_VII_COLUMNS]

        table_csv = output_dir / "table_vii.csv"
        dataframe.to_csv(table_csv, index=False)

        cycles_csv = write_rows(dataframe.to_dict(orient="records"), output_dir / "cycles.csv", TABLE_VII_COLUMNS)
        area_csv = write_rows(dataframe[AREA_COLUMNS].to_dict(orient="records"), output_dir / "energy.csv", AREA_COLUMNS)
        speed_csv = write_rows(dataframe[SPEED_COLUMNS].to_dict(orient="records"), output_dir / "power.csv", SPEED_COLUMNS)

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=cycles_csv,
            energy_csv=area_csv,
            power_csv=speed_csv,
            verification_json=None,
            reports={"table_vii_csv": table_csv},
        )

    @staticmethod
    def _simulate_speed_rows(config: RunnerConfig, study: StudySpec) -> pd.DataFrame:
        model_name = str(study.extra.get("aqlm_model_name", "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf"))
        layer_number = int(study.extra.get("layer_number", 15))
        weight_types = list(
            study.extra.get(
                "weight_types",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        )
        sequence_length = int((config.sequence_lengths or list(study.sequence_lengths))[0])
        mem_width = int(study.extra.get("mem_width", config.mem_width))
        vq_llm_ratio = float(study.extra.get("vq_llm_ratio", 0.5))
        eva_adder_tree_sizes = {
            "EVA 4x8": int(study.extra.get("eva_eu4_adder_tree_size", 4)),
            "EVA 32x8": int(study.extra.get("eva_eu32_adder_tree_size", 32)),
            "EVA scale x 4": int(study.extra.get("eva_scale_x4_adder_tree_size", 128)),
        }

        model = load_quantized_model(model_name)
        totals = {
            "VQ_conflict": Stats(name="vq_conflict"),
            "VQ_LLM 0.5": Stats(name="vq_llm"),
            "OS_SA": Stats(name="vq_no_conflict"),
            "EVA 4x8": Stats(name="eva_eu4_x1"),
            "EVA 32x8": Stats(name="eva_eu32_x1"),
            "EVA scale x 4": Stats(name="eva_eu32_x4"),
        }

        for weight_type in weight_types:
            indices_full = extract_weight_indices(model, layer_number, weight_type)[:, :, 0:1]
            naive_stats = _run_naive_vq(_build_fc_from_indices(weight_type, sequence_length, indices_full), Stats(name="vq_conflict"), indices_full, mem_width)
            totals["VQ_conflict"].accumulate(naive_stats)

            vq_llm_stats = _run_vq_llm_frequency_replicated(
                _build_fc_from_indices(weight_type, sequence_length, indices_full),
                Stats(name="vq_llm"),
                indices_full,
                vq_llm_ratio,
                mem_width,
            )
            totals["VQ_LLM 0.5"].accumulate(vq_llm_stats)

            os_stats = _run_os_systolic_array(
                _build_fc_from_indices(weight_type, sequence_length, indices_full),
                Stats(name="os_sa"),
                mem_width,
            )
            totals["OS_SA"].accumulate(os_stats)

            for entry, adder_tree_size in eva_adder_tree_sizes.items():
                eva_stats = _run_eva_decode(
                    _build_fc_from_indices(weight_type, sequence_length, indices_full),
                    Stats(name=entry.lower().replace(" ", "_")),
                    mem_width,
                    adder_tree_size,
                )
                totals[entry].accumulate(eva_stats)

        baseline_cycles = float(totals["VQ_conflict"].total_cycles)
        records: list[dict[str, float | int | str]] = []
        for entry in ABLATION_ENTRY_ORDER:
            stats = totals[entry]
            records.append(
                {
                    "entry": entry,
                    "total_cycles": int(stats.total_cycles),
                    "compute_cycles": int(stats.compute_cycles),
                    "mem_stall_cycles": int(stats.mem_stall_cycles),
                    "speedup_vs_os_sa": baseline_cycles / float(stats.total_cycles),
                }
            )
        return pd.DataFrame(records)

