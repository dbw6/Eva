import importlib
import inspect

import numpy as np


def patch_transformers_aqlm_loader() -> bool:
    """Bridge the AQLM loader API difference across transformers versions."""
    try:
        quantizer_aqlm_module = importlib.import_module("transformers.quantizers.quantizer_aqlm")
    except Exception:
        return False

    replace_fn = getattr(quantizer_aqlm_module, "replace_with_aqlm_linear", None)
    if replace_fn is None:
        return False

    try:
        signature = inspect.signature(replace_fn)
    except (TypeError, ValueError):
        return False

    if "linear_weights_not_to_quantize" in signature.parameters:
        return False

    def _compat_process_model_before_weight_loading(self, model, **kwargs):
        return replace_fn(
            model,
            modules_to_not_convert=self.quantization_config.linear_weights_not_to_quantize,
            quantization_config=self.quantization_config,
        )

    quantizer_aqlm_module.AqlmHfQuantizer._process_model_before_weight_loading = _compat_process_model_before_weight_loading
    return True


def load_quantized_model(model_name: str):
    from transformers import AutoModelForCausalLM
    import torch

    patch_transformers_aqlm_loader()
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
    try:
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception:
        pass
    model.eval()
    return model


def extract_weight_indices(model, layer_number: int, weight_type: str) -> np.ndarray:
    if weight_type in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        layer_module = model.model.layers[layer_number].self_attn
    elif weight_type in {"up_proj", "gate_proj", "down_proj"}:
        layer_module = model.model.layers[layer_number].mlp
    else:
        raise ValueError(f"Unsupported weight type: {weight_type}")
    weight_tensor = getattr(layer_module, weight_type).codes
    indices_full = weight_tensor.detach().cpu().numpy().astype(np.uint8)
    if indices_full.ndim == 3:
        indices_full = indices_full.transpose(1, 0, 2)
    return indices_full


def compute_average_index_counts(indices_full: np.ndarray) -> np.ndarray:
    n_rows, _, num_codebooks = indices_full.shape
    codebook_entries = 256
    all_counts = []
    for codebook_idx in range(num_codebooks):
        for row_idx in range(n_rows):
            row = indices_full[row_idx, :, codebook_idx].ravel()
            all_counts.append(np.bincount(row, minlength=codebook_entries))
    return np.mean(np.array(all_counts, dtype=np.float64), axis=0)


def compute_avg_unique_per_tile(indices_full: np.ndarray, tile_sizes: list[int]) -> list[tuple[int, float]]:
    n_rows, n_cols, num_codebooks = indices_full.shape
    results: list[tuple[int, float]] = []
    for tile_size in tile_sizes:
        if tile_size > n_cols:
            results.append((tile_size, float("nan")))
            continue
        unique_counts = []
        for codebook_idx in range(num_codebooks):
            for row_idx in range(n_rows):
                row = indices_full[row_idx, :, codebook_idx].ravel()
                num_full_tiles = n_cols // tile_size
                for tile_idx in range(num_full_tiles):
                    tile = row[tile_idx * tile_size : (tile_idx + 1) * tile_size]
                    unique_counts.append(np.unique(tile).size)
        results.append((tile_size, float(np.mean(unique_counts))))
    return results
