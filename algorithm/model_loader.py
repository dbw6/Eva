"""Load AQLM-quantized HuggingFace models with compatibility fixes for transformers >= 5.x.

Fixes two issues introduced by transformers >= 5.0:

1. **lm_head bug**: layers listed in ``linear_weights_not_to_quantize`` are
   incorrectly replaced with QuantizedLinear, discarding original weight tensors.
   Detected post-load and patched back to nn.Linear.

2. **Fused-expert MoE bug**: MoE models (Mixtral, Qwen3MoE) now store expert
   weights as fused 3D tensors (``gate_up_proj``, ``down_proj``), but AQLM
   checkpoints use individual per-expert layers (``experts.{i}.gate_proj``, etc.)
   whose codes/codebooks/scales cannot be fused.  We monkey-patch the Experts
   class to use an unfused ``nn.ModuleList`` *before* calling ``from_pretrained``,
   so parameter names match the checkpoint and AQLM integration works correctly.
"""

import glob
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN


# ---------------------------------------------------------------------------
# Unfused expert replacement for MoE models
# ---------------------------------------------------------------------------

class _ExpertMLP(nn.Module):
    """Single expert MLP with individual gate/up/down projections.

    Args:
        gate_name: attribute name for the gate projection (e.g. "gate_proj" or "w1")
        up_name:   attribute name for the up projection   (e.g. "up_proj" or "w3")
        down_name: attribute name for the down projection (e.g. "down_proj" or "w2")
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        gate_name: str = "gate_proj",
        up_name: str = "up_proj",
        down_name: str = "down_proj",
    ):
        super().__init__()
        self._gate_name = gate_name
        self._up_name = up_name
        self._down_name = down_name
        setattr(self, gate_name, nn.Linear(hidden_dim, intermediate_dim, bias=False))
        setattr(self, up_name, nn.Linear(hidden_dim, intermediate_dim, bias=False))
        setattr(self, down_name, nn.Linear(intermediate_dim, hidden_dim, bias=False))
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = getattr(self, self._gate_name)
        up = getattr(self, self._up_name)
        down = getattr(self, self._down_name)
        return down(self.act_fn(gate(x)) * up(x))


class _UnfusedExperts(nn.Module):
    """Drop-in replacement for fused *Experts classes using per-expert ModuleList.

    Parameter names follow ``{i}.<gate_name>.*``, etc., matching the AQLM
    checkpoint layout (e.g. ``gate_proj``/``up_proj``/``down_proj`` for Qwen3,
    ``w1``/``w3``/``w2`` for Mixtral).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        gate_name: str = "gate_proj",
        up_name: str = "up_proj",
        down_name: str = "down_proj",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        for i in range(num_experts):
            self.add_module(
                str(i),
                _ExpertMLP(hidden_dim, intermediate_dim, act_fn, gate_name, up_name, down_name),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for idx in expert_hit:
            expert_idx = idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert_module = getattr(self, str(expert_idx.item()))
            current_hidden_states = expert_module(current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def _make_unfused_qwen3_experts(config):
    """Create unfused experts matching Qwen3MoeExperts interface."""
    return _UnfusedExperts(
        num_experts=config.num_experts,
        hidden_dim=config.hidden_size,
        intermediate_dim=config.moe_intermediate_size,
        act_fn=ACT2FN[config.hidden_act],
    )


def _make_unfused_mixtral_experts(config):
    """Create unfused experts matching MixtralExperts interface.

    Mixtral checkpoints use w1 (gate), w3 (up), w2 (down) naming.
    """
    return _UnfusedExperts(
        num_experts=config.num_local_experts,
        hidden_dim=config.hidden_size,
        intermediate_dim=config.intermediate_size,
        act_fn=ACT2FN[config.hidden_act],
        gate_name="w1",
        up_name="w3",
        down_name="w2",
    )


@contextmanager
def _patch_moe_experts():
    """Temporarily replace fused Experts classes with unfused versions.

    Must be applied *before* ``from_pretrained()`` so the model is constructed
    with per-expert ModuleList modules whose parameter names align with the
    AQLM checkpoint.
    """
    patches = []

    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as qm
        orig_qwen3_block_init = qm.Qwen3MoeSparseMoeBlock.__init__

        def _qwen3_block_init(self, config):
            nn.Module.__init__(self)
            self.experts = _make_unfused_qwen3_experts(config)
            self.gate = qm.Qwen3MoeTopKRouter(config)

        qm.Qwen3MoeSparseMoeBlock.__init__ = _qwen3_block_init
        patches.append((qm.Qwen3MoeSparseMoeBlock, "__init__", orig_qwen3_block_init))
    except (ImportError, AttributeError):
        pass

    try:
        from transformers.models.mixtral import modeling_mixtral as mm
        orig_mixtral_block_init = mm.MixtralSparseMoeBlock.__init__

        def _mixtral_block_init(self, config):
            nn.Module.__init__(self)
            self.top_k = config.num_experts_per_tok
            self.jitter_noise = config.router_jitter_noise
            self.gate = mm.MixtralTopKRouter(config)
            self.experts = _make_unfused_mixtral_experts(config)

        mm.MixtralSparseMoeBlock.__init__ = _mixtral_block_init
        patches.append((mm.MixtralSparseMoeBlock, "__init__", orig_mixtral_block_init))
    except (ImportError, AttributeError):
        pass

    try:
        yield
    finally:
        for cls, attr, orig in patches:
            setattr(cls, attr, orig)


# ---------------------------------------------------------------------------
# lm_head fix
# ---------------------------------------------------------------------------

def _is_aqlm_quantized(model) -> bool:
    """Check if a loaded model uses AQLM quantization."""
    qconfig = getattr(getattr(model, "config", None), "quantization_config", None)
    if qconfig is None:
        return False
    qt = qconfig.get("quant_method", "") if isinstance(qconfig, dict) else getattr(qconfig, "quant_method", "")
    return qt == "aqlm"


def _fix_unquantized_layers(model, model_path: str) -> None:
    """Replace QuantizedLinear layers that should have been regular Linear.

    Walks the model looking for layers whose names appear in the config's
    ``linear_weights_not_to_quantize`` and are incorrectly quantized.  For
    each such layer, loads the original weight from the checkpoint and
    swaps the QuantizedLinear with a regular nn.Linear.

    This handles both the ``lm_head`` bug and unquantized MoE experts
    (e.g. expert 64 in certain Qwen3-30B-A3B layers).
    """
    config = model.config
    qconfig = getattr(config, "quantization_config", None)
    if qconfig is None:
        return

    if isinstance(qconfig, dict):
        skip_names = qconfig.get("linear_weights_not_to_quantize", [])
    else:
        skip_names = getattr(qconfig, "linear_weights_not_to_quantize", [])

    if not skip_names:
        return

    local_path = snapshot_download(model_path)
    st_files = sorted(glob.glob(os.path.join(local_path, "*.safetensors")))

    skip_set = set(skip_names)
    weight_map: dict[str, torch.Tensor] = {}
    for sf in st_files:
        with safe_open(sf, framework="pt") as f:
            for key in f.keys():
                if key in skip_set:
                    weight_map[key] = f.get_tensor(key)

    for full_name in skip_names:
        if full_name not in weight_map:
            continue

        parts = full_name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        parent_path, attr_name = parts[0], parts[1]
        if attr_name != "weight":
            continue
        module_path = parent_path

        try:
            parent = model
            for part in module_path.split("."):
                parent = getattr(parent, part)
        except AttributeError:
            continue

        if not hasattr(parent, "codes"):
            continue

        w = weight_map[full_name]
        device = next(parent.parameters()).device
        new_linear = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype, device=device)
        new_linear.weight.data.copy_(w.to(device))

        module_parts = module_path.split(".")
        container = model
        for part in module_parts[:-1]:
            container = getattr(container, part)
        setattr(container, module_parts[-1], new_linear)
        print(f"  Fixed {module_path}: QuantizedLinear -> Linear ({w.shape})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_aqlm_moe(model_path: str) -> bool:
    """Check if the model is an AQLM-quantized MoE model."""
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        qconfig = getattr(config, "quantization_config", None)
        is_moe = model_type in ("mixtral", "qwen3_moe")
        is_aqlm = False
        if qconfig is not None:
            qt = qconfig.get("quant_method", "") if isinstance(qconfig, dict) else getattr(qconfig, "quant_method", "")
            is_aqlm = qt == "aqlm"
        return is_moe and is_aqlm
    except Exception:
        return False


def load_aqlm_model(model_path: str, **kwargs):
    """Load an AQLM model, automatically fixing transformers >= 5.x issues.

    Applies two patches when needed:
    - Unfused MoE experts (so per-expert AQLM codes/codebooks/scales load correctly)
    - lm_head de-quantization (restoring the original dense weight)
    """
    defaults = dict(
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    defaults.update(kwargs)

    print(f"Loading model: {model_path}")

    use_moe_patch = _is_aqlm_moe(model_path)
    if use_moe_patch:
        print("  Applying unfused-experts patch for AQLM MoE model...")
        with _patch_moe_experts():
            model = AutoModelForCausalLM.from_pretrained(model_path, **defaults)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **defaults)

    if _is_aqlm_quantized(model):
        _fix_unquantized_layers(model, model_path)

    model.eval()
    return model
