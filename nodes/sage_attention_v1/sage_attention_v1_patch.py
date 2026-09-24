"""SageAttention V1 patch for OmniVoice's Qwen3 LLM backbone.

Replaces Qwen3Attention.forward with SageAttention V1 (Triton-based).
Supports AMD ROCm and older NVIDIA GPUs.

If V2 CUDA kernels are available, they take priority (handled by sage_attention_patch.py).
This module is ONLY loaded when V2 is NOT available.
"""

import logging

import torch

logger = logging.getLogger("OmniVoice")

# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------
SAGE_ATTN_V1_AVAILABLE = False
_sageattn_v1_func = None
_SAGE_V1_SMOOTH_K = True
_SAGE_V1_TARGET_DTYPE = None
_SAGE_V1_IS_CUDA = False

try:
    from sageattention import sageattn as _sageattn_v1_func

    # V1's sageattn signature: sageattn(q, k, v, is_causal=False, smooth_k=True)
    # Verify this is actually V1 by checking that V2-specific functions don't exist
    try:
        from sageattention.core import sageattn_qk_int8_pv_fp16_cuda
        # V2 is available — don't use V1 fallback
        SAGE_ATTN_V1_AVAILABLE = False
        logger.info("SageAttention V2 detected — V1 fallback not needed.")
    except ImportError:
        # V2 not available, V1 is available
        SAGE_ATTN_V1_AVAILABLE = True
        _SAGE_V1_SMOOTH_K = True
        _SAGE_V1_IS_CUDA = torch.cuda.is_available()

        if _SAGE_V1_IS_CUDA:
            major, minor = torch.cuda.get_device_capability()
            if torch.cuda.is_bf16_supported():
                _SAGE_V1_TARGET_DTYPE = torch.bfloat16
            else:
                _SAGE_V1_TARGET_DTYPE = torch.float16
            logger.info(f"SageAttention V1 detected (CUDA)")
        else:
            # ROCm / CPU — use float32 for V1 Triton
            _SAGE_V1_TARGET_DTYPE = torch.float32
            logger.info("SageAttention V1 detected (non-CUDA, using float32)")
except ImportError:
    pass
except Exception as e:
    logger.warning(f"SageAttention V1: import failed: {e}")

if _sageattn_v1_func is None:
    SAGE_ATTN_V1_AVAILABLE = False
    logger.info("SageAttention V1: not installed, skipping patch.")


# ---------------------------------------------------------------------------
# V1 attention forward
# ---------------------------------------------------------------------------
def _v1_sage_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Drop-in replacement for Qwen3Attention.forward using SageAttention V1.

    Falls back to SDPA when attention masks are present (sageattn doesn't
    support masks) or when past_key_values are in use.
    """
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Q/K projections with QKNorm (Qwen3-specific)
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # Check BEFORE the cache update — once past_key_values.update() runs
    # the variable is still not-None, so checking after always skips sage.
    _use_cache = past_key_values is not None
    if _use_cache:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # sageattn V1 does not support attention masks or KV-cache scenarios well.
    # Fall back to SDPA in those cases.
    use_sage = (
        attention_mask is None
        and not _use_cache
        and _sageattn_v1_func is not None
    )

    if use_sage:
        original_dtype = query_states.dtype
        q = query_states.to(_SAGE_V1_TARGET_DTYPE)
        k = key_states.to(_SAGE_V1_TARGET_DTYPE)
        v = value_states.to(_SAGE_V1_TARGET_DTYPE)

        # V1 handles GQA internally when num_key_value_groups > 1,
        # but we still expand for safety since V1's GQA handling
        # may not match V2's quality.
        if self.num_key_value_groups != 1:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # TTS generation is always causal
        is_causal = True
        attn_output = _sageattn_v1_func(
            q, k, v,
            is_causal=is_causal,
            smooth_k=_SAGE_V1_SMOOTH_K,
        )

        # Handle tuple return (V1 sometimes returns extra values)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = attn_output.to(original_dtype)
        attn_weights = None
    else:
        # Standard eager/SDPA path
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
        )
        attn_weights = None

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def set_sage_attention_v1(model):
    """Monkey-patch all Qwen3Attention layers using SageAttention V1.

    Args:
        model: OmniVoice model instance (model.llm contains Qwen3Model)
    """
    if not SAGE_ATTN_V1_AVAILABLE:
        raise ImportError(
            "SageAttention V1 is not installed or GPU not supported.\n"
            "Install with: pip install sageattention==1.06\n"
            "Note: Triton-based, may work on AMD ROCm and some older NVIDIA GPUs."
        )

    if _sageattn_v1_func is None:
        logger.warning(
            "SageAttention V1: no compatible function found. Skipping patch."
        )
        return

    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    patched_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.forward = _v1_sage_attention_forward.__get__(
                module, Qwen3Attention
            )
            patched_count += 1

    logger.info(f"SageAttention V1: patched {patched_count} Qwen3Attention layers.")
    if patched_count == 0:
        logger.warning(
            "SageAttention V1: no Qwen3Attention layers found in model."
        )
