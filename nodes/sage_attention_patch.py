"""SageAttention patch for OmniVoice's Qwen3 LLM backbone.

Replaces Qwen3Attention.forward with a SageAttention-accelerated version.
Falls back to SDPA when attention masks are present (sageattn doesn't support masks).
"""

import logging

import torch

logger = logging.getLogger("OmniVoice")

# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------
SAGE_ATTENTION_AVAILABLE = False
SAGE_ATTENTION_FUNCTION = None
QK_QUANT_GRAN = "per_warp"
PV_ACCUM_DTYPE = "fp32"

try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )

    SAGE_ATTENTION_AVAILABLE = True

    # Select the best kernel for the current GPU architecture
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_code = major * 10 + minor

        if arch_code >= 120:  # Blackwell
            SAGE_ATTENTION_FUNCTION = sageattn_qk_int8_pv_fp8_cuda
            PV_ACCUM_DTYPE = "fp32+fp32"
            logger.info(f"SageAttention: SM{arch_code} (Blackwell) FP8 kernel selected.")
        elif arch_code >= 90:  # Hopper
            SAGE_ATTENTION_FUNCTION = sageattn_qk_int8_pv_fp8_cuda_sm90
            PV_ACCUM_DTYPE = "fp32+fp32"
            logger.info(f"SageAttention: SM{arch_code} (Hopper) FP8 kernel selected.")
        elif arch_code == 89:  # Ada Lovelace
            SAGE_ATTENTION_FUNCTION = sageattn_qk_int8_pv_fp8_cuda
            PV_ACCUM_DTYPE = "fp32+fp32"
            logger.info(f"SageAttention: SM{arch_code} (Ada) FP8 kernel selected.")
        elif arch_code >= 80:  # Ampere
            SAGE_ATTENTION_FUNCTION = sageattn_qk_int8_pv_fp16_cuda
            PV_ACCUM_DTYPE = "fp32"
            logger.info(f"SageAttention: SM{arch_code} (Ampere) FP16 kernel selected.")
        else:
            SAGE_ATTENTION_AVAILABLE = False
            logger.warning(
                f"SageAttention: GPU SM{arch_code} is below SM80. Not supported."
            )
    else:
        SAGE_ATTENTION_AVAILABLE = False
except ImportError:
    pass


def sage_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values = None,
    cache_position = None,
    **kwargs,
):
    """Drop-in replacement for Qwen3Attention.forward using SageAttention.

    Falls back to SDPA when attention_mask is present (sageattn doesn't
    support arbitrary masks) or when past_key_values are in use.
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

    # Check BEFORE the cache update — once past_key_values is updated the
    # variable is still not-None, so checking after would always skip sage.
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

    # sageattn does not support attention masks or KV-cache scenarios well.
    # Fall back to SDPA in those cases.
    use_sage = (
        attention_mask is None
        and not _use_cache
        and SAGE_ATTENTION_FUNCTION is not None
    )

    if use_sage:
        original_dtype = query_states.dtype
        target_dtype = (
            torch.bfloat16
            if hasattr(self.q_proj, "quant_state")
            else self.q_proj.weight.dtype
        )

        q = query_states.to(target_dtype)
        k = key_states.to(target_dtype)
        v = value_states.to(target_dtype)

        # SageAttention handles GQA (num_qo_heads divisible by num_kv_heads)
        # internally, so we do NOT repeat_kv here.
        # TTS generation is always causal — sliding window attention is still
        # causal (tokens attend only to past tokens within the window).
        is_causal = True

        attn_output = SAGE_ATTENTION_FUNCTION(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran=QK_QUANT_GRAN,
            pv_accum_dtype=PV_ACCUM_DTYPE,
        )

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


def set_sage_attention(model):
    """Monkey-patch all Qwen3Attention layers in the OmniVoice LLM backbone.

    Args:
        model: OmniVoice model instance (model.llm contains Qwen3Model)
    """
    if not SAGE_ATTENTION_AVAILABLE:
        raise ImportError(
            "SageAttention is not installed or GPU not supported.\n"
            "Install with: pip install sageattention\n"
            "Requires NVIDIA GPU with compute capability >= 8.0 (Ampere)."
        )

    if SAGE_ATTENTION_FUNCTION is None:
        logger.warning(
            "SageAttention: no compatible kernel found for this GPU. Skipping patch."
        )
        return

    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    patched_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.forward = sage_attention_forward.__get__(
                module, Qwen3Attention
            )
            patched_count += 1

    logger.info(f"SageAttention: patched {patched_count} Qwen3Attention layers.")
    if patched_count == 0:
        logger.warning(
            "SageAttention: no Qwen3Attention layers found in model."
        )
