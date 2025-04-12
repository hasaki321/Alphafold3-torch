import torch
import math
import torch.nn.functional as F

def torch_attn(
    q,  # [batch, tokens_q, heads, dims]
    k,  # [batch, tokens_kv, heads, dims]
    v,  # [batch, tokens_kv, heads, dims]
    bias = None, # [batch, heads, tokens_q, tokens_kv]
    mask = None, # Boolean mask [batch, heads, tokens_q, tokens_kv] or similar
): # [batch, tokens_q, heads, dims]
    """Computes attention."""
    scaler = (1 / math.sqrt(q.shape[-1]))
    logits = torch.einsum('...qhc, ...khc->...hqk', q * scaler, k)  # ([bs], num_head, num_tokens, num_tokens)

    if bias is not None:
        logits += bias

    if mask is not None:
        mask_value = torch.finfo(logits.dtype).min # Minimum value for mask
        logits = torch.where(mask, logits, mask_value) # Apply mask using torch.where

    weights = F.softmax(logits, dim=-1) # Softmax

    weights = weights.to(v.dtype)
    out = torch.einsum("...hqk,...khd->...qhd", weights, v)
    return out

def F_attn(q, k, v, bias=None, mask=None):
    shaper = lambda x: x.transpose(-3, -2).continous() # ([bs], num_head, num_tokens, value_dim_per_head)
    q, k, v = shaper(q), shaper(k), shaper(v)

    if mask is not None:
        if bias is not None:
            bias = bias + (1e9 * (mask.to(bias.dtype) - 1.0)) # ([bs], 1, 1, num_tokens)
        else:
            bias = mask.bool()
    weights = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    return shaper(weights)