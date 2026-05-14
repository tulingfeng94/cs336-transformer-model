import math

import einops
import os
from typing import Any, BinaryIO, IO
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from torch.optim import Optimizer

# building blocks

class Linear(nn.Module):                                                                                                                                                                                                                       
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()                                                           
        self.weights = nn.Parameter(                                                                                                                                                                                                                 
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )                                                                                                                                                                                                                                      
        nn.init.kaiming_uniform_(self.weights, a=5**0.5)
                                                
    def forward(self, x: torch.Tensor) -> torch.Tensor:                                                                                                                                                                                        
        return einops.einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weights

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)
        nn.init.kaiming_uniform_(self.w3, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(W1 x) ⊙ (W3 x), then W2; all gate activations live in d_ff.
        a = einops.einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        b = einops.einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        hidden = torch.nn.functional.silu(a) * b
        return einops.einsum(self.w2, hidden, "d_model d_ff, ... d_ff -> ... d_model")

class RoPE(nn.Module):
    """Rotary positional embeddings; constructed as ``RoPE(theta, d_k, max_seq_len)`` (see ``run_rope``)."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        idx = torch.arange(0, d_k, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (idx / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq, d_k); token_positions: (..., seq) integer positions
        inv = self.inv_freq.to(device=x.device, dtype=x.dtype)
        angles = token_positions[..., None].to(dtype=x.dtype) * inv
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)

def Softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:                                                                                                                                                                              
        # Subtract max for numerical stability (prevents exp overflow)
        x = in_features - in_features.max(dim=dim, keepdim=True).values                                                                                                                                                                            
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

def ScaledDotProductAttention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = einops.einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))
    weights = Softmax(scores, dim=-1)
    return einops.einsum(weights, V, "... q k, ... k d_v -> ... q d_v")

def MultiheadSelfAttention(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Batched multi-head self-attention.

    Supports two call shapes from ``tests/adapters``:

    - ``(d_model, num_heads, q_w, k_w, v_w, o_w, in_features)``
    - ``(d_model, num_heads, max_seq_len, theta, q_w, k_w, v_w, o_w, in_features, token_positions)``
      (RoPE on Q/K; head dim must match ``d_model // num_heads``).
    """
    d_k = d_model // num_heads
    rope = RoPE(float(theta), d_k, max_seq_len)
    q = einops.einsum(q_proj_weight, in_features, "d_out d_in, ... s d_in -> ... s d_out")
    k = einops.einsum(k_proj_weight, in_features, "d_out d_in, ... s d_in -> ... s d_out")
    v = einops.einsum(v_proj_weight, in_features, "d_out d_in, ... s d_in -> ... s d_out")
    q = einops.rearrange(q, "... s (h d) -> ... h s d", h=num_heads, d=d_k)
    k = einops.rearrange(k, "... s (h d) -> ... h s d", h=num_heads, d=d_k)
    v = einops.rearrange(v, "... s (h d) -> ... h s d", h=num_heads, d=d_k)
    if rope is not None and token_positions is not None:
        q = rope(q, token_positions)
        k = rope(k, token_positions)
    scores = einops.einsum(q, k, "... h q d, ... h k d -> ... h q k") / math.sqrt(d_k)
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.tril(
            torch.ones(q_len, k_len, device=scores.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, -float("inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask[..., None, :, :], -float("inf"))
    weights = Softmax(scores, dim=-1)
    out = einops.einsum(weights, v, "... h q k, ... h k d -> ... h q d")
    out = einops.rearrange(out, "... h s d -> ... s (h d)")
    return einops.einsum(o_proj_weight, out, "d_out d_in, ... s d_in -> ... s d_out")

def build_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_features: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Pre-norm block: x + MHA(RMSN₁(x)); then h + FFN(RMSN₂(h)), with RoPE in MHA."""
    x = in_features
    seq_len = x.shape[-2]
    pos_1d = torch.arange(seq_len, device=x.device, dtype=torch.long)
    token_positions = pos_1d.view((1,) * (x.ndim - 2) + (seq_len,)).expand(*x.shape[:-1])

    w_ln1 = weights["ln1.weight"]
    rms_norm_1 = RMSNorm(d_model, device=w_ln1.device, dtype=w_ln1.dtype)
    rms_norm_1.weights.data.copy_(w_ln1)
    normed_1 = rms_norm_1(x)

    attn_out = MultiheadSelfAttention(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        normed_1,
        token_positions,
        mask=None,
        causal=causal,
    )
    h = x + attn_out

    w_ln2 = weights["ln2.weight"]
    rms_norm_2 = RMSNorm(d_model, device=w_ln2.device, dtype=w_ln2.dtype)
    rms_norm_2.weights.data.copy_(w_ln2)
    normed_2 = rms_norm_2(h)

    w1, w2, w3 = weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"]
    ffn = SwiGLU(d_model, d_ff, device=w1.device, dtype=w1.dtype)
    ffn.w1.data.copy_(w1)
    ffn.w2.data.copy_(w2)
    ffn.w3.data.copy_(w3)

    return h + ffn(normed_2)

def _layer_weights(weights: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    """Extract ``layers.{layer}.*`` without ``startswith('layers.1.')`` matching ``layers.10``."""
    prefix = f"layers.{layer}."
    return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}


def build_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: torch.Tensor,
) -> torch.Tensor:
    _ = vocab_size
    tok = weights["token_embeddings.weight"]
    x = tok[in_indices.to(device=tok.device)]

    for layer in range(num_layers):
        x = build_transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            _layer_weights(weights, layer),
            x,
            causal=False,
        )

    w_final = weights["ln_final.weight"]
    ln_final = RMSNorm(d_model, device=w_final.device, dtype=w_final.dtype)
    ln_final.weights.data.copy_(w_final)
    x = ln_final(x)

    return torch.nn.functional.linear(x, weights["lm_head.weight"])

# Training
        
def CrossEntropyLoss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x = inputs - inputs.max(dim=-1, keepdim=True).values
    logits = x[torch.arange(targets.size(0)), targets]
    logits_sum = torch.log(torch.exp(x).sum(dim=-1))

    return (logits_sum - logits).mean()

class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # 1st moment
                    state["v"] = torch.zeros_like(p.data)  # 2nd moment

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # Update biased moments
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias correction
                alpha = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)

                # Update parameters
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha)  # Adam step
                p.data.add_(p.data, alpha=-lr * wd)  # weight decay

        return loss

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    hi = len(dataset) - context_length
    starts = np.random.randint(0, hi, size=(batch_size,))
    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

def save_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, sr
    