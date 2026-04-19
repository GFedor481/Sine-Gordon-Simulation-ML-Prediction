"""
model.py
--------
Spatiotemporal Transformer for sine-Gordon energy localization prediction.

Architecture (your trained model):
  - Input projection: 2 channels (KE, PE) -> d_model=128
  - 2D Rotary Positional Embedding (RoPE) on queries & keys
  - 6 learnable attention sink tokens prepended
  - 8 transformer blocks: 1 head, d_k=128, FFN=512, dropout=0.1
  - Dynamic temporal pooling (MLP + softmax over T timesteps)
  - Classification MLP -> 100 logits (one per pendulum)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------
class RotaryEmbedding2D(nn.Module):
    def __init__(self, d_k: int, T: int, P: int):
        super().__init__()
        assert d_k % 4 == 0, "d_k must be divisible by 4 for 2D RoPE"
        half    = d_k // 2
        quarter = half // 2

        inv_freq = 1.0 / (10000 ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))

        t_pos = torch.arange(T, dtype=torch.float32)
        p_pos = torch.arange(P, dtype=torch.float32)
        t_emb = torch.outer(t_pos, inv_freq)
        p_emb = torch.outer(p_pos, inv_freq)

        t_idx = torch.arange(T).repeat_interleave(P)
        p_idx = torch.arange(P).repeat(T)

        sin = torch.cat([torch.sin(t_emb[t_idx]), torch.sin(p_emb[p_idx])], dim=-1)
        cos = torch.cat([torch.cos(t_emb[t_idx]), torch.cos(p_emb[p_idx])], dim=-1)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q, k):
        sin = self.sin.unsqueeze(0).unsqueeze(0)
        cos = self.cos.unsqueeze(0).unsqueeze(0)
        half = sin.shape[-1]
        q_rot = q[..., :half] * cos + self._rotate_half(q[..., :half]) * sin
        k_rot = k[..., :half] * cos + self._rotate_half(k[..., :half]) * sin
        q = torch.cat([q_rot, q[..., half:]], dim=-1)
        k = torch.cat([k_rot, k[..., half:]], dim=-1)
        return q, k


# ---------------------------------------------------------------------------
# Single-Head Self-Attention with Sink Tokens + RoPE
# ---------------------------------------------------------------------------
class SinkAttention(nn.Module):
    """
    n_heads=1, d_k=128: single head with full embedding dimension.
    The 6 sink tokens absorb noisy/uninformative attention mass,
    keeping physical token attention maps clean and interpretable.
    """

    def __init__(self, d_model, n_heads, d_k, n_sinks, rope, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = d_k
        self.n_sinks = n_sinks
        self.rope    = rope

        self.Wq = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wo = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.sink_tokens = nn.Parameter(torch.randn(n_sinks, d_model) * 0.02)

    def forward(self, x, return_weights=False):
        B, L, D = x.shape
        S = self.n_sinks

        sinks  = self.sink_tokens.unsqueeze(0).expand(B, -1, -1)
        x_full = torch.cat([sinks, x], dim=1)   # (B, S+L, D)

        def split_heads(t):
            return t.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.Wq(x_full))
        k = split_heads(self.Wk(x_full))
        v = split_heads(self.Wv(x_full))

        # RoPE on physical tokens only
        q_phys, k_phys = self.rope(q[:, :, S:, :], k[:, :, S:, :])
        q = torch.cat([q[:, :, :S, :], q_phys], dim=2)
        k = torch.cat([k[:, :, :S, :], k_phys], dim=2)

        if return_weights:
            # Materialise full matrix only when needed for interpretability
            # (not during training — only called from interpret.py)
            scale  = math.sqrt(self.d_k)
            attn_w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
            attn_w = self.dropout(attn_w)
            out    = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(B, S+L, -1)
            out    = self.Wo(out[:, S:, :])
            return out, attn_w[:, :, S:, S:]
        else:
            # Flash attention — never materialises the (B, 1, 1000, 1000) matrix
            # Saves ~1 GB/layer = ~8 GB total at batch_size=256
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(B, S+L, -1)
            return self.Wo(out[:, S:, :])


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_ff, n_sinks, rope, dropout=0.1):
        super().__init__()
        self.attn  = SinkAttention(d_model, n_heads, d_k, n_sinks, rope, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_weights=False):
        if return_weights:
            attn_out, w = self.attn(self.norm1(x), return_weights=True)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x, w
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Dynamic Temporal Pooling
# ---------------------------------------------------------------------------
class DynamicTemporalPooling(nn.Module):
    def __init__(self, d_model, T):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.T = T

    def forward(self, x):
        B, LP, D = x.shape
        P       = LP // self.T
        x_grid  = x.view(B, self.T, P, D)
        weights = F.softmax(self.scorer(x_grid).squeeze(-1), dim=1)
        return (x_grid * weights.unsqueeze(-1)).sum(dim=1)  # (B, P, D)


# ---------------------------------------------------------------------------
# Full Spatiotemporal Transformer
# ---------------------------------------------------------------------------
class SineGordonTransformer(nn.Module):
    """
    Your trained configuration:
        d_model = 128  (embedding dim)
        n_heads = 1    (single head — full 128-dim attention)
        d_k     = 128  (= d_model since n_heads=1)
        d_ff    = 512  (4 * d_model)
        n_layers= 8
        n_sinks = 6    (absorb noisy attention)
        T=10, P=100
    """

    def __init__(
        self,
        T=10, P=100, n_channels=2,
        d_model=128, n_heads=1, d_k=128, d_ff=512,
        n_layers=8, n_sinks=6, dropout=0.1,
    ):
        super().__init__()
        self.T = T
        self.P = P
        self.d_model = d_model

        assert d_model == n_heads * d_k, \
            f"d_model ({d_model}) must equal n_heads ({n_heads}) * d_k ({d_k})"

        self.input_proj    = nn.Linear(n_channels, d_model)
        self.rope          = RotaryEmbedding2D(d_k, T, P)
        self.blocks        = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_k, d_ff, n_sinks, self.rope, dropout)
            for _ in range(n_layers)
        ])
        self.temporal_pool = DynamicTemporalPooling(d_model, T)
        self.classifier    = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, return_all_weights=False):
        B, T, P, C = x.shape
        h = self.input_proj(x).view(B, T * P, self.d_model)

        all_weights = []
        for block in self.blocks:
            if return_all_weights:
                h, w = block(h, return_weights=True)
                all_weights.append(w)
            else:
                h = block(h)

        logits = self.classifier(self.temporal_pool(h)).squeeze(-1)  # (B, P)

        if return_all_weights:
            return logits, all_weights
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = SineGordonTransformer()
    print(f"Parameters:        {model.count_parameters():,}")

    x = torch.randn(4, 10, 100, 2)
    logits = model(x)
    print(f"Output shape:      {logits.shape}")          # (4, 100)

    logits, weights = model(x, return_all_weights=True)
    print(f"Attention layers:  {len(weights)}")          # 8
    print(f"Weight shape (L0): {weights[0].shape}")      # (4, 1, 1000, 1000)