"""
TinyNet - Core TRM Architecture

Ultra-efficient 2-layer transformer with:
- RMSNorm (no bias)
- RoPE positional encoding  
- SwiGLU activation
- ~5M parameters target

Based on Algorithm 3 from TRM paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean centering)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute sin/cos
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to q and k."""
        # q, k: [B, L, H, D_head]
        seq_len = q.shape[1]
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :seq_len]
        sin = self.sin_cached[:, :seq_len]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish-Gated Linear Unit."""
    
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2 / 3 * 4 * in_dim)
        # Make hidden_dim multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)  # Gate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TinyAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, L, H, D_head]
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Attention: [B, H, L, L]
        q = q.transpose(1, 2)  # [B, H, L, D_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v  # [B, H, L, D_head]
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        
        return self.out_proj(out)


class TinyBlock(nn.Module):
    """Single transformer block: Attention + FFN with pre-norm."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = TinyAttention(dim, n_heads, dropout, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TinyNet(nn.Module):
    """
    TinyNet: 2-layer transformer for TRM recursion.
    
    Target: ~5M parameters for extreme efficiency.
    """
    
    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        vocab_size: int = 10,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        input_concat_dim: Optional[int] = None
    ):
        """
        Args:
            dim: Hidden dimension D
            n_layers: Number of transformer layers (default 2)
            n_heads: Number of attention heads
            vocab_size: Output vocabulary size
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            input_concat_dim: If set, project concatenated input to dim
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        # Optional input projection for concatenated inputs (x, y, z)
        if input_concat_dim:
            self.input_proj = nn.Linear(input_concat_dim, dim, bias=False)
        else:
            self.input_proj = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TinyBlock(dim, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through TinyNet.
        
        Args:
            x: Input tensor [B, L, D] or [B, L, concat_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [B, L, D]
        """
        # Project if input is concatenated
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Transformer layers
        for block in self.blocks:
            x = block(x, mask)
        
        return self.norm(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tiny_net(
    dim: int = 512,
    n_layers: int = 2,
    n_heads: int = 8,
    **kwargs
) -> TinyNet:
    """Factory function for TinyNet."""
    model = TinyNet(dim=dim, n_layers=n_layers, n_heads=n_heads, **kwargs)
    print(f"TinyNet created: {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_tiny_net(dim=512, n_layers=2)
    
    # Test forward
    x = torch.randn(2, 128, 512)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Test with concatenated input
    model_concat = TinyNet(dim=512, input_concat_dim=1536)  # 3 * 512
    print(f"TinyNet (concat): {model_concat.count_parameters():,} parameters")
    
    x_concat = torch.randn(2, 128, 1536)
    out_concat = model_concat(x_concat)
    print(f"Concat Input: {x_concat.shape} -> Output: {out_concat.shape}")
