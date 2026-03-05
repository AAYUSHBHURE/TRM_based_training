"""
TRM Recursion Logic

Implements Algorithm 3 from TRM paper:
- latent_recursion: n z-updates + 1 y-refine
- deep_recursion: T-1 no-grad warmups + 1 grad-enabled cycle
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .tiny_net import TinyNet
from .heads import CombinedHead


class LatentRecursion(nn.Module):
    """
    Single recursion cycle: n latent z-updates + 1 y-refine.
    
    z_new = net(cat([x, y, z])) for n steps
    y_new = net(cat([y, z]))
    
    Builds "reasoning memory" in z latent without hierarchy.
    """
    
    def __init__(
        self,
        net: TinyNet,
        dim: int = 512,
        n_latent: int = 6
    ):
        """
        Args:
            net: Shared TinyNet for all operations
            dim: Hidden dimension
            n_latent: Number of z-updates per cycle (default 6)
        """
        super().__init__()
        self.net = net
        self.dim = dim
        self.n_latent = n_latent
        
        # Projections for concatenation
        self.z_proj = nn.Linear(3 * dim, dim, bias=False)  # For z update: [x, y, z]
        self.y_proj = nn.Linear(2 * dim, dim, bias=False)  # For y refine: [y, z]
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recursion cycle.
        
        Args:
            x: Input conditioning [B, L, D]
            y: Current solution proposal [B, L, D]
            z: Latent reasoning state [B, L, D]
            
        Returns:
            y_new: Refined solution [B, L, D]
            z_new: Updated latent [B, L, D]
        """
        # n latent z-updates
        for _ in range(self.n_latent):
            # Concatenate [x, y, z] and project
            concat = torch.cat([x, y, z], dim=-1)  # [B, L, 3D]
            projected = self.z_proj(concat)  # [B, L, D]
            z = self.net(projected)
        
        # 1 y-refine
        concat_y = torch.cat([y, z], dim=-1)  # [B, L, 2D]
        projected_y = self.y_proj(concat_y)  # [B, L, D]
        y_new = self.net(projected_y)
        
        return y_new, z


class DeepRecursion(nn.Module):
    """
    Deep recursion wrapper with gradient control.
    
    Runs T-1 cycles without gradients (warmup), then 1 cycle with gradients.
    Enables training through 42+ effective layers without BPTT memory explosion.
    """
    
    def __init__(
        self,
        net: TinyNet,
        dim: int = 512,
        n_latent: int = 6,
        T_cycles: int = 3,
        vocab_size: int = 10
    ):
        """
        Args:
            net: Shared TinyNet
            dim: Hidden dimension
            n_latent: Latent steps per cycle
            T_cycles: Total recursion cycles (T-1 no-grad + 1 grad)
            vocab_size: Output vocabulary size
        """
        super().__init__()
        self.T_cycles = T_cycles
        
        self.latent_recursion = LatentRecursion(net, dim, n_latent)
        self.heads = CombinedHead(dim, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Deep recursion with gradient control.
        
        Args:
            x: Input conditioning [B, L, D]
            y: Initial solution proposal [B, L, D]
            z: Initial latent state [B, L, D]
            
        Returns:
            (y, z): Updated states
            y_hat: Output logits [B, L, vocab_size]
            q_hat: Halting logits [B, 1]
        """
        # T-1 cycles without gradients (warmup)
        with torch.no_grad():
            for _ in range(self.T_cycles - 1):
                y, z = self.latent_recursion(x, y, z)
        
        # Final cycle with gradients
        y, z = self.latent_recursion(x, y, z)
        
        # Compute outputs
        y_hat, q_hat = self.heads(y)
        
        # Detach states to prevent backprop through time across supervision steps
        # This is critical for the N_sup loop where we optimizer.step() at each iteration
        return (y.detach(), z.detach()), y_hat, q_hat
    
    def forward_all_cycles(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward with all cycles having gradients (for analysis/debugging).
        """
        intermediates = []
        
        for t in range(self.T_cycles):
            y, z = self.latent_recursion(x, y, z)
            if return_intermediates:
                intermediates.append((y.clone(), z.clone()))
        
        y_hat, q_hat = self.heads(y)
        
        if return_intermediates:
            return (y, z), y_hat, q_hat, intermediates
        return (y, z), y_hat, q_hat


class TRMModel(nn.Module):
    """
    Complete TRM Model combining all components.
    
    Encapsulates:
    - TinyNet backbone
    - Deep recursion logic
    - Output/halting heads
    - EMA tracking
    """
    
    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        n_latent: int = 6,
        T_cycles: int = 3,
        vocab_size: int = 10,
        max_seq_len: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Core network
        self.net = TinyNet(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Deep recursion
        self.deep_recursion = DeepRecursion(
            self.net,
            dim=dim,
            n_latent=n_latent,
            T_cycles=T_cycles,
            vocab_size=vocab_size
        )
        
        # Embedding for y_init
        self.y_embedding = nn.Embedding(vocab_size + 1, dim)  # +1 for padding
    
    def forward(
        self,
        x: torch.Tensor,
        y_init: torch.Tensor,
        z_init: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full TRM forward pass.
        
        Args:
            x: Input embedding [B, L, D]
            y_init: Initial solution embedding [B, L, D]
            z_init: Initial latent (zeros) [B, L, D]
            
        Returns:
            (y, z): Final states
            y_hat: Output logits [B, L, vocab_size]
            q_hat: Halting logits [B, 1]
        """
        return self.deep_recursion(x, y_init, z_init)
    
    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_trm_model(**kwargs) -> TRMModel:
    """Factory function for TRMModel."""
    model = TRMModel(**kwargs)
    print(f"TRMModel created: {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test complete model
    model = create_trm_model(dim=512, n_layers=2, n_latent=6, T_cycles=3)
    
    batch_size, seq_len, dim = 4, 81, 512
    x = torch.randn(batch_size, seq_len, dim)
    y_init = torch.randn(batch_size, seq_len, dim)
    z_init = torch.zeros(batch_size, seq_len, dim)
    
    (y, z), y_hat, q_hat = model(x, y_init, z_init)
    
    print(f"Input x: {x.shape}")
    print(f"Output y: {y.shape}")
    print(f"Output z: {z.shape}")
    print(f"y_hat (logits): {y_hat.shape}")
    print(f"q_hat (halt): {q_hat.shape}")
