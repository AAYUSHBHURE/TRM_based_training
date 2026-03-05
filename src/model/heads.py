"""
TRM Output Heads

OutputHead: Projects latent to vocabulary logits for y_hat
QHead: Binary halting prediction for ACT (Adaptive Computation Time)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OutputHead(nn.Module):
    """
    Output head for solution prediction.
    
    Projects [B, L, D] -> [B, L, vocab_size] logits.
    """
    
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Refined solution embedding [B, L, D]
            
        Returns:
            Logits [B, L, vocab_size]
        """
        return self.proj(y)


class QHead(nn.Module):
    """
    Halting head for Adaptive Computation Time (ACT).
    
    Predicts whether current solution is correct (should halt).
    Projects [B, L, D] -> [B, L, 1] halting logits.
    """
    
    def __init__(self, dim: int, pool: str = "mean"):
        """
        Args:
            dim: Input dimension
            pool: Pooling strategy ("mean", "first", "max")
        """
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(dim, 1, bias=False)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Refined solution embedding [B, L, D]
            
        Returns:
            Halting logits [B, 1] (pre-sigmoid)
        """
        if self.pool == "mean":
            pooled = y.mean(dim=1)  # [B, D]
        elif self.pool == "first":
            pooled = y[:, 0]  # [B, D]
        elif self.pool == "max":
            pooled = y.max(dim=1).values  # [B, D]
        else:
            pooled = y.mean(dim=1)
        
        return self.proj(pooled)  # [B, 1]
    
    def predict_halt(self, y: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict whether to halt.
        
        Args:
            y: Refined solution embedding [B, L, D]
            threshold: Halting threshold
            
        Returns:
            Boolean tensor [B] indicating halt decision
        """
        logits = self.forward(y)
        probs = torch.sigmoid(logits).squeeze(-1)  # [B]
        return probs > threshold


class CombinedHead(nn.Module):
    """
    Combined output and halting head for efficiency.
    
    Single forward pass produces both predictions.
    """
    
    def __init__(self, dim: int, vocab_size: int, pool: str = "mean"):
        super().__init__()
        self.output_head = OutputHead(dim, vocab_size)
        self.q_head = QHead(dim, pool)
    
    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y: Refined solution embedding [B, L, D]
            
        Returns:
            y_hat: Output logits [B, L, vocab_size]
            q_hat: Halting logits [B, 1]
        """
        y_hat = self.output_head(y)
        q_hat = self.q_head(y)
        return y_hat, q_hat


if __name__ == "__main__":
    # Test heads
    dim, vocab_size = 512, 10
    batch_size, seq_len = 4, 81
    
    y = torch.randn(batch_size, seq_len, dim)
    
    output_head = OutputHead(dim, vocab_size)
    q_head = QHead(dim)
    combined = CombinedHead(dim, vocab_size)
    
    y_hat = output_head(y)
    q_hat = q_head(y)
    y_hat_c, q_hat_c = combined(y)
    
    print(f"OutputHead: {y.shape} -> {y_hat.shape}")
    print(f"QHead: {y.shape} -> {q_hat.shape}")
    print(f"CombinedHead: {y.shape} -> ({y_hat_c.shape}, {q_hat_c.shape})")
    
    # Test halt prediction
    halt = q_head.predict_halt(y)
    print(f"Halt decisions: {halt}")
