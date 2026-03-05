"""
TRM Training Losses

CE Loss: Cross-entropy for solution accuracy
ACT Loss: Binary cross-entropy for halting prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TRMLoss(nn.Module):
    """
    Combined TRM training loss.
    
    L = CE(y_hat, y_true) + alpha * BCE(q_hat, halt_target)
    
    Where halt_target = 1 if y_hat matches y_true, else 0.
    """
    
    def __init__(
        self,
        act_weight: float = 0.5,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            act_weight: Weight for ACT loss (default 0.5)
            ignore_index: Index to ignore in CE loss (padding)
            label_smoothing: Label smoothing for CE
        """
        super().__init__()
        self.act_weight = act_weight
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        q_hat: torch.Tensor,
        seq_len: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            y_hat: Predicted logits [B, L, vocab_size]
            y_true: Ground truth [B, L]
            q_hat: Halting logits [B, 1]
            seq_len: Valid sequence lengths [B] (optional)
            
        Returns:
            loss: Combined loss scalar
            metrics: Dict with individual losses
        """
        B, L, V = y_hat.shape
        
        # CE loss: flatten for cross entropy
        y_hat_flat = y_hat.view(-1, V)  # [B*L, V]
        y_true_flat = y_true.view(-1)  # [B*L]
        ce = self.ce_loss(y_hat_flat, y_true_flat)
        
        # Compute accuracy for halt target
        with torch.no_grad():
            predictions = y_hat.argmax(dim=-1)  # [B, L]
            
            if seq_len is not None:
                # Masked accuracy
                correct = (predictions == y_true).float()
                mask = torch.arange(L, device=y_true.device)[None, :] < seq_len[:, None]
                accuracy = (correct * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            else:
                accuracy = (predictions == y_true).float().mean(dim=-1)
            
            # Halt target: 1 if fully correct, threshold at 0.95
            halt_target = (accuracy > 0.95).float().unsqueeze(-1)  # [B, 1]
        
        # ACT loss
        act = F.binary_cross_entropy_with_logits(q_hat, halt_target)
        
        # Combined
        loss = ce + self.act_weight * act
        
        metrics = {
            "ce_loss": ce.item(),
            "act_loss": act.item(),
            "total_loss": loss.item(),
            "accuracy": accuracy.mean().item(),
            "halt_rate": torch.sigmoid(q_hat).mean().item()
        }
        
        return loss, metrics


class SudokuLoss(TRMLoss):
    """Specialized loss for Sudoku with constraint checking."""
    
    def __init__(self, act_weight: float = 0.5):
        super().__init__(act_weight=act_weight, ignore_index=0)  # Ignore padding (0)
    
    def check_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Check Sudoku constraints (rows, cols, boxes).
        
        Args:
            predictions: [B, 81] predicted grid values (1-9)
            
        Returns:
            constraint_violations: [B] count of violations
        """
        B = predictions.shape[0]
        grid = predictions.view(B, 9, 9)
        violations = torch.zeros(B, device=predictions.device)
        
        for i in range(9):
            # Row violations
            row = grid[:, i, :]
            violations += 9 - len(torch.unique(row, dim=-1))
            
            # Col violations
            col = grid[:, :, i]
            violations += 9 - len(torch.unique(col, dim=-1))
        
        # Box violations
        for bi in range(3):
            for bj in range(3):
                box = grid[:, bi*3:(bi+1)*3, bj*3:(bj+1)*3].reshape(B, 9)
                violations += 9 - len(torch.unique(box, dim=-1))
        
        return violations


if __name__ == "__main__":
    # Test loss computation
    batch_size, seq_len, vocab_size = 4, 81, 10
    
    y_hat = torch.randn(batch_size, seq_len, vocab_size)
    y_true = torch.randint(0, vocab_size, (batch_size, seq_len))
    q_hat = torch.randn(batch_size, 1)
    
    loss_fn = TRMLoss()
    loss, metrics = loss_fn(y_hat, y_true, q_hat)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
