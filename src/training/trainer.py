"""
TRM Trainer (Fixed)

Training loop with deep supervision - fixed gradient flow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from pathlib import Path
import time

from ..model import TRMModel


class TRMTrainer:
    """TRM Training loop."""
    
    def __init__(
        self,
        model: TRMModel,
        train_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        N_sup: int = 16,
        halt_threshold: float = 0.5,
        ema_beta: float = 0.999,
        use_amp: bool = True,
        device: str = "cuda",
        output_dir: str = "outputs",
        **kwargs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.N_sup = N_sup
        self.halt_threshold = halt_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding for tokens
        self.embedding = nn.Embedding(20, model.dim).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(self.embedding.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # AMP
        self.use_amp = use_amp and device == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # EMA
        self.ema_beta = ema_beta
        self.ema_params = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.ema_params[name] = p.data.clone()
        
        self.global_step = 0
    
    def update_ema(self):
        for name, p in self.model.named_parameters():
            if name in self.ema_params:
                self.ema_params[name] = (
                    self.ema_beta * self.ema_params[name] + 
                    (1 - self.ema_beta) * p.data
                )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict]:
        """Single training step with N_sup Deep Supervision."""
        self.model.train()
        
        x_tokens = batch["x_tokens"].to(self.device)
        y_init_tokens = batch["y_init_tokens"].to(self.device)
        y_true = batch["y_true"].to(self.device)
        
        # Initial states
        # y_init is embedding of initial guess (usually zeros/clues)
        y = self.embedding(y_init_tokens)
        z = torch.zeros_like(y)
        
        total_loss = 0.0
        final_metrics = {}
        
        # Deep Supervision Loop
        for step in range(self.N_sup):
            # 1. Re-embed x with latest weights (since opt.step() happens inside loop)
            x = self.embedding(x_tokens)
            
            # 2. Forward pass (Gradient flows through this step's computation only)
            # The model internally detaches y and z for the next step, but returns
            # attached versions if we needed them. Wait, DeepRecursion returns detached!
            # So y, z are already detached for the NEXT loop iteration.
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                (y_next, z_next), y_hat, q_hat = self.model.deep_recursion(x, y, z)
                
                # Update states for next iteration (detached by model)
                y, z = y_next, z_next
                
                # CE loss
                loss_ce = nn.functional.cross_entropy(
                    y_hat.view(-1, y_hat.size(-1)),
                    y_true.view(-1),
                    ignore_index=0
                )
                
                # ACT (Halting) loss
                with torch.no_grad():
                    preds = y_hat.argmax(-1)
                    # Accuracy per sample [B]
                    acc = (preds == y_true).float().mean(dim=-1)
                    # Target: 1 if solved (acc > 0.99), 0 otherwise
                    halt_target = (acc > 0.99).float().unsqueeze(-1)
                
                loss_halt = nn.functional.binary_cross_entropy_with_logits(q_hat, halt_target)
                step_loss = loss_ce + 0.5 * loss_halt
            
            # 3. Optimize immediately (Iterative Refinement)
            if self.use_amp:
                self.scaler.scale(step_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                step_loss.backward()
                self.optimizer.step()
                
            self.update_ema()
            total_loss += step_loss.item()
            
            # 4. Metrics & Early Stopping
            with torch.no_grad():
                step_acc = (preds == y_true).float().mean().item()
                step_halt = torch.sigmoid(q_hat).mean().item()
                
                # If mean halting probability is high, we can stop supervision early?
                # Paper suggests instance-wise early exit, but for batch training 
                # we usually run fixed steps or wait for batch consensus.
                # For simplicity, we run N_sup steps unless average halt > 0.95
                if step_halt > 0.95:
                    break
        
        final_metrics = {
            "accuracy": step_acc,
            "halt_rate": step_halt,
            "steps": step + 1
        }
        
        return total_loss / (step + 1), final_metrics
    
    def train(
        self,
        max_iters: int = 50000,
        log_interval: int = 100,
        save_interval: int = 1000,
        **kwargs
    ):
        """Main training loop."""
        print(f"Starting training for {max_iters} iterations")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        train_iter = iter(self.train_loader)
        start = time.time()
        
        for step in range(max_iters):
            self.global_step = step
            
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            loss, metrics = self.train_step(batch)
            
            if step % log_interval == 0:
                elapsed = time.time() - start
                print(f"Step {step}/{max_iters} | Loss: {loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.3f} | Time: {elapsed:.1f}s")
            
            if step % save_interval == 0 and step > 0:
                self.save_checkpoint(f"step_{step}.pt")
        
        self.save_checkpoint("final.pt")
        print("Training complete!")
    
    def save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            "model": self.model.state_dict(),
            "embedding": self.embedding.state_dict(),
            "ema": self.ema_params,
            "step": self.global_step,
        }, path)
        print(f"Saved: {path}")
    
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.embedding.load_state_dict(ckpt["embedding"])
        self.ema_params = ckpt.get("ema", {})
        self.global_step = ckpt.get("step", 0)
