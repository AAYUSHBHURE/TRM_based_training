# ============================================================
# CORRECTED TRM TRAINING - Based on Paper Algorithm 3
# Replace your training cells with thisthis code
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import time
import os

# ============================================================
# 1. MODEL ARCHITECTURE (2 LAYERS - NOT 4!)
# ============================================================

# Use the SAME TinyNet, LatentRecursion, TRMModel classes
# But change CONFIG to use 2 layers:

CONFIG = {
    # Model Architecture - CORRECTED
    "dim": 512,
    "n_layers": 2,           # ✓ 2 layers (not 4!)
    "n_heads": 8,
    "n_latent": 6,           # ✓ n=6 latent updates
    "T_cycles": 3,           # ✓ T=3 (2 warmup + 1 grad)
    "vocab_size": 10,
    "max_seq_len": 128,
    "dropout": 0.1,
    
    # Training
    "lr": 3e-4,
    "weight_decay": 0.01,
    "max_iters": 20000,
    "batch_size": 32,
    "warmup_steps": 1000,
    "grad_clip": 1.0,
    
    # Deep Supervision
    "N_sup": 16,             # ✓ Max supervision steps
    "ema_decay": 0.999,      # ✓ EMA for stability
    
    # Checkpointing
    "save_every": 2000,
    "log_every": 100,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Expected training time on A100: ~60-90 minutes")

# ============================================================
# 2. INITIALIZE MODEL (2 LAYERS!)
# ============================================================

model = TRMModel(
    dim=CONFIG["dim"],
    n_layers=2,              # ✓ ONLY 2 LAYERS!
    n_heads=CONFIG["n_heads"],
    n_latent=CONFIG["n_latent"],
    T_cycles=CONFIG["T_cycles"],
    vocab_size=CONFIG["vocab_size"],
    max_seq_len=CONFIG["max_seq_len"],
    dropout=CONFIG["dropout"]
).to(device)

embedding = nn.Embedding(20, CONFIG["dim"]).to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(embedding.parameters()),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]
)

def lr_lambda(step):
    if step < CONFIG["warmup_steps"]:
        return step / CONFIG["warmup_steps"]
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

# ============================================================
# 3. ADD EMA (CRITICAL FOR STABILITY!)
# ============================================================

class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                    (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

ema_model = EMA(model, decay=CONFIG["ema_decay"])
ema_embedding = EMA(embedding, decay=CONFIG["ema_decay"])

print(f"Model parameters: {model.count_parameters():,}")
print(f"Embedding parameters: {sum(p.numel() for p in embedding.parameters()):,}")
print(f"✓ EMA enabled with decay={CONFIG['ema_decay']}")

# ============================================================
# 4. CORRECTED TRAINING LOOP WITH DEEP SUPERVISION
# ============================================================

os.makedirs('/content/drive/MyDrive/TRM', exist_ok=True)

print(f"\nStarting CORRECTED TRM training for {CONFIG['max_iters']} iterations...")
print("Following paper Algorithm 3:")
print(f"  - n={CONFIG['n_latent']} latent updates")
print(f"  - T={CONFIG['T_cycles']} cycles (T-1 warmup + 1 grad)")
print(f"  - N_sup={CONFIG['N_sup']} max supervision steps")
print(f"  - 2-layer tiny network")
print(f"  - EMA for stability")
print("="*60)

model.train()
train_iter = iter(dataloader)
start_time = time.time()
best_acc = 0.0

for step in range(CONFIG["max_iters"]):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(dataloader)
        batch = next(train_iter)
    
    x_tokens = batch["x_tokens"].to(device)
    y_true = batch["y_true"].to(device)
    
    # Initialize y with random guesses for blanks
    y_tokens = x_tokens.clone()
    blank_mask = y_tokens == 0
    y_tokens[blank_mask] = torch.randint(
        1, 10, 
        (blank_mask.sum(),), 
        device=device
    )
    
    # Deep supervision loop (up to N_sup steps)
    total_loss = 0.0
    num_sup_steps = 0
    
    for sup_step in range(CONFIG["N_sup"]):
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Embed
            x = embedding(x_tokens)
            y = embedding(y_tokens)
            z = torch.zeros_like(x)
            
            # Forward through TRM (with T-cycle warmup built-in)
            (y_out, z_out), y_hat, q_hat = model(x, y, z)
            
            # Loss on FINAL output only
            loss = F.cross_entropy(
                y_hat.view(-1, y_hat.size(-1)),
                y_true.view(-1),
                ignore_index=0
            )
        
        # Backprop
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(embedding.parameters()),
            CONFIG["grad_clip"]
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update EMA
        ema_model.update()
        ema_embedding.update()
        
        total_loss += loss.item()
        num_sup_steps += 1
        
        # Update y_tokens for next supervision step
        with torch.no_grad():
            preds = y_hat.argmax(dim=-1)
            # Keep given cells, update predicted cells
            y_tokens = x_tokens.clone()
            y_tokens[blank_mask] = preds[blank_mask]
            
            # Early stop if solved
            if (preds[:, :81] == y_true[:, :81]).all():
                break
    
    # Logging
    if step % CONFIG["log_every"] == 0:
        with torch.no_grad():
            # Apply EMA for evaluation
            ema_model.apply_shadow()
            ema_embedding.apply_shadow()
            
            # Evaluate
            x = ema_embedding(x_tokens)
            y = ema_embedding(batch["y_init_tokens"].to(device))
            z = torch.zeros_like(x)
            (y_out, z_out), y_hat, q_hat = model(x, y, z)
            preds = y_hat[:, :81].argmax(dim=-1)
            acc = (preds == y_true[:, :81]).float().mean().item()
            
            # Restore training weights
            ema_model.restore()
            ema_embedding.restore()
        
        avg_loss = total_loss / num_sup_steps
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]
        
        print(f"Step {step:5d}/{CONFIG['max_iters']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.3f} | "
              f"SupSteps: {num_sup_steps} | "
              f"LR: {lr:.6f} | "
              f"Time: {elapsed:.0f}s")
        
        if acc > best_acc:
            best_acc = acc
            print(f"  🎯 New best accuracy: {best_acc:.3f}")
    
    # Save checkpoint
    if step > 0 and step % CONFIG["save_every"] == 0:
        # Apply EMA before saving
        ema_model.apply_shadow()
        ema_embedding.apply_shadow()
        
        torch.save({
            "model": model.state_dict(),
            "embedding": embedding.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": CONFIG,
            "best_acc": best_acc
        }, f"/content/drive/MyDrive/TRM/step_{step}.pt")
        
        # Restore training weights
        ema_model.restore()
        ema_embedding.restore()
        
        print(f"  ✓ Saved EMA checkpoint step_{step}.pt (best_acc: {best_acc:.3f})")

print("="*60)
print(f"Training complete! Best accuracy: {best_acc:.3f}")
print(f"Expected: 70-87% (paper achieves 87.4%)")
