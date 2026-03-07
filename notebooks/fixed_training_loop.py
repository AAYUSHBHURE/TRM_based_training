# ============================================================
# FIXED: Training Loop - Corrected EMA Usage
# ============================================================

import os
os.makedirs('/content/drive/MyDrive/TRM', exist_ok=True)

print(f"Starting CORRECTED TRM training for {CONFIG['max_iters']} iterations...")
print("Following paper Algorithm 3:")
print(f"  - n={CONFIG['n_latent']} latent updates per recursion")
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
    
    # Initialize y with random guesses
    y_tokens = x_tokens.clone()
    blank_mask = y_tokens == 0
    y_tokens[blank_mask] = torch.randint(1, 10, (blank_mask.sum(),), device=device)
    
    # Deep supervision loop (up to N_sup steps)
    total_loss = 0.0
    num_sup_steps = 0
    
    for sup_step in range(CONFIG["N_sup"]):
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            x = embedding(x_tokens)
            y = embedding(y_tokens)
            z = torch.zeros_like(x)
            
            # Forward (with T-cycle warmup built-in)
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
            
            # Use the original embedding and model (now with EMA weights)
            x = embedding(x_tokens)
            y = embedding(batch["y_init_tokens"].to(device))
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
        
        print(f"  ✓ Saved EMA checkpoint (best_acc: {best_acc:.3f})")

print("="*60)
print(f"Training complete! Best accuracy: {best_acc:.3f}")
print(f"Target: 70-87% (paper: 87.4%)")
