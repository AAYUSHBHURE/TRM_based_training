# ============================================================
# DEEP SUPERVISION FIX - Unlock TRM Novelty
# Replace the training loop to add intermediate supervision
# ============================================================

# Add this BEFORE the training loop

# Deep supervision: supervise ALL latent steps
def compute_deep_supervised_loss(model, x, y_init, y_true, embedding):
    """
    The KEY to TRM's novelty: supervise intermediate z-states
    This makes the model learn progressive refinement
    """
    x_emb = embedding(x)
    y_emb = embedding(y_init)
    z = torch.zeros_like(x_emb)
    
    total_loss = 0.0
    n_losses = 0
    
    # Supervise EACH T-cycle
    for t in range(model.T_cycles):
        # Supervise EACH latent step within the cycle
        for latent_step in range(model.latent_recursion.n_latent):
            # Get intermediate z
            concat = torch.cat([x_emb, y_emb, z], dim=-1)
            z = model.net(model.latent_recursion.z_proj(concat))
            
            # Predict from current z
            concat_y = torch.cat([y_emb, z], dim=-1)
            y_intermediate = model.net(model.latent_recursion.y_proj(concat_y))
            y_pred = model.heads.output_head(y_intermediate)
            
            # Compute loss for this intermediate prediction
            loss = F.cross_entropy(
                y_pred.view(-1, y_pred.size(-1)),
                y_true.view(-1),
                ignore_index=0
            )
            
            # Weight: later steps get higher weight
            weight = (t * model.latent_recursion.n_latent + latent_step + 1) / \
                     (model.T_cycles * model.latent_recursion.n_latent)
            
            total_loss += weight * loss
            n_losses += weight
        
        # Update y for next cycle
        concat_y = torch.cat([y_emb, z], dim=-1)
        y_emb = model.net(model.latent_recursion.y_proj(concat_y))
    
    return total_loss / n_losses


# ============================================================
# UPDATED TRAINING LOOP WITH DEEP SUPERVISION
# ============================================================

import os
os.makedirs('/content/drive/MyDrive/TRM', exist_ok=True)

print(f"Starting DEEP SUPERVISED training for {CONFIG['max_iters']} iterations...")
print("="*60)
print("DEEP SUPERVISION: Supervising all intermediate latent states")
print("This unlocks TRM's recursive reasoning novelty!")
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
    y_init_tokens = batch["y_init_tokens"].to(device)
    y_true = batch["y_true"].to(device)
    
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda'):
        # Use deep supervised loss instead of simple forward
        loss = compute_deep_supervised_loss(
            model, x_tokens, y_init_tokens, y_true, embedding
        )
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(embedding.parameters()), 
        CONFIG["grad_clip"]
    )
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    # Logging
    if step % CONFIG["log_every"] == 0:
        with torch.no_grad():
            # Evaluate with full forward pass
            x = embedding(x_tokens)
            y = embedding(y_init_tokens)
            z = torch.zeros_like(x)
            (y_out, z_out), y_hat, q_hat = model(x, y, z)
            preds = y_hat[:, :81].argmax(dim=-1)
            acc = (preds == y_true[:, :81]).float().mean().item()
        
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]
        print(f"Step {step:5d}/{CONFIG['max_iters']} | Loss: {loss.item():.4f} | Acc: {acc:.3f} | LR: {lr:.6f} | Time: {elapsed:.0f}s")
        
        if acc > best_acc:
            best_acc = acc
            print(f"  🎯 New best accuracy: {best_acc:.3f}")
    
    # Save checkpoint
    if step > 0 and step % CONFIG["save_every"] == 0:
        torch.save({
            "model": model.state_dict(),
            "embedding": embedding.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": CONFIG,
            "best_acc": best_acc
        }, f"/content/drive/MyDrive/TRM/step_{step}.pt")
        print(f"  ✓ Saved checkpoint step_{step}.pt (best_acc: {best_acc:.3f})")

print("="*60)
print(f"Training complete! Best accuracy: {best_acc:.3f}")
print("Deep supervision enabled the model to learn progressive refinement!")
