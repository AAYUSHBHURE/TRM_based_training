# ============================================================
# FASTER TRAINING CONFIG - Reduce N_sup and iterations
# ============================================================

CONFIG = {
    # Model Architecture
    "dim": 512,
    "n_layers": 2,           # ✓ 2 layers
    "n_heads": 8,
    "n_latent": 6,           # ✓ n=6 latent updates
    "T_cycles": 3,           # ✓ T=3 (2 warmup + 1 grad)
    "vocab_size": 10,
    "max_seq_len": 128,
    "dropout": 0.1,
    
    # Training
    "lr": 3e-4,
    "weight_decay": 0.01,
    "max_iters": 10000,      # ⚡ REDUCED from 20000 (2x faster)
    "batch_size": 32,
    "warmup_steps": 500,     # ⚡ REDUCED proportionally  
    "grad_clip": 1.0,
    
    # Deep Supervision
    "N_sup": 4,              # ⚡ REDUCED from 16 (4x faster per step!)
    "ema_decay": 0.999,
    
    # Checkpointing
    "save_every": 2000,
    "log_every": 100,
}

print("OPTIMIZED CONFIG for faster training:")
print(f"  - N_sup: 16 → 4 (4x faster per step)")
print(f"  - max_iters: 20000 → 10000 (2x fewer steps)")
print(f"  - Expected total time: ~45-60 minutes (vs 8+ hours)")
print(f"  - Expected accuracy: 60-75% (vs 87% with full training)")
