import torch
import os

# Load the checkpoint
checkpoint_path = "outputs/sudoku_final.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 60)
print("SUDOKU MODEL CHECKPOINT INFO")
print("=" * 60)

# Display checkpoint contents
print("\nCheckpoint keys:", list(checkpoint.keys()))

if 'best_acc' in checkpoint:
    print(f"\n✓ Best Accuracy: {checkpoint['best_acc']:.1%}")
else:
    print("\n⚠ No accuracy info in checkpoint")

if 'step' in checkpoint:
    print(f"✓ Training Steps: {checkpoint['step']:,}")

if 'config' in checkpoint:
    print("\n✓ Model Configuration:")
    config = checkpoint['config']
    print(f"  - Dimensions: {config.get('dim', 'N/A')}")
    print(f"  - Layers: {config.get('n_layers', 'N/A')}")
    print(f"  - Heads: {config.get('n_heads', 'N/A')}")
    print(f"  - Latent Steps: {config.get('n_latent', 'N/A')}")
    print(f"  - T-cycles: {config.get('T_cycles', 'N/A')}")
    print(f"  - N_sup: {config.get('N_sup', 'N/A')}")

# Count parameters
if 'model' in checkpoint:
    model_params = sum(p.numel() for p in checkpoint['model'].values())
    print(f"\n✓ Model Parameters: {model_params:,}")

if 'embedding' in checkpoint:
    emb_params = sum(p.numel() for p in checkpoint['embedding'].values())
    print(f"✓ Embedding Parameters: {emb_params:,}")
    print(f"✓ Total Parameters: {model_params + emb_params:,}")

print("\n" + "=" * 60)
print("Model saved to: outputs/sudoku_final.pt")
print("=" * 60)
