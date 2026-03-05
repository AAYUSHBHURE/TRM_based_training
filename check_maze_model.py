import torch

# Load maze checkpoint
checkpoint = torch.load("outputs/maze_final.pt", map_location='cpu')

print("=" * 60)
print("MAZE MODEL CHECKPOINT INFO")
print("=" * 60)

print("\nCheckpoint keys:", list(checkpoint.keys()))

if 'acc' in checkpoint:
    print(f"\n[OK] Best Accuracy: {checkpoint['acc']:.1%}")
else:
    print("\n[WARN] No accuracy in checkpoint")

if 'model' in checkpoint:
    model_params = sum(p.numel() for p in checkpoint['model'].values())
    print(f"\n[OK] Model Parameters: {model_params:,}")

if 'emb' in checkpoint:
    emb_params = sum(p.numel() for p in checkpoint['emb'].values())
    print(f"[OK] Embedding Parameters: {emb_params:,}")
    print(f"[OK] Total Parameters: {model_params + emb_params:,}")

print("\n" + "=" * 60)
print("Model saved to: outputs/maze_final.pt")
print("=" * 60)
