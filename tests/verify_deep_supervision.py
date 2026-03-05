
import sys
import os
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import TRMModel
from src.training.trainer import TRMTrainer

def test_deep_supervision_loop():
    print("Setting up dummy TRM model...")
    dim = 64
    model = TRMModel(dim=dim, n_layers=1, n_heads=4, n_latent=2, T_cycles=2, vocab_size=10)
    
    # Dummy data
    B, L = 2, 10
    x_tokens = torch.randint(0, 10, (B, L))
    y_init = torch.randint(0, 10, (B, L))
    y_true = torch.randint(0, 10, (B, L))
    
    dataset = TensorDataset(x_tokens, y_init, y_true)
    # Collate function to return dict as expected by trainer
    def collate_fn(batch):
        xs, yis, yts = zip(*batch)
        return {
            "x_tokens": torch.stack(xs),
            "y_init_tokens": torch.stack(yis),
            "y_true": torch.stack(yts)
        }
        
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    trainer = TRMTrainer(
        model=model,
        train_loader=loader,
        lr=1e-3,
        N_sup=3,  # Run 3 steps per batch
        device="cpu", # Use CPU for simple test
        use_amp=False
    )
    
    print("Running train_step with N_sup=3...")
    
    # Track initial weights to ensure they change
    initial_weight = model.net.blocks[0].ffn.w1.weight.clone()
    
    batch = next(iter(loader))
    loss, metrics = trainer.train_step(batch)
    
    print(f"Final Loss: {loss}")
    print(f"Metrics: {metrics}")
    
    final_weight = model.net.blocks[0].ffn.w1.weight
    
    assert metrics["steps"] <= 3, "Should run at most 3 steps"
    assert not torch.allclose(initial_weight, final_weight), "Weights should have updated"
    
    print("\n✅ Verification Successful: Training loop ran and updated weights.")

if __name__ == "__main__":
    try:
        test_deep_supervision_loop()
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
