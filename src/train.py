"""
TRM Training CLI

Usage:
    python -m src.train --config configs/sudoku_baseline.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch

from src.model import create_trm_model
from src.data import TRMDataset, create_dataloader
from src.training import TRMTrainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train TRM model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--max_iters", type=int, default=None, help="Override max iterations")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.data:
        config["data"]["train_path"] = args.data
    if args.max_iters:
        config["training"]["max_iters"] = args.max_iters
    if args.device:
        config["device"] = args.device
    
    print("=" * 60)
    print("TRM Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {config.get('device', 'cuda')}")
    print(f"Model: {config['model']['n_layers']} layers, dim={config['model']['dim']}")
    print("=" * 60)
    
    # Create model
    model = create_trm_model(
        dim=config["model"]["dim"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        n_latent=config["model"]["n_latent"],
        T_cycles=config["model"]["T_cycles"],
        vocab_size=config["model"]["vocab_size"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"].get("dropout", 0.0)
    )
    
    # Create dataloader
    data_config = config["data"]
    train_loader = create_dataloader(
        data_path=data_config["train_path"],
        domain=data_config["domain"],
        batch_size=config["training"]["batch_size"],
        embed_dim=config["model"]["dim"],
        max_seq_len=config["model"]["max_seq_len"],
        vocab_size=config["model"]["vocab_size"]
    )
    
    # Create trainer
    trainer = TRMTrainer(
        model=model,
        train_loader=train_loader,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        N_sup=config["training"]["N_sup"],
        halt_threshold=config["training"]["halt_threshold"],
        ema_beta=config["training"]["ema_beta"],
        use_amp=config.get("use_amp", True),
        device=config.get("device", "cuda"),
        output_dir=args.output,
        use_wandb=config["logging"].get("use_wandb", False),
        wandb_project=config["logging"].get("wandb_project", "trm")
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        max_iters=config["training"]["max_iters"],
        log_interval=config["logging"]["log_interval"],
        save_interval=config["logging"]["save_interval"],
        val_interval=config["logging"]["val_interval"]
    )


if __name__ == "__main__":
    main()
