"""
TRM Evaluation Script - MVP Demo

Evaluates trained model on Sudoku puzzles and reports accuracy.

Usage:
    python -m src.evaluate --checkpoint outputs/final.pt --samples 100
"""

import argparse
import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from typing import Dict, Tuple

from src.model import create_trm_model
from src.data import TRMDataset


def load_model(checkpoint_path: str, config: dict, device: str) -> Tuple[nn.Module, nn.Embedding]:
    """Load trained model and embedding from checkpoint."""
    model = create_trm_model(
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_latent=config["n_latent"],
        T_cycles=config["T_cycles"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
    )
    
    embedding = nn.Embedding(20, config["dim"])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    embedding.load_state_dict(checkpoint["embedding"])
    
    model = model.to(device)
    embedding = embedding.to(device)
    model.eval()
    
    return model, embedding


def evaluate_puzzle(
    model: nn.Module,
    embedding: nn.Embedding,
    puzzle: torch.Tensor,
    solution: torch.Tensor,
    device: str
) -> Tuple[bool, float, torch.Tensor]:
    """
    Evaluate a single puzzle.
    
    Returns:
        solved: Whether puzzle was solved correctly
        cell_acc: Cell-level accuracy
        prediction: Predicted solution
    """
    with torch.no_grad():
        # Prepare inputs
        x_tokens = puzzle.unsqueeze(0).to(device)  # [1, 81]
        
        # Initial guess (random for blanks)
        y_init = puzzle.clone()
        blank_mask = puzzle == 0
        y_init[blank_mask] = torch.randint(1, 10, (blank_mask.sum(),))
        y_init = y_init.unsqueeze(0).to(device)  # [1, 81]
        
        # Pad to 128
        x_padded = torch.zeros(1, 128, dtype=torch.long, device=device)
        x_padded[0, :81] = x_tokens[0]
        
        y_padded = torch.zeros(1, 128, dtype=torch.long, device=device)
        y_padded[0, :81] = y_init[0]
        
        # Embed
        x = embedding(x_padded)
        y = embedding(y_padded)
        z = torch.zeros_like(x)
        
        # Forward
        (y_out, z_out), y_hat, q_hat = model(x, y, z)
        
        # Decode
        prediction = y_hat[0, :81].argmax(dim=-1).cpu()  # [81]
        
        # For blanks, use prediction; for hints, use puzzle
        final_pred = puzzle.clone()
        final_pred[blank_mask] = prediction[blank_mask]
        
        # Calculate accuracy
        correct_cells = (final_pred == solution).sum().item()
        cell_acc = correct_cells / 81
        solved = correct_cells == 81
        
        return solved, cell_acc, final_pred


def format_grid(grid: torch.Tensor) -> str:
    """Format 81-element tensor as 9x9 Sudoku grid."""
    lines = []
    for i in range(9):
        row = grid[i*9:(i+1)*9].tolist()
        row_str = ""
        for j, val in enumerate(row):
            if j % 3 == 0 and j > 0:
                row_str += "│ "
            row_str += f"{val} "
        lines.append(row_str)
        if i % 3 == 2 and i < 8:
            lines.append("──────┼───────┼──────")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM model on Sudoku")
    parser.add_argument("--checkpoint", type=str, default="outputs/final.pt", help="Checkpoint path")
    parser.add_argument("--data", type=str, default="data/sudoku/train.json", help="Test data path")
    parser.add_argument("--samples", type=int, default=100, help="Number of puzzles to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show_examples", type=int, default=3, help="Number of examples to display")
    args = parser.parse_args()
    
    # Config (must match training)
    config = {
        "dim": 512,
        "n_layers": 2,
        "n_heads": 8,
        "n_latent": 6,
        "T_cycles": 3,
        "vocab_size": 10,
        "max_seq_len": 128,
    }
    
    print("=" * 60)
    print("TRM Sudoku Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Test samples: {args.samples}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, embedding = load_model(args.checkpoint, config, args.device)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Load data
    print("\nLoading test data...")
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    # Sample puzzles
    if len(data) > args.samples:
        indices = torch.randperm(len(data))[:args.samples].tolist()
        test_data = [data[i] for i in indices]
    else:
        test_data = data[:args.samples]
    
    print(f"Loaded {len(test_data)} puzzles")
    
    # Evaluate
    print("\nEvaluating...")
    start_time = time.time()
    
    solved_count = 0
    total_cell_acc = 0.0
    examples = []
    
    for i, item in enumerate(test_data):
        puzzle = torch.tensor(item["puzzle"], dtype=torch.long)
        solution = torch.tensor(item["solution"], dtype=torch.long)
        
        solved, cell_acc, prediction = evaluate_puzzle(
            model, embedding, puzzle, solution, args.device
        )
        
        if solved:
            solved_count += 1
        total_cell_acc += cell_acc
        
        # Store examples
        if len(examples) < args.show_examples:
            examples.append({
                "puzzle": puzzle,
                "solution": solution,
                "prediction": prediction,
                "solved": solved,
                "cell_acc": cell_acc
            })
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")
    
    elapsed = time.time() - start_time
    
    # Results
    puzzle_acc = solved_count / len(test_data) * 100
    cell_acc = total_cell_acc / len(test_data) * 100
    avg_time = elapsed / len(test_data) * 1000
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Cell Accuracy:    {cell_acc:.1f}%")
    print(f"Puzzle Accuracy:  {puzzle_acc:.1f}%", end="")
    if puzzle_acc >= 87:
        print("  ✓ Target met!")
    else:
        print(f"  (Target: 87%)")
    print(f"Avg Inference:    {avg_time:.1f}ms")
    print(f"Total Time:       {elapsed:.1f}s")
    print("=" * 60)
    
    # Show examples
    if args.show_examples > 0:
        print("\n" + "=" * 60)
        print("EXAMPLE PUZZLES")
        print("=" * 60)
        
        for i, ex in enumerate(examples):
            print(f"\n--- Example {i+1} ---")
            status = "✓ SOLVED" if ex["solved"] else f"✗ {ex['cell_acc']*100:.0f}% cells"
            print(f"Status: {status}")
            print("\nPuzzle (0=blank):")
            print(format_grid(ex["puzzle"]))
            print("\nModel Output:")
            print(format_grid(ex["prediction"]))
    
    return puzzle_acc


if __name__ == "__main__":
    main()
