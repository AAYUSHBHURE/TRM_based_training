"""
Sudoku-Extreme Generator for TRM Training

Generates extreme difficulty 9x9 Sudoku puzzles (20-25 hints) using 
backtracking solver with constraint propagation.

Target: 1K train examples, augmentable via rule-preserving shuffles.
Source: huggingface.co/datasets/sapientinc/sudoku-extreme
"""

import numpy as np
import random
from typing import Tuple, List, Optional
import json
from pathlib import Path


def is_valid(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in grid[row]:
        return False
    # Check column
    if num in grid[:, col]:
        return False
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    if num in grid[box_row:box_row+3, box_col:box_col+3]:
        return False
    return True


def solve_sudoku(grid: np.ndarray) -> bool:
    """Solve Sudoku using backtracking with constraint propagation."""
    # Find empty cell with minimum remaining values (MRV heuristic)
    min_options = 10
    best_cell = None
    
    for i in range(9):
        for j in range(9):
            if grid[i, j] == 0:
                options = sum(1 for n in range(1, 10) if is_valid(grid, i, j, n))
                if options < min_options:
                    min_options = options
                    best_cell = (i, j)
                    if options == 0:
                        return False
    
    if best_cell is None:
        return True  # Solved
    
    row, col = best_cell
    nums = list(range(1, 10))
    random.shuffle(nums)
    
    for num in nums:
        if is_valid(grid, row, col, num):
            grid[row, col] = num
            if solve_sudoku(grid):
                return True
            grid[row, col] = 0
    
    return False


def generate_complete_grid() -> np.ndarray:
    """Generate a complete valid Sudoku grid."""
    grid = np.zeros((9, 9), dtype=np.int32)
    solve_sudoku(grid)
    return grid


def count_solutions(grid: np.ndarray, limit: int = 2) -> int:
    """Count solutions up to limit (for uniqueness check)."""
    grid = grid.copy()
    count = [0]
    
    def backtrack():
        if count[0] >= limit:
            return
        
        # Find empty cell
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    for num in range(1, 10):
                        if is_valid(grid, i, j, num):
                            grid[i, j] = num
                            backtrack()
                            grid[i, j] = 0
                    return
        count[0] += 1
    
    backtrack()
    return count[0]


def generate_extreme_puzzle(min_hints: int = 20, max_hints: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an extreme difficulty Sudoku puzzle.
    
    Returns:
        puzzle: 9x9 grid with 0s for blanks
        solution: Complete 9x9 grid
    """
    solution = generate_complete_grid()
    puzzle = solution.copy()
    
    # Remove cells to create puzzle
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    
    removed = 0
    target_blanks = 81 - random.randint(min_hints, max_hints)
    
    for row, col in cells:
        if removed >= target_blanks:
            break
        
        backup = puzzle[row, col]
        puzzle[row, col] = 0
        
        # Check uniqueness
        if count_solutions(puzzle, limit=2) == 1:
            removed += 1
        else:
            puzzle[row, col] = backup
    
    return puzzle, solution


def shuffle_sudoku(puzzle: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rule-preserving shuffle (band/stack swaps, digit permutation)."""
    puzzle = puzzle.copy()
    solution = solution.copy()
    
    # Digit permutation
    perm = list(range(1, 10))
    random.shuffle(perm)
    perm = [0] + perm  # 0 maps to 0
    
    for grid in [puzzle, solution]:
        for i in range(9):
            for j in range(9):
                grid[i, j] = perm[grid[i, j]]
    
    # Row swaps within bands
    for band in range(3):
        rows = list(range(band * 3, band * 3 + 3))
        random.shuffle(rows)
        puzzle[[band*3, band*3+1, band*3+2]] = puzzle[rows]
        solution[[band*3, band*3+1, band*3+2]] = solution[rows]
    
    # Column swaps within stacks
    for stack in range(3):
        cols = list(range(stack * 3, stack * 3 + 3))
        random.shuffle(cols)
        puzzle[:, [stack*3, stack*3+1, stack*3+2]] = puzzle[:, cols]
        solution[:, [stack*3, stack*3+1, stack*3+2]] = solution[:, cols]
    
    return puzzle, solution


def grid_to_constraint_prefix(puzzle: np.ndarray) -> str:
    """Generate constraint prefix for training augmentation."""
    constraints = []
    for i in range(9):
        row_vals = [v for v in puzzle[i] if v != 0]
        if row_vals:
            constraints.append(f"R{i+1}:{','.join(map(str, row_vals))}")
    return "|".join(constraints[:3])  # First 3 row constraints


def generate_dataset(
    n_examples: int = 1000,
    n_augments: int = 1000,
    output_dir: str = "data/sudoku",
    include_constraints: bool = True
) -> None:
    """
    Generate Sudoku-Extreme dataset.
    
    Args:
        n_examples: Number of base puzzles to generate
        n_augments: Number of augmented variants per puzzle
        output_dir: Output directory
        include_constraints: Whether to add constraint prefixes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_data = []
    
    print(f"Generating {n_examples} Sudoku-Extreme puzzles...")
    for i in range(n_examples):
        puzzle, solution = generate_extreme_puzzle()
        
        # Store base example
        example = {
            "id": f"sudoku_{i:05d}",
            "puzzle": puzzle.flatten().tolist(),
            "solution": solution.flatten().tolist(),
            "hints": int(np.sum(puzzle != 0))
        }
        
        if include_constraints:
            example["constraint_prefix"] = grid_to_constraint_prefix(puzzle)
        
        train_data.append(example)
        
        # Generate augmented variants
        for j in range(min(n_augments, 10)):  # Cap at 10 for efficiency
            aug_puzzle, aug_solution = shuffle_sudoku(puzzle, solution)
            aug_example = {
                "id": f"sudoku_{i:05d}_aug{j:02d}",
                "puzzle": aug_puzzle.flatten().tolist(),
                "solution": aug_solution.flatten().tolist(),
                "hints": int(np.sum(aug_puzzle != 0))
            }
            if include_constraints:
                aug_example["constraint_prefix"] = grid_to_constraint_prefix(aug_puzzle)
            train_data.append(aug_example)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_examples} puzzles")
    
    # Save dataset
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f)
    
    print(f"Saved {len(train_data)} examples to {output_path / 'train.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Sudoku-Extreme dataset")
    parser.add_argument("--size", type=int, default=1000, help="Number of puzzles")
    parser.add_argument("--output", type=str, default="data/sudoku", help="Output dir")
    parser.add_argument("--augments", type=int, default=10, help="Augments per puzzle")
    args = parser.parse_args()
    
    generate_dataset(args.size, args.augments, args.output)
