"""
Maze-Hard Generator for TRM Training

Generates hard 30x30 mazes with shortest paths >110 steps using 
recursive division and NetworkX path algorithms.

Target: 1K mazes with 8 dihedral transforms per example.
Source: github.com/understanding-search/maze-dataset
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Dict
import json
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    nx = None
    print("Warning: networkx not installed. Run: pip install networkx")


def generate_maze_recursive_division(width: int = 30, height: int = 30) -> np.ndarray:
    """
    Generate maze using recursive division algorithm.
    
    Returns:
        maze: 2D array where 0=path, 1=wall
    """
    maze = np.zeros((height, width), dtype=np.int32)
    
    # Add border walls
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1
    
    def divide(x1: int, y1: int, x2: int, y2: int, horizontal: bool):
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        
        if horizontal:
            # Horizontal wall
            y = random.randrange(y1 + 1, y2, 2) if (y2 - y1) > 2 else y1 + 1
            if y >= y2:
                return
            for x in range(x1, x2 + 1):
                maze[y, x] = 1
            # Create passage
            passage = random.randrange(x1, x2 + 1, 2) if x2 > x1 else x1
            maze[y, passage] = 0
            
            divide(x1, y1, x2, y - 1, not horizontal)
            divide(x1, y + 1, x2, y2, not horizontal)
        else:
            # Vertical wall
            x = random.randrange(x1 + 1, x2, 2) if (x2 - x1) > 2 else x1 + 1
            if x >= x2:
                return
            for y in range(y1, y2 + 1):
                maze[y, x] = 1
            # Create passage
            passage = random.randrange(y1, y2 + 1, 2) if y2 > y1 else y1
            maze[passage, x] = 0
            
            divide(x1, y1, x - 1, y2, not horizontal)
            divide(x + 1, y1, x2, y2, not horizontal)
    
    divide(1, 1, width - 2, height - 2, random.choice([True, False]))
    
    return maze


def add_imperfections(maze: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """Add random passages to make maze imperfect (multiple paths)."""
    maze = maze.copy()
    height, width = maze.shape
    
    n_breaks = int(ratio * np.sum(maze == 1))
    walls = list(zip(*np.where(maze == 1)))
    random.shuffle(walls)
    
    for y, x in walls[:n_breaks]:
        if 0 < y < height - 1 and 0 < x < width - 1:
            maze[y, x] = 0
    
    return maze


def find_path_bfs(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Find shortest path using BFS."""
    height, width = maze.shape
    visited = set()
    queue = [(start, [start])]
    
    while queue:
        (y, x), path = queue.pop(0)
        
        if (y, x) == end:
            return path
        
        if (y, x) in visited:
            continue
        visited.add((y, x))
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0:
                if (ny, nx) not in visited:
                    queue.append(((ny, nx), path + [(ny, nx)]))
    
    return []  # No path found


def find_hard_endpoints(maze: np.ndarray, min_path_length: int = 110) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Find start/end points with path length >= min_path_length."""
    height, width = maze.shape
    
    # Get all path cells
    path_cells = list(zip(*np.where(maze == 0)))
    if len(path_cells) < 2:
        return None
    
    # Try random pairs to find hard paths
    for _ in range(100):
        start = random.choice(path_cells)
        end = random.choice(path_cells)
        
        if start == end:
            continue
        
        path = find_path_bfs(maze, start, end)
        if len(path) >= min_path_length:
            return (start, end)
    
    # Fallback: use corners
    corners = [
        (1, 1), (1, width - 2),
        (height - 2, 1), (height - 2, width - 2)
    ]
    valid_corners = [(y, x) for y, x in corners if maze[y, x] == 0]
    if len(valid_corners) >= 2:
        return (valid_corners[0], valid_corners[-1])
    
    return None


def generate_hard_maze(
    width: int = 30,
    height: int = 30,
    min_path_length: int = 110,
    max_attempts: int = 50
) -> Optional[Dict]:
    """
    Generate a hard maze with path length >= min_path_length.
    
    Returns:
        Dict with maze, path, start, end, or None if failed
    """
    for _ in range(max_attempts):
        maze = generate_maze_recursive_division(width, height)
        maze = add_imperfections(maze, ratio=0.05)
        
        endpoints = find_hard_endpoints(maze, min_path_length)
        if endpoints is None:
            continue
        
        start, end = endpoints
        path = find_path_bfs(maze, start, end)
        
        if len(path) >= min_path_length:
            return {
                "maze": maze,
                "path": path,
                "start": start,
                "end": end,
                "path_length": len(path)
            }
    
    return None


def apply_dihedral_transform(maze: np.ndarray, path: List[Tuple[int, int]], transform_id: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Apply one of 8 dihedral transforms (4 rotations × 2 reflections)."""
    height, width = maze.shape
    
    def transform_point(y, x, t_id):
        if t_id == 0:  # Identity
            return (y, x)
        elif t_id == 1:  # Rotate 90
            return (x, height - 1 - y)
        elif t_id == 2:  # Rotate 180
            return (height - 1 - y, width - 1 - x)
        elif t_id == 3:  # Rotate 270
            return (width - 1 - x, y)
        elif t_id == 4:  # Flip horizontal
            return (y, width - 1 - x)
        elif t_id == 5:  # Flip vertical
            return (height - 1 - y, x)
        elif t_id == 6:  # Flip diagonal
            return (x, y)
        elif t_id == 7:  # Flip anti-diagonal
            return (height - 1 - x, width - 1 - y)
        return (y, x)
    
    # Transform maze
    if transform_id == 0:
        new_maze = maze.copy()
    elif transform_id == 1:
        new_maze = np.rot90(maze, k=3)
    elif transform_id == 2:
        new_maze = np.rot90(maze, k=2)
    elif transform_id == 3:
        new_maze = np.rot90(maze, k=1)
    elif transform_id == 4:
        new_maze = np.fliplr(maze)
    elif transform_id == 5:
        new_maze = np.flipud(maze)
    elif transform_id == 6:
        new_maze = maze.T
    elif transform_id == 7:
        new_maze = np.rot90(maze.T, k=2)
    else:
        new_maze = maze.copy()
    
    # Transform path
    new_path = [transform_point(y, x, transform_id) for y, x in path]
    
    return new_maze, new_path


def perturb_graph(maze: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """Graph perturbation: add/remove edges (walls) by ±ratio."""
    maze = maze.copy()
    height, width = maze.shape
    
    # Add walls (remove edges)
    path_cells = list(zip(*np.where(maze == 0)))
    n_add = int(ratio * len(path_cells))
    random.shuffle(path_cells)
    for y, x in path_cells[:n_add]:
        if 0 < y < height - 1 and 0 < x < width - 1:
            maze[y, x] = 1
    
    # Remove walls (add edges)
    wall_cells = list(zip(*np.where(maze == 1)))
    n_remove = int(ratio * len(wall_cells))
    random.shuffle(wall_cells)
    for y, x in wall_cells[:n_remove]:
        if 0 < y < height - 1 and 0 < x < width - 1:
            maze[y, x] = 0
    
    return maze


def generate_dataset(
    n_examples: int = 1000,
    output_dir: str = "data/maze",
    width: int = 30,
    height: int = 30
) -> None:
    """
    Generate Maze-Hard dataset with augmentations.
    
    Args:
        n_examples: Number of base mazes to generate
        output_dir: Output directory
        width, height: Maze dimensions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_data = []
    
    print(f"Generating {n_examples} hard mazes ({width}x{height})...")
    generated = 0
    attempts = 0
    
    while generated < n_examples and attempts < n_examples * 10:
        attempts += 1
        result = generate_hard_maze(width, height)
        
        if result is None:
            continue
        
        maze = result["maze"]
        path = result["path"]
        
        # Base example
        example = {
            "id": f"maze_{generated:05d}",
            "maze": maze.flatten().tolist(),
            "path": path,
            "start": result["start"],
            "end": result["end"],
            "path_length": result["path_length"],
            "width": width,
            "height": height
        }
        train_data.append(example)
        
        # 8 dihedral transforms
        for t_id in range(1, 8):  # Skip identity (0)
            t_maze, t_path = apply_dihedral_transform(maze, path, t_id)
            aug_example = {
                "id": f"maze_{generated:05d}_t{t_id}",
                "maze": t_maze.flatten().tolist(),
                "path": t_path,
                "start": t_path[0],
                "end": t_path[-1],
                "path_length": len(t_path),
                "width": width,
                "height": height
            }
            train_data.append(aug_example)
        
        generated += 1
        if generated % 100 == 0:
            print(f"  Generated {generated}/{n_examples} mazes")
    
    # Save dataset
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f)
    
    print(f"Saved {len(train_data)} examples to {output_path / 'train.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Maze-Hard dataset")
    parser.add_argument("--size", type=int, default=1000, help="Number of mazes")
    parser.add_argument("--output", type=str, default="data/maze", help="Output dir")
    parser.add_argument("--width", type=int, default=30, help="Maze width")
    parser.add_argument("--height", type=int, default=30, help="Maze height")
    args = parser.parse_args()
    
    generate_dataset(args.size, args.output, args.width, args.height)
