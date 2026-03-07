# ============================================================
# CORRECTED: Load Maze Dataset from Kaggle
# Format: .txt files with 0=path, 1=wall
# ============================================================

import glob
import numpy as np
from collections import deque

print("Loading Maze dataset from Kaggle...")
print("Dataset: mexwell/maze-dataset")
print("="*60)

# Download dataset
import os
os.environ['KAGGLE_USERNAME'] = 'aayushbhure'  # CHANGE THIS!
os.environ['KAGGLE_KEY'] = 'KGAT_4bb3a6797b5560ffe3b19de1e8035651'

!pip install -q kaggle
!kaggle datasets download -d mexwell/maze-dataset
!unzip -q maze-dataset.zip

print("✓ Dataset downloaded")

# Load maze from .txt file
def load_maze_txt(filepath):
    """Load maze from text file (0=path, 1=wall)"""
    maze = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            maze.append(row)
    return np.array(maze)

# Find shortest path using BFS
def find_path_bfs(maze):
    """
    Find shortest path from top-left corner to bottom-right
    0 = path (walkable), 1 = wall
    """
    rows, cols = maze.shape
    
    # Start and goal
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    
    # BFS
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        (row, col), path = queue.popleft()
        
        if (row, col) == goal:
            return path
        
        # Four directions
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            
            if (0 <= nr < rows and 
                0 <= nc < cols and 
                maze[nr, nc] == 0 and  # 0 = path
                (nr, nc) not in visited):
                
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    
    # No path found
    return [start, goal]

# Load all maze files
maze_files = glob.glob('imperfect_maze/*.txt')
print(f"Found {len(maze_files)} maze files")

train_data = []
skipped = 0

for filepath in maze_files[:5000]:  # Use first 5000
    try:
        maze = load_maze_txt(filepath)
        
        # Filter by size (use only ~30x30 to 100x100)
        if maze.shape[0] < 20 or maze.shape[0] > 100:
            skipped += 1
            continue
        
        if maze.shape[0] != maze.shape[1]:  # Skip non-square
            skipped += 1
            continue
        
        # Find shortest path
        path = find_path_bfs(maze)
        
        if len(path) < 2:  # No valid path
            skipped += 1
            continue
        
        # Resize to 30x30 for consistency
        from PIL import Image
        maze_img = Image.fromarray((maze * 255).astype(np.uint8))
        maze_resized = np.array(maze_img.resize((30, 30), Image.NEAREST)) // 255
        
        # Convert path to 30x30 coordinates (scale)
        scale = 30 / maze.shape[0]
        path_resized = [(int(r * scale), int(c * scale)) for r, c in path]
        
        # Remove duplicates
        path_unique = []
        for p in path_resized:
            if not path_unique or p != path_unique[-1]:
                path_unique.append(p)
        
        # Flatten maze and path
        maze_flat = maze_resized.flatten().tolist()
        path_flat = [r * 30 + c for r, c in path_unique[:100]]
        
        train_data.append({
            "maze": maze_flat,
            "path": path_flat
        })
        
        if len(train_data) % 500 == 0:
            print(f"  Loaded {len(train_data)} valid mazes...")
    
    except Exception as e:
        skipped += 1
        continue

print("="*60)
print(f"✓ Loaded {len(train_data)} mazes!")
print(f"  Skipped {skipped} invalid/unsupported mazes")
print(f"  Grid size: 30x30 (resized)")
print(f"  Avg path length: {np.mean([len(d['path']) for d in train_data]):.1f} steps")

# Show sample
if len(train_data) > 0:
    sample = train_data[0]
    maze_grid = np.array(sample['maze']).reshape(30, 30)
    print(f"\nSample maze (0=path, 1=wall):")
    print(maze_grid[:10, :10])  # Show top-left corner
    print(f"Path length: {len(sample['path'])} steps")
