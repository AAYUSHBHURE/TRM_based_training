# FIXED: Maze loading for extracted dataset

import os, glob, cv2
import numpy as np
from collections import deque

print("="*60)
print("Loading mazes from extracted dataset...")
print("="*60)

# Check what directories exist
print("\nChecking extracted files...")
for root, dirs, files in os.walk('.'):
    if any(f.endswith('.png') for f in files):
        png_count = len([f for f in files if f.endswith('.png')])
        print(f"  Found {png_count} PNG files in: {root}")

# Function to convert image to maze array
def image_to_maze(img_path):
    """Convert maze image to binary array (0=path, 1=wall)"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Threshold: white (>127) = path (0), dark (<127) = wall (1)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    return binary

def path_to_directions(path):
    """Convert path to directions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STOP"""
    if len(path) < 2: return [4]
    dirs = []
    for i in range(len(path)-1):
        r1, c1, r2, c2 = path[i][0], path[i][1], path[i+1][0], path[i+1][1]
        if r2 > r1: dirs.append(1)  # DOWN
        elif r2 < r1: dirs.append(0)  # UP
        elif c2 > c1: dirs.append(3)  # RIGHT
        elif c2 < c1: dirs.append(2)  # LEFT
    dirs.append(4)  # STOP
    return dirs

def find_path_bfs(maze):
    rows, cols = maze.shape
    start, goal = (0,0), (rows-1, cols-1)
    queue, visited = deque([(start, [start])]), {start}
    while queue:
        (r,c), path = queue.popleft()
        if (r,c) == goal: return path
        for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and maze[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                queue.append(((nr,nc), path+[(nr,nc)]))
    return [start, goal]

# Find ALL PNG files recursively
all_images = glob.glob('**/*.png', recursive=True)
print(f"\nTotal PNG files found: {len(all_images)}")

# Load mazes
train_data = []
loaded_count = 0
error_count = 0

for img_path in all_images[:3000]:  # Process up to 3000
    try:
        maze = image_to_maze(img_path)
        if maze is None:
            error_count += 1
            continue
            
        # Only use 10×10 mazes
        if maze.shape != (10, 10):
            continue
        
        # Find path
        path = find_path_bfs(maze)
        if len(path) < 2:
            continue
        
        # Convert to directions
        directions = path_to_directions(path[:30])  # Max 30 steps
        
        train_data.append({
            "maze": maze.flatten().tolist(),
            "directions": directions
        })
        
        loaded_count += 1
        
        if loaded_count % 500 == 0:
            print(f"  Loaded {loaded_count} valid 10×10 mazes...")
            
    except Exception as e:
        error_count += 1
        if error_count < 5:  # Show first few errors
            print(f"  Error loading {img_path}: {e}")
        continue

print("\n" + "="*60)
print(f"✓ Successfully loaded {len(train_data)} valid 10×10 mazes")
print(f"  Errors/skipped: {error_count}")

if len(train_data) > 0:
    avg_len = np.mean([len(d['directions']) for d in train_data])
    print(f"  Avg path length: {avg_len:.1f} steps")
    print(f"  Sample directions: {train_data[0]['directions'][:10]}")
    print(f"  Sample maze shape: {np.array(train_data[0]['maze']).shape}")
else:
    print("\n⚠ WARNING: No valid mazes loaded!")
    print("  Possible issues:")
    print("  - Files not extracted properly")
    print("  - Wrong image format")
    print("  - All mazes are not 10×10")

print("="*60)
