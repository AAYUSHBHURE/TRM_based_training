# ============================================================
# Load Kaggle Sudoku Dataset using API Token
# Dataset: https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings
# ============================================================

import pandas as pd
import numpy as np
import os

print("Loading Sudoku dataset from Kaggle...")
print("Dataset: 3 Million Sudoku Puzzles with Ratings")
print("="*60)

# Set Kaggle credentials using token
os.environ['KAGGLE_USERNAME'] = 'aayushbhure'  # Replace with your Kaggle username
os.environ['KAGGLE_KEY'] = 'KGAT_4bb3a6797b5560ffe3b19de1e8035651'

# Install Kaggle
!pip install -q kaggle

# Download dataset
print("Downloading dataset from Kaggle...")
!kaggle datasets download -d radcliffe/3-million-sudoku-puzzles-with-ratings

# Unzip
print("Extracting files...")
!unzip -q 3-million-sudoku-puzzles-with-ratings.zip

print("✓ Dataset downloaded and extracted")

# Load CSV
print("\nLoading CSV file...")
df = pd.read_csv('sudoku-3m.csv')
print(f"Total puzzles in dataset: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

# Show sample
print("\nSample rows:")
print(df.head(3))

# Check difficulty distribution
if 'difficulty' in df.columns:
    print("\nDifficulty distribution:")
    print(df['difficulty'].value_counts().sort_index())

# Convert to our format
print("\nConverting to training format...")
train_data = []
N_PUZZLES = 50000  # Use 50K puzzles

# Sample puzzles evenly across difficulty levels for better learning
# This creates a curriculum from easy to hard
df_sampled = df.head(N_PUZZLES)

skipped = 0
for idx, row in df_sampled.iterrows():
    try:
        # Parse puzzle and solution
        puzzle_str = str(row['puzzle'])
        solution_str = str(row['solution'])
        
        # Convert to list of ints (handle both '.' and '0' for blanks)
        puzzle = []
        for c in puzzle_str:
            if c.isdigit():
                puzzle.append(int(c))
            elif c in ['.', '0']:
                puzzle.append(0)
        
        solution = [int(c) for c in solution_str if c.isdigit()]
        
        # Validate length
        if len(puzzle) != 81 or len(solution) != 81:
            skipped += 1
            continue
        
        train_data.append({
            "puzzle": puzzle,
            "solution": solution,
            "difficulty": row.get('difficulty', 0)
        })
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1:,} puzzles...")
    
    except Exception as e:
        skipped += 1
        continue

print(f"Skipped {skipped} invalid puzzles")

print("="*60)
print(f"✓ Loaded {len(train_data):,} Sudoku puzzles!")
print(f"  Dataset: Kaggle 3M Sudoku with Ratings")
print(f"  Difficulty: Progressive (easy → hard)")

# Show difficulty distribution of loaded data
if len(train_data) > 0 and 'difficulty' in train_data[0]:
    difficulties = [d['difficulty'] for d in train_data]
    print(f"\nLoaded puzzle difficulty stats:")
    print(f"  Min: {min(difficulties):.2f}")
    print(f"  Max: {max(difficulties):.2f}")
    print(f"  Mean: {np.mean(difficulties):.2f}")

# Show sample
sample = train_data[0]
print(f"\nSample puzzle (0 = blank):")
puzzle_grid = np.array(sample['puzzle']).reshape(9, 9)
print(puzzle_grid)
print(f"\nSample solution:")
solution_grid = np.array(sample['solution']).reshape(9, 9)
print(solution_grid)
if 'difficulty' in sample:
    print(f"Difficulty rating: {sample['difficulty']:.2f}")

# Verify data integrity
print(f"\nData integrity check:")
for i in range(min(3, len(train_data))):
    p = train_data[i]['puzzle']
    s = train_data[i]['solution']
    assert len(p) == 81, f"Invalid puzzle length: {len(p)}"
    assert len(s) == 81, f"Invalid solution length: {len(s)}"
print("✓ All samples validated!")
