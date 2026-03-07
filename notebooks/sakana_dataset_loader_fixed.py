# ============================================================
# FIXED: Load SakanaAI/Sudoku-Bench with Validation
# Replace the data loading cell with this code
# ============================================================

from datasets import load_dataset

print("Loading Sudoku dataset from HuggingFace...")
print("Dataset: SakanaAI/Sudoku-Bench (Challenge Sudoku puzzles)")
print("="*60)

# Load dataset
dataset = load_dataset("SakanaAI/Sudoku-Bench", "challenge_100", split="test")
print(f"Total puzzles available: {len(dataset):,}")

# Convert to our format with validation
train_data = []
skipped = 0

for i, item in enumerate(dataset):
    try:
        # Parse initial_board and solution
        initial_board = item['initial_board']
        solution_str = item['solution']
        
        # Validate lengths
        if len(initial_board) != 81 or len(solution_str) != 81:
            print(f"  Skipping puzzle {i}: invalid length (initial={len(initial_board)}, solution={len(solution_str)})")
            skipped += 1
            continue
        
        # Convert to list of ints ('.' = 0 for blanks)
        puzzle = [int(c) if c != '.' else 0 for c in initial_board]
        solution = [int(c) for c in solution_str]
        
        # Validate all values are 0-9
        if not all(0 <= x <= 9 for x in puzzle + solution):
            print(f"  Skipping puzzle {i}: invalid digit values")
            skipped += 1
            continue
        
        train_data.append({
            "puzzle": puzzle,
            "solution": solution
        })
        
    except Exception as e:
        print(f"  Skipping puzzle {i}: {e}")
        skipped += 1
        continue

print(f"\nLoaded {len(train_data)} valid puzzles (skipped {skipped})")

# Augment data by repeating with different random initializations
# Each puzzle will be seen 100 times with different random guesses
print(f"\nAugmenting dataset by 100x (different random guesses)...")
original_data = train_data.copy()
train_data = original_data * 100

print("="*60)
print(f"✓ Loaded {len(original_data):,} unique Sudoku puzzles!")
print(f"  Total training samples (with augmentation): {len(train_data):,}")
print(f"  Difficulty: Challenge (from sudokupad.app)")
print(f"  Source: SakanaAI/Sudoku-Bench")

# Show sample
sample = train_data[0]
print(f"\nSample puzzle (0 = blank):")
puzzle_grid = np.array(sample['puzzle']).reshape(9, 9)
print(puzzle_grid)
print(f"\nSample solution:")
solution_grid = np.array(sample['solution']).reshape(9, 9)
print(solution_grid)

# Verify data integrity
print(f"\nData integrity check:")
for i in range(min(5, len(train_data))):
    p = train_data[i]['puzzle']
    s = train_data[i]['solution']
    print(f"  Puzzle {i}: len={len(p)}, solution len={len(s)}")
    assert len(p) == 81, f"Invalid puzzle length: {len(p)}"
    assert len(s) == 81, f"Invalid solution length: {len(s)}"
print("✓ All samples validated!")
