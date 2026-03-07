# ============================================================
# UPDATED: Load Dataset from SakanaAI/Sudoku-Bench
# Replace the "Load Dataset from HuggingFace" cell with this code
# ============================================================

from datasets import load_dataset

print("Loading Sudoku dataset from HuggingFace...")
print("Dataset: SakanaAI/Sudoku-Bench (Challenge Sudoku puzzles)")
print("="*60)

# Load dataset - using challenge_100 config
dataset = load_dataset("SakanaAI/Sudoku-Bench", "challenge_100", split="test")
print(f"Total puzzles available: {len(dataset):,}")

# We have 100 puzzles - let's augment by loading other splits too
print("\nLoading additional puzzle sets...")
all_data = []

# Load all available challenge puzzles
configs = ["challenge_100", "challenge_10", "challenge_1"]
for config in configs:
    try:
        ds = load_dataset("SakanaAI/Sudoku-Bench", config, split="test")
        all_data.extend(list(ds))
        print(f"  Loaded {len(ds)} puzzles from {config}")
    except:
        print(f"  Skipping {config} (not available)")

print(f"\nTotal unique puzzles: {len(all_data)}")

# Convert to our format
train_data = []

for item in all_data:
    # Parse initial_board and solution from SakanaAI format
    # initial_board: string with '.' for blanks (like ".3.5..7..")
    # solution: string of 81 digits (like "234567891...")
    initial_board = item['initial_board']
    solution_str = item['solution']
    
    # Convert to list of ints ('.' = 0 for blanks)
    puzzle = [int(c) if c != '.' else 0 for c in initial_board]
    solution = [int(c) for c in solution_str]
    
    train_data.append({
        "puzzle": puzzle,
        "solution": solution
    })

# Augment data by creating variations (since we only have ~300 puzzles)
# We'll repeat the dataset multiple times with different random initializations
print(f"\nAugmenting dataset by 100x (different random guesses)...")
original_data = train_data.copy()
train_data = original_data * 100  # 100x augmentation

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
print(f"\nSolution:")
solution_grid = np.array(sample['solution']).reshape(9, 9)
print(solution_grid)
