# ============================================================
# DIAGNOSTIC: Check SakanaAI Dataset Format
# Run this cell FIRST to see the actual data structure
# ============================================================

from datasets import load_dataset
import numpy as np

# Load one sample
dataset = load_dataset("SakanaAI/Sudoku-Bench", "challenge_100", split="test")
sample = dataset[0]

print("Dataset Sample Structure:")
print("="*60)
for key, value in sample.items():
    print(f"\n{key}:")
    print(f"  Type: {type(value)}")
    if isinstance(value, str):
        print(f"  Length: {len(value)}")
        print(f"  First 100 chars: {value[:100]}...")
    elif isinstance(value, (list, np.ndarray)):
        print(f"  Length: {len(value)}")
        print(f"  Content: {value}")
    else:
        print(f"  Value: {value}")

print("\n" + "="*60)
print("\nTrying to parse initial_board...")
initial = sample['initial_board']
print(f"Initial board type: {type(initial)}")
print(f"Initial board: {initial}")

# Try different parsing methods
print("\n1. Direct conversion:")
try:
    parsed = [int(c) if c != '.' else 0 for c in initial]
    print(f"   Length: {len(parsed)}")
    print(f"   First 20: {parsed[:20]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Check if it's a grid string:")
print(f"   Contains newlines: {'\\n' in initial}")
print(f"   Contains spaces: {' ' in initial}")

print("\n3. Rows/Cols fields:")
print(f"   Rows: {sample.get('rows', 'N/A')}")
print(f"   Cols: {sample.get('cols', 'N/A')}")
