# Debug: Find the actual maze image files
import os
import glob

print("Checking unzipped files...")
print("="*60)

# Check all files in current directory
all_files = glob.glob('**/*', recursive=True)
print(f"\nTotal files found: {len(all_files)}")

# Show directory structure
dirs = [f for f in all_files if os.path.isdir(f)]
print(f"\nDirectories found ({len(dirs)}):")
for d in dirs[:20]:
    print(f"  - {d}")

# Find image files
images = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"\nImage files found ({len(images)}):")
for img in images[:20]:
    print(f"  - {img}")

# Try different patterns
patterns = [
    '*.png',
    '*/*.png',
    '**/*.png',
    'rectangular_mazes_10x10/*.png',
    'mazes_10x10/*.png',
    '10x10/*.png'
]

print(f"\nTrying different glob patterns:")
for pattern in patterns:
    found = glob.glob(pattern, recursive=True)
    print(f"  {pattern}: {len(found)} files")
    if found:
        print(f"    First file: {found[0]}")
        break
