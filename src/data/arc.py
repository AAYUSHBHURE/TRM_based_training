"""
ARC-AGI Dataset Loader for TRM Training

Loads ARC-AGI-1 (800 train / 400 eval) and ARC-AGI-2 (1K train / 120 eval)
from JSON format with augmentation support.

Sources:
- ARC-AGI-1: github.com/fchollet/ARC-AGI, kaggle.com/datasets/evanhislupus/arc-agi-dataset
- ARC-AGI-2: github.com/arcprize/ARC-AGI-2, kaggle.com/datasets/boristown/arc-agi-2
"""

import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import urllib.request
import zipfile
import os


ARC_AGI_1_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_AGI_2_URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"


def download_arc_dataset(version: int = 1, output_dir: str = "data/arc") -> Path:
    """Download ARC-AGI dataset from GitHub."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    url = ARC_AGI_1_URL if version == 1 else ARC_AGI_2_URL
    zip_path = output_path / f"arc_agi_{version}.zip"
    
    if not zip_path.exists():
        print(f"Downloading ARC-AGI-{version}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
    
    # Find data directory
    if version == 1:
        data_dir = output_path / "ARC-AGI-master" / "data"
    else:
        data_dir = output_path / "ARC-AGI-2-main" / "data"
    
    return data_dir


def load_arc_task(task_path: Path) -> Dict[str, Any]:
    """Load a single ARC task from JSON file."""
    with open(task_path, 'r') as f:
        task = json.load(f)
    return task


def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    """Convert ARC grid (list of lists) to numpy array."""
    return np.array(grid, dtype=np.int32)


def pad_grid(grid: np.ndarray, max_size: int = 30) -> np.ndarray:
    """Pad grid to max_size x max_size."""
    h, w = grid.shape
    padded = np.zeros((max_size, max_size), dtype=np.int32)
    padded[:h, :w] = grid
    return padded


def color_permutation(grid: np.ndarray, perm: Optional[List[int]] = None) -> np.ndarray:
    """Apply color permutation (0-9 -> permuted 0-9)."""
    if perm is None:
        perm = list(range(10))
        random.shuffle(perm[1:])  # Keep 0 (background) fixed
    
    result = grid.copy()
    for old, new in enumerate(perm):
        result[grid == old] = new
    return result


def apply_dihedral(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply dihedral transform (8 possibilities)."""
    if transform_id == 0:
        return grid
    elif transform_id == 1:
        return np.rot90(grid, k=1)
    elif transform_id == 2:
        return np.rot90(grid, k=2)
    elif transform_id == 3:
        return np.rot90(grid, k=3)
    elif transform_id == 4:
        return np.fliplr(grid)
    elif transform_id == 5:
        return np.flipud(grid)
    elif transform_id == 6:
        return grid.T
    elif transform_id == 7:
        return np.rot90(grid.T, k=2)
    return grid


def translate_grid(grid: np.ndarray, dy: int, dx: int, max_size: int = 30) -> np.ndarray:
    """Translate grid by (dy, dx) within max_size bounds."""
    h, w = grid.shape
    result = np.zeros((max_size, max_size), dtype=np.int32)
    
    # Compute valid regions
    src_y1, src_y2 = max(0, -dy), min(h, max_size - dy)
    src_x1, src_x2 = max(0, -dx), min(w, max_size - dx)
    dst_y1, dst_y2 = max(0, dy), min(max_size, h + dy)
    dst_x1, dst_x2 = max(0, dx), min(max_size, w + dx)
    
    if src_y2 > src_y1 and src_x2 > src_x1:
        result[dst_y1:dst_y2, dst_x1:dst_x2] = grid[src_y1:src_y2, src_x1:src_x2]
    
    return result


def augment_arc_task(
    task: Dict[str, Any],
    n_augments: int = 1000,
    include_dihedral: bool = True,
    include_color_perm: bool = True,
    include_translation: bool = True
) -> List[Dict[str, Any]]:
    """Generate augmented versions of an ARC task."""
    augmented = [task]  # Include original
    
    for _ in range(n_augments):
        aug_task = {"train": [], "test": []}
        
        # Choose transforms
        transform_id = random.randint(0, 7) if include_dihedral else 0
        color_perm = None
        if include_color_perm:
            color_perm = list(range(10))
            random.shuffle(color_perm[1:])
        dy, dx = 0, 0
        if include_translation:
            dy, dx = random.randint(-5, 5), random.randint(-5, 5)
        
        # Apply to all examples
        for split in ["train", "test"]:
            for example in task.get(split, []):
                inp = grid_to_array(example["input"])
                out = grid_to_array(example["output"])
                
                if include_dihedral:
                    inp = apply_dihedral(inp, transform_id)
                    out = apply_dihedral(out, transform_id)
                
                if include_color_perm:
                    inp = color_permutation(inp, color_perm)
                    out = color_permutation(out, color_perm)
                
                if include_translation:
                    inp = translate_grid(inp, dy, dx)
                    out = translate_grid(out, dy, dx)
                
                aug_task[split].append({
                    "input": inp.tolist(),
                    "output": out.tolist()
                })
        
        augmented.append(aug_task)
    
    return augmented


def load_arc_dataset(
    data_dir: Path,
    split: str = "training",
    max_tasks: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load all tasks from a split directory."""
    split_dir = data_dir / split
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return []
    
    tasks = []
    task_files = sorted(split_dir.glob("*.json"))
    
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    for task_file in task_files:
        task = load_arc_task(task_file)
        task["id"] = task_file.stem
        tasks.append(task)
    
    return tasks


def prepare_arc_for_trm(
    task: Dict[str, Any],
    max_size: int = 30
) -> Dict[str, Any]:
    """
    Prepare ARC task for TRM training.
    
    Flattens demo pairs into conditioning context, extracts test I/O.
    """
    # Collect demo pairs (for x conditioning)
    demo_inputs = []
    demo_outputs = []
    for example in task.get("train", []):
        inp = pad_grid(grid_to_array(example["input"]), max_size)
        out = pad_grid(grid_to_array(example["output"]), max_size)
        demo_inputs.append(inp.flatten())
        demo_outputs.append(out.flatten())
    
    # Test pair
    test_examples = []
    for example in task.get("test", []):
        inp = pad_grid(grid_to_array(example["input"]), max_size)
        out = pad_grid(grid_to_array(example["output"]), max_size)
        test_examples.append({
            "input": inp.flatten().tolist(),
            "output": out.flatten().tolist()
        })
    
    return {
        "id": task.get("id", "unknown"),
        "demo_inputs": [d.tolist() for d in demo_inputs],
        "demo_outputs": [d.tolist() for d in demo_outputs],
        "test_examples": test_examples,
        "n_demos": len(demo_inputs)
    }


def generate_dataset(
    version: int = 1,
    output_dir: str = "data/arc",
    n_augments: int = 1000,
    download: bool = True
) -> None:
    """
    Generate ARC-AGI dataset for TRM training.
    
    Args:
        version: 1 for ARC-AGI-1, 2 for ARC-AGI-2
        output_dir: Output directory
        n_augments: Number of augmentations per task
        download: Whether to download if not present
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if download:
        data_dir = download_arc_dataset(version, output_dir)
    else:
        if version == 1:
            data_dir = output_path / "ARC-AGI-master" / "data"
        else:
            data_dir = output_path / "ARC-AGI-2-main" / "data"
    
    print(f"Loading ARC-AGI-{version} from {data_dir}...")
    
    # Load training tasks
    train_tasks = load_arc_dataset(data_dir, "training")
    print(f"Loaded {len(train_tasks)} training tasks")
    
    # Load evaluation tasks
    eval_tasks = load_arc_dataset(data_dir, "evaluation")
    print(f"Loaded {len(eval_tasks)} evaluation tasks")
    
    # Prepare and augment
    train_data = []
    for task in train_tasks:
        prepared = prepare_arc_for_trm(task)
        train_data.append(prepared)
        
        # Limited augmentation for memory efficiency
        if n_augments > 0:
            aug_tasks = augment_arc_task(task, min(n_augments, 10))
            for i, aug_task in enumerate(aug_tasks[1:]):  # Skip original
                aug_prepared = prepare_arc_for_trm(aug_task)
                aug_prepared["id"] = f"{task['id']}_aug{i:02d}"
                train_data.append(aug_prepared)
    
    eval_data = [prepare_arc_for_trm(task) for task in eval_tasks]
    
    # Save
    with open(output_path / f"arc{version}_train.json", "w") as f:
        json.dump(train_data, f)
    
    with open(output_path / f"arc{version}_eval.json", "w") as f:
        json.dump(eval_data, f)
    
    print(f"Saved {len(train_data)} train, {len(eval_data)} eval to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare ARC-AGI dataset")
    parser.add_argument("--version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--output", type=str, default="data/arc")
    parser.add_argument("--augments", type=int, default=10)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    
    generate_dataset(args.version, args.output, args.augments, not args.no_download)
