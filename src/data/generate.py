"""
TRM Data Generation CLI

Unified interface for generating all TRM training datasets.

Usage:
    python -m src.data.generate --domain sudoku --size 1000 --output data/sudoku
    python -m src.data.generate --domain maze --size 1000 --output data/maze
    python -m src.data.generate --domain arc --version 1 --output data/arc
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate TRM training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Domains:
  sudoku    Sudoku-Extreme (9x9, 20-25 hints, 87% target)
  maze      Maze-Hard (30x30, path >110 steps, 85% target)
  arc       ARC-AGI-1/2 (abstract reasoning, 45%/8% target)

Examples:
  python -m src.data.generate --domain sudoku --size 1000
  python -m src.data.generate --domain maze --size 500 --width 50 --height 50
  python -m src.data.generate --domain arc --version 2 --augments 100
        """
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["sudoku", "maze", "arc"],
        help="Dataset domain to generate"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Number of base examples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/{domain})"
    )
    parser.add_argument(
        "--augments",
        type=int,
        default=10,
        help="Number of augmentations per example"
    )
    
    # Maze-specific
    parser.add_argument("--width", type=int, default=30, help="Maze width")
    parser.add_argument("--height", type=int, default=30, help="Maze height")
    
    # ARC-specific
    parser.add_argument("--version", type=int, default=1, choices=[1, 2], help="ARC version")
    parser.add_argument("--no-download", action="store_true", help="Skip ARC download")
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output is None:
        args.output = f"data/{args.domain}"
    
    # Generate based on domain
    if args.domain == "sudoku":
        from src.data.sudoku import generate_dataset
        generate_dataset(
            n_examples=args.size,
            n_augments=args.augments,
            output_dir=args.output
        )
    
    elif args.domain == "maze":
        from src.data.maze import generate_dataset
        generate_dataset(
            n_examples=args.size,
            output_dir=args.output,
            width=args.width,
            height=args.height
        )
    
    elif args.domain == "arc":
        from src.data.arc import generate_dataset
        generate_dataset(
            version=args.version,
            output_dir=args.output,
            n_augments=args.augments,
            download=not args.no_download
        )
    
    print(f"\n✓ Dataset generation complete: {args.output}")


if __name__ == "__main__":
    main()
