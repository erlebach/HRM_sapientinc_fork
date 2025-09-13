"""Enhanced 4x4 Sudoku dataset builder with YAML configuration and augmentation support.

This is the file to use to build the dataset.
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from common import PuzzleDatasetMetadata
from utils.sudoku_augmentation import simple_digit_augmentation


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_4x4_sudoku_puzzle() -> tuple[np.ndarray, np.ndarray]:
    """Generate a valid 4x4 Sudoku puzzle and solution.

    Returns:
        tuple of (puzzle, solution) where puzzle has some cells blank (0)
    """
    # Create a valid 4x4 Sudoku solution
    solution = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]])

    # Randomly shuffle rows and columns to create variety
    # Shuffle rows within each 2x2 block
    row_perm = np.random.permutation(4)
    solution = solution[row_perm]

    # Shuffle columns within each 2x2 block
    col_perm = np.random.permutation(4)
    solution = solution[:, col_perm]

    # Create puzzle by removing 8-12 cells randomly
    puzzle = solution.copy()
    num_blank = random.randint(8, 12)
    blank_positions = random.sample(range(16), num_blank)

    for pos in blank_positions:
        row, col = pos // 4, pos % 4
        puzzle[row, col] = 0

    return puzzle, solution


def generate_augmented_samples(
    puzzle: np.ndarray,
    solution: np.ndarray,
    num_augmentations: int,
    augmentation_type: str = "digit_permutation",
) -> list[tuple[np.ndarray, np.ndarray, str, np.ndarray]]:
    """Generate augmented samples for a puzzle.

    Args:
        puzzle: Original puzzle
        solution: Original solution
        num_augmentations: Number of augmentations to generate
        augmentation_type: Type of augmentation to use

    Returns:
        list of (augmented_puzzle, augmented_solution, augmentation_id, digit_map)
    """
    augmented_samples = []

    # Add original sample with identity mapping
    identity_map = np.array([0, 1, 2, 3, 4])  # 0→0, 1→1, 2→2, 3→3, 4→4
    augmented_samples.append((puzzle.copy(), solution.copy(), "original", identity_map))

    # Generate augmentations
    for i in range(num_augmentations):
        if augmentation_type == "digit_permutation":
            aug_puzzle, aug_solution, digit_map = simple_digit_augmentation(
                puzzle, solution
            )
        else:
            # Fallback to original if augmentation type not supported
            aug_puzzle, aug_solution = puzzle.copy(), solution.copy()
            digit_map = identity_map

        augmented_samples.append((aug_puzzle, aug_solution, f"aug_{i}", digit_map))

    return augmented_samples


def check_puzzle_equivalence(puzzle1: np.ndarray, puzzle2: np.ndarray) -> bool:
    """Check if two puzzles are equivalent (same structure, different digits).

    Args:
        puzzle1: First puzzle
        puzzle2: Second puzzle

    Returns:
        True if puzzles are equivalent
    """
    # Two puzzles are equivalent if they have the same blank pattern
    return np.array_equal((puzzle1 == 0), (puzzle2 == 0))


def ensure_no_equivalents(
    puzzles: list[np.ndarray], solutions: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove equivalent puzzles from the dataset.

    Args:
        puzzles: list of puzzles
        solutions: list of solutions

    Returns:
        tuple of (filtered_puzzles, filtered_solutions)
    """
    unique_puzzles = []
    unique_solutions = []

    for i, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
        is_equivalent = False
        for existing_puzzle in unique_puzzles:
            if check_puzzle_equivalence(puzzle, existing_puzzle):
                is_equivalent = True
                break

        if not is_equivalent:
            unique_puzzles.append(puzzle)
            unique_solutions.append(solution)

    return unique_puzzles, unique_solutions


def build_dataset(config: dict[str, Any]) -> None:
    """Build the 4x4 Sudoku dataset with augmentations.

    Args:
        config: Configuration dictionary
    """
    data_cfg = config["data_generation"]

    # Set random seed for reproducibility
    random.seed(data_cfg["seed"])
    np.random.seed(data_cfg["seed"])

    # Create output directory
    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate ALL puzzles first (before splitting)
    total_puzzles = (
        data_cfg["train_size"] + data_cfg["val_size"] + data_cfg["test_size"]
    )
    print(f"Generating {total_puzzles} total puzzles...")

    all_puzzles_dict = {}
    for i in range(total_puzzles):
        puzzle, solution = generate_4x4_sudoku_puzzle()
        all_puzzles_dict[i] = [puzzle, solution]

    # Remove equivalent puzzles from entire dataset
    if data_cfg.get("ensure_no_equivalents", False):
        print("Removing equivalent puzzles from entire dataset...")
        all_puzzles = [all_puzzles_dict[i][0] for i in all_puzzles_dict]
        all_solutions = [all_puzzles_dict[i][1] for i in all_puzzles_dict]

        unique_puzzles, unique_solutions = ensure_no_equivalents(
            all_puzzles, all_solutions
        )

        # Rebuild dict with filtered puzzles
        all_puzzles_dict = {
            i: {"id": i, "puzzle": unique_puzzles[i], "solution": unique_solutions[i]}
            for i in range(len(unique_puzzles))
        }

        print(f"After removing equivalents: {len(all_puzzles_dict)} unique puzzles")

    # After removing equivalents and before splitting, generate augmentations
    print("Generating augmentations for all puzzles...")

    # Define identity mapping for original puzzles
    identity_map = np.array([0, 1, 2, 3, 4])  # 0→0, 1→1, 2→2, 3→3, 4→4

    # Generate all the augmentations
    augmentation_type = data_cfg["augmentation_type"]
    num_augmentations = data_cfg["num_augmentations"]

    for key, val in all_puzzles_dict.items():
        puzzle_id = val["id"]
        puzzle = val["puzzle"]
        solution = val["solution"]

        augmented_samples = generate_augmented_samples(
            puzzle, solution, num_augmentations, augmentation_type
        )

        augmented_samples_dict = {}
        for i in range(len(augmented_samples)):
            augmented_samples_dict[i] = {
                "id": (puzzle_id, i),
                "puzzle": augmented_samples[i][0],
                "solution": augmented_samples[i][1],
                # "augmentation_id": augmented_samples[i][2],
                "digit_map": augmented_samples[i][3],
            }
        all_puzzles_dict[key]["augmentations"] = augmented_samples_dict

    print(f"Generated augmentations for {len(all_puzzles_dict)} puzzles")

    # NOW split into train/val/test using the ratios
    original_total = (
        data_cfg["train_size"] + data_cfg["val_size"] + data_cfg["test_size"]
    )
    actual_total = len(all_puzzles_dict)

    # Calculate ratios
    rat1 = data_cfg["train_size"] / original_total  # train ratio
    rat2 = data_cfg["val_size"] / original_total  # val ratio
    rat3 = data_cfg["test_size"] / original_total  # test ratio

    # Apply ratios to actual dataset size
    train_size = int(actual_total * rat1)
    val_size = int(actual_total * rat2)
    test_size = actual_total - train_size - val_size  # Ensure we use all samples

    print(
        f"Original sizes: train={data_cfg['train_size']}, val={data_cfg['val_size']}, test={data_cfg['test_size']}"
    )
    print(f"New sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"Total: {train_size + val_size + test_size} (should equal {actual_total})")

    # Split the puzzles using the new sizes
    base = 0
    train_puzzles_dict = {i: all_puzzles_dict[base + i] for i in range(train_size)}
    base = train_size
    val_puzzles_dict = {i: all_puzzles_dict[base + i] for i in range(val_size)}
    base = train_size + val_size
    test_puzzles_dict = {i: all_puzzles_dict[base + i] for i in range(test_size)}
    # print(test_puzzles_dict)
    # quit()

    # Save the split dictionaries directly
    for split_name, puzzles_dict in [
        ("train", train_puzzles_dict),
        ("val", val_puzzles_dict),
        ("test", test_puzzles_dict),
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        # Save the entire dictionary structure
        with open(split_dir / "puzzles_dict.pkl", "wb") as f:
            pickle.dump(puzzles_dict, f)

        print(f"Saved {split_name} dataset: {len(puzzles_dict)} puzzles")

    print("Dataset generation complete!")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Build 4x4 Sudoku dataset with augmentations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sudoku_data_generation.yaml",
        help="Path to YAML configuration file (default: config/sudoku_data_generation.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Build dataset
    build_dataset(config)

    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
