"""Enhanced 4x4 Sudoku dataset builder with YAML configuration and augmentation support.

This is the file to use to build the dataset.
"""

import argparse
import json
import os
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
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate augmented samples for a puzzle.

    Args:
        puzzle: Original puzzle
        solution: Original solution
        num_augmentations: Number of augmentations to generate
        augmentation_type: Type of augmentation to use

    Returns:
        list of (augmented_puzzle, augmented_solution, augmentation_id)
    """
    augmented_samples = []

    # Add original sample
    augmented_samples.append((puzzle.copy(), solution.copy(), "original"))

    # Generate augmentations
    for i in range(num_augmentations):
        if augmentation_type == "digit_permutation":
            aug_puzzle, aug_solution = simple_digit_augmentation(puzzle, solution)
        else:
            # Fallback to original if augmentation type not supported
            aug_puzzle, aug_solution = puzzle.copy(), solution.copy()

        augmented_samples.append((aug_puzzle, aug_solution, f"aug_{i}"))

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

    # Generate base puzzles
    print(f"Generating {data_cfg['train_size']} training puzzles...")
    train_puzzles_dict = {}
    for i in range(data_cfg["train_size"]):
        puzzle, solution = generate_4x4_sudoku_puzzle()
        train_puzzles_dict[i] = [puzzle, solution]

    print(f"Generating {data_cfg['val_size']} validation puzzles...")
    val_puzzles_dict = {}
    for i in range(data_cfg["val_size"]):
        puzzle, solution = generate_4x4_sudoku_puzzle()
        val_puzzles_dict[i] = [puzzle, solution]

    print(f"Generating {data_cfg['test_size']} test puzzles...")
    test_puzzles_dict = {}
    for i in range(data_cfg["test_size"]):
        puzzle, solution = generate_4x4_sudoku_puzzle()
        test_puzzles_dict[i] = [puzzle, solution]

    # Remove equivalent puzzles if requested
    if data_cfg.get("ensure_no_equivalents", False):
        print("Removing equivalent puzzles...")
        # Convert dicts to lists for equivalence checking
        train_puzzles = [train_puzzles_dict[i][0] for i in train_puzzles_dict]
        train_solutions = [train_puzzles_dict[i][1] for i in train_puzzles_dict]
        val_puzzles = [val_puzzles_dict[i][0] for i in val_puzzles_dict]
        val_solutions = [val_puzzles_dict[i][1] for i in val_puzzles_dict]
        test_puzzles = [test_puzzles_dict[i][0] for i in test_puzzles_dict]
        test_solutions = [test_puzzles_dict[i][1] for i in test_puzzles_dict]

        # Check for equivalents
        train_puzzles, train_solutions = ensure_no_equivalents(
            train_puzzles, train_solutions
        )
        val_puzzles, val_solutions = ensure_no_equivalents(val_puzzles, val_solutions)
        test_puzzles, test_solutions = ensure_no_equivalents(
            test_puzzles, test_solutions
        )

        # Rebuild dicts with filtered puzzles
        train_puzzles_dict = {
            i: [train_puzzles[i], train_solutions[i]] for i in range(len(train_puzzles))
        }
        val_puzzles_dict = {
            i: [val_puzzles[i], val_solutions[i]] for i in range(len(val_puzzles))
        }
        test_puzzles_dict = {
            i: [test_puzzles[i], test_solutions[i]] for i in range(len(test_puzzles))
        }

        print(
            f"After removing equivalents: {len(train_puzzles_dict)} train, {len(val_puzzles_dict)} val, {len(test_puzzles_dict)} test"
        )

    # Generate augmented samples for each split
    for split_name, puzzles_dict in [
        ("train", train_puzzles_dict),
        ("val", val_puzzles_dict),
        ("test", test_puzzles_dict),
    ]:
        print(f"Generating augmented samples for {split_name}...")

        # Create split directory
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        # Generate augmented samples for each puzzle
        for puzzle_id, (puzzle, solution) in puzzles_dict.items():
            augmented_samples = generate_augmented_samples(
                puzzle,
                solution,
                data_cfg["num_augmentations"],
                data_cfg["augmentation_type"],
            )

            # Append augmentations to the puzzle entry
            puzzles_dict[puzzle_id].extend(augmented_samples)

        # Flatten the data for saving
        all_puzzles = []
        all_solutions = []
        all_identifiers = []
        puzzle_groups = {}  # Dictionary: {group_id: [indices]}

        for puzzle_id, puzzle_data in puzzles_dict.items():
            original_puzzle, original_solution = puzzle_data[0], puzzle_data[1]
            augmentations = puzzle_data[2:]  # All augmentations after the original

            # Start a new group for this puzzle
            puzzle_groups[puzzle_id] = []

            # Add original puzzle
            all_puzzles.append(original_puzzle)
            all_solutions.append(original_solution)
            all_identifiers.append(f"{split_name}_{puzzle_id}_original")
            puzzle_groups[puzzle_id].append(len(all_puzzles) - 1)

            # Add augmentations
            for aug_puzzle, aug_solution, aug_id in augmentations:
                all_puzzles.append(aug_puzzle)
                all_solutions.append(aug_solution)
                all_identifiers.append(f"{split_name}_{puzzle_id}_{aug_id}")
                puzzle_groups[puzzle_id].append(len(all_puzzles) - 1)

        # Convert to numpy arrays
        all_puzzles = np.array(all_puzzles)
        all_solutions = np.array(all_solutions)

        # Save data
        np.save(split_dir / "all__inputs.npy", all_puzzles)
        np.save(split_dir / "all__labels.npy", all_solutions)
        np.save(split_dir / "puzzle_groups.npy", puzzle_groups)

        # Save identifiers
        with open(split_dir / "identifiers.json", "w") as f:
            json.dump(all_identifiers, f)

        # Create metadata
        metadata = PuzzleDatasetMetadata(
            num_puzzles=len(puzzles_dict),
            vocab_size=5,
            ignore_label_id=0,
            blank_identifier_id=0,
            seq_len=16,
            mean_puzzle_examples=len(all_puzzles) / len(puzzles_dict),
            sets=["train", "val", "test"],
            pad_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(puzzles_dict),
        )

        with open(split_dir / "dataset.json", "w") as f:
            json.dump(metadata.__dict__, f)

        print(f"Saved {len(all_puzzles)} samples to {split_dir}")
        print(f"  - {len(puzzles_dict)} original puzzles")
        print(f"  - {len(all_puzzles) - len(puzzles_dict)} augmented samples")


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
