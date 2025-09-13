"""
Small 4x4 Sudoku Dataset Builder

This creates a smaller, more appropriate dataset for 4x4 sudoku training.
4x4 sudoku is much simpler than 9x9, so we need fewer examples.
"""

import json
import os
import random
from typing import Optional

import numpy as np
from argdantic import ArgParser
from common import PuzzleDatasetMetadata
from pydantic import BaseModel
from tqdm import tqdm

cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/sudoku-4x4-small"
    train_size: int = 200  # Much smaller for 4x4
    val_size: int = 50  # Smaller validation set
    test_size: int = 50  # Smaller test set
    num_aug: int = 10  # Fewer augmentations


def generate_4x4_sudoku() -> tuple[np.ndarray, np.ndarray]:
    """Generate a valid 4x4 sudoku puzzle and solution."""
    # Start with a valid 4x4 sudoku solution
    solution = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]])

    # Apply random transformations to create variety
    # Random digit permutation (1-4)
    digit_map = np.random.permutation(np.arange(1, 5))
    solution = digit_map[solution - 1]

    # Random row permutation within 2x2 blocks
    # Block 1: rows 0,1
    if random.random() < 0.5:
        solution[[0, 1]] = solution[[1, 0]]
    # Block 2: rows 2,3
    if random.random() < 0.5:
        solution[[2, 3]] = solution[[3, 2]]

    # Random column permutation within 2x2 blocks
    # Block 1: cols 0,1
    if random.random() < 0.5:
        solution[:, [0, 1]] = solution[:, [1, 0]]
    # Block 2: cols 2,3
    if random.random() < 0.5:
        solution[:, [2, 3]] = solution[:, [3, 2]]

    # Create puzzle by removing some cells (0 = blank)
    puzzle = solution.copy()

    # Remove cells randomly, ensuring at least 4 cells remain
    num_to_remove = random.randint(8, 12)  # Remove 8-12 cells
    positions = [(i, j) for i in range(4) for j in range(4)]
    random.shuffle(positions)

    for i in range(min(num_to_remove, len(positions))):
        puzzle[positions[i]] = 0

    return puzzle, solution


def shuffle_4x4_sudoku(
    board: np.ndarray, solution: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random transformations to create augmented versions."""
    # Create a random digit mapping: permutation of 1..4, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 5)), (1, 0))

    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5

    # Generate valid row permutation within 2x2 blocks
    # Block 1: rows 0,1
    block1_rows = np.random.permutation([0, 1])
    # Block 2: rows 2,3
    block2_rows = np.random.permutation([2, 3])
    row_perm = np.concatenate([block1_rows, block2_rows])

    # Similarly for columns (stacks)
    # Block 1: cols 0,1
    block1_cols = np.random.permutation([0, 1])
    # Block 2: cols 2,3
    block2_cols = np.random.permutation([2, 3])
    col_perm = np.concatenate([block1_cols, block2_cols])

    # Build 16->16 mapping
    mapping = np.array([row_perm[i // 4] * 4 + col_perm[i % 4] for i in range(16)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping
        new_board = x.flatten()[mapping].reshape(4, 4).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    """Convert and save a subset of 4x4 sudoku data."""
    # Determine dataset size
    if set_name == "train":
        size = config.train_size
    elif set_name == "val":
        size = config.val_size
    else:  # test
        size = config.test_size

    # Generate puzzles
    inputs = []
    labels = []

    print(f"Generating {size} {set_name} puzzles...")
    for _ in tqdm(range(size), desc=f"Generating {set_name}"):
        puzzle, solution = generate_4x4_sudoku()
        inputs.append(puzzle)
        labels.append(solution)

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"Processing {set_name}"), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_4x4_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)  # All puzzles have same ID for 4x4

        # Push group
        results["group_indices"].append(puzzle_id)

    # Convert to numpy arrays
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        # Ensure values are in range 0-4
        assert np.all((arr >= 0) & (arr <= 4))
        return arr

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata for 4x4 sudoku
    metadata = PuzzleDatasetMetadata(
        seq_len=16,  # 4x4 = 16 cells
        vocab_size=5,  # 0 (blank) + 1,2,3,4
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,  # All puzzles are same type
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"Saved {set_name} dataset with {len(results['inputs'])} examples")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Preprocess small 4x4 sudoku dataset for training."""
    print("Creating Small 4x4 Sudoku Dataset for CPU Training")
    print(f"Output directory: {config.output_dir}")
    print(f"Train size: {config.train_size} puzzles")
    print(f"Val size: {config.val_size} puzzles")
    print(f"Test size: {config.test_size} puzzles")
    print(f"Augmentations per train puzzle: {config.num_aug}")
    print(f"Total training examples: {config.train_size * (1 + config.num_aug)}")

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Process each subset
    convert_subset("train", config)
    convert_subset("val", config)
    convert_subset("test", config)

    print("Dataset creation complete!")


if __name__ == "__main__":
    cli()


