#!/usr/bin/env python3
"""
Test script for 4x4 Sudoku dataset loading

This script tests the dataset loading without requiring PyTorch.
"""

import json
import os

import numpy as np


def test_dataset_loading(data_dir: str = "dataset/data/sudoku-4x4"):
    """Test loading the 4x4 sudoku dataset."""
    print("Testing 4x4 Sudoku Dataset Loading")
    print("=" * 40)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        return False

    # Test each split
    for split in ["train", "val", "test"]:
        print(f"\nTesting {split} split:")
        split_dir = os.path.join(data_dir, split)

        if not os.path.exists(split_dir):
            print(f"  Error: {split} directory does not exist!")
            return False

        # Load data files
        try:
            inputs = np.load(os.path.join(split_dir, "all__inputs.npy"))
            labels = np.load(os.path.join(split_dir, "all__labels.npy"))
            puzzle_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"))

            print(f"  ✓ Loaded {len(inputs)} examples")
            print(f"  ✓ Input shape: {inputs.shape}")
            print(f"  ✓ Label shape: {labels.shape}")
            print(f"  ✓ Puzzle IDs shape: {puzzle_ids.shape}")

            # Check data ranges
            print(f"  ✓ Input range: {inputs.min()} - {inputs.max()}")
            print(f"  ✓ Label range: {labels.min()} - {labels.max()}")
            print(f"  ✓ Puzzle IDs range: {puzzle_ids.min()} - {puzzle_ids.max()}")

            # Show sample
            print(f"  ✓ Sample input: {inputs[0]}")
            print(f"  ✓ Sample label: {labels[0]}")

        except Exception as e:
            print(f"  Error loading {split} data: {e}")
            return False

        # Load metadata
        try:
            with open(os.path.join(split_dir, "dataset.json"), "r") as f:
                metadata = json.load(f)

            print(f"  ✓ Metadata loaded:")
            print(f"    - Sequence length: {metadata['seq_len']}")
            print(f"    - Vocabulary size: {metadata['vocab_size']}")
            print(f"    - Puzzle identifiers: {metadata['num_puzzle_identifiers']}")
            print(f"    - Total groups: {metadata['total_groups']}")

        except Exception as e:
            print(f"  Error loading {split} metadata: {e}")
            return False

    print("\n✓ All dataset loading tests passed!")
    return True


def show_sample_puzzles(
    data_dir: str = "dataset/data/sudoku-4x4", num_samples: int = 3
):
    """Show sample puzzles in a readable format."""
    print(f"\nSample Puzzles from {data_dir}")
    print("=" * 50)

    # Load train data
    train_dir = os.path.join(data_dir, "train")
    inputs = np.load(os.path.join(train_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(train_dir, "all__labels.npy"))

    for i in range(min(num_samples, len(inputs))):
        print(f"\nPuzzle {i+1}:")

        # Format input puzzle
        input_puzzle = inputs[i].reshape(4, 4)
        print("Input:")
        for row in input_puzzle:
            row_str = " ".join(str(x) if x != 0 else "." for x in row)
            print(f"  {row_str}")

        # Format solution
        solution = labels[i].reshape(4, 4)
        print("Solution:")
        for row in solution:
            row_str = " ".join(str(x) for x in row)
            print(f"  {row_str}")

        print("-" * 20)


def main():
    """Main function."""
    data_dir = "dataset/data/sudoku-4x4"

    # Test dataset loading
    if test_dataset_loading(data_dir):
        # Show sample puzzles
        show_sample_puzzles(data_dir, num_samples=3)

        print("\n" + "=" * 50)
        print("Dataset is ready for training!")
        print("To start training, run:")
        print("python3 sudoku4x4.py")
        print("=" * 50)
    else:
        print("\nDataset loading failed. Please check the data directory.")


if __name__ == "__main__":
    main()


