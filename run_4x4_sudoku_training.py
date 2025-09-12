#!/usr/bin/env python3
"""
Complete 4x4 Sudoku Training Pipeline

This script runs the complete pipeline:
1. Generate 4x4 sudoku dataset
2. Train HRM model
3. Evaluate trained model

Usage:
    python run_4x4_sudoku_training.py
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status.

    Args:
        cmd: Command to run as list
        description: Description of what the command does

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def main():
    """Run the complete 4x4 sudoku training pipeline."""
    print("4x4 Sudoku HRM Training Pipeline")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("dataset").exists():
        print("Error: Please run this script from the project root directory")
        print("Expected structure: dataset/, HRM_didactic/, etc.")
        sys.exit(1)

    # Step 1: Generate dataset
    print("\nStep 1: Generating 4x4 Sudoku Dataset")
    success = run_command(
        [
            "python",
            "dataset/build_4x4_sudoku_dataset.py",
            "--output_dir",
            "data/sudoku-4x4",
            "--train_size",
            "1000",
            "--val_size",
            "200",
            "--test_size",
            "200",
            "--num_aug",
            "3",
        ],
        "Generate 4x4 sudoku dataset",
    )

    if not success:
        print("Failed to generate dataset. Exiting.")
        sys.exit(1)

    # Step 2: Train model
    print("\nStep 2: Training HRM Model")
    success = run_command(
        ["python", "train_4x4_sudoku.py"], "Train HRM model on 4x4 sudoku"
    )

    if not success:
        print("Failed to train model. Exiting.")
        sys.exit(1)

    # Step 3: Evaluate model
    print("\nStep 3: Evaluating Trained Model")
    checkpoint_path = "checkpoints_4x4/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"Warning: Best model checkpoint not found at {checkpoint_path}")
        print("Looking for any checkpoint...")
        checkpoint_dir = Path("checkpoints_4x4")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                checkpoint_path = str(checkpoints[0])
                print(f"Using checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found. Skipping evaluation.")
                sys.exit(1)
        else:
            print("No checkpoint directory found. Skipping evaluation.")
            sys.exit(1)

    success = run_command(
        [
            "python",
            "evaluate_4x4_sudoku.py",
            "--checkpoint",
            checkpoint_path,
            "--data_dir",
            "data/sudoku-4x4",
            "--device",
            "cpu",
            "--num_examples",
            "5",
        ],
        "Evaluate trained model",
    )

    if not success:
        print("Failed to evaluate model.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("✓ Dataset generated")
    print("✓ Model trained")
    print("✓ Model evaluated")
    print("\nFiles created:")
    print("- data/sudoku-4x4/ (dataset)")
    print("- checkpoints_4x4/ (model checkpoints)")
    print("\nTo run evaluation again:")
    print(f"python evaluate_4x4_sudoku.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()

