"""
4x4 Sudoku Evaluation Script

This script evaluates a trained HRM model on 4x4 sudoku puzzles.
Features:
- Load trained model
- Generate predictions
- Visualize puzzles and solutions
- Compute accuracy metrics
"""

import argparse
import json
import os

# Import HRM model from didactic implementation
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("HRM_didactic")
from hrm_model import create_hrm_model

# Import dataset
from train_4x4_sudoku import Sudoku4x4Dataset, create_data_loaders


def load_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Create model with same architecture as training
    model = create_hrm_model(
        vocab_size=5,  # 0-4 for 4x4 sudoku
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        max_seq_len=16,
        num_puzzle_ids=1,
        h_layers=2,
        l_layers=2,
        h_cycles=1,
        l_cycles=1,
        halt_max_steps=4,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    return model


def predict_sudoku(
    model: torch.nn.Module,
    input_puzzle: torch.Tensor,
    puzzle_id: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Predict solution for a single sudoku puzzle.

    Args:
        model: Trained HRM model
        input_puzzle: Input puzzle [seq_len]
        puzzle_id: Puzzle identifier [1]
        device: Device to run on

    Returns:
        Predicted solution [seq_len]
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        input_puzzle = input_puzzle.unsqueeze(0).to(device)
        puzzle_id = puzzle_id.unsqueeze(0).to(device)

        # Forward pass
        outputs = model(input_puzzle, puzzle_id)

        # Get predictions (argmax over vocabulary)
        predictions = torch.argmax(outputs["logits"], dim=-1)

        return predictions.squeeze(0)


def format_sudoku_grid(puzzle: np.ndarray, title: str = "Sudoku") -> str:
    """Format sudoku puzzle as a readable grid.

    Args:
        puzzle: 4x4 sudoku puzzle
        title: Title for the grid

    Returns:
        Formatted string representation
    """
    lines = [f"{title}:"]
    lines.append("┌─────┬─────┐")

    for i in range(4):
        row = "│"
        for j in range(4):
            val = puzzle[i, j]
            if val == 0:
                row += " · "
            else:
                row += f" {val} "
            if j == 1:
                row += "│"
        row += "│"
        lines.append(row)

        if i == 1:
            lines.append("├─────┼─────┤")

    lines.append("└─────┴─────┘")
    return "\n".join(lines)


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute accuracy metrics for sudoku predictions.

    Args:
        predictions: Predicted solutions [batch, seq_len]
        targets: Target solutions [batch, seq_len]

    Returns:
        Dictionary of accuracy metrics
    """
    # Overall accuracy (exact match)
    exact_matches = np.all(predictions == targets, axis=1)
    exact_accuracy = np.mean(exact_matches)

    # Cell-wise accuracy (ignoring padding/blank cells)
    # Only count non-zero cells in targets
    non_zero_mask = targets != 0
    if np.any(non_zero_mask):
        cell_accuracy = np.mean(predictions[non_zero_mask] == targets[non_zero_mask])
    else:
        cell_accuracy = 0.0

    # Puzzle completion accuracy (how many puzzles are valid sudokus)
    valid_sudokus = 0
    for i in range(len(predictions)):
        pred = predictions[i].reshape(4, 4)
        if is_valid_4x4_sudoku(pred):
            valid_sudokus += 1

    completion_accuracy = valid_sudokus / len(predictions)

    return {
        "exact_accuracy": exact_accuracy,
        "cell_accuracy": cell_accuracy,
        "completion_accuracy": completion_accuracy,
        "valid_sudokus": valid_sudokus,
        "total_puzzles": len(predictions),
    }


def is_valid_4x4_sudoku(grid: np.ndarray) -> bool:
    """Check if a 4x4 grid is a valid sudoku solution.

    Args:
        grid: 4x4 sudoku grid

    Returns:
        True if valid sudoku, False otherwise
    """
    # Check rows
    for i in range(4):
        row = grid[i]
        if len(set(row)) != 4 or not all(1 <= x <= 4 for x in row):
            return False

    # Check columns
    for j in range(4):
        col = grid[:, j]
        if len(set(col)) != 4 or not all(1 <= x <= 4 for x in col):
            return False

    # Check 2x2 blocks
    for block_i in range(0, 4, 2):
        for block_j in range(0, 4, 2):
            block = grid[block_i : block_i + 2, block_j : block_j + 2].flatten()
            if len(set(block)) != 4 or not all(1 <= x <= 4 for x in block):
                return False

    return True


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    num_examples: int = 10,
) -> Dict[str, Any]:
    """Evaluate model on test set.

    Args:
        model: Trained HRM model
        test_loader: Test data loader
        device: Device to run on
        num_examples: Number of examples to show

    Returns:
        Dictionary of evaluation results
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_inputs = []

    print("Evaluating model...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            puzzle_ids = batch["puzzle_ids"].to(device)

            # Get predictions
            predictions = torch.argmax(model(input_ids, puzzle_ids)["logits"], dim=-1)

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target_ids.cpu().numpy())
            all_inputs.append(input_ids.cpu().numpy())

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Compute accuracy metrics
    metrics = compute_accuracy(all_predictions, all_targets)

    # Show examples
    print(f"\nShowing {min(num_examples, len(all_inputs))} examples:")
    print("=" * 80)

    for i in range(min(num_examples, len(all_inputs))):
        input_puzzle = all_inputs[i].reshape(4, 4)
        target_solution = all_targets[i].reshape(4, 4)
        predicted_solution = all_predictions[i].reshape(4, 4)

        print(f"\nExample {i+1}:")
        print(format_sudoku_grid(input_puzzle, "Input Puzzle"))
        print(format_sudoku_grid(target_solution, "Target Solution"))
        print(format_sudoku_grid(predicted_solution, "Predicted Solution"))

        # Check if prediction is correct
        is_correct = np.array_equal(predicted_solution, target_solution)
        is_valid = is_valid_4x4_sudoku(predicted_solution)

        print(f"Correct: {'✓' if is_correct else '✗'}")
        print(f"Valid Sudoku: {'✓' if is_valid else '✗'}")
        print("-" * 40)

    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate 4x4 Sudoku HRM Model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sudoku-4x4",
        help="Directory containing test data",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument(
        "--num_examples", type=int, default=10, help="Number of examples to show"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Create test data loader
    _, _, test_loader = create_data_loaders(args.data_dir, batch_size=16)

    # Evaluate model
    metrics = evaluate_model(model, test_loader, args.device, args.num_examples)

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Exact Accuracy: {metrics['exact_accuracy']:.3f}")
    print(f"Cell Accuracy: {metrics['cell_accuracy']:.3f}")
    print(f"Completion Accuracy: {metrics['completion_accuracy']:.3f}")
    print(f"Valid Sudokus: {metrics['valid_sudokus']}/{metrics['total_puzzles']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

