#!/usr/bin/env python3
"""
4x4 Sudoku HRM Training Script

This script reads the 4x4 sudoku dataset files and trains the HRM model.
It provides a complete training pipeline optimized for CPU training.

Usage:
    python sudoku4x4.py [options]

Options:
    --data_dir: Directory containing the dataset (default: dataset/data/sudoku-4x4)
    --epochs: Number of training epochs (default: 10)
    --batch_size: Batch size for training (default: 8)
    --learning_rate: Learning rate (default: 1e-4)
    --device: Device to use (default: cpu)
    --save_dir: Directory to save checkpoints (default: checkpoints_4x4)
"""

import argparse
import hashlib
import json
import os

# Import HRM model from didactic implementation
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("HRM_didactic")
from pathlib import Path

import yaml
from hrm_model import create_hrm_model


# Add this configuration loading function
def load_config(config_path: str = "config/sudoku_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class Sudoku4x4Dataset(Dataset):
    """4x4 Sudoku dataset for training."""

    def __init__(self, data_dir: str, split: str = "train"):
        """Initialize dataset.

        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.split = split

        # Load data
        data_path = os.path.join(data_dir, split)
        self.inputs = np.load(os.path.join(data_path, "all__inputs.npy"))
        self.labels = np.load(os.path.join(data_path, "all__labels.npy"))
        self.puzzle_ids = np.load(
            os.path.join(data_path, "all__puzzle_identifiers.npy")
        )

        # Load metadata
        with open(os.path.join(data_path, "dataset.json"), "r") as f:
            self.metadata = json.load(f)

        print(f"Loaded {split} dataset: {len(self.inputs)} examples")
        print(f"Sequence length: {self.metadata['seq_len']}")
        print(f"Vocabulary size: {self.metadata['vocab_size']}")
        print(f"Sample input: {self.inputs[0]}")
        print(f"Sample label: {self.labels[0]}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.

        Returns:
            Dictionary containing:
            - input_ids: Input puzzle [seq_len]
            - target_ids: Target solution [seq_len]
            - puzzle_ids: Puzzle identifier [1]
        """
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.labels[idx], dtype=torch.long),
            "puzzle_ids": torch.tensor(self.puzzle_ids[idx], dtype=torch.long),
        }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train/val/test splits.

    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes (0 for CPU)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = Sudoku4x4Dataset(data_dir, "train")
    val_dataset = Sudoku4x4Dataset(data_dir, "val")
    test_dataset = Sudoku4x4Dataset(data_dir, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    lm_weight: float = 1.0,
    q_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined loss for HRM model.

    Args:
        outputs: Model outputs containing logits and Q-values
        targets: Target token IDs
        lm_weight: Weight for language modeling loss
        q_weight: Weight for Q-learning loss

    Returns:
        Total loss and loss components
    """
    # Language modeling loss (only on non-padding tokens)
    logits = outputs["logits"]
    lm_loss = nn.CrossEntropyLoss(ignore_index=0)(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )

    # Q-learning loss (encourage continuing for more steps)
    q_halt = outputs["q_halt_logits"]
    q_continue = outputs["q_continue_logits"]

    # Target: prefer continuing (1) over halting (0)
    q_targets = torch.ones_like(q_halt)  # Target = 1 (continue)
    q_loss = nn.MSELoss()(q_continue, q_targets)

    # Combined loss
    total_loss = lm_weight * lm_loss + q_weight * q_loss

    return total_loss, {
        "lm_loss": lm_loss.item(),
        "q_loss": q_loss.item(),
        "total_loss": total_loss.item(),
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: HRM model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_q_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        puzzle_ids = batch["puzzle_ids"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, puzzle_ids)

        # Compute loss
        loss, loss_components = compute_loss(outputs, target_ids)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        total_loss += loss_components["total_loss"]
        total_lm_loss += loss_components["lm_loss"]
        total_q_loss += loss_components["q_loss"]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss_components['total_loss']:.4f}",
                "lm": f"{loss_components['lm_loss']:.4f}",
                "q": f"{loss_components['q_loss']:.4f}",
            }
        )

    return {
        "train_loss": total_loss / num_batches,
        "train_lm_loss": total_lm_loss / num_batches,
        "train_q_loss": total_q_loss / num_batches,
    }


def grid_hash(grid: np.ndarray) -> str:
    """Create a hash for a 4x4 grid for voting purposes."""
    return hashlib.sha256(grid.tobytes()).hexdigest()


def inverse_aug_name(aug_name: str) -> str:
    """Extract original puzzle name from augmented name."""
    return aug_name.split("_")[0]


def generate_augmented_samples(
    input_puzzle: np.ndarray, target_solution: np.ndarray, num_augmentations: int = 20
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate augmented samples for voting.

    Args:
        input_puzzle: Original 4x4 puzzle
        target_solution: Original 4x4 solution
        num_augmentations: Number of augmented samples to generate

    Returns:
        List of (augmented_puzzle, augmented_solution, aug_name) tuples
    """
    # Import the augmentation function
    from dataset.build_4x4_sudoku_dataset import shuffle_4x4_sudoku

    samples = []
    for i in range(num_augmentations):
        aug_puzzle, aug_solution = shuffle_4x4_sudoku(input_puzzle, target_solution)
        aug_name = f"puzzle_aug_{i}"
        samples.append((aug_puzzle, aug_solution, aug_name))

    return samples


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_voting: bool = True,
    num_augmentations: int = 20,
) -> Dict[str, float]:
    """Evaluate model on validation set with optional voting.

    Args:
        model: HRM model
        val_loader: Validation data loader
        device: Device to evaluate on
        use_voting: Whether to use augmented sample voting
        num_augmentations: Number of augmented samples for voting

    Returns:
        Dictionary of validation metrics including accuracy
    """
    model.eval()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_q_loss = 0.0
    num_batches = 0

    # Accuracy tracking
    total_correct = 0
    total_examples = 0
    exact_matches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            puzzle_ids = batch["puzzle_ids"].to(device)

            if use_voting:
                # Voting-based evaluation
                batch_predictions = []
                batch_q_values = []

                for i in range(input_ids.size(0)):
                    # Get original puzzle and solution
                    input_puzzle = input_ids[i].cpu().numpy().reshape(4, 4)
                    target_solution = target_ids[i].cpu().numpy().reshape(4, 4)

                    # Generate augmented samples
                    aug_samples = generate_augmented_samples(
                        input_puzzle, target_solution, num_augmentations
                    )

                    # Collect predictions and Q-values for each augmentation
                    pred_hashes = []
                    q_values = []

                    for aug_puzzle, aug_solution, aug_name in aug_samples:
                        # Convert back to tensor format
                        aug_input = (
                            torch.tensor(aug_puzzle.flatten(), dtype=torch.long)
                            .unsqueeze(0)
                            .to(device)
                        )
                        aug_puzzle_id = puzzle_ids[i : i + 1]  # Same puzzle ID

                        # Run inference
                        outputs = model(aug_input, aug_puzzle_id)
                        prediction = (
                            torch.argmax(outputs["logits"], dim=-1)
                            .cpu()
                            .numpy()
                            .reshape(4, 4)
                        )
                        q_halt = outputs["q_halt_logits"].cpu().item()

                        # Create hash for voting
                        pred_hash = grid_hash(prediction)
                        pred_hashes.append(pred_hash)
                        q_values.append(q_halt)

                    # Perform voting
                    vote_map = defaultdict(lambda: [0, 0.0])  # [count, sum_q_values]

                    for pred_hash, q_val in zip(pred_hashes, q_values):
                        vote_map[pred_hash][0] += 1
                        vote_map[pred_hash][1] += q_val

                    # Average Q-values and sort by confidence
                    for pred_hash in vote_map:
                        vote_map[pred_hash][1] /= vote_map[pred_hash][
                            0
                        ]  # Average Q-value

                    # Sort by average Q-value (confidence)
                    sorted_votes = sorted(
                        vote_map.items(), key=lambda x: x[1][1], reverse=True
                    )

                    # Take the most confident prediction
                    best_pred_hash = sorted_votes[0][0]

                    # Find the actual prediction that matches this hash
                    best_prediction = None
                    for aug_puzzle, aug_solution, aug_name in aug_samples:
                        aug_input = (
                            torch.tensor(aug_puzzle.flatten(), dtype=torch.long)
                            .unsqueeze(0)
                            .to(device)
                        )
                        aug_puzzle_id = puzzle_ids[i : i + 1]
                        outputs = model(aug_input, aug_puzzle_id)
                        prediction = (
                            torch.argmax(outputs["logits"], dim=-1)
                            .cpu()
                            .numpy()
                            .reshape(4, 4)
                        )

                        if grid_hash(prediction) == best_pred_hash:
                            best_prediction = prediction
                            break

                    if best_prediction is None:
                        # Fallback to first prediction if voting fails
                        aug_input = (
                            torch.tensor(aug_samples[0][0].flatten(), dtype=torch.long)
                            .unsqueeze(0)
                            .to(device)
                        )
                        aug_puzzle_id = puzzle_ids[i : i + 1]
                        outputs = model(aug_input, aug_puzzle_id)
                        best_prediction = (
                            torch.argmax(outputs["logits"], dim=-1)
                            .cpu()
                            .numpy()
                            .reshape(4, 4)
                        )

                    batch_predictions.append(
                        torch.tensor(best_prediction.flatten(), dtype=torch.long)
                    )

                # Convert to tensor
                predictions = torch.stack(batch_predictions).to(device)

                # Compute loss (use first augmentation for loss computation)
                first_aug_input = (
                    torch.tensor(aug_samples[0][0].flatten(), dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )
                first_aug_puzzle_id = puzzle_ids[0:1]
                outputs = model(first_aug_input, first_aug_puzzle_id)
                loss, loss_components = compute_loss(outputs, target_ids[0:1])

            else:
                # Standard evaluation (no voting)
                outputs = model(input_ids, puzzle_ids)
                loss, loss_components = compute_loss(outputs, target_ids)
                predictions = torch.argmax(outputs["logits"], dim=-1)

            # Update metrics
            total_loss += loss_components["total_loss"]
            total_lm_loss += loss_components["lm_loss"]
            total_q_loss += loss_components["q_loss"]
            num_batches += 1

            # Exact match accuracy (entire puzzle correct)
            exact_match = torch.all(predictions == target_ids, dim=1)
            exact_matches += exact_match.sum().item()

            # Cell-wise accuracy (ignoring padding/blank cells)
            # Only count non-zero cells in targets
            non_zero_mask = target_ids != 0
            if non_zero_mask.any():
                correct_cells = (predictions == target_ids) & non_zero_mask
                total_correct += correct_cells.sum().item()
                total_examples += non_zero_mask.sum().item()

    # Calculate accuracies
    exact_accuracy = exact_matches / (len(val_loader.dataset))
    cell_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

    return {
        "val_loss": total_loss / num_batches,
        "val_lm_loss": total_lm_loss / num_batches,
        "val_q_loss": total_q_loss / num_batches,
        "val_exact_accuracy": exact_accuracy,
        "val_cell_accuracy": cell_accuracy,
    }


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


def show_examples(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_examples: int = 5,
) -> None:
    """Show example predictions.

    Args:
        model: Trained HRM model
        test_loader: Test data loader
        device: Device to run on
        num_examples: Number of examples to show
    """
    model.eval()

    print(f"\nShowing {num_examples} example predictions:")
    print("=" * 80)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_examples:
                break

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            puzzle_ids = batch["puzzle_ids"].to(device)

            # Get predictions
            outputs = model(input_ids, puzzle_ids)
            predictions = torch.argmax(outputs["logits"], dim=-1)

            # Show first example from batch
            input_puzzle = input_ids[0].cpu().numpy().reshape(4, 4)
            target_solution = target_ids[0].cpu().numpy().reshape(4, 4)
            predicted_solution = predictions[0].cpu().numpy().reshape(4, 4)

            print(f"\nExample {i+1}:")
            print(format_sudoku_grid(input_puzzle, "Input Puzzle"))
            print(format_sudoku_grid(target_solution, "Target Solution"))
            print(format_sudoku_grid(predicted_solution, "Predicted Solution"))

            # Check if prediction is correct
            is_correct = np.array_equal(predicted_solution, target_solution)
            print(f"Correct: {'✓' if is_correct else '✗'}")
            print("-" * 40)


def train_model(
    config: Dict[str, Any],
    config_path: str = "config/sudoku_config.yaml",
) -> None:
    """Train HRM model on 4x4 sudoku using configuration.

    Args:
        config: Configuration dictionary
        config_path: Path to configuration file (for logging)
    """
    # Extract configuration sections
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]

    print("=" * 60)
    print("4x4 Sudoku HRM Training")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Device: {training_cfg['device']}")
    print(f"Epochs: {training_cfg['num_epochs']}")
    print(f"Batch size: {training_cfg['batch_size']}")
    print(f"Learning rate: {training_cfg['learning_rate']}")
    print(f"Data directory: {dataset_cfg['data_dir']}")
    print(f"Save directory: {training_cfg['save_dir']}")
    print(f"Use voting: {eval_cfg['use_voting']}")
    print(f"Voting augmentations: {eval_cfg['num_augmentations']}")
    print()

    # Create save directory
    os.makedirs(training_cfg["save_dir"], exist_ok=True)

    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_cfg["data_dir"], training_cfg["batch_size"]
    )

    # Create model using configuration
    print("Creating model...")
    model = create_hrm_model(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        num_heads=model_cfg["num_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_seq_len=model_cfg["max_seq_len"],
        num_puzzle_ids=model_cfg["num_puzzle_ids"],
        h_layers=model_cfg["h_layers"],
        l_layers=model_cfg["l_layers"],
        h_cycles=model_cfg["h_cycles"],
        l_cycles=model_cfg["l_cycles"],
        halt_max_steps=model_cfg["halt_max_steps"],
    )

    model = model.to(training_cfg["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Open log file
    log_path = os.path.join(training_cfg["save_dir"], training_cfg["log_file"])
    with open(log_path, "w", buffering=1) as log_f:
        log_f.write(
            "Epoch,Train_Loss,Val_Loss,Val_Exact_Accuracy,Val_Cell_Accuracy,Time\n"
        )
        log_f.flush()

        print("Starting training...")
        print()

        for epoch in range(1, training_cfg["num_epochs"] + 1):
            start_time = time.time()

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, training_cfg["device"], epoch
            )

            # Validate with voting configuration
            val_metrics = evaluate(
                model,
                val_loader,
                training_cfg["device"],
                use_voting=eval_cfg["use_voting"],
                num_augmentations=eval_cfg["num_augmentations"],
            )

            # Update best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["val_loss"],
                        "val_exact_accuracy": val_metrics["val_exact_accuracy"],
                        "val_cell_accuracy": val_metrics["val_cell_accuracy"],
                    },
                    os.path.join(training_cfg["save_dir"], "best_model.pt"),
                )

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_exact_accuracy": val_metrics["val_exact_accuracy"],
                    "val_cell_accuracy": val_metrics["val_cell_accuracy"],
                },
                os.path.join(training_cfg["save_dir"], f"checkpoint_epoch_{epoch}.pt"),
            )

            # Track losses and accuracies
            train_losses.append(train_metrics["train_loss"])
            val_losses.append(val_metrics["val_loss"])
            val_accuracies.append(val_metrics["val_exact_accuracy"])

            # Calculate epoch time
            epoch_time = time.time() - start_time

            # Log to file (unbuffered)
            log_f.write(
                f"{epoch},{train_metrics['train_loss']:.6f},{val_metrics['val_loss']:.6f},"
                f"{val_metrics['val_exact_accuracy']:.6f},{val_metrics['val_cell_accuracy']:.6f},"
                f"{epoch_time:.2f}\n"
            )
            log_f.flush()  # Force write to disk immediately

            # Print progress
            print(
                f"Epoch {epoch:2d}/{training_cfg['num_epochs']} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_exact_accuracy']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            print(
                f"  LM Loss: {train_metrics['train_lm_loss']:.4f} | "
                f"Q Loss: {train_metrics['train_q_loss']:.4f} | "
                f"Cell Acc: {val_metrics['val_cell_accuracy']:.4f}"
            )
            print()

        # Final evaluation
        print("Final evaluation on test set...")
        test_metrics = evaluate(
            model,
            test_loader,
            training_cfg["device"],
            use_voting=eval_cfg["use_voting"],
            num_augmentations=eval_cfg["num_augmentations"],
        )
        print(f"Test Loss: {test_metrics['val_loss']:.4f}")
        print(f"Test LM Loss: {test_metrics['val_lm_loss']:.4f}")
        print(f"Test Q Loss: {test_metrics['val_q_loss']:.4f}")
        print(f"Test Exact Accuracy: {test_metrics['val_exact_accuracy']:.4f}")
        print(f"Test Cell Accuracy: {test_metrics['val_cell_accuracy']:.4f}")
        print()

        # Log final test results
        log_f.write(f"# Final Test Results\n")
        log_f.write(f"# Test Loss: {test_metrics['val_loss']:.6f}\n")
        log_f.write(
            f"# Test Exact Accuracy: {test_metrics['val_exact_accuracy']:.6f}\n"
        )
        log_f.write(f"# Test Cell Accuracy: {test_metrics['val_cell_accuracy']:.6f}\n")
        log_f.flush()

        # Show examples
        if eval_cfg["show_examples_after"]:
            show_examples(
                model, test_loader, training_cfg["device"], eval_cfg["num_examples"]
            )

        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {max(val_accuracies):.4f}")
        print(f"Model saved to: {training_cfg['save_dir']}")
        print(f"Training log saved to: {log_path}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Train HRM on 4x4 Sudoku")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sudoku_config.yaml",
        help="Path to YAML configuration file (default: config/sudoku_config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--use_voting",
        action="store_true",
        help="Override to enable voting",
    )
    parser.add_argument(
        "--no_voting",
        action="store_true",
        help="Override to disable voting",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    if args.data_dir:
        config["dataset"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.use_voting:
        config["evaluation"]["use_voting"] = True
    if args.no_voting:
        config["evaluation"]["use_voting"] = False

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if data directory exists
    if not os.path.exists(config["dataset"]["data_dir"]):
        print(f"Error: Data directory {config['dataset']['data_dir']} does not exist!")
        print("Please run the dataset generation first:")
        print("python dataset/build_4x4_sudoku_dataset.py")
        return

    # Train the model
    train_model(config, args.config)


if __name__ == "__main__":
    main()
