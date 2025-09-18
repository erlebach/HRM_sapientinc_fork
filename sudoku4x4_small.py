#!/usr/bin/env python3
"""
Small 4x4 Sudoku HRM Training Script

This script trains the HRM model on a smaller, more appropriate 4x4 sudoku dataset.
4x4 sudoku is much simpler than 9x9, so we use fewer examples for faster training.

Usage:
    python sudoku4x4_small.py [options]
"""

import argparse
import json
import os

# Import HRM model from didactic implementation
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("HRM_didactic")
from hrm_model import create_hrm_model


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
    """Create data loaders for train/val/test splits."""
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
    """Compute combined loss for HRM model."""
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
    """Train for one epoch."""
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


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
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

            # Forward pass
            outputs = model(input_ids, puzzle_ids)

            # Compute loss
            loss, loss_components = compute_loss(outputs, target_ids)

            # Update metrics
            total_loss += loss_components["total_loss"]
            total_lm_loss += loss_components["lm_loss"]
            total_q_loss += loss_components["q_loss"]
            num_batches += 1

            # Compute accuracy
            predictions = torch.argmax(outputs["logits"], dim=-1)

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


def train_model(
    data_dir: str = "dataset/data/sudoku-4x4-small",
    num_epochs: int = 15,
    batch_size: int = 16,  # Larger batch size for smaller dataset
    learning_rate: float = 2e-4,  # Slightly higher learning rate
    save_dir: str = "./checkpoints_4x4_small",
    device: str = "cpu",
    log_file: str = "training_log_small.txt",
) -> None:
    """Train HRM model on small 4x4 sudoku dataset."""
    print("=" * 60)
    print("Small 4x4 Sudoku HRM Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()

    # Create model (small for CPU training)
    print("Creating model...")
    model = create_hrm_model(
        vocab_size=5,  # 0-4 for 4x4 sudoku
        #hidden_size=128,  # Small hidden size
        hidden_size=256,  # Small hidden size
        num_heads=8,  # Fewer heads
        intermediate_size=1024,  # Smaller intermediate
        max_seq_len=16,  # 4x4 = 16
        num_puzzle_ids=1,  # Single puzzle type
        h_layers=3,  # Fewer layers
        l_layers=3,
        h_cycles=2,  # Fewer cycles
        l_cycles=2,
        halt_max_steps=8,  # Fewer max steps
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Open log file in unbuffered mode
    log_path = os.path.join(save_dir, log_file)
    with open(log_path, "w", buffering=1) as log_f:
        log_f.write(
            "Epoch,Train_Loss,Val_Loss,Val_Exact_Accuracy,Val_Cell_Accuracy,Time\n"
        )
        log_f.flush()

        print("Starting training...")
        print()

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

            # Validate
            val_metrics = evaluate(model, val_loader, device)

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
                    os.path.join(save_dir, "best_model.pt"),
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
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
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
            log_f.flush()

            # Print progress
            print(
                f"Epoch {epoch:2d}/{num_epochs} | "
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
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_metrics['val_loss']:.4f}")
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

        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {max(val_accuracies):.4f}")
        print(f"Model saved to: {save_dir}")
        print(f"Training log saved to: {log_path}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Train HRM on Small 4x4 Sudoku")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/data/sudoku-4x4-small",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_4x4_small",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_file", type=str, default="training_log_small.txt", help="Log file name"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        print("Please run the dataset generation first:")
        print("python dataset/build_4x4_sudoku_small.py")
        return

    # Train the model
    train_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        device=args.device,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()



