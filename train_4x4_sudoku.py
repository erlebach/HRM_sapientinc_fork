"""
4x4 Sudoku Training Script for CPU

This script trains the HRM model on 4x4 sudoku puzzles, optimized for CPU training.
Features:
- Small model size for fast CPU training
- Small batch sizes
- Simple puzzle dataset
- Progress tracking
- Model checkpointing
"""

import json
import os

# Import HRM model from didactic implementation
import sys
import time
from typing import Any, Dict

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
    batch_size: int = 8,  # Small batch size for CPU
    num_workers: int = 0,  # No multiprocessing for CPU
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
) -> tuple[torch.Tensor, Dict[str, float]]:
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


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: HRM model
        val_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_q_loss = 0.0
    num_batches = 0

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

    return {
        "val_loss": total_loss / num_batches,
        "val_lm_loss": total_lm_loss / num_batches,
        "val_q_loss": total_q_loss / num_batches,
    }


def train_model(
    data_dir: str = "data/sudoku-4x4",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_dir: str = "./checkpoints_4x4",
    device: str = "cpu",
) -> None:
    """Train HRM model on 4x4 sudoku.

    Args:
        data_dir: Directory containing the dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save checkpoints
        device: Device to train on
    """
    print("=" * 60)
    print("4x4 Sudoku HRM Training")
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
        hidden_size=128,  # Small hidden size
        num_heads=4,  # Fewer heads
        intermediate_size=256,  # Smaller intermediate
        max_seq_len=16,  # 4x4 = 16
        num_puzzle_ids=1,  # Single puzzle type
        h_layers=2,  # Fewer layers
        l_layers=2,
        h_cycles=1,  # Fewer cycles
        l_cycles=1,
        halt_max_steps=4,  # Fewer max steps
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
            },
            os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
        )

        # Track losses
        train_losses.append(train_metrics["train_loss"])
        val_losses.append(val_metrics["val_loss"])

        # Print progress
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(
            f"  LM Loss: {train_metrics['train_lm_loss']:.4f} | "
            f"Q Loss: {train_metrics['train_q_loss']:.4f}"
        )
        print()

    # Final evaluation
    print("Final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"Test LM Loss: {test_metrics['val_lm_loss']:.4f}")
    print(f"Test Q Loss: {test_metrics['val_q_loss']:.4f}")
    print()

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Train the model
    train_model(
        data_dir="data/sudoku-4x4",
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-4,
        save_dir="./checkpoints_4x4",
        device="cpu",
    )

