"""
Training Script for Asymmetric HRM Model

This script provides a complete training pipeline for the Asymmetric HRM model including:
- Model initialization with System 1/System 2 architecture
- Loss computation (language modeling + Q-learning)
- Training loop with validation
- Model checkpointing
- Progress tracking
- Support for asymmetric processing strategies
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from asymmetric_hrm import AsymmetricHRMModel, create_asymmetric_hrm_model
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class PuzzleDataset(Dataset):
    """Simple puzzle dataset for training the Asymmetric HRM model."""

    def __init__(
        self,
        size: int = 1000,
        vocab_size: int = 1000,
        seq_len: int = 64,
        num_puzzle_ids: int = 100,
    ):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_puzzle_ids = num_puzzle_ids

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Generate random puzzle data
        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        target_ids = input_ids.clone()
        puzzle_ids = torch.randint(0, self.num_puzzle_ids, (1,)).squeeze()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "puzzle_ids": puzzle_ids,
        }


def create_data_loaders(
    batch_size: int = 32,
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 200,
    vocab_size: int = 1000,
    seq_len: int = 64,
    num_puzzle_ids: int = 100,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    train_dataset = PuzzleDataset(train_size, vocab_size, seq_len, num_puzzle_ids)
    val_dataset = PuzzleDataset(val_size, vocab_size, seq_len, num_puzzle_ids)
    test_dataset = PuzzleDataset(test_size, vocab_size, seq_len, num_puzzle_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class AsymmetricHRMTrainer:
    """Trainer class for the Asymmetric HRM model with System 1/System 2 architecture."""

    def __init__(
        self,
        model: AsymmetricHRMModel,
        learning_rate: float = 1e-4,
        q_learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        print_every: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.q_learning_rate = q_learning_rate
        self.print_every = print_every

        # Separate optimizers for different components
        self.lm_optimizer = optim.AdamW(
            [p for name, p in model.named_parameters() if "q_head" not in name],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.q_optimizer = optim.AdamW(
            [p for name, p in model.named_parameters() if "q_head" in name],
            lr=q_learning_rate,
            weight_decay=weight_decay,
        )

        # Loss functions
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.q_criterion = nn.MSELoss()

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    @beartype
    def compute_loss(
        self, batch: dict[str, Tensor]
    ) -> tuple[Float[Tensor, ""], dict[str, float]]:
        """Compute the total loss including language modeling and Q-learning components.

        Args:
            batch: Dictionary containing input_ids, target_ids, puzzle_ids

        Returns:
            total_loss: Combined loss
            metrics: Dictionary of loss components and metrics
        """
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        puzzle_ids = batch["puzzle_ids"].to(self.device)

        # Forward pass through model
        outputs = self.model(input_ids, puzzle_ids)

        logits = outputs["logits"]
        q_halt_logits = outputs["q_halt_logits"]
        q_continue_logits = outputs["q_continue_logits"]
        segments_taken = outputs["segments_taken"]

        # Language modeling loss
        lm_loss = self.lm_criterion(
            logits.view(-1, logits.size(-1)), target_ids.view(-1)
        )

        # Q-learning loss (simplified - in practice this would be more complex)
        # For now, we'll use a simple reward based on whether the model produces reasonable outputs
        batch_size = input_ids.size(0)

        # Simple reward: higher reward for taking more segments (encouraging exploration)
        # In practice, this would be based on task performance
        target_q_continue = (
            torch.ones_like(q_continue_logits) * 0.7
        )  # Target for continuing
        target_q_halt = torch.ones_like(q_halt_logits) * 0.3  # Target for halting

        # Add some noise to make it more realistic
        target_q_continue += torch.randn_like(target_q_continue) * 0.1
        target_q_halt += torch.randn_like(target_q_halt) * 0.1

        q_continue_loss = self.q_criterion(q_continue_logits, target_q_continue)
        q_halt_loss = self.q_criterion(q_halt_logits, target_q_halt)
        q_loss = q_continue_loss + q_halt_loss

        # Total loss
        total_loss = lm_loss + 0.1 * q_loss  # Weight Q-learning loss

        # Compute metrics
        with torch.no_grad():
            # Accuracy for language modeling
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == target_ids) & (target_ids != 0)  # Ignore padding
            accuracy = correct.float().mean().item()

            # Average segments taken
            avg_segments = segments_taken.float().mean().item()

            # Q-value statistics
            q_continue_mean = q_continue_logits.mean().item()
            q_halt_mean = q_halt_logits.mean().item()

        metrics = {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "q_loss": q_loss.item(),
            "q_continue_loss": q_continue_loss.item(),
            "q_halt_loss": q_halt_loss.item(),
            "accuracy": accuracy,
            "avg_segments": avg_segments,
            "q_continue_mean": q_continue_mean,
            "q_halt_mean": q_halt_mean,
        }

        return total_loss, metrics

    @beartype
    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Perform a single training step."""
        self.model.train()

        # Compute loss
        total_loss, metrics = self.compute_loss(batch)

        # Backward pass
        self.lm_optimizer.zero_grad()
        self.q_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.lm_optimizer.step()
        self.q_optimizer.step()

        self.step += 1
        return metrics

    @beartype
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate the model on validation set."""
        self.model.eval()

        total_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                _, metrics = self.compute_loss(batch)

                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value

                num_batches += 1

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics

    @beartype
    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_metrics = {}
        num_batches = 0

        for batch in train_loader:
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

            num_batches += 1

            # Print progress every n steps
            if self.step % self.print_every == 0:
                print(
                    f"Step {self.step}: Loss = {metrics['total_loss']:.4f}, "
                    f"Acc = {metrics['accuracy']:.3f}, "
                    f"Segments = {metrics['avg_segments']:.1f}, "
                    f"Q-continue = {metrics['q_continue_mean']:.3f}, "
                    f"Q-halt = {metrics['q_halt_mean']:.3f}"
                )

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "lm_optimizer_state_dict": self.lm_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.lm_optimizer.load_state_dict(checkpoint["lm_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]


def train_model(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    q_learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    save_dir: str = "./asymmetric_hrm_checkpoints",
    device: str | None = None,
    print_every: int = 10,
    # Asymmetric HRM specific parameters
    L_blocks: int = 2,
    H_blocks: int = 4,
    H_memory_size: int = 64,
    T_cycles: int = 2,
    M_segments: int = 16,
):
    """Train the Asymmetric HRM model with the specified parameters.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for language modeling
        q_learning_rate: Learning rate for Q-learning
        weight_decay: Weight decay for regularization
        save_dir: Directory to save checkpoints
        device: Device to use for training
        print_every: Print progress every n steps
        L_blocks: Number of transformer blocks in Low-level module
        H_blocks: Number of transformer blocks in High-level module
        H_memory_size: Working memory size in High-level module
        T_cycles: Number of cycles per segment
        M_segments: Maximum number of segments
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Printing progress every {print_every} steps")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=batch_size, train_size=800, val_size=100, test_size=100
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create Asymmetric HRM model
    print("Creating Asymmetric HRM model...")
    model = create_asymmetric_hrm_model(
        vocab_size=1000,
        L_hidden_size=256,  # Smaller for faster training
        H_hidden_size=256,  # Smaller for faster training
        num_heads=8,
        intermediate_size=1024,
        max_seq_len=64,
        num_puzzle_ids=100,
        L_blocks=L_blocks,
        H_blocks=H_blocks,
        H_memory_size=H_memory_size,
        T_cycles=T_cycles,
        M_segments=M_segments,
    )

    # Print model information
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"L_module parameters: {model_info['L_parameters']:,}")
    print(f"H_module parameters: {model_info['H_parameters']:,}")
    print(f"L_blocks: {model_info['L_blocks']}, H_blocks: {model_info['H_blocks']}")
    print(f"T_cycles: {model_info['T_cycles']}, M_segments: {model_info['M_segments']}")

    # Create trainer
    trainer = AsymmetricHRMTrainer(
        model=model,
        learning_rate=learning_rate,
        q_learning_rate=q_learning_rate,
        weight_decay=weight_decay,
        device=device,
        print_every=print_every,
    )

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Update epoch
        trainer.epoch = epoch

        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(
            f"  Train Loss: {train_metrics['total_loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}"
        )
        print(
            f"  Val Loss: {val_metrics['total_loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}"
        )
        print(f"  Avg Segments: {train_metrics['avg_segments']:.1f}")
        print(
            f"  Q-values - Continue: {train_metrics['q_continue_mean']:.3f}, Halt: {train_metrics['q_halt_mean']:.3f}"
        )

        # Save checkpoint if validation loss improved
        if val_metrics["total_loss"] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics["total_loss"]
            checkpoint_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"  New best model saved to {checkpoint_path}")

        print("-" * 60)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['total_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test Avg Segments: {test_metrics['avg_segments']:.1f}")

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pt")
    trainer.save_checkpoint(final_path)
    print(f"\nFinal model saved to {final_path}")

    return trainer, test_metrics


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Train Asymmetric HRM model")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for language modeling (default: 1e-4)",
    )
    parser.add_argument(
        "--q_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Q-learning (default: 1e-3)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization (default: 0.01)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./asymmetric_hrm_checkpoints",
        help="Directory to save checkpoints (default: ./asymmetric_hrm_checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu) (default: auto-detect)",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every n steps (default: 10)",
    )
    # Asymmetric HRM specific arguments
    parser.add_argument(
        "--L_blocks",
        type=int,
        default=2,
        help="Number of transformer blocks in Low-level module (default: 2)",
    )
    parser.add_argument(
        "--H_blocks",
        type=int,
        default=4,
        help="Number of transformer blocks in High-level module (default: 4)",
    )
    parser.add_argument(
        "--H_memory_size",
        type=int,
        default=64,
        help="Working memory size in High-level module (default: 64)",
    )
    parser.add_argument(
        "--T_cycles",
        type=int,
        default=2,
        help="Number of cycles per segment (default: 2)",
    )
    parser.add_argument(
        "--M_segments",
        type=int,
        default=16,
        help="Maximum number of segments (default: 16)",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Train the model
    trainer, test_metrics = train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        q_learning_rate=args.q_learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        device=args.device,
        print_every=args.print_every,
        L_blocks=args.L_blocks,
        H_blocks=args.H_blocks,
        H_memory_size=args.H_memory_size,
        T_cycles=args.T_cycles,
        M_segments=args.M_segments,
    )

    print("\nTraining completed!")
    print(f"Final test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Final test loss: {test_metrics['total_loss']:.4f}")


if __name__ == "__main__":
    main()


