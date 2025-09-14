#!/usr/bin/env python3
"""4x4 Sudoku HRM Training Script.

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
import pickle

# Import HRM model from didactic implementation
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Add import
from dataset.sudoku_dataloader import create_dataloaders, create_evaluation_dataloader
from HRM_didactic.hrm_model import create_hrm_model
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.sudoku_augmentation import shuffle_4x4_sudoku, simple_digit_augmentation


# ---
def debug_voting_process(input_ids, target_ids, model, device, num_augmentations=3):
    """Debug the entire voting process step by step."""
    print("=== DEBUGGING VOTING PROCESS ===")

    # Take just the first sample for debugging
    input_puzzle = input_ids[0].cpu().numpy().reshape(4, 4)
    target_solution = target_ids[0].cpu().numpy().reshape(4, 4)

    print(f"Original puzzle:\n{input_puzzle}")
    print(f"Target solution:\n{target_solution}")

    # Test no-voting prediction first
    with torch.no_grad():
        no_vote_output = model(input_ids[0:1], torch.tensor([0], device=device))
        no_vote_pred = torch.argmax(no_vote_output["logits"], dim=-1)
        no_vote_pred_np = no_vote_pred[0].cpu().numpy().reshape(4, 4)
        print(f"No-voting prediction:\n{no_vote_pred_np}")
        print(f"No-voting correct: {np.array_equal(no_vote_pred_np, target_solution)}")

    # Now test voting
    all_inputs = [input_ids[0]]
    all_inv_maps = [torch.arange(5, device=device)]

    for i in range(num_augmentations):
        aug_puzzle, aug_solution, _, digit_map = generate_augmented_samples(
            input_puzzle, target_solution, 1
        )[0]

        print(f"\nAugmentation {i+1}:")
        print(f"Digit map: {digit_map}")
        print(f"Aug puzzle:\n{aug_puzzle}")
        print(f"Aug solution:\n{aug_solution}")

        # Create inverse mapping
        inv_map = torch.zeros_like(torch.tensor(digit_map, device=device))
        inv_map[torch.tensor(digit_map, device=device)] = torch.arange(5, device=device)
        print(f"Inverse map: {inv_map.cpu().numpy()}")

        # Test the mapping
        test_forward = digit_map[target_solution.flatten()]
        test_backward = inv_map[torch.tensor(test_forward, device=device)].cpu().numpy()
        print(f"Target forward: {test_forward}")
        print(f"Target backward: {test_backward}")
        print(
            f"Mapping correct: {np.array_equal(test_backward, target_solution.flatten())}"
        )

        all_inputs.append(
            torch.tensor(aug_puzzle.flatten(), dtype=torch.long, device=device)
        )
        all_inv_maps.append(inv_map)

    # Run inference
    all_inputs = torch.stack(all_inputs)
    with torch.no_grad():
        outputs = model(
            all_inputs, torch.zeros(len(all_inputs), device=device, dtype=torch.long)
        )
        logits = outputs["logits"]

    print(f"\nLogits shape: {logits.shape}")

    # Show individual predictions before voting
    print("\nIndividual predictions before voting:")
    for k in range(logits.size(0)):
        pred = torch.argmax(logits[k], dim=-1).cpu().numpy().reshape(4, 4)
        print(f"Sample {k} prediction:\n{pred}")
        if k == 0:
            print("(This is the original - should match no-voting prediction)")

    # Remap logits
    remapped_logits = []
    for k in range(logits.size(0)):
        inv_perm = all_inv_maps[k]
        remapped = logits[k, :, inv_perm]
        remapped_logits.append(remapped)
        print(
            f"Sample {k} - Original logits max: {logits[k].max():.3f}, Remapped max: {remapped.max():.3f}"
        )

    remapped_logits = torch.stack(remapped_logits, dim=0)

    # Show remapped predictions
    print("\nRemapped predictions before voting:")
    for k in range(remapped_logits.size(0)):
        pred = torch.argmax(remapped_logits[k], dim=-1).cpu().numpy().reshape(4, 4)
        print(f"Sample {k} remapped prediction:\n{pred}")

    # Vote
    voted_logits = remapped_logits.sum(dim=0)  # Sum across all samples
    voted_pred = voted_logits.argmax(dim=-1)
    voted_pred_np = voted_pred.cpu().numpy().reshape(4, 4)

    print(f"\nVoted prediction:\n{voted_pred_np}")
    print(f"Voted correct: {np.array_equal(voted_pred_np, target_solution)}")

    print("=== END DEBUG ===")


# ---


# Add this debug function
def debug_digit_mapping():
    """Debug digit mapping logic."""
    # Test with a simple example
    digit_map = np.array([0, 3, 1, 4, 2])  # 0→0, 1→3, 2→1, 3→4, 4→2
    print(f"Original digit_map: {digit_map}")

    # Create inverse mapping
    inv_map = np.zeros_like(digit_map)
    inv_map[digit_map] = np.arange(5)
    print(f"Inverse mapping: {inv_map}")

    # Test round-trip
    test_values = np.array([0, 1, 2, 3, 4])
    forward = digit_map[test_values]
    backward = inv_map[forward]
    print(f"Test values: {test_values}")
    print(f"Forward: {forward}")
    print(f"Backward: {backward}")
    print(f"Round-trip correct: {np.array_equal(test_values, backward)}")


# Add this function after the debug_digit_mapping function (around line 167):


def verify_inverse_mapping():
    """Verify that the inverse mapping computation is correct."""
    print("=== VERIFYING INVERSE MAPPING ===")

    # Test with a known digit_map
    digit_map = np.array([0, 3, 1, 4, 2])  # 0→0, 1→3, 2→1, 3→4, 4→2
    print(f"Original digit_map: {digit_map}")

    # Current method
    inv_map = np.zeros_like(digit_map)
    inv_map[digit_map] = np.arange(5)
    print(f"Computed inverse_map: {inv_map}")

    # Test round-trip
    test_values = np.array([0, 1, 2, 3, 4])
    forward = digit_map[test_values]
    backward = inv_map[forward]
    print(f"Test values: {test_values}")
    print(f"Forward mapping: {forward}")
    print(f"Backward mapping: {backward}")
    print(f"Round-trip correct: {np.array_equal(test_values, backward)}")

    # Verify the inverse property
    print(f"\nVerification:")
    for i in range(5):
        original = i
        mapped = digit_map[original]
        unmapped = inv_map[mapped]
        print(
            f"{original} → {mapped} → {unmapped} {'✓' if original == unmapped else '✗'}"
        )

    print("=== END VERIFICATION ===")


# Call this in your evaluate function before voting


# Add this configuration loading function
def load_config(config_path: str = "config/sudoku_config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)
    return config


# Remove the old Sudoku4x4Dataset class since we're using the new dataloader


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    lm_weight: float = 1.0,
    q_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
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


def compute_diagnostic_quantity(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute a diagnostic quantity to verify training consistency.

    Args:
        predictions: Model predictions [batch_size, seq_len]
        targets: Target values [batch_size, seq_len]

    Returns:
        Diagnostic quantity (sum of all prediction values)
    """
    return predictions.sum().item()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
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
        puzzle_ids = batch["puzzle_id"]
        puzzles = batch["puzzle"]
        solutions = batch["solution"]
        solutions = solutions.to(device)
        puzzles = puzzles.reshape(puzzles.shape[0], -1).to(device)
        puzzle_ids = torch.zeros(puzzles.shape[0], dtype=torch.long).to(device)
        target_ids = solutions.reshape(solutions.shape[0], -1).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(puzzles, puzzle_ids)

        # ADD DEBUG STATEMENTS HERE
        predictions = torch.argmax(outputs["logits"], dim=-1)

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
) -> list[tuple[np.ndarray, np.ndarray, str, np.ndarray]]:
    """Generate augmented samples for voting.

    Args:
        input_puzzle: Original 4x4 puzzle
        target_solution: Original 4x4 solution
        num_augmentations: Number of augmented samples to generate

    Returns:
        List of (augmented_puzzle, augmented_solution, aug_name, digit_mapping) tuples
    """
    # Use simple digit-only augmentation for truly equivalent puzzles
    samples = []
    for i in range(num_augmentations):
        aug_puzzle, aug_solution, digit_map = simple_digit_augmentation(
            input_puzzle, target_solution
        )
        aug_name = f"puzzle_aug_{i}"
        samples.append((aug_puzzle, aug_solution, aug_name, digit_map))

    return samples


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_voting: bool = True,
    num_augmentations: int = 20,
) -> dict[str, float]:
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

    # Initialize GPU tensors for accumulation
    exact_matches_tensor = torch.tensor(0, dtype=torch.long, device=device)
    total_correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
    total_examples_tensor = torch.tensor(0, dtype=torch.long, device=device)

    # Check that batch size is 1 for evaluation
    batch_size = val_loader.batch_size
    if batch_size != 1:
        raise ValueError(
            f"Evaluation requires batch_size=1 for correct voting logic, but got batch_size={batch_size}. "
            "Please set batch_size=1 in your DataLoader for validation."
        )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            puzzles = batch["puzzles"].to(device)  # Shape: [batch_size, 14, 4, 4]
            solutions = batch["solutions"].to(device)  # Shape: [batch_size, 14, 4, 4]
            puzzle_ids = batch["puzzle_id"]  # Single puzzle group ID
            global_ids = batch["global_ids"]  # List of (puzzle_id, aug_idx) tuples
            digit_maps = batch["digit_maps"]  # Shape: [batch_size, 14, 5]

            # For evaluation, we want to use the original puzzle (index 0) as input
            input_ids = puzzles[:, 0, :, :].flatten(1)  # Shape: [batch_size, 16]
            target_ids = solutions[:, 0, :, :].flatten(1)  # Shape: [batch_size, 16]

            # For puzzle_ids, since all samples in a group belong to the same puzzle,
            # we can use a constant or the actual puzzle group ID
            # Extract the scalar value from puzzle_ids if it's a tensor
            if isinstance(puzzle_ids, torch.Tensor):
                puzzle_id_value = puzzle_ids.item()
            else:
                puzzle_id_value = puzzle_ids

            # puzzle_ids_tensor = torch.full(
            #     (puzzles.size(0),), puzzle_id_value, device=device, dtype=torch.long
            # )
            # print("puzzles.size(0)= ", puzzles.size(0))
            puzzle_ids_tensor = torch.zeros(
                puzzles.size(0), device=device, dtype=torch.long
            )

            if use_voting:
                # Generate all augmented samples for the batch
                all_inputs = []
                all_inv_maps = []  # Store inverse digit mappings
                vocab_size = 5  # 0, 1, 2, 3, 4

                for i in range(input_ids.size(0)):
                    # Get original puzzle and solution
                    input_puzzle = input_ids[i].cpu().numpy().reshape(4, 4)
                    target_solution = target_ids[i].cpu().numpy().reshape(4, 4)

                    # Add original sample (identity mapping)
                    all_inputs.append(input_ids[i])
                    id_map = torch.arange(vocab_size, device=device)
                    all_inv_maps.append(id_map)

                    # Generate augmented samples
                    for _ in range(num_augmentations):
                        aug_puzzle, aug_solution, aug_name, digit_map = (
                            generate_augmented_samples(
                                input_puzzle, target_solution, 1
                            )[0]
                        )

                        # Convert back to tensor format
                        aug_input = torch.tensor(
                            aug_puzzle.flatten(), dtype=torch.long
                        ).to(device)
                        all_inputs.append(aug_input)

                        # Create inverse mapping
                        inv_map = torch.empty_like(
                            torch.tensor(digit_map, device=device),
                        )
                        inv_map[torch.tensor(digit_map, device=device)] = torch.arange(
                            vocab_size, device=device
                        )
                        all_inv_maps.append(inv_map)
                        # Verify that the inverse mapping is correct. DO NOT REMOVE.
                        # verify_inverse_mapping()  # Verify the inverse mapping is correct

                # Stack all samples (original + augmented)
                all_inputs = torch.stack(all_inputs)  #  [batch*(A+1), seq_len]

                # Run inference on all samples
                outputs = model(
                    all_inputs,
                    puzzle_ids_tensor.repeat_interleave(num_augmentations + 1),
                )
                logits = outputs["logits"]  # [batch*(A+1), seq_len, vocab]

                # Remap each augmented sample's logit channels back to original label space
                remapped_logits = []
                for k in range(logits.size(0)):
                    inv_perm = all_inv_maps[k]  # [vocab]
                    remapped_logits.append(logits[k, :, inv_perm])  # [seq_len, vocab]
                remapped_logits = torch.stack(
                    remapped_logits, dim=0
                )  # [batch*(A+1), seq_len, vocab]

                # Reshape and vote by logit summation
                batch_size = input_ids.size(0)
                remapped_logits = remapped_logits.view(
                    batch_size, num_augmentations + 1, -1, vocab_size
                )
                voted_logits = remapped_logits.sum(dim=1)  # [batch, seq_len, vocab]
                predictions = voted_logits.argmax(dim=-1)  # [batch, seq_len]

                # Compute loss on original samples only (using original targets)
                original_outputs = model(input_ids, puzzle_ids_tensor)
                loss, loss_components = compute_loss(original_outputs, target_ids)

            else:
                # Standard evaluation (no voting)
                outputs = model(input_ids, puzzle_ids_tensor)
                # for k, v in outputs.items():
                #     print(f"==> {k}={v.device}")
                # print(f"==> {target_ids.device=}")
                # print(f"==> {target_ids.shape=}")
                # print(f"==> {target_ids}")
                # quit()
                loss, loss_components = compute_loss(outputs, target_ids)
                predictions = torch.argmax(outputs["logits"], dim=-1)

            # Update metrics
            total_loss += loss_components["total_loss"]
            total_lm_loss += loss_components["lm_loss"]
            total_q_loss += loss_components["q_loss"]
            num_batches += 1

            # Exact match accuracy (entire puzzle correct) - compare voted predictions to true solutions
            exact_match = torch.all(predictions == target_ids, dim=1)
            exact_matches_tensor += exact_match.sum()  # Stay on GPU

            # Cell-wise accuracy (ignoring padding/blank cells)
            # Only count non-zero cells in targets
            non_zero_mask = target_ids != 0
            if non_zero_mask.any():
                correct_cells = (predictions == target_ids) & non_zero_mask
                total_correct_tensor += correct_cells.sum()  # Stay on GPU
                total_examples_tensor += non_zero_mask.sum()  # Stay on GPU

    # Convert to CPU only at the end
    exact_matches = exact_matches_tensor.cpu().item()
    total_correct = total_correct_tensor.cpu().item()
    total_examples = total_examples_tensor.cpu().item()

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
    config: dict[str, Any], config_path: str = "config/sudoku_config.yaml"
) -> None:
    """Train HRM model on 4x4 sudoku using configuration."""
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
    from pathlib import Path

    Path(training_cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    # Create data loaders based on configuration
    use_augmentations = config["training"].get("use_augmentations", True)
    eval_batch_size = config["evaluation"].get(
        "batch_size", 1
    )  # NEW: Get eval batch size

    train_loader, val_loader = create_dataloaders(
        data_dir=dataset_cfg["data_dir"],
        batch_size=training_cfg["batch_size"],
        eval_batch_size=eval_batch_size,  # NEW: Pass eval batch size
        max_train_samples=dataset_cfg.get("max_train_samples"),
        max_val_samples=dataset_cfg.get("max_val_samples"),
        use_augmentations=use_augmentations,
        shuffle_train=True,
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
    log_path = Path(training_cfg["save_dir"]) / training_cfg["log_file"]
    with Path(log_path).open("w", buffering=1) as log_f:
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
                    Path(training_cfg["save_dir"]) / "best_model.pt",
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
                Path(training_cfg["save_dir"]) / f"checkpoint_epoch_{epoch}.pt",
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
        test_loader = create_evaluation_dataloader(
            data_dir=dataset_cfg["data_dir"],
            split="test",
            batch_size=eval_batch_size,  # NEW: Use eval batch size
            max_samples=dataset_cfg.get("max_test_samples"),
        )
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
    """Parse command line arguments and run the main training routine."""
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
        "--save_dir",
        type=str,
        help="Override save directory from config",
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
    if args.save_dir:  # NEW: Add this line
        config["training"]["save_dir"] = args.save_dir
    if args.use_voting:
        config["evaluation"]["use_voting"] = True
    if args.no_voting:
        config["evaluation"]["use_voting"] = False

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Save the final configuration to save directory
    save_dir = config["training"]["save_dir"]  # Use save_dir instead of data_dir
    config_filename = Path(args.config).name  # Get just the filename
    config_output_path = Path(save_dir) / config_filename

    print(f"Saving final configuration to: {config_output_path}", flush=True)
    with Path(config_output_path).open("w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    # Save source files if specified in config
    if "storage" in config and "save_files" in config["storage"]:
        files_to_save = config["storage"]["save_files"]
        print(f"Saving {len(files_to_save)} source files to: {save_dir}")

        for file_path in files_to_save:
            source_file = Path(file_path)
            if source_file.exists():
                dest_file = Path(save_dir) / source_file.name
                print(f"  Copying {file_path} -> {dest_file}")

                # Copy the file
                with source_file.open("r") as src, dest_file.open("w") as dst:
                    dst.write(src.read())
            else:
                print(f"  Warning: File {file_path} not found, skipping...")

        print("Source files saved successfully!")
    else:
        print("No source files specified in config.storage.save_files")

    # Check if data directory exists
    if not Path(save_dir).exists():
        print(f"Error: Data directory {save_dir} does not exist!")
        print("Please run the dataset generation first:")
        print("python dataset/build_4x4_sudoku_dataset.py")
        return

    # Train the model
    train_model(config, args.config)


if __name__ == "__main__":
    main()
