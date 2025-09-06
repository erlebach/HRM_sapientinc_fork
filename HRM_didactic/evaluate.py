"""
Evaluation Script for HRM Model

This script provides evaluation functionality for the trained HRM model,
including inference on new puzzles and analysis of the model's reasoning process.
"""

import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from hrm_model import HRMModel, create_hrm_model
from puzzle_dataset import PuzzleDataset, create_data_loaders
from torch.utils.data import DataLoader


class HRMEvaluator:
    """Evaluator class for the HRM model."""

    def __init__(
        self,
        model: HRMModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate a single batch and return metrics."""
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        puzzle_ids = batch["puzzle_ids"].to(self.device)

        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_ids, puzzle_ids)

            logits = outputs["logits"]
            q_halt_logits = outputs["q_halt_logits"]
            q_continue_logits = outputs["q_continue_logits"]
            steps_taken = outputs["steps_taken"]

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == target_ids) & (target_ids != 0)  # Ignore padding
            accuracy = correct.float().mean().item()

            # Compute perplexity
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            target_log_probs = (
                target_log_probs * (target_ids != 0).float()
            )  # Ignore padding
            perplexity = torch.exp(
                -target_log_probs.sum() / (target_ids != 0).sum()
            ).item()

            # Q-value statistics
            q_continue_mean = q_continue_logits.mean().item()
            q_halt_mean = q_halt_logits.mean().item()
            avg_steps = steps_taken.float().mean().item()

            return {
                "accuracy": accuracy,
                "perplexity": perplexity,
                "avg_steps": avg_steps,
                "q_continue_mean": q_continue_mean,
                "q_halt_mean": q_halt_mean,
                "halt_probability": torch.sigmoid(q_halt_logits).mean().item(),
            }

    def evaluate_dataset(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a full dataset."""
        total_metrics = {}
        num_batches = 0

        for batch in data_loader:
            metrics = self.evaluate_batch(batch)

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

    def generate_response(
        self, input_text: str, puzzle_id: int = 0, max_steps: int = None
    ) -> Dict[str, any]:
        """Generate a response for a given input text."""
        # Simple tokenization (in practice, you'd use a proper tokenizer)
        input_tokens = (
            [1]
            + [ord(c) % 1000 + 10 for c in input_text if c.isalnum() or c in " .,!?"]
            + [2]
        )
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        puzzle_ids = torch.tensor([puzzle_id], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, puzzle_ids, max_steps=max_steps)

            logits = outputs["logits"]
            q_halt_logits = outputs["q_halt_logits"]
            q_continue_logits = outputs["q_continue_logits"]
            steps_taken = outputs["steps_taken"]

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Convert back to text (simplified)
            response_tokens = predictions[0].cpu().tolist()
            response_text = "".join(
                [
                    chr((t - 10) % 256) if 10 <= t < 1000 else " "
                    for t in response_tokens
                ]
            )

            return {
                "input_text": input_text,
                "response_text": response_text.strip(),
                "steps_taken": steps_taken.item(),
                "q_halt_logit": q_halt_logits[0].item(),
                "q_continue_logit": q_continue_logits[0].item(),
                "halt_probability": torch.sigmoid(q_halt_logits[0]).item(),
            }


def load_model_from_checkpoint(checkpoint_path: str, device: str = None) -> HRMModel:
    """Load a trained model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model with same architecture as training
    model = create_hrm_model(
        vocab_size=1000,
        hidden_size=256,
        num_heads=8,
        intermediate_size=1024,
        max_seq_len=64,
        num_puzzle_ids=100,
        h_layers=2,
        l_layers=2,
        h_cycles=2,
        l_cycles=2,
        halt_max_steps=8,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def run_evaluation(
    checkpoint_path: str, batch_size: int = 32, device: str = None
) -> Dict[str, float]:
    """Run full evaluation on test set."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device)

    print("Creating data loaders...")
    _, _, test_loader = create_data_loaders(
        batch_size=batch_size,
        train_size=100,  # Small for demo
        val_size=50,
        test_size=50,
    )

    print("Running evaluation...")
    evaluator = HRMEvaluator(model, device)
    metrics = evaluator.evaluate_dataset(test_loader)

    return metrics


def interactive_demo(checkpoint_path: str, device: str = None):
    """Interactive demo where you can input puzzles and see the model's response."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device)
    evaluator = HRMEvaluator(model, device)

    print("\nInteractive HRM Demo")
    print("=" * 40)
    print("Enter puzzle descriptions and see how the model responds.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            input_text = input("Enter puzzle: ").strip()
            if input_text.lower() == "quit":
                break

            if not input_text:
                continue

            # Generate response
            response = evaluator.generate_response(input_text)

            print(f"\nModel Response:")
            print(f"  Text: {response['response_text']}")
            print(f"  Steps taken: {response['steps_taken']}")
            print(f"  Halt probability: {response['halt_probability']:.3f}")
            print(
                f"  Q-values - Halt: {response['q_halt_logit']:.3f}, Continue: {response['q_continue_logit']:.3f}"
            )
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Demo ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HRM model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "demo"],
        default="eval",
        help="Evaluation mode: eval for full evaluation, demo for interactive",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")

    args = parser.parse_args()

    if args.mode == "eval":
        metrics = run_evaluation(args.checkpoint, args.batch_size, args.device)
        print("\nEvaluation Results:")
        print("=" * 30)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    elif args.mode == "demo":
        interactive_demo(args.checkpoint, args.device)
