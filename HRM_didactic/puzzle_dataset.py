"""
Simple Puzzle Dataset for HRM Training

This module provides a simplified dataset for training the HRM model on various
types of reasoning puzzles including arithmetic, pattern completion, and logic puzzles.
"""

import math
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class PuzzleDataset(Dataset):
    """
    A simple dataset that generates various types of reasoning puzzles.

    Each puzzle consists of:
    - Input sequence: Problem description and context
    - Output sequence: Step-by-step solution
    - Puzzle type: Category of reasoning required
    """

    def __init__(
        self,
        num_samples: int = 1000,
        max_seq_len: int = 64,
        vocab_size: int = 1000,
        num_puzzle_types: int = 10,
    ):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_puzzle_types = num_puzzle_types

        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.SEP_TOKEN = 3

        # Generate all samples upfront for simplicity
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Dict]:
        """Generate all puzzle samples."""
        samples = []

        for i in range(self.num_samples):
            puzzle_type = i % self.num_puzzle_types

            if puzzle_type == 0:
                sample = self._generate_arithmetic_puzzle()
            elif puzzle_type == 1:
                sample = self._generate_pattern_puzzle()
            elif puzzle_type == 2:
                sample = self._generate_logic_puzzle()
            elif puzzle_type == 3:
                sample = self._generate_sequence_puzzle()
            elif puzzle_type == 4:
                sample = self._generate_word_puzzle()
            else:
                sample = self._generate_arithmetic_puzzle()  # Default fallback

            samples.append(sample)

        return samples

    def _generate_arithmetic_puzzle(self) -> Dict:
        """Generate arithmetic reasoning puzzles."""
        # Generate two numbers and an operation
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(["+", "-", "*", "/"])

        if op == "+":
            result = a + b
            problem = f"Calculate {a} + {b}"
            solution = f"Step 1: {a} + {b} = {result}"
        elif op == "-":
            result = a - b
            problem = f"Calculate {a} - {b}"
            solution = f"Step 1: {a} - {b} = {result}"
        elif op == "*":
            result = a * b
            problem = f"Calculate {a} * {b}"
            solution = f"Step 1: {a} * {b} = {result}"
        else:  # division
            result = a // b
            remainder = a % b
            problem = f"Calculate {a} / {b}"
            solution = f"Step 1: {a} / {b} = {result} remainder {remainder}"

        return {
            "input": problem,
            "output": solution,
            "puzzle_type": 0,
            "puzzle_id": random.randint(0, 99),
        }

    def _generate_pattern_puzzle(self) -> Dict:
        """Generate pattern completion puzzles."""
        # Generate a simple number pattern
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        length = random.randint(3, 6)

        pattern = [start + i * step for i in range(length)]
        missing_pos = random.randint(1, length - 2)
        pattern_with_gap = pattern.copy()
        pattern_with_gap[missing_pos] = "?"

        problem = f"Complete the pattern: {pattern_with_gap}"
        solution = f"Step 1: Pattern increases by {step}, missing number is {pattern[missing_pos]}"

        return {
            "input": problem,
            "output": solution,
            "puzzle_type": 1,
            "puzzle_id": random.randint(0, 99),
        }

    def _generate_logic_puzzle(self) -> Dict:
        """Generate simple logic puzzles."""
        puzzles = [
            (
                "If all birds can fly and penguins are birds, can penguins fly?",
                "Step 1: This is a logical fallacy - penguins are birds but cannot fly",
            ),
            (
                "If A > B and B > C, what is the relationship between A and C?",
                "Step 1: A > C by transitivity",
            ),
            (
                "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                "Step 1: Not necessarily - the ground could be wet for other reasons",
            ),
        ]

        problem, solution = random.choice(puzzles)
        return {
            "input": problem,
            "output": solution,
            "puzzle_type": 2,
            "puzzle_id": random.randint(0, 99),
        }

    def _generate_sequence_puzzle(self) -> Dict:
        """Generate sequence completion puzzles."""
        # Generate Fibonacci-like or arithmetic sequences
        if random.random() < 0.5:
            # Fibonacci-like
            a, b = random.randint(1, 5), random.randint(1, 5)
            seq = [a, b, a + b, b + (a + b)]
            problem = f"Complete the sequence: {seq[:3]} ..."
            solution = f"Step 1: Each number is sum of previous two, next is {seq[3]}"
        else:
            # Arithmetic sequence
            start = random.randint(1, 10)
            step = random.randint(2, 5)
            seq = [start + i * step for i in range(4)]
            problem = f"Complete the sequence: {seq[:3]} ..."
            solution = f"Step 1: Sequence increases by {step}, next is {seq[3]}"

        return {
            "input": problem,
            "output": solution,
            "puzzle_type": 3,
            "puzzle_id": random.randint(0, 99),
        }

    def _generate_word_puzzle(self) -> Dict:
        """Generate simple word puzzles."""
        puzzles = [
            (
                "What word becomes shorter when you add two letters?",
                "Step 1: 'Short' becomes 'shorter' - it gets longer, not shorter",
            ),
            (
                "I am taken from a mine and shut in a wooden case. What am I?",
                "Step 1: Pencil lead (graphite) is mined and put in wooden cases",
            ),
            ("What has keys but no locks?", "Step 1: A piano has keys but no locks"),
        ]

        problem, solution = random.choice(puzzles)
        return {
            "input": problem,
            "output": solution,
            "puzzle_type": 4,
            "puzzle_id": random.randint(0, 99),
        }

    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs (simplified tokenization)."""
        # Simple character-based tokenization
        tokens = [self.START_TOKEN]
        for char in text:
            if char.isalnum() or char in " .,!?+-*/=<>":
                token_id = min(
                    ord(char) % (self.vocab_size - 10) + 10, self.vocab_size - 1
                )
                tokens.append(token_id)
        tokens.append(self.END_TOKEN)
        return tokens

    def _pad_sequence(self, tokens: List[int]) -> List[int]:
        """Pad or truncate sequence to max_seq_len."""
        if len(tokens) > self.max_seq_len:
            return tokens[: self.max_seq_len]
        else:
            return tokens + [self.PAD_TOKEN] * (self.max_seq_len - len(tokens))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single puzzle sample."""
        sample = self.samples[idx]

        # Convert text to tokens
        input_tokens = self._text_to_tokens(sample["input"])
        output_tokens = self._text_to_tokens(sample["output"])

        # Pad sequences
        input_tokens = self._pad_sequence(input_tokens)
        output_tokens = self._pad_sequence(output_tokens)

        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "target_ids": torch.tensor(output_tokens, dtype=torch.long),
            "puzzle_ids": torch.tensor(sample["puzzle_id"], dtype=torch.long),
            "puzzle_type": torch.tensor(sample["puzzle_type"], dtype=torch.long),
        }


def create_data_loaders(
    batch_size: int = 32,
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, validation, and test data loaders."""
    from torch.utils.data import DataLoader, random_split

    # Create full dataset
    full_dataset = PuzzleDataset(num_samples=train_size + val_size + test_size)

    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = PuzzleDataset(num_samples=10)

    print("Dataset created successfully!")
    print(f"Number of samples: {len(dataset)}")

    # Show a few examples
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Target shape: {sample['target_ids'].shape}")
        print(f"Puzzle ID: {sample['puzzle_ids'].item()}")
        print(f"Puzzle type: {sample['puzzle_type'].item()}")

    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=4)

    print(f"\nData loaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Show a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

