"""
Data loaders for 4x4 Sudoku with and without augmentations.
"""

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def load_data_config(
    config_path: str = "config/sudoku_data_generation.yaml",
) -> dict[str, Any]:
    """Load data configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class SudokuDataset(Dataset):
    """Base Sudoku dataset class."""

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        max_samples: int = None,
        config_path: str = "config/sudoku_data_generation.yaml",
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to dataset directory (if None, read from config)
            split: Dataset split (train/val/test)
            max_samples: Maximum number of samples to use (None for all)
            config_path: Path to YAML configuration file
        """
        # Load data directory from config if not provided
        if data_dir is None:
            config = load_data_config(config_path)
            data_dir = config["data_generation"]["output_dir"]

        self.data_dir = data_dir
        self.split = split

        # Load data
        self.puzzles = np.load(f"{data_dir}/{split}/all__inputs.npy")
        self.solutions = np.load(f"{data_dir}/{split}/all__labels.npy")
        self.digit_maps = np.load(f"{data_dir}/{split}/all__digit_maps.npy")
        with open(f"{data_dir}/{split}/puzzle_groups.pkl", "rb") as f:
            self.puzzle_groups: dict = pickle.load(f)

        # Limit samples if requested
        if max_samples is not None:
            self.puzzles = self.puzzles[:max_samples]
            self.solutions = self.solutions[:max_samples]
            self.digit_maps = self.digit_maps[:max_samples]

        # Get original puzzle indices (every 21st sample if 20 augmentations)
        self.original_indices = self._get_original_indices()

    def _get_original_indices(self) -> List[int]:
        """Get indices of original puzzles (not augmentations)."""
        original_indices = []
        for puzzle_id, group_indices in self.puzzle_groups.items():
            # First index in each group is the original puzzle
            original_indices.append(group_indices[0])
        return sorted(original_indices)

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.long)
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)
        digit_map = torch.tensor(self.digit_maps[idx], dtype=torch.long)

        return {
            "input_ids": puzzle.flatten(),
            "target_ids": solution.flatten(),
            "puzzle_ids": torch.tensor(0, dtype=torch.long),  # Always use 0 for now
            "digit_map": digit_map,  # Include digit mapping for voting
        }

    def _get_puzzle_group_id(self, idx: int) -> int:
        """Get the puzzle group ID for a given dataset index."""
        for puzzle_id, group_indices in self.puzzle_groups.items():
            if idx in group_indices:
                return puzzle_id
        return 0  # Default fallback

    def get_puzzle_group_digit_maps(self, puzzle_id: int) -> List[torch.Tensor]:
        """Get all digit maps for a puzzle group.

        Args:
            puzzle_id: The puzzle group ID

        Returns:
            List of digit maps for all samples in the group
        """
        if puzzle_id not in self.puzzle_groups:
            return []

        group_indices = self.puzzle_groups[puzzle_id]
        return [
            torch.tensor(self.digit_maps[idx], dtype=torch.long)
            for idx in group_indices
        ]


class SudokuOriginalOnlyDataset(SudokuDataset):
    """Dataset that only uses original puzzles (no augmentations)."""

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        max_samples: int = None,
        config_path: str = "config/sudoku_data_generation.yaml",
    ):
        super().__init__(data_dir, split, max_samples, config_path)

        # Only keep original puzzles
        self.puzzles = self.puzzles[self.original_indices]
        self.solutions = self.solutions[self.original_indices]
        self.digit_maps = self.digit_maps[
            self.original_indices
        ]  # Also filter digit maps

        # Update puzzle groups to only include originals
        self.puzzle_groups = {i: [i] for i in range(len(self.puzzles))}

    def __len__(self) -> int:
        return len(self.puzzles)


class SudokuWithAugmentationsDataset(SudokuDataset):
    """Dataset that uses all samples including augmentations."""

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        max_samples: int | None = None,
        config_path: str = "config/sudoku_data_generation.yaml",
    ) -> None:
        # 3rd argument can be None. The function has incorrect type hint
        super().__init__(data_dir, split, max_samples, config_path)
        # Use all samples (original + augmentations)
        pass


def create_dataloaders(
    data_dir: str = None,
    batch_size: int = 8,
    eval_batch_size: int = 1,  # NEW: Add eval_batch_size parameter
    max_train_samples: int = None,
    max_val_samples: int = None,
    use_augmentations: bool = True,
    shuffle_train: bool = True,
    config_path: str = "config/sudoku_data_generation.yaml",
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        data_dir: Path to dataset directory (if None, read from config)
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (val/test)
        max_train_samples: Maximum training samples
        max_val_samples: Maximum validation samples
        use_augmentations: Whether to use augmentations in training
        shuffle_train: Whether to shuffle training data
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Choose dataset class based on augmentation preference
    dataset_class = (
        SudokuWithAugmentationsDataset
        if use_augmentations
        else SudokuOriginalOnlyDataset
    )

    # Create datasets
    train_dataset = dataset_class(data_dir, "train", max_train_samples, config_path)
    val_dataset = dataset_class(data_dir, "val", max_val_samples, config_path)

    # Create data loaders with separate batch sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Use training batch size
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,  # NEW: Use evaluation batch size
        shuffle=False,  # Never shuffle validation
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_evaluation_dataloader(
    data_dir: str = None,
    split: str = "val",
    batch_size: int = 1,  # Process one puzzle group at a time
    max_samples: int = None,  # NEW: Add max_samples parameter
    config_path: str = "config/sudoku_data_generation.yaml",
) -> DataLoader:
    """Create evaluation data loader that processes puzzle groups.

    Args:
        data_dir: Path to dataset directory (if None, read from config)
        split: Dataset split (val/test)
        batch_size: Batch size (should be 1 for group processing)
        max_samples: Maximum number of samples to use (None for all)
        config_path: Path to YAML configuration file

    Returns:
        Data loader for evaluation
    """
    dataset = SudokuWithAugmentationsDataset(data_dir, split, max_samples, config_path)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )


if __name__ == "__main__":
    # Test with configuration-based path
    create_evaluation_dataloader(split="test", batch_size=1, max_samples=10)
