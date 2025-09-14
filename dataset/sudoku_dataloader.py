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
    """Base dataset class for 4x4 Sudoku puzzles."""

    def __init__(self, data_dir: str, split: str, max_samples: int | None = None):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples

        # Load the dictionary-based dataset
        with Path(f"{data_dir}/{split}/puzzles_dict.pkl").open("rb") as f:
            self.puzzles_dict = pickle.load(f)

        # Convert to lists for compatibility
        self.puzzles = []
        self.solutions = []
        self.digit_maps = []
        self.puzzle_ids = []

        for puzzle_data in self.puzzles_dict.values():
            # Add original puzzle with global ID (puzzle_group_id, 0)
            self.puzzles.append(puzzle_data["puzzle"])
            self.solutions.append(puzzle_data["solution"])
            self.puzzle_ids.append((puzzle_data["id"], 0))  # (puzzle_group_id, 0)

            # Add identity map for original
            identity_map = np.array([0, 1, 2, 3, 4])
            self.digit_maps.append(identity_map)

            # Add augmentations (skip the first one since it's the original)
            for aug_data in puzzle_data["augmentations"].values():
                if (
                    aug_data["id"][1] > 0
                ):  # Only add augmentations, not the original (0)
                    self.puzzles.append(aug_data["puzzle"])
                    self.solutions.append(aug_data["solution"])
                    self.puzzle_ids.append(aug_data["id"])  # (puzzle_group_id, aug_idx)
                    self.digit_maps.append(aug_data["digit_map"])

        # Convert to numpy arrays
        self.puzzles = np.array(self.puzzles)
        self.solutions = np.array(self.solutions)
        self.digit_maps = np.array(self.digit_maps)
        # print(f"self.puzzles={self.puzzles}")
        # print(f"self.digit_maps={self.digit_maps}")
        # print(f"self.puzzle_ids={self.puzzle_ids}")
        self.puzzle_ids = np.array(self.puzzle_ids)

        # Limit samples if specified
        if max_samples is not None:
            self.puzzles = self.puzzles[:max_samples]
            self.solutions = self.solutions[:max_samples]
            self.digit_maps = self.digit_maps[:max_samples]
            self.puzzle_ids = self.puzzle_ids[:max_samples]

        print(f"Loaded {len(self.puzzles)} samples from {split} split")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.puzzles)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, np.ndarray]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            tuple of (puzzle, solution, puzzle_id, digit_map)
        """
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.long)
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)
        puzzle_id = self.puzzle_ids[idx]
        digit_map = self.digit_maps[idx]

        return {
            "puzzle": puzzle,
            "solution": solution,
            "puzzle_id": puzzle_id,
            "digit_map": digit_map,
        }

    # def _get_puzzle_group_id(self, idx: int) -> int:
    #     """Get the puzzle group ID for a given dataset index."""
    #     for puzzle_id, group_indices in self.puzzle_groups.items():
    #         if idx in group_indices:
    #             return puzzle_id
    #     return 0  # Default fallback

    # def get_puzzle_group_digit_maps(self, puzzle_id: int) -> List[torch.Tensor]:
    #     """Get all digit maps for a puzzle group.

    #     Args:
    #         puzzle_id: The puzzle group ID

    #     Returns:
    #         List of digit maps for all samples in the group
    #     """
    #     if puzzle_id not in self.puzzle_groups:
    #         return []

    #     group_indices = self.puzzle_groups[puzzle_id]
    #     return [
    #         torch.tensor(self.digit_maps[idx], dtype=torch.long)
    #         for idx in group_indices
    #     ]


class SudokuOriginalOnlyDataset(SudokuDataset):
    """Dataset that only uses original puzzles (no augmentations)."""

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        max_samples: int = None,
        config_path: str = "config/sudoku_data_generation.yaml",
    ):
        super().__init__(data_dir, split, max_samples)

        # Find indices of original puzzles (those with augmentation index 0)
        original_indices = []
        for i, puzzle_id in enumerate(self.puzzle_ids):
            if puzzle_id[1] == 0:  # Check if augmentation index is 0
                original_indices.append(i)

        # Only keep original puzzles
        self.puzzles = self.puzzles[original_indices]
        self.solutions = self.solutions[original_indices]
        self.digit_maps = self.digit_maps[original_indices]
        self.puzzle_ids = self.puzzle_ids[original_indices]

        print(f"Filtered to {len(self.puzzles)} original puzzles (no augmentations)")

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
        super().__init__(data_dir, split, max_samples)
        # Use all samples (original + augmentations)
        pass


class SudokuValidationDataset(Dataset):
    """Dataset for validation that groups puzzles with their augmentations."""

    def __init__(self, data_dir: str, split: str, max_samples: int | None = None):
        """Initialize the validation dataset.

        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('val', 'test')
            max_samples: Maximum number of puzzle groups to load (None for all)
        """
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples

        # Load the dictionary-based dataset
        file_path = Path(f"{data_dir}") / f"{split}" / "puzzles_dict.pkl"
        with Path(file_path).open("rb") as f:
            self.puzzles_dict = pickle.load(f)

        # Create puzzle groups
        self.puzzle_groups = []
        for puzzle_data in self.puzzles_dict.values():
            puzzle_group = {"puzzle_id": puzzle_data["id"], "samples": []}

            # Add original puzzle
            puzzle_group["samples"].append(
                {
                    "puzzle": puzzle_data["puzzle"],
                    "solution": puzzle_data["solution"],
                    "global_id": (puzzle_data["id"], 0),
                    "digit_map": np.array([0, 1, 2, 3, 4]),
                }
            )

            # Add augmentations (skip the first one since it's the original)
            for aug_data in puzzle_data["augmentations"].values():
                if (
                    aug_data["id"][1] > 0
                ):  # Only add augmentations, not the original (0)
                    puzzle_group["samples"].append(
                        {
                            "puzzle": aug_data["puzzle"],
                            "solution": aug_data["solution"],
                            "global_id": aug_data["id"],
                            "digit_map": aug_data["digit_map"],
                        }
                    )

            self.puzzle_groups.append(puzzle_group)

        # Limit puzzle groups if specified
        if max_samples is not None:
            self.puzzle_groups = self.puzzle_groups[:max_samples]

        print(f"Loaded {len(self.puzzle_groups)} puzzle groups from {split} split")

    def __len__(self) -> int:
        """Return the number of puzzle groups."""
        return len(self.puzzle_groups)

    def __getitem__(self, idx: int) -> dict:
        """Get a puzzle group.

        Args:
            idx: Puzzle group index

        Returns:
            Dictionary containing the puzzle group data
        """
        puzzle_group = self.puzzle_groups[idx]

        # Convert to tensors
        puzzles = []
        solutions = []
        global_ids = []
        digit_maps = []

        for sample in puzzle_group["samples"]:
            puzzles.append(torch.tensor(sample["puzzle"], dtype=torch.long))
            solutions.append(torch.tensor(sample["solution"], dtype=torch.long))

            # Ensure global_id is a clean tuple
            global_id = sample["global_id"]
            if isinstance(global_id, tuple):
                global_ids.append(global_id)
            else:
                # Convert to tuple if it's not
                global_ids.append(tuple(global_id))

            digit_maps.append(sample["digit_map"])

        print("after gordon")

        return {
            "puzzle_id": puzzle_group["puzzle_id"],
            "puzzles": torch.stack(puzzles),  # Shape: [14, 4, 4]
            "solutions": torch.stack(solutions),  # Shape: [14, 4, 4]
            "global_ids": global_ids,  # List of tuples
            "digit_maps": np.array(digit_maps),  # Shape: [14, 5]
        }


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
    train_dataset = dataset_class(data_dir, "train", max_train_samples)
    val_dataset = SudokuValidationDataset(data_dir, "val", max_val_samples)

    # Create data loaders with separate batch sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Use training batch size
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,  # NEW: Use evaluation batch size
        shuffle=False,  # Never shuffle validation
        num_workers=0,
        pin_memory=True,
        drop_last=True,
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
    dataset = SudokuWithAugmentationsDataset(data_dir, split, max_samples)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )


def test_validation_dataloader(val_dataloader) -> None:
    print(f"\n\nVal dataset size: {len(val_loader.dataset)} puzzle groups")

    # Test val loader - should get 14 samples per puzzle group
    print("\n--- Val Batch ---")
    for i, batch_data in enumerate(val_loader):
        print(f"==> Batch {i}:")
        print(f"  Puzzle ID: {batch_data['puzzle_id']}")
        print(f"  Number of samples: {batch_data['puzzles'].shape[0]}")
        print(f"  Puzzles shape: {batch_data['puzzles'].shape}")
        print(f"  Solutions shape: {batch_data['solutions'].shape}")

        # Convert tensor global_ids back to tuples for display
        # val_loader.dataset[0]["global_ids"] shows the structure as defined in __getitem__
        # However, val_loader iterator shows data as post-processed by the data loader.
        global_ids_display = []
        for gid in batch_data["global_ids"]:
            if isinstance(gid, list) and isinstance(gid[0], torch.Tensor):
                global_ids_display.append((gid[0].item(), gid[1].item()))
            else:
                global_ids_display.append(gid)

        print(f"  Global IDs: {global_ids_display}")
        print(f"  Digit maps shape: {batch_data['digit_maps'].shape}")

        # Show first few global IDs to verify structure
        print(f"  First 5 Global IDs: {global_ids_display[:5]}")

        if i >= 3:  # Only show first 2 batches
            break


def test_training_dataloader(train_dataloader) -> None:
    print(f"\n\nTrain dataset size: {len(val_loader.dataset)} puzzle groups")

    # Test val loader - should get 14 samples per puzzle group
    print("\n--- Train Batch ---")
    for i, batch_data in enumerate(train_dataloader):
        # print(f"{batch_data=}")
        print(f"==> Batch {i}:")
        print(f"  Puzzle ID: {batch_data['puzzle_id']}")
        print(f"  puzzle: {batch_data['puzzle']}")
        print(f"  solution: {batch_data['solution']}")
        print(f"  digit_map: {batch_data['digit_map']}")

        if i >= 1:  # Only show first 2 batches
            break


if __name__ == "__main__":
    # Test with configuration-based path
    config = load_data_config()
    data_dir = config["data_generation"]["output_dir"]

    print("=== Testing Validation Dataloader ===")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=2,
        eval_batch_size=1,
        # max_train_samples=5,
        # max_val_samples=3,
        use_augmentations=True,  # True,
    )

    test_validation_dataloader(val_loader)
    test_training_dataloader(train_loader)

    # nb puzzle groups = # unique puzzle_ids * (1 + num_augmentations)
    print(f"Train dataset size: {len(train_loader.dataset)} puzzle groups")
    # nb puzzle groups = # unique puzzle_ids
