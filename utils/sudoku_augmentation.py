"""
Sudoku augmentation utilities for voting mechanism.
"""

import random

import numpy as np


def simple_digit_augmentation(
    board: np.ndarray, solution: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply simple digit permutation augmentation.

    Only permutes digits (1,2,3,4) while keeping 0 (blank) unchanged.
    This preserves the spatial structure and should be truly equivalent.

    Args:
        board: 4x4 puzzle board
        solution: 4x4 solution board

    Returns:
        tuple containing:
            - aug_board: Augmented puzzle board with permuted digits
            - aug_solution: Augmented solution board with permuted digits
            - digit_map: 5-element array mapping original digits to new digits
                        Index 0 maps to 0 (preserves blanks)
                        Indices 1-4 map to a random permutation of [1,2,3,4]
                        Example: [0, 3, 1, 4, 2] means 0→0, 1→3, 2→1, 3→4, 4→2
    """
    # Create a random digit mapping: permutation of 1..4, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 5)), (1, 0))

    # Apply digit mapping to both board and solution
    aug_board = digit_map[board]
    aug_solution = digit_map[solution]

    return aug_board, aug_solution, digit_map


def shuffle_4x4_sudoku(
    board: np.ndarray, solution: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random transformations to create augmented versions.

    Args:
        board: 4x4 puzzle board
        solution: 4x4 solution board

    Returns:
        Augmented puzzle and solution
    """
    # Create a random digit mapping: permutation of 1..4, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 5)), (1, 0))

    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5

    # Generate valid row permutation within 2x2 blocks
    # Block 1: rows 0,1
    block1_rows = np.random.permutation([0, 1])
    # Block 2: rows 2,3
    block2_rows = np.random.permutation([2, 3])
    row_perm = np.concatenate([block1_rows, block2_rows])

    # Similarly for columns (stacks)
    # Block 1: cols 0,1
    block1_cols = np.random.permutation([0, 1])
    # Block 2: cols 2,3
    block2_cols = np.random.permutation([2, 3])
    col_perm = np.concatenate([block1_cols, block2_cols])

    # Build 16->16 mapping
    mapping = np.array([row_perm[i // 4] * 4 + col_perm[i % 4] for i in range(16)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping
        new_board = x.flatten()[mapping].reshape(4, 4).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)
