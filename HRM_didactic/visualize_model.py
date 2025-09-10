"""
Model Visualization Script for HRM

This script provides visual representations of the HRM model architecture
and helps understand the data flow through the hierarchical reasoning process.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Try to import the model, but don't fail if PyTorch isn't available
try:
    from hrm_model import create_hrm_model

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - creating visualization without model instantiation")


def create_architecture_diagram():
    """Create a visual diagram of the HRM architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define components and their positions
    components = {
        "Input": (1, 7),
        "Token Embed": (1, 6),
        "Puzzle Embed": (1, 5),
        "Combined Embed": (1, 4),
        "H Module": (3, 3),
        "L Module": (5, 3),
        "Q Head": (3, 1),
        "LM Head": (5, 1),
        "Output": (4, 0),
    }

    # Draw components
    for name, (x, y) in components.items():
        if name in ["H Module", "L Module"]:
            # Draw as rectangles for modules
            rect = plt.Rectangle(
                (x - 0.4, y - 0.3),
                0.8,
                0.6,
                facecolor="lightblue",
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                x, y, name, ha="center", va="center", fontsize=10, fontweight="bold"
            )
        else:
            # Draw as circles for other components
            circle = plt.Circle(
                (x, y), 0.3, facecolor="lightgreen", edgecolor="black", linewidth=2
            )
            ax.add_patch(circle)
            ax.text(x, y, name, ha="center", va="center", fontsize=9)

    # Draw arrows showing data flow
    arrows = [
        ((1, 6.7), (1, 6.3)),  # Input -> Token Embed
        ((1, 5.7), (1, 5.3)),  # Token Embed -> Puzzle Embed
        ((1, 4.7), (1, 4.3)),  # Puzzle Embed -> Combined
        ((1.3, 4), (2.6, 3.3)),  # Combined -> H Module
        ((3.3, 3), (4.6, 3)),  # H Module -> L Module
        ((3, 2.7), (3, 1.3)),  # H Module -> Q Head
        ((5, 2.7), (5, 1.3)),  # L Module -> LM Head
        ((3, 0.7), (3.6, 0.3)),  # Q Head -> Output
        ((5, 0.7), (4.4, 0.3)),  # LM Head -> Output
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2, color="red"),
        )

    # Add title and labels
    ax.set_title("HRM Architecture Overview", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add description
    description = """
    Key Features:
    • Hierarchical reasoning with H (high-level) and L (low-level) modules
    • Adaptive computation time via Q-learning
    • Input injection into low-level module
    • Separate optimizers for different components
    """

    ax.text(
        0.5,
        2,
        description,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def create_data_flow_diagram():
    """Create a diagram showing the data flow through the model."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Define the flow steps
    steps = [
        "Input Tokens\n+ Puzzle ID",
        "Embeddings\n(Token + Puzzle)",
        "H State\nInitialization",
        "L State\nInitialization",
        "L Module\n(L cycles)",
        "H Module\n(H cycles)",
        "Q-Values\nComputation",
        "Halt Decision",
        "Output\nGeneration",
    ]

    # Draw the flow
    y_positions = np.linspace(8, 0, len(steps))

    for i, (step, y) in enumerate(zip(steps, y_positions)):
        # Draw step box
        rect = plt.Rectangle(
            (0.5, y - 0.4),
            2,
            0.8,
            facecolor="lightblue" if i % 2 == 0 else "lightgreen",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(1.5, y, step, ha="center", va="center", fontsize=10)

        # Draw arrow to next step
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(1.5, y_positions[i + 1] + 0.4),
                xytext=(1.5, y - 0.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
            )

    # Add side annotations for key concepts
    side_notes = [
        (3.5, 7.5, "Input Processing"),
        (3.5, 6.5, "State Initialization"),
        (3.5, 5.5, "Hierarchical\nReasoning"),
        (3.5, 4.5, "Low-level\nComputation"),
        (3.5, 3.5, "High-level\nPlanning"),
        (3.5, 2.5, "Adaptive\nHalting"),
        (3.5, 1.5, "Decision Making"),
        (3.5, 0.5, "Output Generation"),
    ]

    for x, y, note in side_notes:
        ax.text(
            x,
            y,
            note,
            ha="left",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 9)
    ax.set_title("HRM Data Flow", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    return fig


def create_training_flow_diagram():
    """Create a diagram showing the training process."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Training steps
    training_steps = [
        "Load Batch\n(Input + Target)",
        "Forward Pass\n(HRM Model)",
        "Compute Loss\n(LM + Q-learning)",
        "Backward Pass\n(Gradients)",
        "Update Parameters\n(Optimizers)",
        "Validation\n(Periodic)",
        "Checkpoint\n(Save Model)",
    ]

    # Draw training loop
    x_positions = np.linspace(1, 11, len(training_steps))

    for i, (step, x) in enumerate(zip(training_steps, x_positions)):
        # Draw step box
        rect = plt.Rectangle(
            (x - 0.8, 4),
            1.6,
            1,
            facecolor="lightcoral" if "Loss" in step else "lightblue",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(x, 4.5, step, ha="center", va="center", fontsize=9)

        # Draw arrow to next step
        if i < len(training_steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.8, 4.5),
                xytext=(x + 0.8, 4.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
            )

    # Add loss components
    loss_components = [
        (2, 2.5, "Language Modeling\nLoss (Cross-Entropy)"),
        (4, 2.5, "Q-Learning Loss\n(MSE)"),
        (6, 2.5, "Total Loss\n(Combined)"),
    ]

    for x, y, component in loss_components:
        ax.text(
            x,
            y,
            component,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    # Add arrows from loss components to training steps
    ax.annotate(
        "",
        xy=(2, 4),
        xytext=(2, 3.5),
        arrowprops=dict(arrowstyle="->", lw=1, color="blue"),
    )
    ax.annotate(
        "",
        xy=(4, 4),
        xytext=(4, 3.5),
        arrowprops=dict(arrowstyle="->", lw=1, color="blue"),
    )
    ax.annotate(
        "",
        xy=(6, 4),
        xytext=(6, 3.5),
        arrowprops=dict(arrowstyle="->", lw=1, color="blue"),
    )

    ax.set_xlim(0, 12)
    ax.set_ylim(1, 6)
    ax.set_title("HRM Training Process", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    return fig


def create_model_comparison():
    """Create a comparison between HRM and standard transformer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Standard Transformer
    ax1.set_title("Standard Transformer", fontsize=14, fontweight="bold")

    # Draw transformer layers
    for i in range(4):
        y = 6 - i
        rect = plt.Rectangle(
            (1, y - 0.3), 2, 0.6, facecolor="lightblue", edgecolor="black", linewidth=1
        )
        ax1.add_patch(rect)
        ax1.text(2, y, f"Layer {i+1}", ha="center", va="center", fontsize=10)

        if i < 3:
            ax1.annotate(
                "",
                xy=(2, y - 0.3),
                xytext=(2, y + 0.3),
                arrowprops=dict(arrowstyle="->", lw=1, color="red"),
            )

    ax1.text(2, 7, "Input", ha="center", va="center", fontsize=12, fontweight="bold")
    ax1.text(2, 0.5, "Output", ha="center", va="center", fontsize=12, fontweight="bold")

    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 8)
    ax1.axis("off")

    # HRM Model
    ax2.set_title("HRM Model", fontsize=14, fontweight="bold")

    # Draw H and L modules
    h_rect = plt.Rectangle(
        (0.5, 4), 1.5, 1, facecolor="lightgreen", edgecolor="black", linewidth=2
    )
    ax2.add_patch(h_rect)
    ax2.text(1.25, 4.5, "H Module\n(High-level)", ha="center", va="center", fontsize=10)

    l_rect = plt.Rectangle(
        (2, 4), 1.5, 1, facecolor="lightcoral", edgecolor="black", linewidth=2
    )
    ax2.add_patch(l_rect)
    ax2.text(2.75, 4.5, "L Module\n(Low-level)", ha="center", va="center", fontsize=10)

    # Draw Q-head
    q_rect = plt.Rectangle(
        (0.5, 2), 1.5, 0.8, facecolor="yellow", edgecolor="black", linewidth=2
    )
    ax2.add_patch(q_rect)
    ax2.text(1.25, 2.4, "Q-Head\n(Halt/Continue)", ha="center", va="center", fontsize=9)

    # Draw arrows
    ax2.annotate(
        "",
        xy=(1.25, 4),
        xytext=(1.25, 3.2),
        arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    )
    ax2.annotate(
        "",
        xy=(2.75, 4),
        xytext=(2.75, 3.2),
        arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    )

    ax2.text(1.25, 6, "Input", ha="center", va="center", fontsize=12, fontweight="bold")
    ax2.text(
        1.25, 1, "Output", ha="center", va="center", fontsize=12, fontweight="bold"
    )

    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 7)
    ax2.axis("off")

    plt.tight_layout()
    return fig


def main():
    """Create all visualizations."""
    print("Creating HRM Model Visualizations...")

    # Create output directory
    os.makedirs("visualizations", exist_ok=True)

    # Create diagrams
    diagrams = [
        (create_architecture_diagram, "architecture.png"),
        (create_data_flow_diagram, "data_flow.png"),
        (create_training_flow_diagram, "training_flow.png"),
        (create_model_comparison, "model_comparison.png"),
    ]

    for create_func, filename in diagrams:
        try:
            fig = create_func()
            filepath = os.path.join("visualizations", filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ Created {filepath}")
        except Exception as e:
            print(f"✗ Failed to create {filename}: {e}")

    print(f"\nVisualizations saved to 'visualizations/' directory")
    print(
        "You can view these images to understand the HRM architecture and training process."
    )


if __name__ == "__main__":
    main()

