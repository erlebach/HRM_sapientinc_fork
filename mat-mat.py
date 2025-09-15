#!/usr/bin/env python3
"""GPU Matrix-Matrix Multiplication Timing Script.

This script benchmarks matrix-matrix multiplication on GPU backends (CUDA and MPS)
with configurable matrix sizes and reproducible random number generation.
"""

import argparse
import time
import torch
import numpy as np
from typing import Tuple, Dict, Any


def setup_gpu_environment() -> Tuple[torch.device, str]:
    """Detect and setup the best available GPU backend.
    
    Returns:
        Tuple of (device, backend_name) for the best available GPU.
        
    Raises:
        RuntimeError: If no GPU backend is available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "CUDA"
        print(f"Using CUDA backend: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "MPS"
        print("Using MPS backend (Apple Silicon)")
    else:
        raise RuntimeError("No GPU backend available (CUDA or MPS)")
    
    return device, backend


def create_random_matrices(
    size: int, 
    device: torch.device, 
    dtype: torch.dtype = torch.float32,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create two random square matrices for multiplication.
    
    Args:
        size: Size of the square matrices (size x size).
        device: PyTorch device to create tensors on.
        dtype: Data type for the tensors.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (matrix_a, matrix_b) both of shape (size, size).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create random matrices on the specified device
    matrix_a = torch.randn(size, size, device=device, dtype=dtype)
    matrix_b = torch.randn(size, size, device=device, dtype=dtype)
    
    return matrix_a, matrix_b


def time_matrix_multiplication(
    matrix_a: torch.Tensor, 
    matrix_b: torch.Tensor,
    num_warmup: int = 10,
    num_trials: int = 100
) -> Dict[str, Any]:
    """Time matrix-matrix multiplication with proper GPU synchronization.
    
    Args:
        matrix_a: First matrix for multiplication.
        matrix_b: Second matrix for multiplication.
        num_warmup: Number of warmup iterations.
        num_trials: Number of timing trials.
        
    Returns:
        Dictionary containing timing statistics and matrix info.
    """
    device = matrix_a.device
    
    # Warmup runs to ensure GPU is ready
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = torch.matmul(matrix_a, matrix_b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    
    # Timing trials
    print(f"Running {num_trials} timing trials...")
    times = []
    
    for i in range(num_trials):
        start_time = time.perf_counter()
        
        result = torch.matmul(matrix_a, matrix_b)
        
        # Synchronize to ensure operation completes
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_trials} trials")
    
    # Calculate statistics
    times = np.array(times)
    stats = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "median_time": np.median(times),
        "p95_time": np.percentile(times, 95),
        "p99_time": np.percentile(times, 99),
        "matrix_size": matrix_a.shape[0],
        "dtype": str(matrix_a.dtype),
        "device": str(device)
    }
    
    return stats


def calculate_theoretical_flops(size: int) -> int:
    """Calculate theoretical FLOPs for matrix-matrix multiplication.
    
    Args:
        size: Size of the square matrices.
        
    Returns:
        Number of floating point operations.
    """
    # For A (n×n) × B (n×n) = C (n×n)
    # Each element of C requires n multiplications and n-1 additions
    # Total: n² × (2n - 1) ≈ 2n³ for large n
    return 2 * size ** 3


def print_timing_results(stats: Dict[str, Any]) -> None:
    """Print formatted timing results.
    
    Args:
        stats: Dictionary containing timing statistics.
    """
    print("\n" + "="*60)
    print("MATRIX-MATRIX MULTIPLICATION TIMING RESULTS")
    print("="*60)
    print(f"Matrix Size: {stats['matrix_size']}×{stats['matrix_size']}")
    print(f"Data Type: {stats['dtype']}")
    print(f"Device: {stats['device']}")
    print(f"Number of Elements: {stats['matrix_size']**2:,}")
    print(f"Theoretical FLOPs: {calculate_theoretical_flops(stats['matrix_size']):,}")
    print()
    
    print("Timing Statistics (seconds):")
    print(f"  Mean:     {stats['mean_time']:.6f} ± {stats['std_time']:.6f}")
    print(f"  Median:   {stats['median_time']:.6f}")
    print(f"  Min:      {stats['min_time']:.6f}")
    print(f"  Max:      {stats['max_time']:.6f}")
    print(f"  95th %ile: {stats['p95_time']:.6f}")
    print(f"  99th %ile: {stats['p99_time']:.6f}")
    print()
    
    # Calculate performance metrics
    flops = calculate_theoretical_flops(stats['matrix_size'])
    gflops = flops / (stats['mean_time'] * 1e9)
    print(f"Performance: {gflops:.2f} GFLOPS")
    print("="*60)


def main():
    """Main function to run the matrix multiplication timing benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark matrix-matrix multiplication on GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "size", 
        type=int, 
        help="Size of the square matrices (size x size)"
    )
    parser.add_argument(
        "--dtype", 
        choices=["float32", "float64"], 
        default="float32",
        help="Data type for the matrices"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=100,
        help="Number of timing trials"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if args.size <= 0:
        print("Error: Matrix size must be positive")
        return 1
    
    if args.size > 10000:
        print(f"Warning: Large matrix size ({args.size}) may cause memory issues")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return 0
    
    try:
        # Setup GPU environment
        device, backend = setup_gpu_environment()
        
        # Convert dtype string to torch dtype
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        
        print(f"Creating {args.size}×{args.size} matrices...")
        print(f"Data type: {args.dtype}")
        print(f"Random seed: {args.seed}")
        
        # Create random matrices
        matrix_a, matrix_b = create_random_matrices(
            args.size, device, dtype, args.seed
        )
        
        # Time the multiplication
        stats = time_matrix_multiplication(
            matrix_a, matrix_b, args.warmup, args.trials
        )
        
        # Print results
        print_timing_results(stats)
        
        return 0
        
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
