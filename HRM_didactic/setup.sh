#!/bin/bash

# Setup script for HRM Didactic Implementation
echo "Setting up HRM Didactic Implementation..."
echo "========================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

echo "✓ pip3 found"

# Install PyTorch
echo "Installing PyTorch..."
pip3 install torch

if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed successfully"
else
    echo "✗ Failed to install PyTorch"
    exit 1
fi

# Test the installation
echo "Testing installation..."
python3 test_structure.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Setup completed successfully!"
    echo ""
    echo "You can now run:"
    echo "  python3 example.py          # Run examples"
    echo "  python3 train.py            # Train the model"
    echo "  python3 evaluate.py --help  # See evaluation options"
else
    echo "✗ Setup failed - check the errors above"
    exit 1
fi
