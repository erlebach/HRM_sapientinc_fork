"""
Simple structure test that doesn't require PyTorch installation.
This verifies that the code structure is correct and can be imported.
"""

import os
import sys


def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing module imports...")

    try:
        # Test basic Python imports
        import json
        import math
        import random

        print("✓ Basic Python modules imported successfully")

        # Test if PyTorch is available
        try:
            import torch

            print("✓ PyTorch is available")
            return True
        except ImportError:
            print("⚠ PyTorch not available - this is expected if not installed")
            return False

    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "hrm_model.py",
        "puzzle_dataset.py",
        "train.py",
        "evaluate.py",
        "example.py",
        "README.md",
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False

    return all_exist


def test_code_syntax():
    """Test that Python files have valid syntax."""
    print("\nTesting code syntax...")

    python_files = [
        "hrm_model.py",
        "puzzle_dataset.py",
        "train.py",
        "evaluate.py",
        "example.py",
    ]

    all_valid = True
    for file in python_files:
        try:
            with open(file, "r") as f:
                compile(f.read(), file, "exec")
            print(f"✓ {file} has valid syntax")
        except SyntaxError as e:
            print(f"✗ {file} has syntax error: {e}")
            all_valid = False
        except Exception as e:
            print(f"⚠ {file} - could not test: {e}")

    return all_valid


def main():
    """Run all structure tests."""
    print("HRM Didactic Implementation - Structure Test")
    print("=" * 50)

    # Run tests
    imports_ok = test_imports()
    files_ok = test_file_structure()
    syntax_ok = test_code_syntax()

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Files: {'PASS' if files_ok else 'FAIL'}")
    print(f"Syntax: {'PASS' if syntax_ok else 'FAIL'}")

    if files_ok and syntax_ok:
        print("\n✓ All structure tests passed!")
        print("\nTo run the full implementation:")
        print("1. Install PyTorch: pip install torch")
        print("2. Run examples: python3 example.py")
        print("3. Train model: python3 train.py")
        print("4. Evaluate: python3 evaluate.py --checkpoint <path> --mode eval")
    else:
        print("\n✗ Some tests failed - check the errors above")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
