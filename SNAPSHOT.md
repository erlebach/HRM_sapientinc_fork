# Project Snapshot - 2025-01-27

## Current Architecture
- **HRM Model**: Hierarchical Reasoning Machine with High-level (H) and Low-level (L) reasoning modules
- **4x4 Sudoku Task**: Constraint satisfaction problem with 16 cells, 4 digits (1-4), 0 for blank
- **Dataset Structure**: Dictionary-based with global indexing and persistent puzzle IDs
- **Voting Mechanism**: Implemented but performs worse due to constraint violations
- **Configuration System**: YAML-based configuration for all parameters
- **Status**: ✅ DATASET ARCHITECTURE REBUILT - Clean, maintainable structure

## Active Features
- **Dataset Generation**: 972 train, 194 val, 196 test puzzles with proper deduplication
- **Data Structure**: `puzzle_dict[global_idx] = {"id": puzzle_id, "puzzle": ..., "augmentations": {...}}`
- **Augmentation System**: Composite keys `(puzzle_id, aug_idx)` for unique identification
- **Voting Analysis**: 3-4% worse performance due to constraint violation in majority voting
- **Puzzle ID System**: Persistent IDs that travel with puzzles across splits
- **Flexible Storage**: Dictionary-based pickle files instead of flattened arrays

## File Structure
```
/Users/erlebach/src/2025/HRM/HRM_sapientinc_fork/
├── sudoku4x4.py                    # Main training script with YAML config support ✅
├── config/sudoku_config.yaml       # Centralized configuration file ✅
├── dataset/sudoku_dataloader.py    # Updated with separate batch sizes ✅
├── HRM_didactic/hrm_model.py       # HRM model implementation
├── dataset/build_4x4_sudoku_dataset.py    # Dataset generation (1000 train)
├── dataset/build_4x4_sudoku_small.py      # Small dataset generation (200 train)
├── checkpoints_4x4/                # Model checkpoints and logs
└── checkpoints_4x4_small/          # Small model checkpoints
```

## Key Components
- **Model**: 256 hidden size, 1024 intermediate, 3 H/L layers, 2 cycles each
- **Training**: AdamW optimizer, 1e-4 learning rate, 8 batch size, 50 epochs
- **Evaluation**: Voting with 20 augmentations, grid-level Q-value weighting, batch size 1
- **Data**: 4x4 Sudoku puzzles with 8-12 blank cells per puzzle
- **Configuration**: YAML file with dataset, model, training, and evaluation sections

## Recent Changes
- **✅ Separate Batch Sizes**: Training (8) and evaluation (1) batch sizes implemented
- **✅ Enhanced Debugging**: Single sample evaluation for easier debugging
- **✅ Memory Efficiency**: Lower memory usage during evaluation
- **✅ Error Isolation**: Single sample processing prevents batch-level errors
- **✅ Configuration Updates**: Added evaluation.batch_size parameter
- **✅ Function Updates**: Updated dataloader functions with new parameters

## Current Status
- **Implementation**: ✅ COMPLETE - All features implemented and tested
- **Baseline Performance**: 24% exact accuracy, 82.7% cell accuracy
- **Debugging Ready**: Batch size 1 evaluation enables easier issue tracing
- **Next Steps**: Debug voting mechanism issues with single sample processing
- **Expected Improvement**: 60-80% exact accuracy with proper voting and GPU resources
- **Ready for**: H100 GPU deployment with larger model capacity

## Usage Examples
```bash
# Use default configuration (training batch=8, eval batch=1)
python sudoku4x4.py

# Override specific parameters
python sudoku4x4.py --epochs 100 --learning_rate 0.001 --use_voting

# Use custom config file
python sudoku4x4.py --config config/my_config.yaml

# Disable voting for baseline comparison
python sudoku4x4.py --no_voting
```

## Configuration Structure
```yaml
# Training Configuration
training:
  batch_size: 8  # For efficient training
  learning_rate: 0.0001
  num_epochs: 50

# Evaluation Configuration  
evaluation:
  batch_size: 1  # For clean debugging
  use_voting: true
  num_augmentations: 20
  show_examples_after: true
```
