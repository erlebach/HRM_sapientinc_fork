# Project Snapshot - 2025-01-27

## Current Architecture
- **HRM Model**: Hierarchical Reasoning Machine with High-level (H) and Low-level (L) reasoning modules
- **4x4 Sudoku Task**: Constraint satisfaction problem with 16 cells, 4 digits (1-4), 0 for blank
- **Dual-Format Dataloader**: Training uses tuples, validation uses grouped dictionaries
- **Global ID System**: Proper (puzzle_id, aug_idx) tuple structure for puzzle grouping
- **Configuration System**: YAML-based configuration for all parameters
- **Status**: ✅ DATALOADER ARCHITECTURE COMPLETE - Ready for model training

## Active Features
- **Dataset Generation**: 972 train, 194 val, 196 test puzzles with proper deduplication
- **Dual-Format Dataloader**: Training and validation use different data structures
- **Puzzle Grouping**: 14 samples per group (1 original + 13 augmentations)
- **Global ID System**: Clean tuple format (13, 0), (13, 1), etc.
- **Augmentation Control**: Flexible training with/without augmentations
- **Validation Grouping**: Always uses all augmentations for proper evaluation

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
