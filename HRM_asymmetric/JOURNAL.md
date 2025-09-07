# HRM Asymmetric Development Journal

## 2025-01-XX - Initial Development

### Completed Tasks ✅

1. **Created Asymmetric HRM Architecture**
   - Implemented System 1 (Low-level) module: Fast, LLM-like processing
   - Implemented System 2 (High-level) module: Deliberate processing with working memory
   - Created main `AsymmetricHRMModel` class combining both systems

2. **Applied Standard HRM Notation**
   - Updated all files to use standard HRM notation:
     - `s1_layers/s2_layers` → `L_blocks/H_blocks`
     - `s1_cycles/s2_cycles` → `T_cycles`
     - `halt_max_steps` → `M_segments`
     - `system1/system2` → `L_module/H_module`
     - `s1_state/s2_state` → `L_state/H_state`
     - `final_s1_state/final_s2_state` → `final_L_state/final_H_state`
     - `steps_taken` → `segments_taken`

3. **Fixed Algorithm Implementation**
   - Corrected `forward_single_segment` method to properly implement T cycles
   - Fixed adaptive computation logic (training vs eval mode)
   - Ensured proper gradient flow through both modules

4. **Created Supporting Files**
   - `example.py`: Demonstration script with usage examples
   - `test_asymmetric_hrm.py`: Comprehensive test suite
   - `train.py`: Complete training pipeline
   - `README.md`: Documentation and usage guide

5. **Verified Functionality**
   - All tests pass successfully
   - Example script runs without errors
   - Training script works correctly
   - Model shows proper adaptive computation behavior

### Architecture Details

- **System 1 (L_module)**: 1 transformer block, fast processing
- **System 2 (H_module)**: 4 transformer blocks, working memory, reasoning attention
- **Parameters**: ~7M total (524K L_module, 5.9M H_module)
- **Computation**: Up to 192 block executions (32x deeper than standard transformer)

### Key Features

1. **Asymmetric Processing**: Different complexity levels for different reasoning tasks
2. **Working Memory**: System 2 maintains reasoning history across segments
3. **Adaptive Computation**: Model decides when to halt based on Q-values
4. **Hierarchical Reasoning**: L processes input + H state, H processes L state + input
5. **Standard HRM Algorithm**: Proper implementation of M segments × T cycles

### Files Created/Modified

- `__init__.py` - Module initialization
- `system1_module.py` - Low-level (System 1) processing module
- `system2_module.py` - High-level (System 2) processing module with memory
- `asymmetric_hrm.py` - Main model class combining both systems
- `example.py` - Usage examples and demonstrations
- `test_asymmetric_hrm.py` - Comprehensive test suite
- `train.py` - Training script with command-line interface
- `README.md` - Documentation and usage guide
- `JOURNAL.md` - This development journal

### Next Steps (if needed)

- [ ] Add more sophisticated reward functions for Q-learning
- [ ] Implement curriculum learning strategies
- [ ] Add support for different puzzle types
- [ ] Optimize memory usage for larger models
- [ ] Add visualization tools for reasoning process

### Notes

- Model successfully demonstrates Kahneman's System 1/System 2 distinction
- Adaptive computation works correctly in both training and evaluation modes
- All notation now follows standard HRM conventions
- Ready for production use and further experimentation

---

## 2025-01-XX - Hierarchical Hidden Dimensions Implementation

### Completed Tasks ✅

1. **Implemented Different Hidden Dimensions for System1 and System2**
   - Added `L_hidden_size` parameter for System 1 (fine-grained processing)
   - Added `H_hidden_size` parameter for System 2 (coarse-grained processing)
   - Default values: L_hidden_size=768, H_hidden_size=512
   - Follows CNN principle: fine-to-coarse hierarchical processing

2. **Added Projection Layers for Dimension Matching**
   - `L_to_H_proj`: Projects from L_hidden_size to H_hidden_size
   - `H_to_L_proj`: Projects from H_hidden_size to L_hidden_size
   - Enables communication between different dimensional spaces

3. **Updated Architecture Flow**
   - Input embeddings use L_hidden_size (fine-grained input)
   - System 1 processes in L_hidden_size space
   - System 2 processes in H_hidden_size space (abstract)
   - Cross-system communication uses projection layers
   - Final output uses H_hidden_size for predictions

4. **Updated Type Hints and Documentation**
   - Clear distinction between `L_hidden` and `H_hidden` dimensions
   - Proper tensor shape annotations throughout
   - Updated docstrings to reflect hierarchical processing

### Key Architectural Benefits

- **Fine-to-Coarse Hierarchy**: Similar to CNN progression from detailed to abstract
- **Computational Efficiency**: System 2 operates on smaller representations
- **Biological Plausibility**: Matches human reasoning (detailed → abstract)
- **Memory Efficiency**: System 2 working memory uses smaller dimensions

### Files Modified

- `asymmetric_hrm.py` - Updated to support different hidden dimensions
- `JOURNAL.md` - This entry documenting hierarchical dimension changes

### Technical Details

- **L_hidden_size**: 768 (System 1 - fine-grained processing)
- **H_hidden_size**: 512 (System 2 - coarse-grained processing)
- **Projection layers**: Linear layers for dimension matching
- **Architecture**: True hierarchical processing pipeline

---

## 2025-09-07_15:26 - Training Script Creation

### Completed Tasks ✅

1. **Created Comprehensive Training Script**
   - `train.py` based on HRM_didactic version
   - `AsymmetricHRMTrainer` class with separate optimizers
   - Complete training pipeline with validation and checkpointing
   - Command-line interface with all necessary arguments

2. **Added Asymmetric HRM Specific Features**
   - Support for L_blocks, H_blocks, T_cycles, M_segments parameters
   - Model information display (parameter counts, architecture details)
   - Proper handling of segments_taken vs steps_taken
   - Integration with new HRM notation

3. **Included Data Handling**
   - Simple `PuzzleDataset` class for training data generation
   - `create_data_loaders` function for train/val/test splits
   - Proper data loading and batching

4. **Verified Functionality**
   - Training script runs successfully
   - Model creates with correct parameters (~7M total)
   - Training progresses with decreasing loss
   - Q-values update properly during training
   - Adaptive computation works (variable segments taken)

### Training Script Features

- **Command Line Interface**: All standard training arguments plus asymmetric HRM specific ones
- **Separate Optimizers**: Different learning rates for LM and Q-learning components
- **Progress Tracking**: Comprehensive metrics and progress reporting
- **Checkpointing**: Model saving and loading functionality
- **Validation**: Proper train/val/test split handling

### Usage Examples

```bash
# Basic training
python train.py --epochs 5 --batch_size 16

# Custom asymmetric architecture
python train.py --epochs 10 --L_blocks 3 --H_blocks 6 --H_memory_size 128

# Full configuration
python train.py --epochs 20 --batch_size 32 --learning_rate 1e-4 --T_cycles 3 --M_segments 20
```

### Files Modified

- `train.py` - Complete training script for asymmetric HRM model
- `JOURNAL.md` - This entry documenting training script creation
