```python
python dataset/build_4x4_sudoku_dataset.py --config config/sudoku_data_generation.yaml
# Only required for testing. It is used by sudoku4x4.py
# python dataset/sudoku_dataloader.py
python sudoku4x4.py --config sudoku.yaml --batch_size=4
```
