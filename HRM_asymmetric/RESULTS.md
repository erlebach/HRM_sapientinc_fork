# RUN
```
============================================================================================================================================
FINAL MEMORY OPERATIONS TIMING TABLE
============================================================================================================================================
Update\Retrieval | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token
--------------------------------------------------------------------------------------------------------------------------------------------
                | Simple                               | Attention                            | Sophisticated
--------------------------------------------------------------------------------------------------------------------------------------------
Simple          | 0.4          0.8          0.4          | 1.5          2.9          1.5          | 1.6          3.1          1.6          |
Attention       | 1.6          3.1          1.6          | 2.5          4.9          2.5          | 2.4          4.8          2.4          |
Sophisticated   | 0.8          1.6          0.8          | 1.9          3.7          1.9          | 2.0          3.9          2.0          |
--------------------------------------------------------------------------------------------------------------------------------------------
TIMING METRICS EXPLANATION:
--------------------------------------------------------------------------------------------------------------------------------------------
Raw (ms):        Total memory operation time in milliseconds
ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)
                 - Measures computational efficiency per element processed
                 - Lower values indicate more efficient memory operations
                 - Should be roughly constant across different problem sizes if operations scale linearly

ns/token:        Nanoseconds per token (batch_size × seq_len)
                 - Measures computational cost per token in the sequence
                 - Shows how memory operations scale with sequence length
                 - Linear scaling: ns/token remains constant as seq_len increases
                 - Quadratic scaling: ns/token increases linearly with seq_len

Problem size: batch=1, seq_len=1000, hidden_size=512
Total elements: 512,000
Total tokens: 1,000
```

# RUN
```
============================================================================================================================================
FINAL MEMORY OPERATIONS TIMING TABLE
============================================================================================================================================
Update\Retrieval | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token
--------------------------------------------------------------------------------------------------------------------------------------------
                | Simple                               | Attention                            | Sophisticated
--------------------------------------------------------------------------------------------------------------------------------------------
Simple          | 0.5          0.5          0.3          | 2.5          2.4          1.2          | 2.5          2.5          1.3          |
Attention       | 2.3          2.2          1.2          | 4.0          3.9          2.0          | 4.7          4.6          2.3          |
Sophisticated   | 1.0          0.9          0.5          | 3.0          3.0          1.5          | 3.4          3.3          1.7          |
--------------------------------------------------------------------------------------------------------------------------------------------
TIMING METRICS EXPLANATION:
--------------------------------------------------------------------------------------------------------------------------------------------
Raw (ms):        Total memory operation time in milliseconds
ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)
                 - Measures computational efficiency per element processed
                 - Lower values indicate more efficient memory operations
                 - Should be roughly constant across different problem sizes if operations scale linearly

ns/token:        Nanoseconds per token (batch_size × seq_len)
                 - Measures computational cost per token in the sequence
                 - Shows how memory operations scale with sequence length
                 - Linear scaling: ns/token remains constant as seq_len increases
                 - Quadratic scaling: ns/token increases linearly with seq_len

Problem size: batch=1, seq_len=2000, hidden_size=512
Total elements: 1,024,000
Total tokens: 2,000
============================================================================================================================================
```

# RUN

```
============================================================================================================================================
FINAL MEMORY OPERATIONS TIMING TABLE
============================================================================================================================================
Update\Retrieval | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token
--------------------------------------------------------------------------------------------------------------------------------------------
                | Simple                               | Attention                            | Sophisticated
--------------------------------------------------------------------------------------------------------------------------------------------
Simple          | 0.8          0.4          0.2          | 4.4          2.1          1.1          | 4.7          2.3          1.2          |
Attention       | 4.2          2.1          1.1          | 7.6          3.7          1.9          | 7.2          3.5          1.8          |
Sophisticated   | 1.5          0.7          0.4          | 4.4          2.1          1.1          | 4.9          2.4          1.2          |
--------------------------------------------------------------------------------------------------------------------------------------------
TIMING METRICS EXPLANATION:
--------------------------------------------------------------------------------------------------------------------------------------------
Raw (ms):        Total memory operation time in milliseconds
ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)
                 - Measures computational efficiency per element processed
                 - Lower values indicate more efficient memory operations
                 - Should be roughly constant across different problem sizes if operations scale linearly

ns/token:        Nanoseconds per token (batch_size × seq_len)
                 - Measures computational cost per token in the sequence
                 - Shows how memory operations scale with sequence length
                 - Linear scaling: ns/token remains constant as seq_len increases
                 - Quadratic scaling: ns/token increases linearly with seq_len

Problem size: batch=1, seq_len=4000, hidden_size=512
Total elements: 2,048,000
Total tokens: 4,000
============================================================================================================================================
```

# RUN (`seq_len`=8000, `D`=512)

```
============================================================================================================================================
FINAL MEMORY OPERATIONS TIMING TABLE
============================================================================================================================================
Update\Retrieval | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token
--------------------------------------------------------------------------------------------------------------------------------------------
                | Simple                               | Attention                            | Sophisticated
--------------------------------------------------------------------------------------------------------------------------------------------
Simple          | 1.4          0.3          0.2          | 7.2          1.8          0.9          | 7.5          1.8          0.9          |
Attention       | 6.4          1.6          0.8          | 13.3         3.3          1.7          | 13.4         3.3          1.7          |
Sophisticated   | 2.2          0.5          0.3          | 8.2          2.0          1.0          | 9.1          2.2          1.1          |
--------------------------------------------------------------------------------------------------------------------------------------------
TIMING METRICS EXPLANATION:
--------------------------------------------------------------------------------------------------------------------------------------------
Raw (ms):        Total memory operation time in milliseconds
ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)
                 - Measures computational efficiency per element processed
                 - Lower values indicate more efficient memory operations
                 - Should be roughly constant across different problem sizes if operations scale linearly

ns/token:        Nanoseconds per token (batch_size × seq_len)
                 - Measures computational cost per token in the sequence
                 - Shows how memory operations scale with sequence length
                 - Linear scaling: ns/token remains constant as seq_len increases
                 - Quadratic scaling: ns/token increases linearly with seq_len

Problem size: batch=1, seq_len=8000, hidden_size=512
Total elements: 4,096,000
Total tokens: 8,000
============================================================================================================================================
```

# RUN (`seq_len`=8000, `D`=256)
```
============================================================================================================================================
FINAL MEMORY OPERATIONS TIMING TABLE
============================================================================================================================================
Update\Retrieval | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token
--------------------------------------------------------------------------------------------------------------------------------------------
                | Simple                               | Attention                            | Sophisticated
--------------------------------------------------------------------------------------------------------------------------------------------
Simple          | 0.8          0.4          0.1          | 4.1          2.0          0.5          | 4.4          2.2          0.6          |
Attention       | 5.3          2.6          0.7          | 8.0          3.9          1.0          | 9.4          4.6          1.2          |
Sophisticated   | 1.5          0.8          0.2          | 4.4          2.2          0.6          | 4.7          2.3          0.6          |
--------------------------------------------------------------------------------------------------------------------------------------------
TIMING METRICS EXPLANATION:
--------------------------------------------------------------------------------------------------------------------------------------------
Raw (ms):        Total memory operation time in milliseconds
ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)
                 - Measures computational efficiency per element processed
                 - Lower values indicate more efficient memory operations
                 - Should be roughly constant across different problem sizes if operations scale linearly

ns/token:        Nanoseconds per token (batch_size × seq_len)
                 - Measures computational cost per token in the sequence
                 - Shows how memory operations scale with sequence length
                 - Linear scaling: ns/token remains constant as seq_len increases
                 - Quadratic scaling: ns/token increases linearly with seq_len

Problem size: batch=1, seq_len=8000, hidden_size=256
Total elements: 2,048,000
Total tokens: 8,000
============================================================================================================================================v
```

# RUN (`seq_len`=8000, `D`=256, `M`=64) (`M` was 128 above)

```
============================================================================================================================================
FINAL MEMORY OPERATIONS TIMING TABLE
============================================================================================================================================
Update\Retrieval | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token     | Raw (ms)     ns/element   ns/token
--------------------------------------------------------------------------------------------------------------------------------------------
                | Simple                               | Attention                            | Sophisticated
--------------------------------------------------------------------------------------------------------------------------------------------
Simple          | 0.7          0.4          0.1          | 3.1          1.5          0.4          | 3.6          1.8          0.4          |
Attention       | 3.4          1.7          0.4          | 6.2          3.0          0.8          | 6.7          3.3          0.8          |
Sophisticated   | 1.3          0.6          0.2          | 3.8          1.9          0.5          | 4.2          2.1          0.5          |
--------------------------------------------------------------------------------------------------------------------------------------------
TIMING METRICS EXPLANATION:
--------------------------------------------------------------------------------------------------------------------------------------------
Raw (ms):        Total memory operation time in milliseconds
ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)
                 - Measures computational efficiency per element processed
                 - Lower values indicate more efficient memory operations
                 - Should be roughly constant across different problem sizes if operations scale linearly

ns/token:        Nanoseconds per token (batch_size × seq_len)
                 - Measures computational cost per token in the sequence
                 - Shows how memory operations scale with sequence length
                 - Linear scaling: ns/token remains constant as seq_len increases
                 - Quadratic scaling: ns/token increases linearly with seq_len

Problem size: batch=1, seq_len=8000, hidden_size=256
Total elements: 2,048,000
Total tokens: 8,000
============================================================================================================================================
```
