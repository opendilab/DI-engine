# Testing Log Probability Methods with GPU

The script `test_log_prob_fn` benchmarks different methods for calculating log probabilities (`naive_method`, `efficient_method`, `less_efficient_method`) using both `float32` and `bfloat16` precision formats on a GPU.

## Overview

The script performs benchmarks on three different methods for calculating log probabilities and reports the time and peak GPU memory usage for each method:

- **Naive Method**
- **Efficient Method**
- **Less Efficient Method**

It runs the tests for two types of precision:

- `float32`
- `bfloat16`

## Test Functions

There are two main test functions in the script:

1. **`test_log_prob_methods_float32`**: This function benchmarks the three methods using `float32` precision.
2. **`test_log_prob_methods_bfloat16`**: This function benchmarks the three methods using `bfloat16` precision.

### Workflow

1. **Parameters Setup**: The tests are executed using a batch size of `16`, a sequence length of `1024`, and a dictionary size of `32768`. The data is randomly generated for benchmarking.
2. **GPU Memory Tracking**: The GPU memory is tracked using `torch.cuda.max_memory_allocated()` to measure the peak memory usage during the benchmark.
3. **Method Execution**: Each method is run multiple times (10 iterations) to measure the execution time and to ensure stability.
4. **Results Validation**: The results from each method are compared with the `Naive` method to check for correctness, with a tolerance value applied for `bfloat16` precision.

### Benchmarked Methods:

- **Naive Method**: The basic, unoptimized method for calculating log probabilities.
- **Efficient Method**: An optimized version of the naive method to reduce memory usage.
- **Less Efficient Method**: A method with a higher memory consumption compared to the efficient method.

### GPU Memory Usage:

The function `get_gpu_memory()` is used to fetch the current peak GPU memory usage during the execution of each method.

## Output Example

### Testing with `float32` Precision

```
==================================================
Testing with float32 precision
==================================================

Naive:
Time: 5.07 ± 0.83 ms
Peak GPU Memory: 4096.31 MB

Efficient:
Time: 15.76 ± 21.19 ms
Peak GPU Memory: 2176.44 MB

Less_Efficient:
Time: 14.63 ± 5.06 ms
Peak GPU Memory: 4608.39 MB
PASSED [100%]
```

### Testing with `bfloat16` Precision

```
==================================================
Testing with bfloat16 precision
==================================================

Naive:
Time: 1.42 ± 0.00 ms
Peak GPU Memory: 2048.22 MB

Efficient:
Time: 1.83 ± 0.01 ms
Peak GPU Memory: 1152.25 MB

Less_Efficient:
Time: 8.67 ± 0.07 ms
Peak GPU Memory: 2560.27 MB
```

## Results Analysis

- Execution Time
  - The Naive method is the fastest in both precisions but sacrifices memory efficiency.
  - The Efficient method balances memory usage and execution time, though it is slower than the Naive method.
  - The Less Efficient method is slower than both the Naive and Efficient methods and consumes the most memory, making it the least desirable for both speed and memory usage.
- GPU Memory
  - The Efficient method consistently uses the least memory, especially in the `bfloat16` precision where it achieves the lowest memory consumption.
  - The Naive method uses more memory than the Efficient method but has lower execution times.
  - The Less Efficient method consumes the most memory in both precision formats.

## How to Run the Tests

To run the tests:

```bash
pytest -v -s test_log_prob_fn.py
```

