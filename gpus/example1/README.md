# SAXPY

The vector calculation `a * x + y` (where a is a scalar, x is a vector, and y is a vector) is an extremely important calculation called SAXPY (scalar a x plus y) 
It is used in a variety of settings ranging from:
- solving mathematics for engineering
- simulating problems (such as physics)
- machine learning
- etc.

The problem consists of iterating through the vectors x and y, scaling the value at the index in x, and summing with y.

On CPUs and GPUs we deal typically will parallelize this problem across hardware by using an abstraction called a thread. 
A thread shares memory with other threads in a process, but executes in parallel. 
By dividing a task across multiple threads we are able to execute in parallel and speedup our processing.

In order to parallelize this problem on modern hardware, we typically use a parallel pattern called map. When performing a map we break 
down the operation we need to perform into smaller computations which we then map to each thread we have.

For example when performing:  `1.0 * [10 11] + [1 2]` we could map the first index to a thread and the second index to a second thread.
This would lead to thread one calculating `1.0 * 10 + 1` and thread two calculating `1.0 * 11 + 2`.

We are not limited to mapping a single task to a single thread, sometimes we may map ranges of operations. For example:
`1.0 * [1 2 3 4] + [1 2 3 4]` may be mapped to two threads as calculating `1.0 * [1 2] + [1 2]` and `1.0 * [3 4] + [3 4]`. This enables
us to get an ideal speedup proportional to the number of threads that we can parallelize on.

To learn more about the map pattern and how it can be used in standard C++ [read here](../docs/Map.md).

For large problems, the GPU will enable us to parallelize more than the CPU can. While CPUs are limited to hundreds of cores, GPUs
can easily exceed thousands of cores for specialized problems such as floating point operations or integer arithmetic.

In order to use a GPU to get improvements in performance, we must copy memory from the CPU to the GPU, perform a calculation with 
code we write, and copy results in memory back from the GPU to the CPU.

To learn more about the performing the map pattern in a high level library on the GPU [read here](../docs/MapOnGPU.md).

In this example we demonstrate how to do this on a CPU at a high level, a high level way to do this on the GPU using [Thrust](https://thrust.github.io), and how this
operation could be written as a kernel (GPU code) in [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

## Results of Running

When running on a machine with a GTX 1660 Super and Intel Core i7,
we can get the following results.

Note a hand optimized cuda version can perform best, but thrust
is able to achieve close to the performance of hand optimizations
while being significantly more programmable.

```
Duration of parallel cpp version (ms):      8.73581
Duration of parallel tbb version (ms):      9.52444
Duration of parallel thrust version (ms):   1.99218
Duration of parallel cuda version (ms):     1.62102
Duration of sequential version (ms):        52.951
Speedup of best:                            32.6651
```
