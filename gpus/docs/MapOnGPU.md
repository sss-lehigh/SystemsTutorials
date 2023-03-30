# Map On GPU

The [map pattern](Map.md) is agnostic to the architecture and can also be used on the GPU.

## In the Thrust library

The Thrust library is a CUDA library that functions as a high level way to execute common patterns on the GPU.

Thrust introduces `thrust::transform`, which can execute the transform operation depending on the location of the
memory (GPU or CPU).

When programming with GPUs we call the CPU the host and the GPU the device.

Memory is on the CPU if it is given in a `thrust::host_vector` and memory is on the GPU if it is given in a `thrust::device_vector`.

Memory can be moved between the CPU and GPU by assigning `host_vector`s and `device_vector`s or by using `thrust::copy`, which functions
similarly to [`std::copy`](https://en.cppreference.com/w/cpp/algorithm/copy), but will also handle moving memory between the GPU and CPU.

In our scalar a x plus y (SAXPY) implementation given in [example 1](../example1) this can be used to output the result by doing:

```{c++}
thrust::host_vector<float> x;
thrust::host_vector<float> y;
thrust::host_vector<float> output;
// ...
thrust::device_vector<float> x_d = x;
thrust::device_vector<float> y_d = y;
thrust::device_vector<float> output_d;
thrust::transform(x_d.begin(), x_d.end(), y_d.begin(), output_d.begin(), op);
output = output_d;
```

Where op multiplies the element of x by a, adds it with the element of y, and stores it in output.

## Writing Operations that Can Run on GPU

Thrust transform calls a kernel (GPU code which can be launched from the GPU). The operator op will be called from this kernel, but
must be compiled for the GPU.

In order to compile a function for the GPU we can tag it with `__device__`. To make it compile for the GPU and CPU we can tag it `__host__ __device__`.

In our examples we use the `--expt-extended-lambda` flag when compiling with our CUDA compiler `nvcc`. This flag enables us to write lambda functions tagged
with host and device. If you are unaware of lambda expressions please [read this](https://learn.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-170).

When writing lambda expressions for the GPU we cannot capture by reference, since the references would be to CPU memory. Instead we must capture by value, which will be
copied to and accessible by the GPU.

For example we can write the following lambda for the GPU:

```{c++}
int a = 2;
auto func = [a] __device__ (int x) {
    return x + a;
};
```

This lambda will add the value of a to the input x given, and will compile into GPU executable code.

If you are interested in learning more about CUDA feel free to read the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

If you are interested in learning more about Thrust feel free to [read the documentation](https://thrust.github.io).

