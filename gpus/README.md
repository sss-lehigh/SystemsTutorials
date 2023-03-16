# GPU Tutorial

This tutorial uses CUDA and TBB to demonstrate how to use GPUs and what it means to use a GPU.

## Examples

- [system\_example](system_example/)
    - example code that can find edges in an image and output the resulting image
    - compiles to image\_process\_cpu and image\_process\_gpu
- [example1](example1/)
    - example of how to write a parallel vector `z = a * x + y` on both CPUs and GPUs
    - compiles to example1\_saxpy
- [example2](example2/)
    - example of how to write a parallel dot product `a = x dot y` on both CPUs and GPUs
    - compiles to example2\_dot
- [example3](example3/)
    - example of how to write a parallel group by and agregation on both CPUs and GPUs
    - compiles to example3\_groupby

Each of these examples have a `README.md` explaining the code and the example code has comments
explaining what is going on in the code.

## Compiling

You must start by installing cmake, conan, and [CUDA](https://developer.nvidia.com/cuda-downloads) 11 or greater.

- cmake can be installed from [cmake.org](https://cmake.org/download/)
- conan 1.58 can be installed from [pip](https://pip.pypa.io/en/stable/installation/#supported-methods) or 
wherever you get your python packages (e.g., `pip install conan==1.58`)

Once installed navigate to the [`cutlass`](cutlass) directory (`cd cutlass`) and run `conan create .` then return to the gpus directory (`cd ..`).
Next make a directory called build (`mkdir build`). Navigate to this directory (`cd build`).
In this directory run `conan install .. --build=missing` and then `cmake ..`. After this run make using `make -j`.
At this point you will have compiled the binaries in build and be able to run them.
Run `ls` to see the binaries.

