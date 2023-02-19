# GPU Tutorial

This tutorial uses CUDA and TBB to demonstrate how to use GPUs and what it means to use a GPU.

## Examples

- example1
    - example of how to write a parallel vector `z = a * x + y` on both CPUs and GPUs
    - compiles to example1\_saxpy
- example2
    - example of how to write a parallel dot product `a = x dot y` on both CPUs and GPUs
    - compiles to example2\_dot

## Compiling

You must start by installing cmake, conan, and [CUDA](https://developer.nvidia.com/cuda-downloads) 11 or greater.

- cmake can be installed from [cmake.org](https://cmake.org/download/)
- conan can be installed from [pip](https://pip.pypa.io/en/stable/installation/#supported-methods) or 
wherever you get your python packages (e.g., `pip install conan`)

Once installed make a directory called build (`mkdir build`).
In this directory run `conan install ..` and then `cmake ..`. After this run make using `make`.
At this point you will have compiled the binaries in build and be able to run them.

