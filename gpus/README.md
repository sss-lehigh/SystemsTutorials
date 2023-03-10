# GPU Tutorial

This tutorial uses CUDA and TBB to demonstrate how to use GPUs and what it means to use a GPU.

## Examples

- system\_example
    - example code that can find edges in an image and output the resulting image
    - compiles to image\_process\_cpu and image\_process\_gpu
- example1
    - example of how to write a parallel vector `z = a * x + y` on both CPUs and GPUs
    - compiles to example1\_saxpy
- example2
    - example of how to write a parallel dot product `a = x dot y` on both CPUs and GPUs
    - compiles to example2\_dot
- example3
    - example of how to write a parallel group by and agregation on both CPUs and GPUs
    - compiles to example3\_groupby

Each of these examples have a read me explaining the code and the example code has comments
explaining what is going on in the code.

## Compiling

You must start by installing cmake, conan, and [CUDA](https://developer.nvidia.com/cuda-downloads) 11 or greater.

- cmake can be installed from [cmake.org](https://cmake.org/download/)
- conan can be installed from [pip](https://pip.pypa.io/en/stable/installation/#supported-methods) or 
wherever you get your python packages (e.g., `pip install conan`)

Once installed navigate to the cutlass directory and run `conan create .`. 
Next make a directory called build (`mkdir build`).
In this directory run `conan install .. --build=missing` and then `cmake ..`. After this run make using `make -j`.
At this point you will have compiled the binaries in build and be able to run them.

