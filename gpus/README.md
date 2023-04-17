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
    - This is an extra example that is not needed before trying your hand at GPU parallelism

Each of these examples have a `README.md` explaining the code and the example code has comments
explaining what is going on in the code.

## What You Should Do

- Compile the code
- Test the system example
- Read the README of example1 and read through the example1 code
- Read the README of example2 and read through the code
- Do [you try](you_try/)

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

## Learning Parallel Patterns

In the `docs` folder there are markdown files explaining how to use parallel patterns. This includes:

- [map](docs/Map.md) which is a parallel pattern used to perform data parallel execution
- [map on gpu](docs/MapOnGPU.md) describes how to use the map pattern on a GPU
- [reduce](docs/Reduce.md) describes the reduce parallel pattern and map-reduce


## Sunlab

For Lehigh students who want to compile for sunlab, use the Dockerfile to build with `docker build -t sunlab .`. After this run with `docker run -v$(pwd):/root/tmp -it sunlab`.
Copy the resulting tar.gz file output to `/root/tmp`. After this exit the container and scp this tar.gz file to sunlab. Extract and unzip the file (`tar xzf <file>`). After this
you can run the code in the resulting `bin` folder. If you modify the code on your local machine you must copy this procedure again to get results on sunlab.
