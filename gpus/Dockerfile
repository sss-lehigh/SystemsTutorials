FROM nvidia/cuda:11.4.3-devel-rockylinux8

SHELL ["/usr/bin/env", "bash", "-c"]

RUN dnf install -y gcc-toolset-10 cmake python39 git vim perl-Thread-Queue
RUN python3 -m pip install conan==1.58

COPY . /root/gpus

WORKDIR /root/gpus

RUN mkdir build
RUN cd cutlass && scl enable gcc-toolset-10 "conan create ." && cd ..
RUN scl enable gcc-toolset-10 "cd build && conan install .. --build=missing && cmake .. && make -j && cpack"

