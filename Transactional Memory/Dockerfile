FROM ubuntu:devel
LABEL authors="冀卓疌"
LABEL description="a small transactional memory tutorial"

RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && \
    apt -qy install apt-utils

RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && \
    apt -qy lsb-release wget software-properties-common gnupg

RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && \
    apt -q update && \
    apt -qy install git make cmake ninja-build zlib1g mold && \
    bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" && \
    rm -r /var/lib/apt/lists/*

RUN git clone https://github.com/WowbaggersLiquidLunch/llvm-project.git && \
    mkdir -p llvm-project/build/Ninja-Release && \
    cd llvm-project/build/Ninja-Release && \

RUN cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lldb;lld;polly;libc;libclc;mlir;openmp;pstl" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt;libc;libcxx;libcxxabi;libunwind;openmp" \
    -DLLVM_USE_LINKER=mold \
    ../../llvm

RUN ninja install

RUN clang --version

RUN cd ~ && git clone https://github.com/sss-lehigh/SystemsTutorials.git
