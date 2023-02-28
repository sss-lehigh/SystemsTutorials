cmake_minimum_required(VERSION 3.18)

project(GPUS LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}")

set(CMAKE_CXX_STANDARD 17)

find_package(TBB REQUIRED)

add_executable(example1_saxpy example1/example1.cu)
target_link_libraries(example1_saxpy PUBLIC TBB::tbb)
target_compile_options(example1_saxpy PUBLIC "--expt-extended-lambda")

add_executable(example2_dot example2/example2.cu)
target_link_libraries(example2_dot PUBLIC TBB::tbb)
target_compile_options(example2_dot PUBLIC "--expt-extended-lambda")

add_executable(solution you_try/solution/program_here.cu)
target_link_libraries(solution PUBLIC TBB::tbb)
target_compile_options(solution PUBLIC "--expt-extended-lambda")

add_executable(your_solution you_try/program_here.cu)
target_link_libraries(your_solution PUBLIC TBB::tbb)
target_compile_options(your_solution PUBLIC "--expt-extended-lambda")
