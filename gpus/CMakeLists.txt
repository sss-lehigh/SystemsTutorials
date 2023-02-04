cmake_minimum_required(VERSION 3.18)

project(GPUS LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}")

set(CMAKE_CXX_STANDARD 17)

#include(CheckLanguage)
#
#check_language(CUDA)
#
#if(CMAKE_CUDA_COMPILER)
#    enable_language(CUDA)
#else()
#    message(STATUS "No CUDA support")
#endif()

find_package(TBB REQUIRED)

add_executable(example1_using_cpp example1/example1_using_cpp.cu)
target_link_libraries(example1_using_cpp PUBLIC TBB::tbb)
target_compile_options(example1_using_cpp PUBLIC "--expt-extended-lambda")

add_executable(example2_using_cpp example2/example2.cu)
target_link_libraries(example2_using_cpp PUBLIC TBB::tbb)
target_compile_options(example2_using_cpp PUBLIC "--expt-extended-lambda")

