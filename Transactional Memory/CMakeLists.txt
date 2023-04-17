cmake_minimum_required(VERSION 3.25)
project(Transactional_Memory)

set(CMAKE_CXX_STANDARD 23)

add_executable(linked-list linked-list-main.cpp)
target_include_directories(linked-list PRIVATE include/)
target_sources(linked-list PRIVATE src/LinkedList.cpp)

add_executable(concurrent-linked-list concurrent-linked-list-main.cpp)
target_include_directories(concurrent-linked-list PRIVATE include/)
target_sources(concurrent-linked-list PRIVATE src/ConcurrentLinkedList.cpp)

add_executable(transactional-memory-concurrent-linked-list transactional-memory-concurrent-linked-list-main.cpp)
target_include_directories(transactional-memory-concurrent-linked-list PRIVATE include/)
target_sources(transactional-memory-concurrent-linked-list PRIVATE src/LinkedList.cpp)
set_target_properties(transactional-memory-concurrent-linked-list PROPERTIES COMPILE_FLAGS "-ftm-ts")
