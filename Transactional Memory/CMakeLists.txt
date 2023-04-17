cmake_minimum_required(VERSION 3.25)
project(Transactional_Memory)

set(CMAKE_CXX_STANDARD 23)

add_executable(linked-list linked-list-main.cpp)
target_include_directories(linked-list PRIVATE include/)
target_sources(linked-list PRIVATE src/LinkedList.cpp)

add_executable(concurrent-linked-list concurrent-linked-list-main.cpp)
target_include_directories(concurrent-linked-list PRIVATE include/)
target_sources(concurrent-linked-list PRIVATE src/ConcurrentLinkedList.cpp)

