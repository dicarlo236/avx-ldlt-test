cmake_minimum_required(VERSION 3.0)
project(avx_ldlt_functions)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -ggdb -march=core-avx2")

add_executable(avx_ldlt_functions main.cpp reference_implementation.cpp avx_implementation.cpp)