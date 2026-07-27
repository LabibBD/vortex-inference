cmake_minimum_required(VERSION 3.24)

project(
    VORTEX
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 Laptop GPU - Ada Lovelace, Compute Capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(
    vortex_cuda_check
    examples/cuda_check.cu
)

target_compile_options(
    vortex_cuda_check
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
