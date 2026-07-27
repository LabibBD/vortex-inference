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

add_executable(
    vortex_vector_add
    examples/vector_add.cu
)

target_include_directories(
    vortex_vector_add
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    vortex_vector_add
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(
    vortex_scalar_mul
    examples/scalar_mul.cu
)

target_include_directories(
    vortex_scalar_mul
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    vortex_scalar_mul
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(
    vortex_relu
    examples/relu.cu
)

target_include_directories(
    vortex_relu
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    vortex_relu
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
