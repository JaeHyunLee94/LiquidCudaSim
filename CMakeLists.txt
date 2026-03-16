cmake_minimum_required(VERSION 4.2)
project(LiquidCudaSim CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(LiquidCudaSim main.cu)

set_target_properties(LiquidCudaSim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
