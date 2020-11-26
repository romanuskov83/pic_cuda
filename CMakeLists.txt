cmake_minimum_required(VERSION 2.8)
project(pic)

find_package(CUDA)

add_compile_options(-std=c++11)
include_directories(.)


cuda_add_executable(pic.ex
        main.cpp
        grid.cpp
        random.cpp
        gpu_code.cu
        )



