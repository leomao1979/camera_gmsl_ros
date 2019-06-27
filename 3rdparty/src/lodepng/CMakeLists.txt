# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

project(lodepng C CXX)

set(SOURCES
    # version 20170917
    src/lodepng.cpp
)

add_library(lodepng-src STATIC ${SOURCES})
target_include_directories(lodepng-src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
