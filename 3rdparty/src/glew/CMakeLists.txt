# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

project(glew C CXX)

set(SOURCES
    # release-2.1.0
    src/glew.c
)

add_library(glew-src STATIC ${SOURCES})
target_include_directories(glew-src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add GL/GLU link dependencies this way - don't use FindPackage(OpenGL), as this conflicts with certain
# unusal OpenGL library locations (cross-compilation, Optimus/Bumblebee laptops)
target_link_libraries(glew-src INTERFACE GLU GL)
