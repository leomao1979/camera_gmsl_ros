# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

project(glfw-src C CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_INSTALL OFF CACHE BOOL "")
set(GLFW_VULKAN_STATIC OFF CACHE BOOL "")
set(GLFW_DOCUMENT_INTERNALS OFF CACHE BOOL "")
set(GLFW_USE_WAYLAND OFF CACHE BOOL "")
set(GLFW_USE_MIR OFF CACHE BOOL "")
set(GLFW_USE_SCREEN OFF CACHE BOOL "")

mark_as_advanced(BUILD_SHARED_LIBS
                 GLFW_BUILD_EXAMPLES
                 GLFW_BUILD_TESTS
                 GLFW_BUILD_DOCS
                 GLFW_INSTALL
                 GLFW_VULKAN_STATIC
                 GLFW_DOCUMENT_INTERNALS
                 GLFW_USE_WAYLAND
                 GLFW_USE_MIR
                 GLFW_USE_SCREEN)

if(VIBRANTE_V5Q)
  set(GLFW_USE_SCREEN ON) # qnx requires screen-port of glfw (missing X11)
endif()

add_subdirectory(3.2-screen)
