# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/src/lodepng)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/src/glfw)

if(VIBRANTE)
    set(vibrante_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/vibrante" CACHE PATH '' FORCE)
    find_package(vibrante REQUIRED CONFIG)
    if(NOT VIBRANTE_V5Q)
        set(vibrante_Xlibs_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/vibrante_Xlibs" CACHE PATH '' FORCE)
        find_package(vibrante_Xlibs CONFIG REQUIRED)
    endif()
    find_package(EGL REQUIRED)
    add_definitions(-DDW_USE_NVMEDIA)
    add_definitions(-DDW_USE_EGL)
    set(DW_USE_NVMEDIA ON)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
else()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/src/glew)
endif()

# Hide settings in default cmake view
mark_as_advanced(vibrante_DIR vibrante_Xlibs_DIR)
