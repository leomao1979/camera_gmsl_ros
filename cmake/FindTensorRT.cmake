# - Try to find TENSORRT
# Once done this will define
#
#  TENSORRT_FOUND - system has libusb
#  TENSORRT_INCLUDE_DIRS - the libusb include directory
#  TENSORRT_LIBRARIES - Link these to use libusb
#  TENSORRT_DEFINITIONS - Compiler switches required for using libusb
#
#  Adapted from cmake-modules Google Code project
#
#  Copyright (c) 2006 Andreas Schneider <mail@cynapses.org>
#
#  (Changes for libusb) Copyright (c) 2008 Kyle Machulis <kyle@nonpolynomial.com>
#
# Redistribution and use is allowed according to the terms of the New BSD license.
#
# CMake-Modules Project New BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the CMake-Modules Project nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

if (TENSORRT_LIBRARIES AND TENSORRT_INCLUDE_DIRS)
  # in cache already
  set(TENSORRT_FOUND TRUE)
else (TENSORRT_LIBRARIES AND TENSORRT_INCLUDE_DIRS)

  set(TENSORRT_TARGET "x86_64-linux-gnu")
  if (CMAKE_CROSSCOMPILING)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(TENSORRT_TARGET "aarch64-linux-gnu")
    endif()
  endif()

  # TENSORRT_LIB_PATH is useful for RPATH for cross-compiling.
  set(TENSORRT_PARENT_PATH /usr/local/nvidia/tensorrt/targets/${TENSORRT_TARGET})
  set(TENSORRT_INCLUDE_PATH ${TENSORRT_PARENT_PATH}/include)
  set(TENSORRT_LIB_PATH ${TENSORRT_PARENT_PATH}/lib)

  find_path(TENSORRT_INCLUDE_DIR
    NAMES
      NvInfer.h
    PATHS
      ${TENSORRT_INCLUDE_PATH}
    PATH_SUFFIXES
      libnvinfer
  )

  find_library(TENSORRT_LIBRARY
    NAMES
      nvinfer
    PATHS
      ${TENSORRT_LIB_PATH}
  )

  find_library(TENSORRT_CAFFEPARSER_LIBRARY
    NAMES
      nvcaffe_parser
    PATHS
      /usr/local/nvidia/tensorrt/targets/${TENSORRT_TARGET}/lib
  )

  set(TENSORRT_INCLUDE_DIRS
    ${TENSORRT_INCLUDE_DIR}
  )
  set(TENSORRT_LIBRARIES
    ${TENSORRT_LIBRARY}
    ${TENSORRT_CAFFEPARSER_LIBRARY}
  )

  if (TENSORRT_INCLUDE_DIRS AND TENSORRT_LIBRARIES)
     set(TENSORRT_FOUND TRUE)
  endif (TENSORRT_INCLUDE_DIRS AND TENSORRT_LIBRARIES)

  if (TENSORRT_FOUND)
    if (NOT TENSORRT_FIND_QUIETLY)
      message(STATUS "Found TensorRT:")
      message(STATUS " - Includes: ${TENSORRT_INCLUDE_DIRS}")
      message(STATUS " - Libraries: ${TENSORRT_LIBRARIES}")
    endif (NOT TENSORRT_FIND_QUIETLY)
    else (TENSORRT_FOUND)
      message(FATAL_ERROR "Could not find TensorRT")
  endif (TENSORRT_FOUND)

  # show the TENSORRT_INCLUDE_DIRS and TENSORRT_LIBRARIES variables only in the advanced view
  mark_as_advanced(TENSORRT_INCLUDE_DIRS TENSORRT_LIBRARIES)

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${TENSORRT_LIB_PATH}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${TENSORRT_LIB_PATH}")

  include_directories(SYSTEM ${TENSORRT_INCLUDE_DIRS})
endif (TENSORRT_LIBRARIES AND TENSORRT_INCLUDE_DIRS)
