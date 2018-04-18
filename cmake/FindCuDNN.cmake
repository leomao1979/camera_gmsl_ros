# - Try to find libcudnn
# Once done this will define
#
#  CUDNN_FOUND - system has libusb
#  CUDNN_INCLUDE_DIRS - the libusb include directory
#  CUDNN_LIBRARIES - Link these to use libusb
#  CUDNN_DEFINITIONS - Compiler switches required for using libusb
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

if (CUDNN_LIBRARIES AND CUDNN_INCLUDE_DIRS)
  # in cache already
  set(CUDNN_FOUND TRUE)
else (CUDNN_LIBRARIES AND CUDNN_INCLUDE_DIRS)

  set(CUDNN_TARGET "x86_64-linux")
  if (CMAKE_CROSSCOMPILING)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(CUDNN_TARGET "aarch64-linux")
    endif()
  endif()

  # CUDNN_LIB_PATH is useful for RPATH for cross-compiling.
  set(CUDNN_PARENT_PATH ${CUDA_TOOLKIT_ROOT_DIR}/targets/${CUDNN_TARGET})
  set(CUDNN_INCLUDE_PATH ${CUDNN_PARENT_PATH}/include)
  set(CUDNN_LIB_PATH ${CUDNN_PARENT_PATH}/lib)

  find_path(CUDNN_INCLUDE_DIR
    NAMES
      cudnn.h
    PATHS
      ${CUDNN_INCLUDE_PATH}
    PATH_SUFFIXES
      libcudnn
  )

  find_library(CUDNN_LIBRARY
    NAMES
      cudnn
    PATHS
      ${CUDNN_LIB_PATH}
  )

  set(CUDNN_INCLUDE_DIRS
    ${CUDNN_INCLUDE_DIR}
  )
  set(CUDNN_LIBRARIES
    ${CUDNN_LIBRARY}
  )

  if (CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
     set(CUDNN_FOUND TRUE)
  endif (CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)

  if (CUDNN_FOUND)
    if (NOT CUDNN_FIND_QUIETLY)
      message(STATUS "Found libcudnn:")
      message(STATUS " - Includes: ${CUDNN_INCLUDE_DIRS}")
      message(STATUS " - Libraries: ${CUDNN_LIBRARIES}")
    endif (NOT CUDNN_FIND_QUIETLY)
    else (CUDNN_FOUND)
      message(FATAL_ERROR "Could not find libcudnn")
  endif (CUDNN_FOUND)

  # show the CUDNN_INCLUDE_DIRS and CUDNN_LIBRARIES variables only in the advanced view
  mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${CUDNN_LIB_PATH}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${CUDNN_LIB_PATH}")

  include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
endif (CUDNN_LIBRARIES AND CUDNN_INCLUDE_DIRS)
