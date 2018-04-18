cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE Debug)
include(SamplesSetBuildType) # Set the build type before project is created
set(SDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SDK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#-------------------------------------------------------------------------------
# Root project
#-------------------------------------------------------------------------------
project(camera_gmsl_raw_ros C CXX)

#-------------------------------------------------------------------------------
# System packages
#-------------------------------------------------------------------------------
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
# Basic configuration
#-------------------------------------------------------------------------------
include(ArchConfiguration)
include(CommonConfiguration)
include(SamplesConfiguration)
include(Samples3rdparty)
include(SamplesInstallConfiguration)

#-------------------------------------------------------------------------------
# Driveworks SDK
#-------------------------------------------------------------------------------
find_package(Driveworks REQUIRED)
include_directories(${Driveworks_INCLUDE_DIR})
list(APPEND Driveworks_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
if(LINUX)
    list(APPEND Driveworks_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} dl rt)
elseif(VIBRANTE)
    list(APPEND Driveworks_LIBRARIES ${vibrante_LIBRARIES} ${vibrante_Xlibs_LIBRARIES} nvmedia ${EGL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

#-------------------------------------------------------------------------------
# Catkin
#-------------------------------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
)
include_directories(${catkin_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Samples
#-------------------------------------------------------------------------------
if(CMAKE_CROSSCOMPILING)
  set(DRIVEWORKS_DATAPATH "../data")
else()
  set(DRIVEWORKS_DATAPATH "${CMAKE_CURRENT_LIST_DIR}/../data")
endif()
include_directories(${SDK_BINARY_DIR}/configured/samples)
configure_file(src/framework/DataPath.hpp.in
               ${SDK_BINARY_DIR}/configured/samples/framework/DataPath.hpp)

set(SAMPLES framework)
foreach(SAMPLE ${SAMPLES})
    add_subdirectory(src/${SAMPLE})
endforeach()

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
	src/LMImagePublisher.cpp
)

set(LIBRARIES
	glfw
    dw_samples_framework
    ${Driveworks_LIBRARIES}
    lodepng
	${catkin_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# ------------------------------------------------------------------------------
# Install target
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})

#-------------------------------------------------------------------------------
# Cuda dependencies
#-------------------------------------------------------------------------------
# Create a target to update cuda dependencies
add_cuda_dependencies_target(update_cuda_deps ON)
set_property(TARGET update_cuda_deps PROPERTY FOLDER "Support")

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES camera_gmsl_raw
  CATKIN_DEPENDS cv_bridge image_transport roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)
