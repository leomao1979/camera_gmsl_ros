# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

project(camera_multiple_gmsl C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    main.cpp
    ../common/LMImagePublisher.cpp
)

set(LIBRARIES
    glfw
    dw_samples_framework
    ${Driveworks_LIBRARIES}
    lodepng-src
    gpujpeg
    ${catkin_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

# ------------------------------------------------------------------------------
# Install target
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})
