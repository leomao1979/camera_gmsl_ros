# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

project(dw_samples_framework C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    Checks.hpp
    SampleFramework.hpp
    SampleFramework.cpp

    DriveWorksSample.hpp
    DriveWorksSample.cpp

    ProgramArguments.hpp
    ProgramArguments.cpp

    Window.hpp
    Window.cpp

    Grid.hpp
    Grid.cpp

    WindowGLFW.hpp
    WindowGLFW.cpp

    MathUtils.hpp
    MathUtils.cpp
    MouseView3D.cpp
    MouseView3D.hpp

    Mat3.hpp
    Mat4.hpp

    Log.hpp
    Log.cpp

    StatsCounter.hpp
    CudaTimer.hpp

    ProfilerCUDA.hpp
    ProfilerCUDA.cpp

    SimpleStreamer.hpp
    SimpleCamera.hpp
    SimpleCamera.cpp

    SimpleRenderer.cpp
    SimpleRenderer.hpp
    SimpleRecordingPlayer.hpp
    SimpleRecordingPlayer.cpp

    ScreenshotHelper.hpp
    ScreenshotHelper.cpp
    RenderUtils.hpp
    RenderUtils.cpp
    ${SDK_BINARY_DIR}/configured/samples/framework/DataPath.hpp
)

set(LIBRARIES
    glfw
    lodepng-src
    )

if(NOT VIBRANTE)
    list(APPEND LIBRARIES
        glew-src
    )
endif()

if(VIBRANTE)
    list(APPEND SOURCES
         WindowEGL.hpp
         WindowEGL.cpp
    )
    if(VIBRANTE_V5Q)
        list(APPEND SOURCES
            WindowQNXEGL.hpp
            WindowQNXEGL.cpp
        )
        list(APPEND LIBRARIES
            screen
        )
    elseif(VIBRANTE_V5L)
        list(APPEND SOURCES
            WindowLinuxEGL.hpp
            WindowLinuxEGL.cpp
        )
        list(APPEND LIBRARIES
            drm
        )
    endif()
endif()

if(LINUX)
    list(APPEND LIBRARIES
    )
elseif(VIBRANTE)
    list(APPEND LIBRARIES
        vibrante
        nvmedia
        ${EGL_LIBRARIES}
    )

    if(VIBRANTE_V5L)
        list(APPEND LIBRARIES
            vibrante_Xlibs
        )
    endif()
endif()

list(APPEND LIBRARIES
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
target_include_directories(${PROJECT_NAME}
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/..
                           ${CMAKE_CURRENT_SOURCE_DIR}/../..
                           ${CUDA_INCLUDE_DIRS}
                           ${SDK_BINARY_DIR}/configured/samples)
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")
