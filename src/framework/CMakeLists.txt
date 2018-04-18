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

    Grid.hpp
    Grid.cpp

    WindowGLFW.hpp
    WindowGLFW.cpp

    MathUtils.hpp
    MathUtils.cpp
    MouseView3D.cpp
    MouseView3D.hpp

    Mat4.hpp

    Log.hpp
    Log.cpp

    StatsCounter.hpp
    CudaTimer.hpp

    ProfilerCUDA.hpp
    ProfilerCUDA.cpp

    GenericImage.hpp
    GenericImage.cpp

    SimpleStreamer.hpp
    SimpleCamera.hpp
    SimpleCamera.cpp

    SimpleFormatConverter.hpp
    SimpleRenderer.cpp
    SimpleRenderer.hpp
    SimpleRecordingPlayer.hpp
    SimpleRecordingPlayer.cpp

    ${SDK_BINARY_DIR}/configured/samples/framework/DataPath.hpp
)

set(LIBRARIES glfw)

if(NOT VIBRANTE)
    list(APPEND LIBRARIES
        GLEW::glew_s
    )
endif()

if(VIBRANTE)
    list(APPEND SOURCES
        WindowEGL.hpp
        WindowEGL.cpp
    )
    list(APPEND LIBRARIES
        drm
    )
endif()

if(LINUX)
    list(APPEND LIBRARIES
        ${X11_LIBRARIES} ${X11_Xrandr_LIB}
        ${X11_Xcursor_LIB} ${X11_Xxf86vm_LIB}
        ${X11_Xinerama_LIB} ${X11_Xi_LIB}
    )
elseif(VIBRANTE)
    list(APPEND LIBRARIES
        vibrante
        vibrante_Xlibs
        nvmedia
        ${EGL_LIBRARIES}
    )
endif()

list(APPEND LIBRARIES 
    ${CMAKE_THREAD_LIBS_INIT} 
    dl
    rt
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
                           ${lodepng_DIR}/include
                           ${CUDA_INCLUDE_DIRS}
                           ${SDK_BINARY_DIR}/configured/samples)
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")
