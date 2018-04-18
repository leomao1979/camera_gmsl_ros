/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2014-2016 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef SAMPLES_COMMON_WINDOWGLFW_HPP__
#define SAMPLES_COMMON_WINDOWGLFW_HPP__

#include "Window.hpp"

#ifdef VIBRANTE
    #define GLFW_INCLUDE_ES3
    #define GLFW_EXPOSE_NATIVE_X11
    #define GLFW_EXPOSE_NATIVE_EGL
    #include <EGL/egl.h>
#endif

#include <GLFW/glfw3.h>
#include <memory>

class WindowGLFW : public WindowBase
{
  public:
    // create an X11 window
    //   width: width of window
    //   height: height of window
    WindowGLFW(const char* title, int width, int height, bool invisible = false);
    WindowGLFW(int w, int h, bool invisible = false)
        : WindowGLFW("DriveWorks", w, h, invisible)
    {}

    // release window
    ~WindowGLFW();

    // swap back and front buffers - return false if failed, i.e. window need close
    bool swapBuffers() override;

    // reset EGL context
    void resetContext() override;

    EGLContext createSharedContext() const override;

    // make window context current to the calling thread
    bool makeCurrent() override;

    // remove current window context from the calling thread
    bool resetCurrent() override;

    bool shouldClose() override { return glfwWindowShouldClose(m_hWindow) != 0; }

    // Set the window size
    bool setWindowSize(int width, int height) override;

    // get EGL display
    EGLDisplay getEGLDisplay(void) override;
    EGLContext getEGLContext(void) override;

    void onKeyCallback(int key, int scancode, int action, int mods);
    void onMouseButtonCallback(int button, int action, int mods);
    void onMouseMoveCallback(double x, double y);
    void onMouseWheelCallback(double dx, double dy);
    void onResizeWindowCallback(int width, int height);

  protected:
    GLFWwindow *m_hWindow;

#ifdef VIBRANTE
    EGLDisplay m_display;
    EGLContext m_context;
    std::unique_ptr<EGLConfig[]> m_config;
#endif
};

#endif // SAMPLES_COMMON_WINDOWGLFW_HPP__
