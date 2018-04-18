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

#include "WindowGLFW.hpp"

#include <iostream>
#include <cstring>

#ifdef VIBRANTE
    #include <GLFW/glfw3native.h>
    #include <EGL/eglext.h>
#endif

// -----------------------------------------------------------------------------
WindowGLFW::WindowGLFW(const char* title, int width, int height, bool invisible)
    : WindowBase(width, height)
#ifdef VIBRANTE
    , m_display(EGL_NO_DISPLAY)
    , m_context(EGL_NO_CONTEXT)
#endif
{
    if (glfwInit() == 0) {
        std::cout << "WindowGLFW: Failed initialize GLFW " << std::endl;
        throw std::exception();
    }

    // Create a windowed mode window and its OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 0);     // Disable MSAA
    glfwWindowHint(GLFW_DEPTH_BITS, 24); // Enable

    if (invisible) {
         glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    }

#ifdef _GLESMODE
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#else
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif

    m_hWindow = glfwCreateWindow(width, height, title, NULL, NULL);

    if (!m_hWindow) {
        glfwTerminate();
        std::cout << "WindowGLFW: Failed create window" << std::endl;
        throw std::exception();
    }

    glfwMakeContextCurrent(m_hWindow);

#ifdef USE_GLEW
    // dwRenderer requires glewExperimental
    // because it calls glGenVertexArrays()
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        glfwDestroyWindow(m_hWindow);
        glfwTerminate();
        std::cout << "WindowGLFW: Failed to init GLEW: " << glewGetErrorString(err) << std::endl;
        throw std::exception();
    }
    glGetError(); // clears error on init
#endif

    // No vsync
    glfwSwapInterval(0);

    glfwSetInputMode(m_hWindow, GLFW_STICKY_KEYS, GL_FALSE);

    //Callbacks
    glfwSetWindowUserPointer(m_hWindow, this);
    glfwSetKeyCallback(m_hWindow, [](GLFWwindow *win, int key, int scancode, int action, int mods) {
        WindowGLFW *window = reinterpret_cast<WindowGLFW *>(glfwGetWindowUserPointer(win));
        window->onKeyCallback(key, scancode, action, mods);
    });
    glfwSetMouseButtonCallback(m_hWindow, [](GLFWwindow *win, int button, int action, int mods) {
        WindowGLFW *window = reinterpret_cast<WindowGLFW *>(glfwGetWindowUserPointer(win));
        window->onMouseButtonCallback(button, action, mods);
    });
    glfwSetCursorPosCallback(m_hWindow, [](GLFWwindow *win, double x, double y) {
        WindowGLFW *window = reinterpret_cast<WindowGLFW *>(glfwGetWindowUserPointer(win));
        window->onMouseMoveCallback(x, y);
    });
    glfwSetScrollCallback(m_hWindow, [](GLFWwindow *win, double dx, double dy) {
        WindowGLFW *window = reinterpret_cast<WindowGLFW *>(glfwGetWindowUserPointer(win));
        window->onMouseWheelCallback(dx, dy);
    });
    glfwSetFramebufferSizeCallback(m_hWindow, [](GLFWwindow *win, int width, int height) {
        WindowGLFW *window = reinterpret_cast<WindowGLFW *>(glfwGetWindowUserPointer(win));
        window->onResizeWindowCallback(width, height);
    });

#ifdef VIBRANTE
    m_display = glfwGetEGLDisplay();
    m_context = glfwGetEGLContext(m_hWindow);

    // Get configuration
    EGLint num_config;
    eglGetConfigs(m_display, nullptr, 0, &num_config);
    m_config.reset(new EGLConfig[num_config]);
    if(eglGetConfigs(m_display, m_config.get(), num_config, &num_config) == EGL_FALSE) {
        glfwTerminate();
        std::cout << "WindowGLFW: Failed to get configs" << std::endl;
        throw std::exception();
    }
#endif
}

// -----------------------------------------------------------------------------
WindowGLFW::~WindowGLFW(void)
{
    glfwDestroyWindow(m_hWindow);
    glfwTerminate();
}

// -----------------------------------------------------------------------------
EGLDisplay WindowGLFW::getEGLDisplay(void)
{
#ifdef VIBRANTE
    return m_display;
#else
    return 0;
#endif
}

// -----------------------------------------------------------------------------
EGLContext WindowGLFW::getEGLContext(void)
{
#ifdef VIBRANTE
    return m_context;
#else
    return 0;
#endif
}

// -----------------------------------------------------------------------------
void WindowGLFW::onKeyCallback(int key, int scancode, int action, int mods)
{
    if (!m_keyPressCallback)
        return;

    (void)scancode;

    if ((action == GLFW_PRESS || action == GLFW_REPEAT) && mods == 0)
        m_keyPressCallback(key);
}

// -----------------------------------------------------------------------------
void WindowGLFW::onMouseButtonCallback(int button, int action, int mods)
{
    (void)mods;

    double x, y;
    glfwGetCursorPos(m_hWindow, &x, &y);
    if (action == GLFW_PRESS) {
        if (!m_mouseDownCallback)
            return;
        m_mouseDownCallback(button, (float)x, (float)y);
    } else if (action == GLFW_RELEASE) {
        if (!m_mouseUpCallback)
            return;
        m_mouseUpCallback(button, (float)x, (float)y);
    }
}

// -----------------------------------------------------------------------------
void WindowGLFW::onMouseMoveCallback(double x, double y)
{
    if (!m_mouseMoveCallback)
        return;
    m_mouseMoveCallback((float)x, (float)y);
}

// -----------------------------------------------------------------------------
void WindowGLFW::onMouseWheelCallback(double dx, double dy)
{
    if (!m_mouseWheelCallback)
        return;
    m_mouseWheelCallback((float)dx, (float)dy);
}

// -----------------------------------------------------------------------------
void WindowGLFW::onResizeWindowCallback(int width, int height)
{
    m_width  = width;
    m_height = height;

    if (!m_resizeWindowCallback)
        return;
    m_resizeWindowCallback(width, height);
}

// -----------------------------------------------------------------------------
bool WindowGLFW::swapBuffers(void)
{
    glfwPollEvents();
    glfwSwapBuffers(m_hWindow);
    return true;
}

// -----------------------------------------------------------------------------
void WindowGLFW::resetContext()
{
}

// -----------------------------------------------------------------------------
EGLContext WindowGLFW::createSharedContext() const {
#ifdef VIBRANTE
    // -----------------------
    std::cout << "WindowGLFW: create shared EGL context" << std::endl;

    EGLint ctxAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT, EGL_FALSE,
        EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT, EGL_NO_RESET_NOTIFICATION_EXT,
        EGL_NONE, EGL_NONE};

    EGLContext shared = eglCreateContext(m_display, *m_config.get(), m_context, ctxAttribs);

    if (shared == EGL_NO_CONTEXT) {
        std::cout << "WindowGLFW: Failed to create shared EGL context " << eglGetError() << std::endl;
        throw std::exception();
    }

    EGLBoolean status = eglMakeCurrent(m_display, EGL_NO_SURFACE, EGL_NO_SURFACE, shared);
    if (status != EGL_TRUE) {
        std::cout << "WindowGLFW: Failed to make shared EGL context current: " << eglGetError() << std::endl;
        throw std::exception();
    }
    return shared;
#else
    return 0;
#endif
}

// -----------------------------------------------------------------------------
bool WindowGLFW::makeCurrent()
{
    // Make the window's context current
    glfwMakeContextCurrent(m_hWindow);

    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::resetCurrent()
{
    glfwMakeContextCurrent(nullptr);

    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::setWindowSize(int width, int height)
{
    // Set the window size
    glfwSetWindowSize(m_hWindow, width, height);
    return true;
}
