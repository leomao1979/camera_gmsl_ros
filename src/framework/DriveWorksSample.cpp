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
// Copyright (c) 2015-2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "DriveWorksSample.hpp"

#include <signal.h>

#include <framework/WindowGLFW.hpp>
#include <framework/MathUtils.hpp>
#include <framework/Log.hpp>
#include <framework/DataPath.hpp>
#include <framework/SimpleRenderer.hpp>

#ifdef VIBRANTE
#include <framework/WindowEGL.hpp>
#endif

// System includes
#include <thread>

#if (!WINDOWS)
#include <execinfo.h>
#include <unistd.h>
#include <csignal>
#endif


namespace dw_samples
{
namespace common
{

//------------------------------------------------------------------------------
DriveWorksSample* DriveWorksSample::g_instance  = nullptr;

//------------------------------------------------------------------------------
DriveWorksSample::DriveWorksSample(const ProgramArguments& args)
    : m_profiler()
    , m_args(args)
    , m_run(true)
    , m_pause(false)
    , m_playSingleFrame(false)
    , m_reset(false)
    , m_runIterationPeriod(0)
    , m_frameIdx(-1)
{
    // ----------- Singleton -----------------
    if (g_instance)
        throw std::runtime_error("Can only create one app in the process.");
    g_instance = this;

    // ----------- Signals -----------------
    struct sigaction action = {};
    action.sa_handler = DriveWorksSample::globalSigHandler;

    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command
    sigaction(SIGSTOP, &action, NULL); // kill command

    // ----------- Initialization -----------------
    cudaFree(0);
}

//------------------------------------------------------------------------------
void DriveWorksSample::initializeWindow(const char *title, int width, int height, bool offscreen)
{
    m_title     = title;
    m_width     = width;
    m_height    = height;

    // -------------------------------------------
    // Initialize GL
    // -------------------------------------------
#ifdef VIBRANTE
    if (offscreen) {
        m_window.reset(new WindowOffscreenEGL(m_width, m_height));
    }
#endif
    if (!m_window) m_window.reset(new WindowGLFW(m_title.c_str(), m_width, m_height, offscreen));

    m_window->makeCurrent();
    m_window->setOnKeypressCallback(processKeyCb);
    m_window->setOnMouseUpCallback(mouseUpCb);
    m_window->setOnMouseDownCallback(mouseDownCb);
    m_window->setOnMouseMoveCallback(mouseMoveCb);
    m_window->setOnMouseWheelCallback(mouseWheelCb);
    m_window->setOnResizeWindowCallback(resizeCb);

    glClearColor(0, 0, 0, 0);
    CHECK_GL_ERROR();
}




//------------------------------------------------------------------------------
bool DriveWorksSample::shouldRun()
{
    if (m_window) {
        return !m_window->shouldClose() && m_run;
    }

    return m_run;
}

//------------------------------------------------------------------------------
void DriveWorksSample::setProcessRate(int loopsPerSecond)
{
    m_runIterationPeriod = std::chrono::duration_cast<myclock_t::duration>(
                std::chrono::nanoseconds(static_cast<std::chrono::nanoseconds::rep>(1e9 / loopsPerSecond)));
}

//------------------------------------------------------------------------------
void DriveWorksSample::processKey(int key)
{
    // stop application
    if (key == GLFW_KEY_ESCAPE)
        m_run = false;
    else if (key == GLFW_KEY_SPACE)
        m_pause = !m_pause;
    else if (key == GLFW_KEY_F5) {
        m_playSingleFrame = !m_playSingleFrame;
        m_pause = m_playSingleFrame;
    }
    else if (key == GLFW_KEY_R)
        m_reset = true;

    onProcessKey(key);
}

//------------------------------------------------------------------------------
void DriveWorksSample::mouseDown(int button, float x, float y)
{
    m_mouseView.mouseDown(button, x, y);
    onMouseDown(button,x,y);
}

//------------------------------------------------------------------------------
void DriveWorksSample::mouseUp(int button, float x, float y)
{
    m_mouseView.mouseUp(button, x, y);
    onMouseUp(button,x,y);
}

//------------------------------------------------------------------------------
void DriveWorksSample::mouseMove(float x, float y)
{
    m_mouseView.mouseMove(x, y);
    onMouseMove(x,y);
}

//------------------------------------------------------------------------------
void DriveWorksSample::mouseWheel(float x, float y)
{
    m_mouseView.mouseWheel(x, y);
    onMouseWheel(x,y);
}

//------------------------------------------------------------------------------
void DriveWorksSample::resize(int width, int height)
{
    m_mouseView.setWindowAspect((float)width / height);
    onResizeWindow(width, height);
}

//------------------------------------------------------------------------------
EGLDisplay DriveWorksSample::getEGLDisplay() const
{
    if (m_window) return m_window->getEGLDisplay();
    return 0;
}

//------------------------------------------------------------------------------
int DriveWorksSample::getWindowWidth() const
{
    if (m_window) return m_window->width();
    return 0;
}

//------------------------------------------------------------------------------
int DriveWorksSample::getWindowHeight() const
{
    if (m_window) return m_window->height();
    return 0;
}

//------------------------------------------------------------------------------
void DriveWorksSample::tryToSleep()
{
    // This is the time that the previous iteration took
    auto timeSinceUpdate = myclock_t::now() - m_lastRunIterationTime;

    // Count FPS
    if(!m_pause)
    {
        m_fpsBuffer[m_fpsSampleIdx] = timeSinceUpdate.count();
        m_fpsSampleIdx = (m_fpsSampleIdx + 1) % FPS_BUFFER_SIZE;

        float32_t totalTime = 0;
        for(uint32_t i=0; i<FPS_BUFFER_SIZE; i++)
            totalTime += m_fpsBuffer[i];

        myclock_t::duration meanTime(static_cast<myclock_t::duration::rep>(totalTime/FPS_BUFFER_SIZE));
        m_currentFPS = 1e6f / static_cast<float32_t>(std::chrono::duration_cast<std::chrono::microseconds>(meanTime).count());
    }

    // Limit framerate, sleep if necessary
    // Sleep just before draw() so that the screen update is as regular as possible
    if (timeSinceUpdate < m_runIterationPeriod)
    {
        auto sleepDuration = m_runIterationPeriod - timeSinceUpdate;
        std::this_thread::sleep_for(sleepDuration);
    }

    // Start the iteration timer ticking here to take rendering into account
    m_lastRunIterationTime = myclock_t::now();
}

//------------------------------------------------------------------------------
int DriveWorksSample::run()
{
    if (!onInitialize()) return -1;

    // Main program loop
    m_run = true;
    m_lastRunIterationTime = myclock_t::now() - m_runIterationPeriod;

    while (shouldRun())
    {
        if (m_reset)
        {
            onReset();
            m_reset = false;
        }

        // Iteration
        if (!m_pause)
        {
            m_frameIdx++;

            onProcess();
            if (m_playSingleFrame)
                m_pause = true;
        }else{
            onPause();
        }

        // if we expect to slow down the execution of the code
        if (m_runIterationPeriod != myclock_t::duration::zero())
        {
            tryToSleep();
        }

        if (shouldRun() && !m_reset && m_window)
        {
            onRender();

            m_window->swapBuffers();
        }
    }

    // Show timings
    if (!m_profiler.empty())
    {
        m_profiler.collectTimers();

        std::stringstream ss;
        ss << "Timing results:\n" << m_profiler << "\n";
        std::cout << ss.str();
        //log(DW_LOG_VERBOSE, ss.str().c_str());
    }

    onRelease();

    return 0;
}


}
}
