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
// Copyright (c) 2015-2018 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <thread>
#include <queue>
#include <string>
#include <chrono>
#include <signal.h>

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>
#include <dw/core/NvMedia.h>

// HAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/SensorSerializer.h>
#include <dw/sensors/camera/Camera.h>

// Image
#include <dw/image/ImageStreamer.h>

// Renderer
#include <dw/renderer/RenderEngine.h>

// Sample Includes
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/WindowGLFW.hpp>
#include <framework/MathUtils.hpp>

#include <ros/ros.h>
#include <libgpujpeg/gpujpeg.h>
#include "ROSImagePublisher.hpp"

using namespace std;
using namespace dw_samples::common;

#define MAX_PORTS_COUNT 4

///------------------------------------------------------------------------------
///------------------------------------------------------------------------------
class CameraMultiGMSLSample : public DriveWorksSample
{
private:

    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_sdk                  = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                      = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine    = DW_NULL_HANDLE;
    uint32_t m_tileVideo[MAX_PORTS_COUNT * 4];

    uint32_t m_activeCamerasPerPort[4] = {0};
    std::unique_ptr<ScreenshotHelper> m_screenshot;

    struct gpujpeg_encoder *m_jpegEncoder = nullptr;
    struct gpujpeg_parameters m_gpujpeg_param;
    struct gpujpeg_image_parameters m_gpujpeg_param_image;

    ROSImagePublisher *m_imagePublisher[MAX_PORTS_COUNT * 4] = {nullptr};
    ros::NodeHandle& m_node;
    // which camera is connected to which port (for displaying the name on screen)
    const char* cameraToPort[MAX_PORTS_COUNT];
public:

    const char* portIDNames[MAX_PORTS_COUNT] = {"a", "c", "e", "g"};

    dwRenderEngineColorRGBA m_colorPerPort[MAX_PORTS_COUNT];

    dwSensorHandle_t m_camera[MAX_PORTS_COUNT] = {DW_NULL_HANDLE};
    dwImageProperties m_cameraImageProperties[MAX_PORTS_COUNT];
    dwCameraProperties m_cameraProperties[MAX_PORTS_COUNT];
    dwImageStreamerHandle_t m_streamerCUDAtoGL[MAX_PORTS_COUNT] = {DW_NULL_HANDLE};

    dwRectf m_renderRanges[MAX_PORTS_COUNT * 4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    dwRenderEngineParams params{};

    // holds a copy converted image from native to NvMedia rgba
    dwImageHandle_t m_rgbaFrame[MAX_PORTS_COUNT] = {DW_NULL_HANDLE};
    dwImageHandle_t m_rgbFrame[MAX_PORTS_COUNT]  = {DW_NULL_HANDLE};

    /// -----------------------------
    /// Initialize application
    /// -----------------------------
    CameraMultiGMSLSample(const ProgramArguments& args, ros::NodeHandle& node) : DriveWorksSample(args), m_node(node) {}

    void initializeCameras()
    {
        uint32_t totalCameras = 0;

        std::string selectorMask = getArgument("selector-mask");
        uint32_t portID = 0;

        // go through the mask and count 1, per CSI port
        for (uint32_t i = 0; i < selectorMask.length() && i < 16; i++) {
            const char s = selectorMask[i];
            if (s == '1') {
                portID = i / 4;
                m_activeCamerasPerPort[portID]++;
                totalCameras++;
            }
        }

        // create and initialize all cameras
        for (portID = 0; portID < MAX_PORTS_COUNT; ++portID) {
            if (m_activeCamerasPerPort[portID] > 0) {
                std::string parameterString;
                std::string cameraName = getArgument((std::string("type-") + portIDNames[portID]).c_str());

                parameterString += std::string("csi-port=") + portIDNames[portID];
                parameterString += ",camera-type=" + cameraName;
                parameterString += ",camera-count=4"; // when using the mask, just ask for all cameras, mask will select properly

                if (selectorMask.size() >= portID*4) {
                    parameterString += ",camera-mask="+ selectorMask.substr(portID*4, std::min(selectorMask.size() - portID*4, size_t{4}));
                }

                parameterString += ",slave="  + getArgument("tegra-slave");

                dwSensorParams params;
                params.parameters = parameterString.c_str();
                params.protocol = "camera.gmsl";
                CHECK_DW_ERROR(dwSAL_createSensor(&m_camera[portID], params, m_sal));

                CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&m_cameraProperties[portID], m_camera[portID]));
                log("Successfully initialized %d cameras of type %s in port %s with resolution of %dx%d at framerate of %f FPS\n",
                    m_activeCamerasPerPort[portID], cameraName.c_str(), portIDNames[portID],
                    m_cameraProperties[portID].resolution.x, m_cameraProperties[portID].resolution.y, m_cameraProperties[portID].framerate);

                cameraToPort[portID] = cameraName.c_str();
            }
        }

        // start all cameras together
        for (portID = 0; portID < MAX_PORTS_COUNT; ++portID) {
            if (m_activeCamerasPerPort[portID] > 0) {
                CHECK_DW_ERROR(dwSensor_start(m_camera[portID]));
                log("Successfully started cameras in port %s\n", portIDNames[portID]);
            }
        }
    }

    /// -----------------------------
    /// Initialize Renderer, Sensors, and Image Streamers, Egomotion
    /// -----------------------------
    bool onInitialize() override
    {
        // -----------------------------------------
        // Initialize DriveWorks context and SAL
        // -----------------------------------------
        {
            // initialize logger to print verbose message on console in color
            dwLogger_initialize(getConsoleLoggerCallback(true));
            dwLogger_setLogLevel(DW_LOG_VERBOSE);

            // initialize SDK context, using data folder
            dwContextParameters sdkParams = {};
            sdkParams.dataPath = DataPath::get_cstr();

            #ifdef VIBRANTE
            sdkParams.eglDisplay = getEGLDisplay();
            #endif

            CHECK_DW_ERROR(dwInitialize(&m_sdk, DW_VERSION, &sdkParams));
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));
        }

        //------------------------------------------------------------------------------
        // initializes camera
        // - the SensorCamera module
        // -----------------------------------------
        {
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));
            initializeCameras();
        }


        // -----------------------------
        // Initialize Renderer
        // -----------------------------
        {
            m_colorPerPort[0] = {1,0,0,1};
            m_colorPerPort[1] = {0,1,0,1};
            m_colorPerPort[2] = {0,0,1,1};
            m_colorPerPort[3] = {0,0,0,1};

            uint32_t totalCameras = 0;
            for (uint32_t i = 0; i < MAX_PORTS_COUNT; ++i) {
                    totalCameras += m_activeCamerasPerPort[i];
            }

            log("Total cameras %d\n", totalCameras);

            CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
            params.defaultTile.lineWidth = 2.0f;
            params.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_24;
            params.maxBufferCount = 1;

            float32_t windowSize[2] = {static_cast<float32_t>(getWindowWidth()), static_cast<float32_t>(getWindowHeight())};
            params.bounds     = {0, 0};

            uint32_t tilesPerRow = 1;
            params.bounds.width = windowSize[0];
            params.bounds.height = windowSize[1];
            switch (totalCameras) {
            case 1 :
                tilesPerRow = 1;
                break;
            case 2 :
                params.bounds.height = (windowSize[1] / 2);
                params.bounds.y = (windowSize[1] / 2);
                tilesPerRow = 2;
                break;
            case 3 :
                tilesPerRow = 2;
                break;
            case 4 :
                tilesPerRow = 2;
                break;
            default :
                tilesPerRow = 4;
                break;
            }

            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_sdk));

            dwRenderEngineTileState paramList[MAX_PORTS_COUNT * 4];
            for (uint32_t i = 0; i < totalCameras; ++i) {
                dwRenderEngine_initTileState(&paramList[i]);
                paramList[i].modelViewMatrix = DW_IDENTITY_TRANSFORMATION;
                paramList[i].font = DW_RENDER_ENGINE_FONT_VERDANA_24;
            }

            dwRenderEngine_addTilesByCount(m_tileVideo, totalCameras, tilesPerRow, paramList, m_renderEngine);
        }

        //------------------------------------------------------------------------------
        // initializes streamer
        // -----------------------------------------
        for (uint32_t i = 0; i < MAX_PORTS_COUNT; ++i){
            if (m_activeCamerasPerPort[i] == 0) continue;

            dwImageProperties rgbaImageProperties{};
            dwSensorCamera_getImageProperties(&rgbaImageProperties, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, m_camera[i]);
            rgbaImageProperties.format = DW_IMAGE_FORMAT_RGBA_UINT8;
            // Leo added
            rgbaImageProperties.type   = DW_IMAGE_CUDA;

            // create an image to hold the conversion from native to rgba, fit for streaming to gl
            CHECK_DW_ERROR(dwImage_create(&m_rgbaFrame[i], rgbaImageProperties, m_sdk));
            CHECK_DW_ERROR(dwImageStreamer_initialize(&m_streamerCUDAtoGL[i], &rgbaImageProperties, DW_IMAGE_GL, m_sdk));
            // create an image to hold the conversion from RGBA to RGB for ROS publish
            dwImageProperties rgbImageProperties = rgbaImageProperties;
            rgbImageProperties.format = DW_IMAGE_FORMAT_RGB_UINT8;
            CHECK_DW_ERROR(dwImage_create(&m_rgbFrame[i], rgbImageProperties, m_sdk));
        }

        //--------------------------------------------------------------------------
        // initializes ROS publishers
        // -------------------------------------------------------------------------
        std::string selectorMask = getArgument("selector-mask");
        for (uint32_t i = 0; i < selectorMask.length() && i < 16; i++) {
            const char s = selectorMask[i];
            if (s == '1') {
                string rosTopicName = "ros-topic-" + i;
                m_imagePublisher[i] = new ROSImagePublisher(m_node, getArgument(rosTopicName.c_str()), enabled("compressed"));
            }
        }

        m_screenshot.reset(new ScreenshotHelper(m_sdk, m_sal, getWindowWidth(), getWindowHeight(), "CameraGMSL_Multi"));
        return true;
    }

    ///------------------------------------------------------------------------------
    /// Free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {

        if (m_renderEngine) {
            CHECK_DW_ERROR(dwRenderEngine_release(&m_renderEngine));
        }

        // stop all sensors
        for (uint32_t i = 0; i < MAX_PORTS_COUNT; ++i) {
            if (m_camera[i]) {
                if (m_activeCamerasPerPort[i] > 0) {
                    dwSensor_stop(m_camera[i]);
                }
            }
        }

        for (uint32_t i = 0; i < MAX_PORTS_COUNT; ++i) {
            if (m_rgbFrame[i]) {
                CHECK_DW_ERROR(dwImage_destroy(&m_rgbFrame[i]));
            }
            if (m_rgbaFrame[i]) {
                CHECK_DW_ERROR(dwImage_destroy(&m_rgbaFrame[i]));
            }
            if (m_streamerCUDAtoGL[i]) {
                dwImageStreamer_release(&m_streamerCUDAtoGL[i]);
            }
            if (m_camera[i]) {
                if (m_activeCamerasPerPort[i] > 0) {
                    dwSAL_releaseSensor(&m_camera[i]);
                }
            }
        }

        // Release ROS publishers
        for (uint32_t i = 0; i < MAX_PORTS_COUNT * 4; ++i) {
            if (m_imagePublisher[i] != nullptr) {
                delete m_imagePublisher[i];
                m_imagePublisher[i] = nullptr;
            }
        }

        dwSAL_release(&m_sal);
        dwRelease(&m_sdk);
        dwLogger_release();
    }

    void onProcess() override
    {}

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        {
            dwRenderEngine_reset(m_renderEngine);
            dwRectf rect;
            rect.width  = width;
            rect.height = height;
            rect.x      = 0;
            rect.y      = 0;
            dwRenderEngine_setBounds(rect, m_renderEngine);
        }

        log("window resized to %dx%d\n", width, height);
    }

    void onKeyDown(int key, int scancode, int mods) override
    {
        (void)scancode;
        (void)mods;

        if (key == GLFW_KEY_S) {
            m_screenshot->takeScreenshot();
        }
    }

    ///------------------------------------------------------------------------------
    /// Main processing of the sample
    ///     - collect sensor data
    ///     - push data to egomotion
    ///     - update egomotion filter in certain interval
    ///     - extract latest filter state
    ///     - integrate relative poses to an absolute one
    ///------------------------------------------------------------------------------
    void onRender() override
    {
        dwTime_t timeout = 66000;

        for (uint32_t csiPort = 0; csiPort < MAX_PORTS_COUNT; ++csiPort) {
            if (m_activeCamerasPerPort[csiPort] == 0) continue;

            uint32_t tileIndex = 0;
            for (uint32_t t = 0; t < csiPort; ++t) {
                tileIndex += m_activeCamerasPerPort[t];
            }

            for (uint32_t cameraSiblingID = 0; cameraSiblingID < m_activeCamerasPerPort[csiPort]; ++cameraSiblingID){
                ros::Time stamp = ros::Time::now();
                dwCameraFrameHandle_t frame;

                // read from camera will update the internal active frame of the camera
                // those frames are images with NATIVE properties that depend on the type and sensor properties set at creation
                dwStatus status = DW_NOT_READY;
                do {
                    status = dwSensorCamera_readFrame(&frame, cameraSiblingID, timeout, m_camera[csiPort]);
                } while(status == DW_NOT_READY || status == DW_TIME_OUT);

                if (status != DW_SUCCESS) {
                    throw std::runtime_error("Camera error");
                }

                CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[cameraSiblingID + tileIndex], m_renderEngine));
                CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
                // get an image with the desired output format
/**
                dwImageHandle_t frameNvMedia;
                // see sample_camera_gmsl for how to directly grab a CUDA processed image from Camera. Because of a limitation
                // on the PX2 DGPU, streaming CUDA->GL for rendering has performance issues, so for this sample
                // we grab native processed (with properties of NvMedia YUV420 planar), convert it to RGBA and
                // stream it to GL.
                CHECK_DW_ERROR(dwSensorCamera_getImage(&frameNvMedia, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, frame));

                // convert native (yuv420 planar nvmedia) to rgba nvmedia
                CHECK_DW_ERROR(dwImage_copyConvert(m_rgbaFrame[csiPort], frameNvMedia, m_sdk));
*/
                // dwImageHandle_t frameCUDA;
                CHECK_DW_ERROR(dwSensorCamera_getImage(&m_rgbaFrame[csiPort], DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
                // Publish RGB image to ROS
                CHECK_DW_ERROR(dwImage_copyConvert(m_rgbFrame[csiPort], m_rgbaFrame[csiPort], m_sdk));
                dwImageCUDA* rgbImageCUDA;
                CHECK_DW_ERROR(dwImage_getCUDA(&rgbImageCUDA, m_rgbFrame[csiPort]));
                uint32_t cameraIndex = csiPort * 4 + cameraSiblingID;
                publish_image(m_imagePublisher[cameraIndex], *rgbImageCUDA, stamp);

                // stream that image to the GL domain
                CHECK_DW_ERROR(dwImageStreamer_producerSend(m_rgbaFrame[csiPort], m_streamerCUDAtoGL[csiPort]));

                // receive the streamed image as a handle
                dwImageHandle_t frameGL;
                CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&frameGL, timeout, m_streamerCUDAtoGL[csiPort]));

                // get the specific image struct to be able to access texture ID and target
                dwImageGL* imageGL;
                CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

                // render received texture
                {
                    dwVector2f range{};
                    range.x = imageGL->prop.width;
                    range.y = imageGL->prop.height;
                    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
                    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0,0,range.x,range.y}, m_renderEngine));
                    dwRenderEngine_setColor(m_colorPerPort[csiPort], m_renderEngine);

                    std::string tileString = std::string(portIDNames[csiPort]) + std::string("-") +
                            std::to_string(cameraSiblingID) + std::string(", ") +
                            std::string(cameraToPort[csiPort]);

                    CHECK_DW_ERROR(dwRenderEngine_renderText2D(tileString.c_str(), {25, range.y - 25}, m_renderEngine));
                }

                // returned the consumed image
                CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&frameGL, m_streamerCUDAtoGL[csiPort]));

                // notify the producer that the work is done
                CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, m_streamerCUDAtoGL[csiPort]));

                // return frame
                CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frame));
            }
        }
    }

    void publish_image(ROSImagePublisher *imagePublisher, dwImageCUDA& rgbImageCUDA, ros::Time& stamp) {
        if (imagePublisher == nullptr) return;
        std::vector<uint8_t> cpuData;
        cpuData.resize(rgbImageCUDA.prop.width * rgbImageCUDA.prop.height * 3);
        cudaMemcpy2D(cpuData.data(), rgbImageCUDA.prop.width*3, rgbImageCUDA.dptr[0], rgbImageCUDA.pitch[0], rgbImageCUDA.prop.width*3, rgbImageCUDA.prop.height, cudaMemcpyDeviceToHost);
        imagePublisher->publish_image(cpuData.data(), stamp, rgbImageCUDA.prop.width, rgbImageCUDA.prop.height);
    } 

};

//------------------------------------------------------------------------------
int main(int argc, const char *argv[])
{

    ProgramArguments args(argc, argv,
    {
        ProgramArguments::Option_t("type-a", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)\n"),
        ProgramArguments::Option_t("type-c", "ar0231-rccb-bae-sf3324", "camera gmsl type\n"),
        ProgramArguments::Option_t("type-e", "ar0231-rccb-bae-sf3324", "camera gmsl type\n"),
        ProgramArguments::Option_t("type-g", "ar0231-rccb-bae-sf3324", "camera gmsl type\n"),

        ProgramArguments::Option_t("selector-mask", "000100010001", "Mask for camera selection [default one camera for each port]:\n"
                              "otherwise 0/1 based on camera number and port, ordered as 3210-3210-3210 (AB-CD-EF). Note that "
                              "the port is serial so camera N can be activated only of camera N-1 is active\n"
                              ),

        ProgramArguments::Option_t("tegra-slave", "0", "Optional parameter used only for Tegra B, enables slave mode.\n"),
        ProgramArguments::Option_t("offscreen",  "false"),
        ProgramArguments::Option_t("compressed", "false"),
        ProgramArguments::Option_t("ros-topic-0",  "/camera/image"),
        ProgramArguments::Option_t("ros-topic-1",  "/camera/image/1"),
        ProgramArguments::Option_t("ros-topic-2",  "/camera/image/2"),
        ProgramArguments::Option_t("ros-topic-3",  "/camera/image/3"),
        ProgramArguments::Option_t("ros-topic-4",  "/camera/front_image"),
        ProgramArguments::Option_t("ros-topic-5",  "/camera/image/5"),
        ProgramArguments::Option_t("ros-topic-6",  "/camera/image/6"),
        ProgramArguments::Option_t("ros-topic-7",  "/camera/image/7"),
        ProgramArguments::Option_t("ros-topic-8",  "/camera/side_image"),
        ProgramArguments::Option_t("ros-topic-9",  "/camera/image/9"),
        ProgramArguments::Option_t("ros-topic-10",  "/camera/image/10"),
        ProgramArguments::Option_t("ros-topic-11",  "/camera/image/11"),

    }, "DriveWorks camera GMSL sample");

    cout << "selector-mask: " << args.get("selector-mask") << endl;
    ros::init(argc, const_cast<char **>(argv), "camera_multiple_gmsl", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    // -------------------
    // initialize and start a window application (with offscreen support if required)
    CameraMultiGMSLSample app(args, nh);

    app.initializeWindow("Camera GMSL sample", 1200, 800, args.enabled("offscreen"));

    return app.run();
}
