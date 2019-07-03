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

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>

// HAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/SensorSerializer.h>
#include <dw/sensors/camera/Camera.h>

// Image
#include <dw/image/ImageStreamer.h>

// ISP
#include <dw/isp/SoftISP.h>

// Renderer
#include <dw/renderer/Renderer.h>

// Sample Includes
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/DataPath.hpp>
#include <framework/WindowGLFW.hpp>

#include <ros/ros.h>
#include "LMImagePublisher.hpp"

using namespace std;
using namespace dw_samples::common;

static const string ROS_TOPIC_IMAGE = "/camera/image";
static const string ROS_TOPIC_IMAGE_COMPRESSED = "/camera/image/compressed";

///------------------------------------------------------------------------------
///------------------------------------------------------------------------------
class CameraGMSLRawSample : public DriveWorksSample
{
private:

    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_sdk                  = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                      = DW_NULL_HANDLE;
    dwRendererHandle_t m_renderer            = DW_NULL_HANDLE;

    std::unique_ptr<ScreenshotHelper> m_screenshot;

    LMImagePublisher *m_imagePublisher = nullptr;

public:

    dwSoftISPHandle_t m_isp = DW_NULL_HANDLE;
    dwSensorHandle_t m_camera = DW_NULL_HANDLE;
    dwImageProperties m_cameraImageProperties;
    dwCameraProperties m_cameraProperties;
    dwImageStreamerHandle_t m_streamerCUDAtoGL = DW_NULL_HANDLE;
    dwSensorSerializerHandle_t m_serializer;

    dwImageHandle_t m_rcbImage = DW_NULL_HANDLE;
    dwImageCUDA* m_rcbCUDAImage;

    dwImageHandle_t m_rgbaImage = DW_NULL_HANDLE;
    dwImageCUDA* m_rgbaCUDAImage;
    dwImageHandle_t m_rgbImage  = DW_NULL_HANDLE;

    uint32_t m_ispOutput;

    bool m_recordCamera = false;

    /// -----------------------------
    /// Initialize application
    /// -----------------------------
    CameraGMSLRawSample(const ProgramArguments& args)
        : DriveWorksSample(args)
    {
    }

    void onProcess() override
    {}

    /// -----------------------------
    /// Initialize Renderer, Sensors, and Image Streamers, Egomotion
    /// -----------------------------
    bool onInitialize() override
    {
        // -----------------------------------------
        // Initialize DriveWorks SDK context and SAL
        // -----------------------------------------
        {
            // initialize logger to print verbose message on console in color
            dwLogger_initialize(getConsoleLoggerCallback(true));
            dwLogger_setLogLevel(DW_LOG_VERBOSE);

            // initialize SDK context, using data folder
            dwContextParameters sdkParams = {};

            #ifdef VIBRANTE
            sdkParams.eglDisplay = getEGLDisplay();
            #endif

            CHECK_DW_ERROR_MSG(dwInitialize(&m_sdk, DW_VERSION, &sdkParams),
                               "Cannot initialize Drive-Works SDK Context");
        }

        //------------------------------------------------------------------------------
        // initializes rendering subpart
        // - the rendering module
        // - the render buffers
        // - projection and modelview matrices
        // - renderer settings
        // -----------------------------------------
        {
            CHECK_DW_ERROR( dwRenderer_initialize(&m_renderer, m_sdk) );

            dwRect rect;
            rect.width  = getWindowWidth();
            rect.height = getWindowHeight();
            rect.x      = 0;
            rect.y      = 0;
            dwRenderer_setRect(rect, m_renderer);
        }

        //------------------------------------------------------------------------------
        // initializes camera
        // - the SensorCamera module
        // -----------------------------------------
        {

            m_ispOutput = DW_SOFTISP_PROCESS_TYPE_DEMOSAIC | DW_SOFTISP_PROCESS_TYPE_TONEMAP;

            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));


            if (getArgument("camera-type").compare("ar0144-cccc-none-gazet1") == 0) {
                // ar0144 only supports direct tonemap output
                m_ispOutput = DW_SOFTISP_PROCESS_TYPE_TONEMAP;
            }

            const char* csiPortString = getArgument("camera-port").c_str();

            dwSensorParams params;
            std::string parameterString = std::string("output-format=raw+data,camera-type=") +
                    std::string(getArgument("camera-type"));
            parameterString             += std::string(",csi-port=") + csiPortString;
            parameterString             += std::string(",format=") + std::string(getArgument("serializer-type"));
            parameterString             += std::string(",fifo-size=") + std::string(getArgument("camera-fifo-size"));
            parameterString             += std::string(",slave=") + std::string(getArgument("tegra-slave"));
            params.parameters           = parameterString.c_str();
            params.protocol             = "camera.gmsl";

            CHECK_DW_ERROR(dwSAL_createSensor(&m_camera, params, m_sal));

            // sensor can take some time to start, it's possible to call the read function and check if the return status is ok
            // before proceding
            CHECK_DW_ERROR(dwSensor_start(m_camera));

            dwCameraFrameHandle_t frame;
            dwStatus status = DW_NOT_READY;
            do {
                status = dwSensorCamera_readFrame(&frame, 0, 66000, m_camera);
            } while (status == DW_NOT_READY);

            // something wrong happened, aborting
            if (status != DW_SUCCESS) {
                throw std::runtime_error("Cameras did not start correctly");
            }

            CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frame));

            CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&m_cameraProperties, m_camera));
            log("Successfully initialized camera with resolution of %dx%d at framerate of %f FPS\n",
                m_cameraProperties.resolution.x, m_cameraProperties.resolution.y, m_cameraProperties.framerate);

            m_screenshot.reset(new ScreenshotHelper(m_sdk, m_sal, getWindowWidth(), getWindowHeight(), "CameraGMSL_Raw"));
        }

        //------------------------------------------------------------------------------
        // initializes software ISP for processing RAW RCCB images
        // - the SensorCamera module
        // -----------------------------------------
        {
            dwSoftISPParams softISPParams;
            CHECK_DW_ERROR(dwSoftISP_initParamsFromCamera(&softISPParams, &m_cameraProperties));
            CHECK_DW_ERROR(dwSoftISP_initialize(&m_isp, &softISPParams, m_sdk));

            if ((m_ispOutput & DW_SOFTISP_PROCESS_TYPE_DEMOSAIC) &&
                std::stoi(getArgument("interpolationDemosaic")) > 0) {
                dwSoftISP_setDemosaicMethod(DW_SOFTISP_DEMOSAIC_METHOD_INTERPOLATION, m_isp);
            }
        }

        //------------------------------------------------------------------------------
        // initializes camera
        // - the SensorCamera module
        // -----------------------------------------
        {
            // we need to allocate memory for a demosaic image and bind it to the ISP
            dwImageProperties rcbProperties{};
            if (m_ispOutput & DW_SOFTISP_PROCESS_TYPE_DEMOSAIC) {
                // getting the properties directly from the ISP
                CHECK_DW_ERROR(dwSoftISP_getDemosaicImageProperties(&rcbProperties, m_isp));
                CHECK_DW_ERROR(dwImage_create(&m_rcbImage, rcbProperties, m_sdk));
                CHECK_DW_ERROR(dwImage_getCUDA(&m_rcbCUDAImage, m_rcbImage));
                // bind the image as the output for demosaic process to the ISP, will be filled at the call of
                // dwSoftISP_processDeviceAsync
                CHECK_DW_ERROR(dwSoftISP_bindOutputDemosaic(m_rcbCUDAImage, m_isp));
            }

            // in order ot visualize we prepare the properties of the tonemapped image
            dwImageProperties rgbaImageProperties{};
            rgbaImageProperties.format = DW_IMAGE_FORMAT_RGBA_UINT8;
            rgbaImageProperties.type = DW_IMAGE_CUDA;

            if (m_ispOutput & DW_SOFTISP_PROCESS_TYPE_DEMOSAIC) {
                rgbaImageProperties.width = rcbProperties.width;
                rgbaImageProperties.height = rcbProperties.height;
            } else {
                // In case no demosaic operation is performed, assume
                // the width/height of the raw image.
                dwImageProperties rawImageProperties{};
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(
                            &rawImageProperties, DW_CAMERA_OUTPUT_CUDA_RAW_UINT16, m_camera));

                rgbaImageProperties.width = rawImageProperties.width;
                rgbaImageProperties.height = rawImageProperties.height;
            }

            // allocate the rgba image
            CHECK_DW_ERROR(dwImage_create(&m_rgbaImage, rgbaImageProperties, m_sdk));
            CHECK_DW_ERROR(dwImage_getCUDA(&m_rgbaCUDAImage, m_rgbaImage));
            CHECK_DW_ERROR(dwSoftISP_bindOutputTonemap(m_rgbaCUDAImage, m_isp));

            // alloate the rgb image
            dwImageProperties rgbImageProperties = rgbaImageProperties;
            rgbImageProperties.format = DW_IMAGE_FORMAT_RGB_UINT8;
            CHECK_DW_ERROR(dwImage_create(&m_rgbImage, rgbImageProperties, m_sdk));

            CHECK_DW_ERROR(dwImageStreamer_initialize(&m_streamerCUDAtoGL, &rgbaImageProperties, DW_IMAGE_GL, m_sdk));
        }

        //------------------------------------------------------------------------------
        // initializes serializer
        // -----------------------------------------
        {
            m_recordCamera = !getArgument("write-file").empty();

            if (m_recordCamera) {
                dwSerializerParams serializerParams;
                serializerParams.parameters = "";
                std::string newParams = "";
                newParams += std::string("format=") + std::string(getArgument("serializer-type"));
                newParams += std::string(",type=disk,file=") + std::string(getArgument("write-file"));

                serializerParams.parameters = newParams.c_str();
                serializerParams.onData     = nullptr;

                CHECK_DW_ERROR(dwSensorSerializer_initialize(&m_serializer, &serializerParams, m_camera));
                CHECK_DW_ERROR(dwSensorSerializer_start(m_serializer));
            }
        }

        //--------------------------------------------------------------------------
        // initializes ROS publisher
        // -------------------------------------------------------------------------
        {
            bool compress_mode = false;
            string rosTopic = getArgument("ros-topic");
            if (rosTopic.find(ROS_TOPIC_IMAGE_COMPRESSED) == 0) {
                compress_mode = true; 
            }
            m_imagePublisher = new LMImagePublisher(rosTopic, compress_mode);
        }

        return true;
    }

    ///------------------------------------------------------------------------------
    /// Free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {

        if (m_rcbImage) {
            dwImage_destroy(&m_rcbImage);
        }

        if (m_rgbaImage) {
            dwImage_destroy(&m_rgbaImage);
        }

        if (m_isp) {
            dwSoftISP_release(&m_isp);
        }

        if (m_streamerCUDAtoGL) {
            dwImageStreamer_release(&m_streamerCUDAtoGL);
        }

        if (m_camera) {
            dwSensor_stop(m_camera);
            dwSAL_releaseSensor(&m_camera);
        }

        if (m_renderer)
        {
            dwRenderer_release(&m_renderer);
        }

        dwSAL_release(&m_sal);
        dwRelease(&m_sdk);
        dwLogger_release();
    }


    ///------------------------------------------------------------------------------
    /// Main processing of the sample (combined processing and renering for more clarity)
    ///     - read from camera
    ///     - get an image with a useful format
    ///     - use softISP to convert from raw and tonemap
    ///     - render
    ///------------------------------------------------------------------------------
    void onRender() override
    {
        dwTime_t timeout = 66000;

        // read from camera
        ros::Time stamp = ros::Time::now();
        uint32_t cameraSiblingID = 0;
        dwCameraFrameHandle_t frame;
        CHECK_DW_ERROR(dwSensorCamera_readFrame(&frame, cameraSiblingID, timeout, m_camera));

        // get an image with the desired output format
        dwImageHandle_t frameCUDA;
        CHECK_DW_ERROR(dwSensorCamera_getImage(&frameCUDA, DW_CAMERA_OUTPUT_CUDA_RAW_UINT16, frame));

        if (m_recordCamera) {
            dwStatus status = dwSensorSerializer_serializeCameraFrameAsync(frame, m_serializer);
            if (status == DW_BUFFER_FULL)
            {
                logError("SensorSerializer failed to serialize data, aborting.");
                stop();
            }
            else
            {
                CHECK_DW_ERROR(status);
            }
        }

        // raw images need to be processed through the softISP
        dwImageCUDA* rawImageCUDA;
        CHECK_DW_ERROR(dwImage_getCUDA(&rawImageCUDA, frameCUDA));
        CHECK_DW_ERROR(dwSoftISP_bindInputRaw(rawImageCUDA, m_isp));
        // request the softISP to perform a demosaic and a tonemap. This is for edmonstration purposes, the demosaic
        // output will not be used in this sample, only the tonemap output
        CHECK_DW_ERROR(dwSoftISP_setProcessType(m_ispOutput,
                                                    m_isp));
        CHECK_DW_ERROR(dwSoftISP_processDeviceAsync(m_isp));

        // Publish RGB image to ROS
        CHECK_DW_ERROR(dwImage_copyConvert(m_rgbImage, m_rgbaImage, m_sdk));
        dwImageCUDA* rgbImageCUDA;
        CHECK_DW_ERROR(dwImage_getCUDA(&rgbImageCUDA, m_rgbImage));
        publish_image(*rgbImageCUDA, stamp);

        // stream that tonamap image to the GL domain
        CHECK_DW_ERROR(dwImageStreamer_producerSend(m_rgbaImage, m_streamerCUDAtoGL));

        // receive the streamed image as a handle
        dwImageHandle_t frameGL;
        CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&frameGL, timeout, m_streamerCUDAtoGL));

        // get the specific image struct to be able to access texture ID and target
        dwImageGL* imageGL;
        CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

        // render received texture
        CHECK_DW_ERROR(dwRenderer_renderTexture(imageGL->tex, imageGL->target, m_renderer));

        // returned the consumed image
        CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&frameGL, m_streamerCUDAtoGL));

        // notify the producer that the work is done
        CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, m_streamerCUDAtoGL));

        // return frame
        CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frame));
    }

    void publish_image(dwImageCUDA& rgbImageCUDA, ros::Time& stamp) {
        std::vector<uint8_t> cpuData;
        cpuData.resize(rgbImageCUDA.prop.width * rgbImageCUDA.prop.height * 3);
        m_imagePublisher->publish_image(cpuData.data(), stamp, rgbImageCUDA.prop.width, rgbImageCUDA.prop.height);
    }

    void onKeyDown(int key, int scancode, int mods) override
    {
        (void)scancode;
        (void)mods;

        if (key == GLFW_KEY_S) {
            m_screenshot->takeScreenshot();
        }
    }
};


//------------------------------------------------------------------------------
int main(int argc, const char **argv)
{

    ProgramArguments args(argc, argv,
    {
        ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)\n"),

        ProgramArguments::Option_t("camera-port", "a", "Camera CSI port [default 0]\n"
                              "a - port AB on px2, A on ddpx\n"
                              "c - port CD on px2, C on ddpx\n"
                              "e - port EF on px2, E on ddpx\n"
                              "g - G on ddpx only\n"
                              ),

        ProgramArguments::Option_t("interpolationDemosaic", "0", "activates softISP interpolation at full resolution"),
        ProgramArguments::Option_t("serializer-type", "raw", "Serialization type for raw images, either raw or lraw"),
        ProgramArguments::Option_t("write-file", "", "If this string is not empty, then the serializer will record in this location\n"),
        ProgramArguments::Option_t("tegra-slave", "0", "Optional parameter used only for Tegra B, enables slave mode."),
        ProgramArguments::Option_t("camera-fifo-size", "3", "Size of the internal camera fifo (minimum 3). "
                              "A larger value might be required during recording due to slowdown"),
        ProgramArguments::Option_t("ros-topic", ""),
        ProgramArguments::Option_t("node-name", ""),
        ProgramArguments::Option_t("offscreen", "false"),

    }, "DriveWorks camera GMSL Raw sample");

    string nodeName = args.get("nodeName");
    if (nodeName == "") {
        nodeName = "gmsl_camera_image_publisher";     
    }
    cout << "nodeName: " << nodeName << endl;
    ros::init(argc, const_cast<char **>(argv), nodeName);

    // -------------------
    // initialize and start a window application (with offscreen support if required)
    CameraGMSLRawSample app(args);

    app.initializeWindow("Camera GMSL Raw sample", 1280, 800, args.enabled("offscreen"));

    return app.run();
}

