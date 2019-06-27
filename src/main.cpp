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
// Copyright (c) 2015-2016 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <signal.h>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <memory>

#ifdef LINUX
#include <execinfo.h>
#include <unistd.h>
#endif

#include <cstring>
#include <functional>
#include <list>
#include <iomanip>
#include <thread>

#include <chrono>
#include <mutex>
#include <condition_variable>

// SAMPLE COMMON
#include <framework/ProgramArguments.hpp>
#include <framework/Log.hpp>
#include <framework/Checks.hpp>
#include <framework/WindowGLFW.hpp>
#ifdef VIBRANTE
#include <framework/WindowEGL.hpp>
#endif

// CORE
#include <dw/core/Context.h>
#include <dw/core/Logger.h>

// RENDERER
#include <dw/renderer/Renderer.h>

// HAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/SensorSerializer.h>
#include <dw/sensors/camera/Camera.h>

// IMAGE
#include <dw/image/FormatConverter.h>
#include <dw/image/ImageStreamer.h>
#include "framework/SimpleFormatConverter.hpp"

// RCCB
#include <dw/isp/SoftISP.h>

#include <lodepng.h>
#include <libgpujpeg/gpujpeg.h>
 
#include <ros/ros.h>
#include "LMImagePublisher.hpp"

using namespace std;
using namespace dw_samples::common;
typedef std::chrono::high_resolution_clock myclock_t;
typedef std::chrono::time_point<myclock_t> timepoint_t;

static const string ROS_TOPIC_IMAGE             = "/camera/image";
static const string ROS_TOPIC_IMAGE_COMPRESSED  = "/camera/image/compressed";

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
// 1KB should be plenty for data lines from any sensor
// Actual size is returned during runtime
#define MAX_EMBED_DATA_SIZE (1024 * 1024)

static volatile bool g_run = true;
NvMediaISCEmbeddedData sensorData;

cudaStream_t g_cudaStream  = 0;
bool gTakeScreenshot = false;
uint32_t gScreenshotCount = 0;

bool g_RCCB = false;

// Program arguments
ProgramArguments g_arguments(
    {
        ProgramArguments::Option_t("camera-type", "ar0231-rccb-ssc"),
        ProgramArguments::Option_t("csi-port", "ab"),
        ProgramArguments::Option_t("interpolationDemosaic", "0"),
        ProgramArguments::Option_t("write-file", ""),
        ProgramArguments::Option_t("serializer-type", "raw"),
        ProgramArguments::Option_t("slave", "0"),
        ProgramArguments::Option_t("fifo-size", "3"),
        ProgramArguments::Option_t("ros-topic", ""),
        ProgramArguments::Option_t("node-name", ""),
        ProgramArguments::Option_t("offscreen", "false"),
    });

//------------------------------------------------------------------------------
// Method declarations
//------------------------------------------------------------------------------
int main(int argc, const char **argv);
void parseArguments(int argc, const char **argv);
void initGL(WindowBase **window, bool offscreen);
void initSdk(dwContextHandle_t *context, WindowBase *window);
void initRenderer(dwRendererHandle_t *renderer, dwContextHandle_t context, WindowBase *window);
void initSensors(dwSALHandle_t *sal, dwSensorHandle_t *camera, dwImageProperties *cameraImageProperties, dwCameraProperties* cameraProperties, dwContextHandle_t context);

void runNvMedia_pipeline(WindowBase *window, dwRendererHandle_t renderer, dwSensorHandle_t camera, dwImageProperties *rawImageProps, dwCameraProperties *cameraProps, dwContextHandle_t sdk, float32_t framerate);

void sig_int_handler(int sig);
void keyPressCallback(int key);

void initGPUJpegEncoder(struct gpujpeg_encoder **rgba2rgbConverter, const dwImageProperties& rgbaImageProperties);
void initConverter(GenericSimpleFormatConverter **rgba2rgbConverter, const dwImageProperties& rgbaImageProperties, dwContextHandle_t context);
void publish_image(LMImagePublisher *publisher, const ros::Time& stamp, struct gpujpeg_encoder* encoder, const dwImageCUDA& rgbaImage);

//------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    //SDK objects
    WindowBase *window            = nullptr;
    dwContextHandle_t sdk         = DW_NULL_HANDLE;
    dwRendererHandle_t renderer   = DW_NULL_HANDLE;
    dwSALHandle_t sal             = DW_NULL_HANDLE;
    dwSensorHandle_t cameraSensor = DW_NULL_HANDLE;

    // Set up linux signal handlers
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = sig_int_handler;
    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command

    //Init
    g_run = true;

    parseArguments(argc, argv);
    initGL(&window, g_arguments.enabled("offscreen"));
    initSdk(&sdk, window);
    initRenderer(&renderer, sdk, window);

    // create HAL and camera
    dwImageProperties rawImageProps;
    dwCameraProperties cameraProps;
    initSensors(&sal, &cameraSensor, &rawImageProps, &cameraProps, sdk);

    if (cameraProps.rawFormat == DW_CAMERA_RAW_FORMAT_RCCB || cameraProps.rawFormat == DW_CAMERA_RAW_FORMAT_CRBC ||
        cameraProps.rawFormat == DW_CAMERA_RAW_FORMAT_CBRC || cameraProps.rawFormat == DW_CAMERA_RAW_FORMAT_BCCR) {
        g_RCCB = true;
    }

    if(rawImageProps.type != DW_IMAGE_NVMEDIA)
    {
        std::cerr << "Error: Expected nvmedia image type, received "
                  << rawImageProps.type << " instead." << std::endl;
        exit(-1);
    }

    // Allocate buffer for parsed embedded data
    sensorData.top.data    = new uint8_t[MAX_EMBED_DATA_SIZE];
    sensorData.bottom.data = new uint8_t[MAX_EMBED_DATA_SIZE];
    sensorData.top.bufferSize    = MAX_EMBED_DATA_SIZE;
    sensorData.bottom.bufferSize = MAX_EMBED_DATA_SIZE;

    float32_t framerate = cameraProps.framerate;
    string nodeName = g_arguments.get("node-name");
    if (nodeName == "") {
        nodeName = "gmsl_camera_image_publisher"; 
    }
    cout << "nodeName: " << nodeName << endl;
    ros::init(argc, const_cast<char **>(argv), nodeName);
    runNvMedia_pipeline(window, renderer, cameraSensor, &rawImageProps, &cameraProps, sdk, framerate);

    dwRenderer_release(&renderer);
    // release used objects in correct order
    dwSAL_releaseSensor(&cameraSensor);
    dwSAL_release(&sal);

    dwRelease(&sdk);
    dwLogger_release();

    delete[] sensorData.top.data;
    delete[] sensorData.bottom.data;

    delete window;

    return 0;
}

//------------------------------------------------------------------------------
void parseArguments(int argc, const char **argv)
{
    if (!g_arguments.parse(argc, argv))
        exit(-1); // Exit if not all require arguments are provided

    std::cout << "Program Arguments:\n" << g_arguments.printList() << std::endl;
}

//------------------------------------------------------------------------------
void initGL(WindowBase **window, bool offscreen)
{
    if (!*window) *window = new WindowGLFW("Camera GMSL", 1280, 800, offscreen);

    (*window)->makeCurrent();
    (*window)->setOnKeypressCallback(keyPressCallback);
}

//------------------------------------------------------------------------------
void initSdk(dwContextHandle_t *context, WindowBase *window)
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams;
    memset(&sdkParams, 0, sizeof(dwContextParameters));

#ifdef VIBRANTE
    sdkParams.eglDisplay = window->getEGLDisplay();
#else
    (void)window;
#endif

    dwInitialize(context, DW_VERSION, &sdkParams);
}

//------------------------------------------------------------------------------
void initRenderer(dwRendererHandle_t *renderer, dwContextHandle_t context, WindowBase *window)
{
    dwStatus result;

    result = dwRenderer_initialize(renderer, context);
    if (result != DW_SUCCESS)
        throw std::runtime_error(std::string("Cannot init renderer: ") + dwGetStatusName(result));

    // Set some renderer defaults
    dwRect rect;
    rect.width  = window->width();
    rect.height = window->height();
    rect.x      = 0;
    rect.y      = 0;

    dwRenderer_setRect(rect, *renderer);
}

//------------------------------------------------------------------------------
void initSensors(dwSALHandle_t *sal, dwSensorHandle_t *camera, dwImageProperties *cameraImageProperties, dwCameraProperties* cameraProperties, dwContextHandle_t context)
{
    dwStatus result;

    result = dwSAL_initialize(sal, context);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot initialize SAL: " << dwGetStatusName(result) << std::endl;
        exit(1);
    }

    // create GMSL Camera interface
    dwSensorParams params;
    std::string parameterString = g_arguments.parameterString();
    parameterString             += ",output-format=raw+data";
    params.parameters           = parameterString.c_str();
    params.protocol             = "camera.gmsl";
    result = dwSAL_createSensor(camera, params, *sal);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot create driver: camera.gmsl with params: "
                  << params.parameters << std::endl
                  << "Error: " << dwGetStatusName(result) << std::endl;
        exit(1);
    }

    dwSensorCamera_getImageProperties(cameraImageProperties,
                                     DW_CAMERA_RAW_IMAGE,
                                    *camera);
    dwSensorCamera_getSensorProperties(cameraProperties, *camera);
}

//------------------------------------------------------------------------------
void runNvMedia_pipeline(WindowBase *window, dwRendererHandle_t renderer, dwSensorHandle_t camera, dwImageProperties *rawImageProps, dwCameraProperties *cameraProps, dwContextHandle_t sdk, float32_t framerate)
{
    // the raw image coming from sensor contains embedded datalines, so its resolution is higher than
    // the pixel data only
    dwImageProperties rawPxlDataOnlyImageProps = *rawImageProps;
    rawPxlDataOnlyImageProps.width = cameraProps->resolution.x;
    rawPxlDataOnlyImageProps.height = cameraProps->resolution.y;

    // set streamer from Nvmedia -> CUDA
    dwImageStreamerHandle_t nvm2cuda;
    dwImageStreamer_initialize(&nvm2cuda, &rawPxlDataOnlyImageProps, DW_IMAGE_CUDA, sdk);

    // RCCB specific variables
    dwSoftISPHandle_t pipelineRCCB;
    if (g_RCCB) {
        dwSoftISPParams softISPParams;
        dwSoftISP_initParamsFromCamera(&softISPParams, *cameraProps);
        CHECK_DW_ERROR(dwSoftISP_initialize(&pipelineRCCB, softISPParams, sdk));
        dwSoftISP_setCUDAStream(g_cudaStream, pipelineRCCB);
        if( std::stoi(g_arguments.get("interpolationDemosaic")) > 0 ) {
            dwSoftISP_setDemosaicMethod(DW_SOFT_ISP_DEMOSAIC_METHOD_INTERPOLATION, pipelineRCCB);
        }
    }

    // serializer for video output
    dwSensorSerializerHandle_t serializer;
    dwSerializerParams serializerParams;
    serializerParams.parameters = "";
    bool recordCamera = !g_arguments.get("write-file").empty();
    if (recordCamera) {
        std::string newParams = "";
        if (g_arguments.has("serializer-type")) {
            newParams +=
                std::string("format=") + std::string(g_arguments.get("serializer-type"));
        }
        newParams += std::string(",type=disk,file=") + std::string(g_arguments.get("write-file"));
        newParams += ",slave="  + g_arguments.get("slave");

        serializerParams.parameters = newParams.c_str();
        serializerParams.onData     = nullptr;

        dwSensorSerializer_initialize(&serializer, &serializerParams, camera);
        dwSensorSerializer_start(serializer);
    }

    dwStatus result = DW_FAILURE;

    g_run = g_run && dwSensor_start(camera) == DW_SUCCESS;

    // time
    auto frameDuration         = std::chrono::milliseconds((int)(1000 / framerate));
    timepoint_t lastUpdateTime = myclock_t::now();

    // Allocate output images

    // RGBA image properties, this image is the result of conversion from raw to processed
    dwImageProperties rgbaImageProperties = rawPxlDataOnlyImageProps;
    rgbaImageProperties.type = DW_IMAGE_CUDA;
    // RCCB specific variables
    dwImageProperties rcbProperties;
    dwImageCUDA rcbImage;
    if (g_RCCB) {
        dwSoftISP_getDemosaicImageProperties(&rcbProperties, pipelineRCCB);
        rcbImage.prop = rcbProperties;
        rcbImage.layout = DW_IMAGE_CUDA_PITCH;

        cudaMallocPitch(&rcbImage.dptr[0], &rcbImage.pitch[0], rcbProperties.width * dwSizeOf(rcbProperties.pxlType), rcbProperties.height * rcbProperties.planeCount);
        rcbImage.pitch[1] = rcbImage.pitch[2] = rcbImage.pitch[0];
        rcbImage.dptr[1] = reinterpret_cast<uint8_t*>(rcbImage.dptr[0]) + rcbProperties.height * rcbImage.pitch[0];
        rcbImage.dptr[2] = reinterpret_cast<uint8_t*>(rcbImage.dptr[1]) + rcbProperties.height * rcbImage.pitch[1];

        dwSoftISP_bindDemosaicOutput(&rcbImage, pipelineRCCB);
        rgbaImageProperties = rcbProperties;
    }

    cout << "Camera image with " << rgbaImageProperties.width << "x" << rgbaImageProperties.height << " at " << cameraProps->framerate << " FPS" << endl;

    //RGBA image to display
    rgbaImageProperties.pxlFormat         = DW_IMAGE_RGBA;
    rgbaImageProperties.pxlType           = DW_TYPE_UINT8;
    rgbaImageProperties.planeCount        = 1;
    dwImageCUDA rgbaImage;
    rgbaImage.prop = rgbaImageProperties,
    rgbaImage.layout = DW_IMAGE_CUDA_PITCH;

    cudaMallocPitch(&rgbaImage.dptr[0], &rgbaImage.pitch[0],
                    rgbaImageProperties.width * dwSizeOf(DW_TYPE_UINT8) * 4, rgbaImageProperties.height);

    dwSoftISP_bindTonemapOutput(&rgbaImage, pipelineRCCB);

    dwImageFormatConverterHandle_t convert2RGBA = DW_NULL_HANDLE;
    // for RCCB to format converter converts from already Debayered RCB to RGBA
    // for non-RCCB the format converter calls generic Debayering from RAW to RGBA
    dwImageFormatConverter_initialize(&convert2RGBA, DW_IMAGE_CUDA, sdk);

    // Setup the streamer depending on display size
    dwImageStreamerHandle_t cuda2gl = DW_NULL_HANDLE;
    dwImageProperties glProperties = rgbaImageProperties;

    // raw images come from the stream with embedded lines. In order to get the image one needs to shift
    // the pointer depending on the number of datalines. This image will point to the actual data, skipping
    // the lines
    dwImageCUDA realCudaImage;

    result = dwImageStreamer_initialize(&cuda2gl, &glProperties, DW_IMAGE_GL, sdk);
    if (result == DW_SUCCESS) {
        bool compress_mode = false;
        string rosTopic = g_arguments.get("ros-topic");
        if (rosTopic.find(ROS_TOPIC_IMAGE_COMPRESSED) == 0) {
            compress_mode = true; 
        } else if (rosTopic.empty()) {
            rosTopic = ROS_TOPIC_IMAGE; 
        }
        cout << "rosTopic: " << rosTopic << endl;
        LMImagePublisher *publisher = new LMImagePublisher(rosTopic, compress_mode);
        struct gpujpeg_encoder *encoder = nullptr;
        if (compress_mode) {
            initGPUJpegEncoder(&encoder, rgbaImageProperties);
        }

        GenericSimpleFormatConverter *converter = nullptr;
        initConverter(&converter, rgbaImageProperties, sdk);

        while (g_run && !window->shouldClose()) {
            std::this_thread::yield();

            // run with at most 30FPS
            std::chrono::milliseconds timeSinceUpdate =
                std::chrono::duration_cast<std::chrono::milliseconds>(myclock_t::now() - lastUpdateTime);
            if (timeSinceUpdate < frameDuration)
                continue;

            lastUpdateTime = myclock_t::now();

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            dwCameraFrameHandle_t frameHandle;
            result = dwSensorCamera_readFrame(&frameHandle, 0, 1000000, camera);
            if (result == DW_END_OF_STREAM) {
                std::cout << "Camera reached end of stream" << std::endl;
                dwSensor_reset(camera);
                continue;
            }
            if (result != DW_SUCCESS) {
                std::cerr << "Cannot read frame: " << dwGetStatusName(result) << std::endl;
                continue;
            }

            dwImageCUDA *rawImageCUDA;
            dwImageNvMedia *rawImageNvMedia;

            result = dwSensorCamera_getImageNvMedia(&rawImageNvMedia, DW_CAMERA_RAW_IMAGE, frameHandle);
            if (result != DW_SUCCESS) {
                std::cerr << "Cannot get raw image: " << dwGetStatusName(result) << std::endl;
                continue;
            }
            ros::Time stamp = ros::Time::now();

            // std::cout << "Exposure Time (s): " << rawImageNvMedia->prop.meta.exposureTime << std::endl;

            if (recordCamera)
                dwSensorSerializer_serializeCameraFrameAsync(frameHandle, serializer);

            // process
            result = dwImageStreamer_postNvMedia(rawImageNvMedia, nvm2cuda);

            if( result != DW_SUCCESS ) {
                std::cout << "Cannot post Nvmedia" << std::endl;
            }

            result = dwImageStreamer_receiveCUDA(&rawImageCUDA, 10000, nvm2cuda);

            if( result != DW_SUCCESS ) {
                std::cout << "Cannot receive cuda" << std::endl;
            }

            if (g_RCCB) {
                // Raw -> RCB directly
                dwSoftISP_bindRawInput(rawImageCUDA, pipelineRCCB);
                result = dwSoftISP_processDeviceAsync(DW_SOFT_ISP_PROCESS_TYPE_DEMOSAIC | DW_SOFT_ISP_PROCESS_TYPE_TONEMAP, pipelineRCCB);

                if (result != DW_SUCCESS) {
                    std::cerr << "Cannot run rccb pipeline: " << dwGetStatusName(result) << std::endl;
                    g_run = false;
                    continue;
                }

            } else {
                result = dwImageFormatConverter_copyConvertCUDA(&rgbaImage, &realCudaImage, convert2RGBA, g_cudaStream);
            }

            if (result != DW_SUCCESS) {
                std::cerr << "Something wrong happened " << dwGetStatusName(result) << std::endl;
                g_run = false;
                continue;
            }
            // Publish RGB image to ROS
            dwImageCUDA* rgbImage = GenericImage::toDW<dwImageCUDA>(converter->convert(GenericImage::fromDW(&rgbaImage)));
            publish_image(publisher, stamp, encoder, *rgbImage);

            if (gTakeScreenshot) {
                char fname[128];
                {
                    std::vector<uint16_t> cpuData;
                    sprintf(fname, "screenshot_raw_%04d.png", gScreenshotCount);
                    cpuData.resize(realCudaImage.prop.width*realCudaImage.prop.height);
                    cudaMemcpy2D(cpuData.data(), realCudaImage.prop.width*2, realCudaImage.dptr[0], realCudaImage.pitch[0], realCudaImage.prop.width*2, realCudaImage.prop.height, cudaMemcpyDeviceToHost);
                    lodepng_encode_file(fname, reinterpret_cast<uint8_t*>(cpuData.data()), realCudaImage.prop.width, realCudaImage.prop.height, LCT_GREY, 16);

                    std::cout << "RAW SCREENSHOT TAKEN to " << fname << "\n";
                }

                {
                    std::vector<uint32_t> cpuData;
                    sprintf(fname, "screenshot_rgba_%04d.png", gScreenshotCount);
                    cpuData.resize(rgbaImage.prop.width*rgbaImage.prop.height);
                    cudaMemcpy2D(cpuData.data(), rgbaImage.prop.width*4, rgbaImage.dptr[0],
                            rgbaImage.pitch[0], rgbaImage.prop.width*4, rgbaImage.prop.height,
                            cudaMemcpyDeviceToHost);
                    lodepng_encode32_file(fname, reinterpret_cast<uint8_t*>(cpuData.data()), rgbaImage.prop.width, rgbaImage.prop.height);

                    std::cout << "RGBA SCREENSHOT TAKEN to " << fname << "\n";
                }

                gScreenshotCount++;
                gTakeScreenshot = false;
            }

            dwImageStreamer_returnReceivedCUDA(rawImageCUDA, nvm2cuda);
            dwImageStreamer_waitPostedNvMedia(&rawImageNvMedia, 10000, nvm2cuda);

            // render received texture
            dwImageGL *frameGL;
            dwImageStreamer_postCUDA(&rgbaImage, cuda2gl);
            result = dwImageStreamer_receiveGL(&frameGL, 10000, cuda2gl);
            if( result == DW_SUCCESS ) {
                dwRenderer_renderTexture(frameGL->tex, frameGL->target, renderer);

                dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
            }

            dwImageCUDA *returnedFrame;
            dwImageStreamer_waitPostedCUDA(&returnedFrame, 10000, cuda2gl);

            dwSensorCamera_returnFrame(&frameHandle);

            window->swapBuffers();
        }
        dwImageStreamer_release(&cuda2gl);
        delete converter;
        if (encoder != nullptr) {
            gpujpeg_encoder_destroy(encoder);
        }
    } else {
        std::cerr << "Cannot create CUDA -> GL streamer" << std::endl;
    }

    cudaFree(rgbaImage.dptr[0]);
    cudaFree(rcbImage.dptr[0]);

    dwImageFormatConverter_release(&convert2RGBA);
    dwSensor_stop(camera);

    if (recordCamera) {
        dwSensorSerializer_stop(serializer);
        dwSensorSerializer_release(&serializer);
    }

    if (g_RCCB) {
        dwSoftISP_release(&pipelineRCCB);
    }

    dwImageStreamer_release(&nvm2cuda);
}

//------------------------------------------------------------------------------
void sig_int_handler(int sig)
{
    (void)sig;

    g_run = false;
}

//------------------------------------------------------------------------------
void keyPressCallback(int key)
{
    // stop application
    if (key == GLFW_KEY_ESCAPE)
        g_run = false;

    if (key == GLFW_KEY_S)
        gTakeScreenshot = true;
}

//-----------------------------------------------------------------------------
void initGPUJpegEncoder(struct gpujpeg_encoder **encoder, const dwImageProperties& rgbaImageProperties) {
    // init jpeg encoder
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);  // quality:75, restart int:8, interleaved:0
    param.quality = 80;
    param.restart_interval = 8;
    param.interleaved = 1;  

    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = rgbaImageProperties.width;
    param_image.height = rgbaImageProperties.height;
    param_image.comp_count = 3;
    // (for now, it must be 3)
    param_image.color_space = GPUJPEG_RGB;
    param_image.sampling_factor = GPUJPEG_4_4_4;
    *encoder = gpujpeg_encoder_create(&param, &param_image);

    gpujpeg_print_devices_info();
    int gpu_device_id = 0;
    int retcode = gpujpeg_init_device(gpu_device_id, 0);
    if (retcode != 0) {
        std::cout << "Failed to init device. Ret code: " << retcode << std::endl;
        return;
    }
}

//-----------------------------------------------------------------------------
void initConverter(GenericSimpleFormatConverter **rgba2rgbConverter, const dwImageProperties& rgbaImageProperties, dwContextHandle_t context) {
    dwImageProperties rgbImageProperties = rgbaImageProperties;
    rgbImageProperties.pxlFormat = DW_IMAGE_RGB;

    *rgba2rgbConverter = new GenericSimpleFormatConverter(rgbaImageProperties, rgbImageProperties, context);
}

//-----------------------------------------------------------------------------
void publish_image(LMImagePublisher *publisher, const ros::Time& stamp, struct gpujpeg_encoder* encoder, const dwImageCUDA& rgbImage) {
    if (encoder == nullptr) {
        std::vector<uint8_t> cpuData;
        cpuData.resize(rgbImage.prop.width * rgbImage.prop.height * 3);
        cudaMemcpy2D(cpuData.data(), rgbImage.prop.width*3, rgbImage.dptr[0], rgbImage.pitch[0], rgbImage.prop.width*3, rgbImage.prop.height, cudaMemcpyDeviceToHost);

        publisher->publish_image(cpuData.data(), stamp, rgbImage.prop.width, rgbImage.prop.height);
    } else {
        // Compress with lodepng
        // std::vector<uint8_t> cpuData;
        // cpuData.resize(rgbImage.prop.width * rgbImage.prop.height * 3);
        // cudaMemcpy2D(cpuData.data(), rgbImage.prop.width*3, rgbImage.dptr[0], rgbImage.pitch[0], rgbImage.prop.width*3, rgbImage.prop.height, cudaMemcpyDeviceToHost);
        // unsigned char* compressed_image = NULL;
        // size_t compressed_image_size = 0;
        // lodepng_encode24(&compressed_image, &compressed_image_size, cpuData.data(), rgbImage.prop.width, rgbImage.prop.height);

        // Compress with libgpujpeg
        timepoint_t t0 = myclock_t::now();
        uint8_t* compressed_image = nullptr;
        int compressed_image_size = 0;
        struct gpujpeg_encoder_input encoder_input;
        gpujpeg_encoder_input_set_image(&encoder_input, reinterpret_cast<uint8_t*>(rgbImage.dptr[0]));
        int retcode = gpujpeg_encoder_encode(encoder, &encoder_input, &compressed_image, &compressed_image_size, true);
        if (retcode != 0) {
            cerr << "Failed to encode. Error code: " << retcode << endl;
            return;
        }
        std::chrono::milliseconds encoding_time = std::chrono::duration_cast<std::chrono::milliseconds>(myclock_t::now() - t0);
        cout << "Image size: " << rgbImage.prop.width * rgbImage.prop.height * 3 
             << "; Compressed size: " << compressed_image_size 
             << "; Encoding time: " << std::to_string(encoding_time.count()) << "ms" << endl;

        publisher->publish_compressed_image(compressed_image, stamp, "jpeg", compressed_image_size);

        // free(compressed_image);
    }
}
