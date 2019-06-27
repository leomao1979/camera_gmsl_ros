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

#include "SimpleCamera.hpp"

namespace dw_samples
{
namespace common
{

SimpleCamera::SimpleCamera(const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx,
                           dwCameraOutputType outputType)
    : m_ctx(ctx)
    , m_sal(sal)
    , m_converter(DW_NULL_HANDLE)
    , m_pendingFrame(nullptr)
    , m_outputType(outputType)
    , m_converterRgba(DW_NULL_HANDLE)
    , m_pendingFrameRgba(nullptr)
    , m_pendingFrameRgbaGL(nullptr)
    , m_started(false)
{

    CHECK_DW_ERROR( dwSAL_createSensor(&m_sensor, params, m_sal) );
    CHECK_DW_ERROR( dwSensorCamera_getSensorProperties(&m_cameraProperties, m_sensor) );
    CHECK_DW_ERROR( dwSensorCamera_getImageProperties(&m_imageProperties, m_outputType, m_sensor) );

    m_outputProperties = m_imageProperties;

    std::cout << "Camera image: " << m_imageProperties.width << "x" << m_imageProperties.height << std::endl;
}

SimpleCamera::SimpleCamera(const dwImageProperties &outputProperties, const dwSensorParams &params,
                           dwSALHandle_t sal, dwContextHandle_t ctx,
                           dwCameraOutputType outputType)
    : SimpleCamera(params, sal, ctx, outputType)
{
    setOutputProperties(outputProperties);
}

SimpleCamera::~SimpleCamera()
{
    if (m_converter)
        dwImage_destroy(&m_converter);

    if (m_converterRgba)
        dwImage_destroy(&m_converterRgba);

    if(m_pendingFrame)
        releaseFrame();

    if(m_sensor) {
        if (m_started)
            dwSensor_stop(m_sensor);
        dwSAL_releaseSensor(&m_sensor);
    }
}

void SimpleCamera::setOutputProperties(const dwImageProperties &outputProperties)
{
    m_outputProperties = outputProperties;
    m_outputProperties.width = m_imageProperties.width;
    m_outputProperties.height = m_imageProperties.height;

    if(m_imageProperties.type != outputProperties.type)
    {
        m_streamer.reset(new SimpleImageStreamer<>(m_imageProperties, outputProperties.type, 66666, m_ctx));
    }

    if (m_imageProperties.format != outputProperties.format)
    {
        dwImage_create(&m_converter, m_outputProperties, m_ctx);
    }
}

dwImageHandle_t SimpleCamera::readFrame()
{
    if(!m_started)
    {
        CHECK_DW_ERROR( dwSensor_start(m_sensor));
        m_started = true;
    }

    if(m_pendingFrame)
        releaseFrame();

    dwStatus status = dwSensorCamera_readFrame(&m_pendingFrame, 0, 1000000, m_sensor);

    if (status == DW_END_OF_STREAM) {
        std::cout << "Camera reached end of stream." << std::endl;
        return nullptr;
    } else if (status == DW_NOT_READY){
        while (status == DW_NOT_READY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            status = dwSensorCamera_readFrame(&m_pendingFrame, 0, 1000000, m_sensor);
        }
    } else if (status != DW_SUCCESS) {
        throw std::runtime_error("Error reading from camera");
    }

    dwImageHandle_t img;
    CHECK_DW_ERROR( dwSensorCamera_getImage(&img, m_outputType, m_pendingFrame));

    dwImageHandle_t imgOutput = img;
    if(m_streamer)
    {
        imgOutput = m_streamer->post(img);
    }

    if (m_converter)
    {
        dwImage_copyConvert(m_converter, imgOutput, m_ctx);
        imgOutput = m_converter;
    }

    // OpenGL
    if(isGLOutputEnabled())
    {
        dwImage_copyConvert(m_converterRgba, img, m_ctx);
        m_pendingFrameRgbaGL = m_streamerGL->post(m_converterRgba);
    }

    return imgOutput;
}

void SimpleCamera::releaseFrame()
{
    if(m_pendingFrame) {
        CHECK_DW_ERROR(dwSensorCamera_returnFrame(&m_pendingFrame));
        m_pendingFrame = nullptr;
    }
}

void SimpleCamera::resetCamera()
{
    CHECK_DW_ERROR(dwSensor_reset(m_sensor));
}

void SimpleCamera::enableGLOutput()
{
    dwImageProperties propsRgba = m_imageProperties;
    propsRgba.format = DW_IMAGE_FORMAT_RGBA_UINT8;

    dwImage_create(&m_converterRgba, propsRgba, m_ctx);
    m_streamerGL.reset(new SimpleImageStreamer<>(propsRgba, DW_IMAGE_GL, 60000, m_ctx));
}

//////////////////////////////////////////////////////////////////////////////////////
/// RawSimpleCamera
///

RawSimpleCamera::RawSimpleCamera(const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx, cudaStream_t stream, 
        dwCameraOutputType outputType, dwSoftISPDemosaicMethod demosaicMethod)
    :SimpleCamera(params, sal, ctx, DW_CAMERA_OUTPUT_NATIVE_RAW)
    , m_converter_final(DW_NULL_HANDLE)
{
    m_doTonemap = false;

    if (outputType == DW_CAMERA_OUTPUT_NATIVE_PROCESSED) {
        m_doTonemap = true;
    }

    // set output type so it streams to cuda after reading raw frame
    dwImageProperties cameraImageProps = SimpleCamera::m_imageProperties;
    cameraImageProps.type = DW_IMAGE_CUDA;
    SimpleCamera::setOutputProperties(cameraImageProps);

    dwSoftISPParams softISPParams;
    CHECK_DW_ERROR(dwSoftISP_initParamsFromCamera(&softISPParams, &m_cameraProperties));
    CHECK_DW_ERROR(dwSoftISP_initialize(&m_softISP, &softISPParams, ctx));

    // Initialize Raw pipeline
    CHECK_DW_ERROR(dwSoftISP_setCUDAStream(stream, m_softISP));
    CHECK_DW_ERROR(dwSoftISP_setDemosaicMethod(demosaicMethod, m_softISP));
    CHECK_DW_ERROR(dwSoftISP_getDemosaicImageProperties(&m_rawOutputProperties, m_softISP));

    // RCB image to get output from the RawPipeline
    CHECK_DW_ERROR(dwImage_create(&m_RCBImage, m_rawOutputProperties, m_ctx));
    dwImageCUDA* RCB_cuda;
    dwImage_getCUDA(&RCB_cuda, m_RCBImage);
    dwSoftISP_bindOutputDemosaic(RCB_cuda, m_softISP);

    if (m_doTonemap) {
        dwImageProperties rgbProps = m_rawOutputProperties;
        rgbProps.format = DW_IMAGE_FORMAT_RGB_UINT8_PLANAR;

        CHECK_DW_ERROR(dwImage_create(&m_RGBImage, rgbProps, m_ctx));
        dwImageCUDA* RGB_cuda;
        dwImage_getCUDA(&RGB_cuda, m_RGBImage);
        dwSoftISP_bindOutputTonemap(RGB_cuda, m_softISP);
    }
}

RawSimpleCamera::RawSimpleCamera(const dwImageFormat &outputISPFormat,
        const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx, cudaStream_t stream,
        dwCameraOutputType outputType, dwSoftISPDemosaicMethod demosaicMethod)
    : RawSimpleCamera(params, sal, ctx, stream, outputType, demosaicMethod)
{
    if (DW_IMAGE_FORMAT_RGB_UINT8_PLANAR != outputISPFormat)
    {
        if (m_RGBImage)
            CHECK_DW_ERROR(dwImage_destroy(&m_RGBImage));

        if (m_doTonemap) {
            dwImageProperties rgbProps = m_rawOutputProperties;
            rgbProps.format = outputISPFormat;

            CHECK_DW_ERROR(dwImage_create(&m_RGBImage, rgbProps, m_ctx));
            dwImageCUDA* RGB_cuda;
            dwImage_getCUDA(&RGB_cuda, m_RGBImage);
            dwSoftISP_bindOutputTonemap(RGB_cuda, m_softISP);
        }
    }
}

RawSimpleCamera::RawSimpleCamera(const dwImageProperties &outputProperties, 
        const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx, cudaStream_t stream,
        dwCameraOutputType outputType, dwSoftISPDemosaicMethod demosaicMethod)
    : RawSimpleCamera(params, sal, ctx, stream, outputType, demosaicMethod)
{
    if (m_outputProperties.format != outputProperties.format)
    {
        dwImageProperties newProperties = m_rawOutputProperties;
        newProperties.format = outputProperties.format;

        CHECK_DW_ERROR(dwImage_create(&m_converter_final, newProperties, ctx));
        m_rawOutputProperties = newProperties;
    }
}

RawSimpleCamera::~RawSimpleCamera()
{
    if (m_converter_final) {
        dwImage_destroy(&m_converter_final);
    }

    if (m_RCBImage) {
        dwImage_destroy(&m_RCBImage);
    }

    if (m_RGBImage) {
        dwImage_destroy(&m_RGBImage);
    }

    dwSoftISP_release(&m_softISP);
}

dwImageHandle_t RawSimpleCamera::readFrame()
{
    // see sample raw_pipeline for full use and explanation
    dwImageHandle_t rawImage = SimpleCamera::readFrame();

    if (rawImage == nullptr) {
        return nullptr;
    }

    int32_t processType = DW_SOFTISP_PROCESS_TYPE_DEMOSAIC;
    dwImageHandle_t output = m_RCBImage;
    if (m_doTonemap) {
        processType |= DW_SOFTISP_PROCESS_TYPE_TONEMAP;
        output = m_RGBImage;
    }

    dwImageCUDA* rawImageCUDA;
    dwImage_getCUDA(&rawImageCUDA, rawImage);
    dwSoftISP_bindInputRaw(rawImageCUDA, m_softISP);

    dwSoftISP_setProcessType(processType, m_softISP);
    CHECK_DW_ERROR(dwSoftISP_processDeviceAsync(m_softISP));

    dwImageHandle_t imgOutput = output;
    if (m_converter_final)  {
        dwImage_copyConvert(m_converter_final, imgOutput, m_ctx);
        imgOutput = m_converter_final;
    }

    return imgOutput;
}
}
}
