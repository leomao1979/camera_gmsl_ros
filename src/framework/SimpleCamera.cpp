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
    , m_pendingFrame(nullptr)
    , m_outputType(outputType)
    , m_pendingFrameRgba(nullptr)
    , m_pendingFrameRgbaGL(nullptr)
{

    CHECK_DW_ERROR( dwSAL_createSensor(&m_sensor, params, m_sal) );
    CHECK_DW_ERROR( dwSensorCamera_getSensorProperties(&m_cameraProperties, m_sensor) );
    CHECK_DW_ERROR( dwSensorCamera_getImageProperties(&m_imageProperties, m_outputType, m_sensor) );

    m_outputProperties = m_imageProperties;

    std::cout << "Camera image: " << m_imageProperties.width << "x" << m_imageProperties.height << std::endl;

    CHECK_DW_ERROR( dwSensor_start(m_sensor));
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
    if(m_pendingFrame)
        releaseFrame();

    if(m_sensor) {
        dwSensor_stop(m_sensor);
        dwSAL_releaseSensor(&m_sensor);
    }
}

void SimpleCamera::setOutputProperties(const dwImageProperties &outputProperties)
{
    m_outputProperties = outputProperties;
    m_outputProperties.width = m_imageProperties.width;
    m_outputProperties.height = m_imageProperties.height;

    dwImageProperties newProperties = m_imageProperties;
    if(m_imageProperties.type != outputProperties.type)
    {
        m_streamer.reset(new GenericSimpleImageStreamer(m_imageProperties, outputProperties.type, 66666, m_ctx));
        newProperties.type = m_outputProperties.type;
    }

    if (((m_imageProperties.pxlFormat != outputProperties.pxlFormat) ||
         (m_imageProperties.pxlType != outputProperties.pxlType) ||
         (m_imageProperties.planeCount != outputProperties.planeCount)))
    {
        m_converter.reset(new GenericSimpleFormatConverter(newProperties, m_outputProperties, m_ctx));
    }
}

dwImageGeneric *SimpleCamera::readFrame()
{
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

    dwImageGeneric *img;
    switch(m_imageProperties.type)
    {
    case DW_IMAGE_CPU: {
        dwImageCPU *img_;
        CHECK_DW_ERROR( dwSensorCamera_getImageCPU(&img_, m_outputType, m_pendingFrame));
        img = GenericImage::fromDW(img_);
        break;
    }
    case DW_IMAGE_CUDA: {
        dwImageCUDA *img_;
        CHECK_DW_ERROR( dwSensorCamera_getImageCUDA(&img_, m_outputType, m_pendingFrame));
        img = GenericImage::fromDW(img_);
        break;
    }
    #ifdef VIBRANTE
    case DW_IMAGE_NVMEDIA: {
        dwImageNvMedia *img_;
        CHECK_DW_ERROR( dwSensorCamera_getImageNvMedia(&img_, m_outputType, m_pendingFrame));
        img = GenericImage::fromDW(img_);
        break;
    }
    #endif
    default:
        throw std::runtime_error("Invalid image type");
    }

    dwImageGeneric *imgOutput = img;
    if(m_streamer)
    {
        imgOutput = m_streamer->post(img);
    }

    if (m_converter)
    {
        imgOutput = m_converter->convert(imgOutput);
    }

    // OpenGL
    if(isGLOutputEnabled())
    {
        m_pendingFrameRgba = m_converterRgba->convert(img);
        m_pendingFrameRgbaGL = GenericImage::toDW<dwImageGL>(m_streamerGL->post(m_pendingFrameRgba));
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
    propsRgba.pxlFormat = DW_IMAGE_RGBA;
    propsRgba.pxlType = DW_TYPE_UINT8;
    propsRgba.planeCount = 1;

    m_converterRgba.reset(new GenericSimpleFormatConverter(m_imageProperties, propsRgba, m_ctx));
    m_streamerGL.reset(new GenericSimpleImageStreamer(propsRgba, DW_IMAGE_GL, 60000, m_ctx));
}

//////////////////////////////////////////////////////////////////////////////////////
/// RawSimpleCamera
///

RawSimpleCamera::RawSimpleCamera(const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx, cudaStream_t stream, dwCameraOutputType outputType)
    :SimpleCamera(params, sal, ctx, DW_CAMERA_RAW_IMAGE)
{
    m_doTonemap = false;

    if (outputType == DW_CAMERA_PROCESSED_IMAGE) {
        m_doTonemap = true;
    }

    // set output type so it streams to cuda after reading raw frame
    dwImageProperties cameraImageProps = SimpleCamera::m_imageProperties;
    cameraImageProps.type = DW_IMAGE_CUDA;
    SimpleCamera::setOutputProperties(cameraImageProps);

    dwSoftISPParams softISPParams;
    CHECK_DW_ERROR(dwSoftISP_initParamsFromCamera(&softISPParams, SimpleCamera::m_cameraProperties));
    CHECK_DW_ERROR(dwSoftISP_initialize(&m_softISP, softISPParams, ctx));

    // Initialize Raw pipeline
    CHECK_DW_ERROR(dwSoftISP_setCUDAStream(stream, m_softISP));

    CHECK_DW_ERROR(dwSoftISP_getDemosaicImageProperties(&m_rawOutputProperties, m_softISP));

    // RCB image to get output from the RawPipeline
    dwImageCUDA_create(&m_RCBImage, &m_rawOutputProperties, DW_IMAGE_CUDA_PITCH);
    dwSoftISP_bindDemosaicOutput(&m_RCBImage, m_softISP);

    if (m_doTonemap) {
        dwImageProperties rgbProps = m_rawOutputProperties;
        rgbProps.pxlFormat = DW_IMAGE_RGB;
        rgbProps.pxlType = DW_TYPE_UINT8;

        dwImageCUDA_create(&m_RGBImage, &rgbProps, DW_IMAGE_CUDA_PITCH);
        dwSoftISP_bindTonemapOutput(&m_RGBImage, m_softISP);
    }
}

RawSimpleCamera::~RawSimpleCamera()
{
    dwSoftISP_release(&m_softISP);
}

dwImageGeneric *RawSimpleCamera::readFrame()
{
    // see sample raw_pipeline for full use and explanation
    dwImageCUDA* rawImageCUDA = GenericImage::toDW<dwImageCUDA>(SimpleCamera::readFrame());

    if (rawImageCUDA == nullptr) {
        return nullptr;
    }

    int32_t processType = DW_SOFT_ISP_PROCESS_TYPE_DEMOSAIC;
    dwImageCUDA* output = &m_RCBImage;
    if (m_doTonemap) {
        processType |= DW_SOFT_ISP_PROCESS_TYPE_TONEMAP;
        output = &m_RGBImage;
    }

    dwSoftISP_bindRawInput(rawImageCUDA, m_softISP);
    CHECK_DW_ERROR(dwSoftISP_processDeviceAsync(processType, m_softISP));

    return GenericImage::fromDW<dwImageCUDA>(output);
}
}
}
