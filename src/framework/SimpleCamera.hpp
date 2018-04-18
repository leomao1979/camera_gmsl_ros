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

#ifndef COMMON_SIMPLECAMERA_HPP__
#define COMMON_SIMPLECAMERA_HPP__

// Driveworks
#include <dw/core/Context.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/image/Image.h>
#include <dw/image/FormatConverter.h>
#include <dw/isp/SoftISP.h>

// C++ Std
#include <memory>
#include <vector>
#include <type_traits>
#include <chrono>
#include <thread>

// Common
#include <framework/Checks.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SimpleFormatConverter.hpp>

namespace dw_samples
{
namespace common
{

//-------------------------------------------------------------------------------
/**
* Simple class to get images from a camera. It supports streaming and converting (once, in that order)
* so the returned image is in the expected format. It returns the generic dwImageGeneric type that points
* to an underlying concrete dwImageXXX struct. The real type matches the type requested in the output properties.
*
* Usage:
* \code
* SimpleCamera camera(propsOut, sensorParams, sal, ctx);
*
* for(dwImageGeneric *img = camera.readFrame(); img!=nullptr; img=camera.readFrame())
* {
*   // Do things with img
* }
*
* \endcode
*
* NOTE: for tutorials and best practices about Camera sensors, please see sensors/camera samples
*/
class SimpleCamera
{
  public:
    /// creates a simple camera that outputs a frame with the properties of the camera image
    SimpleCamera(const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx,
                 dwCameraOutputType outputType = DW_CAMERA_PROCESSED_IMAGE);
    /**
     * creates a simple camera and also sets up image streamer and format converter to output a
     * converted image, with properties different from the properties of the camera image
    **/
    SimpleCamera(const dwImageProperties &outputProperties, const dwSensorParams &params, dwSALHandle_t sal,
                 dwContextHandle_t ctx, dwCameraOutputType outputType = DW_CAMERA_PROCESSED_IMAGE);

    virtual ~SimpleCamera();

    /**
     * sets up streamer and converter to be used when acquiring a frame if outputProperties are different
     * from input properties
    **/
    void setOutputProperties(const dwImageProperties &outputProperties);

    const dwCameraProperties &getCameraProperties() const { return m_cameraProperties; }
    const dwImageProperties &getImageProperties() const { return m_imageProperties; }
    virtual const dwImageProperties &getOutputProperties() const {return m_outputProperties;}

    virtual dwImageGeneric *readFrame();

    template <class T>
    T *readFrameTyped()
    {
        return GenericImage::toDW<T>(readFrame());
    }

    /// Releases the frame returned by readFrame. Calling this is optional.
    void releaseFrame();

    void resetCamera();

    /// Enables conversion and streaming directly to GL for each frame read
    /// After this is enabled, getFrameGL() will return the GL image for the last read frame.
    void enableGLOutput();

    bool isGLOutputEnabled() const {return m_streamerGL.get() != nullptr;}

    /// Returns the frame converted to RGBA format of the same type as the input image
    /// Only valid when GL output has been enabled
    dwImageGeneric *getFrameRgba() const {return m_pendingFrameRgba;}

    /// Returns the frame converted to RGBA format as a GL frame
    /// Only valid when GL output has been enabled
    dwImageGL *getFrameRgbaGL() const {return m_pendingFrameRgbaGL;}

protected:
    dwContextHandle_t m_ctx;
    dwSALHandle_t m_sal;

    dwSensorHandle_t m_sensor;

    dwCameraProperties m_cameraProperties;
    dwImageProperties m_imageProperties;
    dwImageProperties m_outputProperties;

    std::unique_ptr<GenericSimpleImageStreamer> m_streamer;
    std::unique_ptr<GenericSimpleFormatConverter> m_converter;

    dwCameraFrameHandle_t m_pendingFrame;

    dwCameraOutputType m_outputType;

    std::unique_ptr<GenericSimpleFormatConverter> m_converterRgba;
    std::unique_ptr<GenericSimpleImageStreamer> m_streamerGL;
    dwImageGeneric *m_pendingFrameRgba;
    dwImageGL *m_pendingFrameRgbaGL;
};

/**
 * Extention of the SimpleCamera that reads a RAW image and applies the raw pipeline. Calling readFrame will
 * a RCB/RCC frame.
 *
 * NOTE for tutorial and details about raw pipeline and raw cameras, see sample_camera_gmsl_raw and sample_raw_pipeline
 */
class RawSimpleCamera : public SimpleCamera
{
public:
    RawSimpleCamera(const dwSensorParams &params, dwSALHandle_t sal, dwContextHandle_t ctx, cudaStream_t stream, dwCameraOutputType outputType);
    ~RawSimpleCamera();

    dwImageGeneric *readFrame() override final;

    virtual const dwImageProperties &getOutputProperties() const override final {return m_rawOutputProperties;}
private :
    dwSoftISPHandle_t m_softISP;
    dwImageCUDA m_RCBImage, m_RGBImage;

    dwImageProperties m_rawOutputProperties;

    bool m_doTonemap;
};
}
}

#endif
