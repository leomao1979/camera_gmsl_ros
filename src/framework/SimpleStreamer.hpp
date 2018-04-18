/* Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef SAMPLES_COMMON_SIMPLESTREAMER_HPP__
#define SAMPLES_COMMON_SIMPLESTREAMER_HPP__

#include <dw/image/ImageStreamer.h>
#include <framework/Checks.hpp>
#include <framework/GenericImage.hpp>

namespace dw_samples
{
namespace common
{

/**
 * A wrapper for the streamer classes. It sacrifices performance to provide a very simple interface.
 * Posting an image blocks and directly returns the image.
 *
 * Usage:
 * \code
 * SimpleImageStreamer streamer(propsIn, DW_IMAGE_GL, 66000, ctx);
 *
 * dwImageCUDA inputImg = getImgFromSomewhere();
 *
 * dwImageGL *outputImg = streamer.post(&inputImg);
 * ...do GL stuff...
 * streamer.release();
 *
 * \endcode
 *
 * NOTE: we strongly encourage to use the real dwImageStreamer, please see the samples in Image for a
 *       complete tutorial
 */
class GenericSimpleImageStreamer
{
public:
    GenericSimpleImageStreamer(const dwImageProperties &imageProps, dwImageType typeOut, dwTime_t timeout, dwContextHandle_t ctx)
        : m_timeout(timeout)
        , m_srcType(imageProps.type)
        , m_dstType(typeOut)
        , m_pendingReturn(nullptr)
    {
        CHECK_DW_ERROR( dwImageStreamer_initialize(&m_streamer, &imageProps, typeOut, ctx) );
    }

    ~GenericSimpleImageStreamer()
    {
#ifndef DW_USE_NVMEDIA
        if (m_pendingReturn)
            release();
#endif
        CHECK_DW_ERROR( dwImageStreamer_release(&m_streamer) );
    }

    /// Posts the input image, blocks until the output image is available, returns the output image.
    dwImageGeneric *post(const dwImageGeneric *imgS_)
    {
        if (m_pendingReturn)
            release();

        // Note: our public interface should accept const
        dwImageGeneric *imgS(const_cast<dwImageGeneric*>(imgS_));

        switch(m_srcType)
        {
        case DW_IMAGE_CPU:  innerPost(GenericImage::toDW<dwImageCPU>(imgS)); break;
        case DW_IMAGE_CUDA: innerPost(GenericImage::toDW<dwImageCUDA>(imgS)); break;
        case DW_IMAGE_GL:   innerPost(GenericImage::toDW<dwImageGL>(imgS)); break;
        #ifdef DW_USE_NVMEDIA
        case DW_IMAGE_NVMEDIA: innerPost(GenericImage::toDW<dwImageNvMedia>(imgS)); break;
        #endif
        default: throw std::runtime_error("Invalid src type");
        }

        switch(m_dstType)
        {
        case DW_IMAGE_CPU: {
            dwImageCPU *imgD_;
            innerReceive(&imgD_);
            m_pendingReturn = GenericImage::fromDW(imgD_);
            break;
        }
        case DW_IMAGE_CUDA: {
            dwImageCUDA *imgD_;
            innerReceive(&imgD_);
            m_pendingReturn = GenericImage::fromDW(imgD_);
            break;
        }
        case DW_IMAGE_GL: {
            dwImageGL *imgD_;
            innerReceive(&imgD_);
            m_pendingReturn = GenericImage::fromDW(imgD_);
            break;
        }
        #ifdef VIBRANTE
        case DW_IMAGE_NVMEDIA: {
            dwImageNvMedia *imgD_;
            innerReceive(&imgD_);
            m_pendingReturn = GenericImage::fromDW(imgD_);
            break;
        }
        #endif
        default: throw std::runtime_error("Invalid src type");
        }

        if (!m_pendingReturn)
            throw std::runtime_error("Cannot receive image");

        return m_pendingReturn;
    }

    /// Returns the previously received image to the real streamer.
    /// This method is optional. Either post() or the destructor will also return the image.
    dwImageGeneric *release()
    {
        if (!m_pendingReturn)
            throw std::runtime_error("Nothing to release");

        switch(m_dstType)
        {
        case DW_IMAGE_CPU: innerReturnReceived(GenericImage::toDW<dwImageCPU>(m_pendingReturn)); break;
        case DW_IMAGE_CUDA: innerReturnReceived(GenericImage::toDW<dwImageCUDA>(m_pendingReturn)); break;
        case DW_IMAGE_GL: innerReturnReceived(GenericImage::toDW<dwImageGL>(m_pendingReturn)); break;
        #ifdef VIBRANTE
        case DW_IMAGE_NVMEDIA: innerReturnReceived(GenericImage::toDW<dwImageNvMedia>(m_pendingReturn)); break;
        #endif
        default: throw std::runtime_error("Invalid dst type");
        }

        m_pendingReturn = nullptr;

        dwImageGeneric *imgSS;
        switch(m_srcType)
        {
        case DW_IMAGE_CPU: {
            dwImageCPU *imgS;
            innerWaitPosted(&imgS);
            imgSS = GenericImage::fromDW(imgS);
            break;
        }
        case DW_IMAGE_CUDA: {
            dwImageCUDA *imgS;
            innerWaitPosted(&imgS);
            imgSS = GenericImage::fromDW(imgS);
            break;
        }
        case DW_IMAGE_GL: {
            dwImageGL *imgS;
            innerWaitPosted(&imgS);
            imgSS = GenericImage::fromDW(imgS);
            break;
        }
        #ifdef VIBRANTE
        case DW_IMAGE_NVMEDIA: {
            dwImageNvMedia *imgS;
            innerWaitPosted(&imgS);
            imgSS = GenericImage::fromDW(imgS);
            break;
        }
        #endif
        default: throw std::runtime_error("Invalid src type");
        }

        return imgSS;
    }

private:
    dwImageStreamerHandle_t m_streamer;
    dwTime_t m_timeout;

    dwImageType m_srcType;
    dwImageType m_dstType;
    
    dwImageGeneric *m_pendingReturn;

    /////////////////////////////////////////////////////////////////////
    // Pose overloads

    void innerPost(dwImageCPU *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_postCPU(img, m_streamer) );
    }

    void innerPost(dwImageCUDA *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_postCUDA(img, m_streamer) );
    }

    void innerPost(dwImageGL *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_postGL(img, m_streamer) );
    }

#ifdef DW_USE_NVMEDIA
    void innerPost(dwImageNvMedia *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_postNvMedia(img, m_streamer) );
    }
#endif

    /////////////////////////////////////////////////////////////////////
    // Return received overloads

    void innerReturnReceived(dwImageCPU *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_returnReceivedCPU(img, m_streamer) );
    }

    void innerReturnReceived(dwImageCUDA *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_returnReceivedCUDA(img, m_streamer) );
    }

    void innerReturnReceived(dwImageGL *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_returnReceivedGL(img, m_streamer) );
    }

#ifdef DW_USE_NVMEDIA
    void innerReturnReceived(dwImageNvMedia *img)
    {
        CHECK_DW_ERROR( dwImageStreamer_returnReceivedNvMedia(img, m_streamer) );
    }
#endif

    /////////////////////////////////////////////////////////////////////
    // Receive overloads

    void innerReceive(dwImageCPU **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_receiveCPU(img, m_timeout, m_streamer) );
    }

    void innerReceive(dwImageCUDA **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_receiveCUDA(img, m_timeout, m_streamer) );
    }

    void innerReceive(dwImageGL **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_receiveGL(img, m_timeout, m_streamer) );
    }

#ifdef DW_USE_NVMEDIA
    void innerReceive(dwImageNvMedia **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_receiveNvMedia(img, m_timeout, m_streamer) );
    }
#endif

    /////////////////////////////////////////////////////////////////////
    // WaitReceived overloads

    void innerWaitPosted(dwImageCPU **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_waitPostedCPU(img, m_timeout, m_streamer) );
    }

    void innerWaitPosted(dwImageCUDA **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_waitPostedCUDA(img, m_timeout, m_streamer) );
    }

    void innerWaitPosted( dwImageGL **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_waitPostedGL(img, m_timeout, m_streamer) );
    }

#ifdef DW_USE_NVMEDIA
    void innerWaitPosted(dwImageNvMedia **img)
    {
        CHECK_DW_ERROR( dwImageStreamer_waitPostedNvMedia(img, m_timeout, m_streamer) );
    }
#endif
};

template <class ImageSource_t, class ImageDestination_t>
class SimpleImageStreamer
{
public:
    SimpleImageStreamer(const dwImageProperties &imageProps, dwTime_t timeout, dwContextHandle_t ctx)
        : m_streamer(imageProps, ImageTypeInfo<ImageDestination_t>::value, timeout, ctx)
    {}

    ImageDestination_t *post(const ImageSource_t *imgS)
    {
        return GenericImage::toDW<ImageDestination_t>(m_streamer.post(GenericImage::fromDW(imgS)));
    }

    ImageSource_t *release()
    {
        return GenericImage::toDW<ImageSource_t>(m_streamer.release());
    }

private:
    GenericSimpleImageStreamer m_streamer;
};

}
}

#endif
