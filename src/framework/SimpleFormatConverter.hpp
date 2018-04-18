/* Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <dw/image/FormatConverter.h>
#include <framework/Checks.hpp>
#include <framework/GenericImage.hpp>
#include <stdexcept>

namespace dw_samples
{
namespace common
{

/**
 * A wrapper for the format converter classes. It provides a very simple interface.
 * Converting an image blocks and directly returns the converted image. The class has its own internal pool
 * of output images (size==1).
 *
 * The generic version uses dwImageGeneric types to be image type agnostic. The typed SimpleFormatConverter<T>
 * receives and returns concrete image structs.
 *
 * Usage:
 * \code
 * GenericSimpleFormatConverter converter(propsIn, propsOut, ctx);
 *
 * dwImageGeneric *inputImg = getImgFromSomewhere();
 * dwImageGeneric *outputImg = converter.convert(&inputImg);
 *
 * \endcode
 */
class GenericSimpleFormatConverter
{
public:
    GenericSimpleFormatConverter(const dwImageProperties &imagePropsIn, const dwImageProperties &imagePropsOut, dwContextHandle_t ctx)
        : m_stream(0)
        , m_srcType(imagePropsIn)
        , m_dstType(imagePropsOut)
    {
        if(imagePropsIn.type != imagePropsOut.type)
            throw std::runtime_error("Generic format converter: input and output types do not match");

        CHECK_DW_ERROR( dwImageFormatConverter_initialize(&m_converter, m_srcType.type, ctx) );

        switch(m_srcType.type)
        {
        case DW_IMAGE_CUDA:
        {
            CHECK_DW_ERROR( dwImageCUDA_create(&m_outImageCUDA, &m_dstType, DW_IMAGE_CUDA_PITCH) );
            m_outImage = GenericImage::fromDW(&m_outImageCUDA);
            break;
        }

        #ifdef USE_NVMEDIA
        case DW_IMAGE_NVMEDIA:
        {
            CHECK_DW_ERROR( dwImageNvMedia_create(&m_outImageNvMedia, &m_dstType, ctx) );
            m_outImage = GenericImage::fromDW(&m_outImageNvMedia);
            break;
        }
        #endif

        default:
            throw std::runtime_error("GenericSimpleFormatConverter: Unsupported input type");
        }
    }

    ~GenericSimpleFormatConverter()
    {
        switch(m_srcType.type)
        {
        case DW_IMAGE_CUDA:
        {
            CHECK_DW_ERROR( dwImageCUDA_destroy(&m_outImageCUDA) );
            break;
        }

        #ifdef USE_NVMEDIA
        case DW_IMAGE_NVMEDIA:
        {
            CHECK_DW_ERROR( dwImageNvMedia_destroy(&m_outImageNvMedia) );
            break;
        }
        #endif

        default:
            throw std::runtime_error("GenericSimpleFormatConverter: Unsupported input type");
        }

        CHECK_DW_ERROR( dwImageFormatConverter_release(&m_converter) );
    }

    /// Posts the input image, blocks until the output image is available, returns the output image.
    dwImageGeneric *convert(const dwImageGeneric *imgS)
    {
        switch(m_srcType.type)
        {
        case DW_IMAGE_CUDA: innerConvert(GenericImage::toDW<dwImageCUDA>(imgS)); break;

        #ifdef USE_NVMEDIA
        case DW_IMAGE_NVMEDIA: innerConvert(GenericImage::toDW<dwImageNvMedia>(imgS)); break;
        #endif

        default: throw std::runtime_error("Invalid src type");
        }

        return m_outImage;
    }

private:
    dwImageFormatConverterHandle_t m_converter;
    cudaStream_t m_stream;

    dwImageProperties m_srcType;
    dwImageProperties m_dstType;
    
    // Currently converter only supports CUDA or NvMedia
    dwImageCUDA m_outImageCUDA;
    #ifdef USE_NVMEDIA
    dwImageNvMedia m_outImageNvMedia;
    #endif

    dwImageGeneric *m_outImage; // Points to the concrete image (e.g. m_outImageCUDA)

    /////////////////////////////////////////////////////////////////////
    // Post overloads

    void innerConvert(const dwImageCUDA *img)
    {
        CHECK_DW_ERROR( dwImageFormatConverter_copyConvertCUDA(&m_outImageCUDA, img, m_converter, m_stream) );
    }

    #ifdef USE_NVMEDIA
    void innerConvert(const dwImageNvMedia *img)
    {
        CHECK_DW_ERROR( dwImageFormatConverter_copyConvertNvMedia(&m_outImageNvMedia, img, m_converter) );
    }
    #endif
};

/**
* Typed version of the GenericSimpleFormatConverter.
*
* Usage:
* \code
* SimpleFormatConverter<dwImageCUDA> converter(propsIn, propsOut, ctx);
*
* dwImageCUDA inputImg = getImgFromSomewhere();
* dwImageCUDA *outputImg = converter.convert(&inputImg);
*
* \endcode
*/
template <class ImageSource_t>
class SimpleFormatConverter
{
public:
    SimpleFormatConverter(const dwImageProperties &imagePropsIn, const dwImageProperties &imagePropsOut, dwContextHandle_t ctx)
        : m_converter(imagePropsIn, imagePropsOut, ctx)
    {}

    ImageSource_t *convert(const ImageSource_t *imgS)
    {
        return GenericImage::toDW<ImageSource_t>(m_converter.convert(GenericImage::fromDW(imgS)));
    }

private:
    GenericSimpleFormatConverter m_converter;
};

}
}
