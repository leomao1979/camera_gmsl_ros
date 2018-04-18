/* Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "GenericImage.hpp"

namespace dw_samples
{
namespace common
{

namespace GenericImage
{
    dwImageProperties *getProperties(dwImageGeneric *img)
    {
        return reinterpret_cast<dwImageProperties*>(img);
    }
    dwImageProperties const *getProperties(dwImageGeneric const *img)
    {
        return reinterpret_cast<dwImageProperties const*>(img);
    }

    dwTime_t getTimestamp(dwImageGeneric *img)
    {
        switch(getProperties(img)->type)
        {
        case DW_IMAGE_CPU: return toDW<dwImageCPU>(img)->timestamp_us;
        case DW_IMAGE_CUDA: return toDW<dwImageCUDA>(img)->timestamp_us;
        case DW_IMAGE_GL: return toDW<dwImageGL>(img)->timestamp_us;
#ifdef USE_NVMEDIA
        case DW_IMAGE_NVMEDIA: return toDW<dwImageNvMedia>(img)->timestamp_us;
#endif
        default:
            throw std::runtime_error("GenericImage: invalid image type");
        }
    }
}

}
}
