/* Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#pragma once

#include <dw/image/Image.h>
#include <cstddef>
#include <type_traits>
#include <stdexcept>

namespace dw_samples
{
namespace common
{

/**
* Helper class that can translate an image type into the corresponding enum value.
* It is specialized for other types (dwImageCUDA, dwImageCPU, etc.)
*/
template<class T>
struct ImageTypeInfo;

#define LinkImageTypeInfo(Type, EnumValue)         \
    template<>                                     \
    struct ImageTypeInfo<Type>                     \
    {                                              \
     static const dwImageType value = EnumValue;    \
    };

LinkImageTypeInfo(dwImageCPU, DW_IMAGE_CPU);
LinkImageTypeInfo(dwImageCUDA, DW_IMAGE_CUDA);
LinkImageTypeInfo(dwImageGL, DW_IMAGE_GL);
#ifdef VIBRANTE
    LinkImageTypeInfo(dwImageNvMedia, DW_IMAGE_NVMEDIA);
#endif

/////////////////////////////////////////////////////////////
/// Generic image
/// This opaque type and other methdos allow treating all
/// DW images as a single type
struct dwImageGeneric; //Opaque

namespace GenericImage
{
    template<class TImg>
    dwImageGeneric *fromDW(TImg *img);

    template<class TImg>
    dwImageGeneric const *fromDW(TImg const *img);

    dwImageProperties *getProperties(dwImageGeneric *img);
    dwImageProperties const *getProperties(dwImageGeneric const *img);

    dwTime_t getTimestamp(dwImageGeneric *img);

    template <class TImg>
    TImg *toDW(dwImageGeneric *img);

    template <class TImg>
    TImg const *toDW(dwImageGeneric const *img);
}


}
}

///////////////////////////////////////////////
/// Implementation

namespace dw_samples
{
namespace common
{
namespace GenericImage
{

namespace detail
{
    /// See: http://stackoverflow.com/questions/8866194/getting-the-type-of-a-member
    template <class T, class M> M get_member_type(M T:: *);
}

template<class TImg>
dwImageGeneric *fromDW(TImg *img)
{
    static_assert(std::is_same<dwImageProperties, decltype(detail::get_member_type(&TImg::prop))>::value, "Prop must be at the beginning of the struct");
    static_assert(offsetof(TImg, prop)==0, "Prop must be at the beginning of the struct");
    return reinterpret_cast<dwImageGeneric *>(img);
}

template<class TImg>
dwImageGeneric const *fromDW(TImg const *img)
{
    static_assert(std::is_same<dwImageProperties, decltype(detail::get_member_type(&TImg::prop))>::value, "Prop must be at the beginning of the struct");
    static_assert(offsetof(TImg, prop)==0, "Prop must be at the beginning of the struct");
    return reinterpret_cast<dwImageGeneric const*>(img);
}


template<class TImg>
TImg *toDW(dwImageGeneric *img)
{
    if(!img)
        return nullptr;

    if(getProperties(img)->type != ImageTypeInfo<TImg>::value)
        throw std::runtime_error("Image type doesn't match");
    return reinterpret_cast<TImg*>(img);
}

template<class TImg>
TImg const *toDW(dwImageGeneric const*img)
{
    if(!img)
        return nullptr;

    if(getProperties(img)->type != ImageTypeInfo<TImg>::value)
        throw std::runtime_error("Image type doesn't match");
    return reinterpret_cast<TImg const*>(img);
}

}
}
}
