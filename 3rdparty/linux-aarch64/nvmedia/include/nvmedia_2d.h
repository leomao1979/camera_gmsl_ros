/*
 * Copyright (c) 2013-2017, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/**
 * \file
 * \brief <b> NVIDIA Media Interface: 2D Processing Control </b>
 *
 * @b Description: This file contains the \ref image_2d_api "Image 2D Processing API".
 */

#ifndef _NVMEDIA_2D_H
#define _NVMEDIA_2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_common.h"
#include "nvmedia_image.h"
/**
 * \defgroup image_2d_api Image 2D Processing
 *
 * The Image 2D Processing API encompasses all NvMedia 2D image processing
 * related functionality.
 *
 * @ingroup nvmedia_image_top
 * @{
 */

/** \brief Major Version number */
#define NVMEDIA_2D_VERSION_MAJOR   2
/** \brief Minor Version number */
#define NVMEDIA_2D_VERSION_MINOR   1


/** \defgroup filter Surface Filtering */
/**
 * \brief Filtering mode used for stretched blits.
 * \ingroup filter
 */
typedef enum
{
    /** Disable the horizontal and vertical filtering */
    NVMEDIA_2D_STRETCH_FILTER_OFF = 0x1,
    /** Enable low quality filtering */
    NVMEDIA_2D_STRETCH_FILTER_LOW,
    /** Enable media quality filtering */
    NVMEDIA_2D_STRETCH_FILTER_MEDIUM,
    /** Enable the best quality filtering */
    NVMEDIA_2D_STRETCH_FILTER_HIGH
} NvMedia2DStretchFilter;

/** \brief Operation flags that affect blit behavior.
 * \ingroup blit
 **/
typedef enum
{
    /** Compute and return CRC value. */
    NVMEDIA_2D_BLIT_FLAG_RETURN_CRC             = (1u << 0)
} NvMedia2DBlitFlags;

/*---------------------------------------------------------*/
/** \defgroup blit Blits
 *
 * Blit functions define valid parameters for a blit.
 */

/**
 * \brief Bit-mask for validFields in \ref NvMedia2DBlitParameters
 * \ingroup blit
 */
typedef enum
{
    /** Enable use of stretch filter */
    NVMEDIA_2D_BLIT_PARAMS_FILTER               = (1u << 0),
    /** Enable use of blit flags */
    NVMEDIA_2D_BLIT_PARAMS_FLAGS                = (1u << 1),
    /** Enable use of destination transform */
    NVMEDIA_2D_BLIT_PARAMS_DST_TRANSFORM        = (1u << 2),
    /** Enable use of color space conversion standard */
    NVMEDIA_2D_BLIT_PARAMS_COLOR_STD            = (1u << 3)
} NvMedia2DBlitParamField;

/**
 * Struct for setting the additional parameters for a blit.
 * ValidFields is a mask which indicates which fields of the struct
 * should be read.
 *
 * \ingroup blit
 */
typedef struct
{
    /*! Valid fields in this structure. This determines which structure
        members are used. The following bit-masks can be ORed:
        \n \ref NVMEDIA_2D_BLIT_PARAMS_FILTER
        \n \ref NVMEDIA_2D_BLIT_PARAMS_FLAGS
        \n \ref NVMEDIA_2D_BLIT_PARAMS_DST_TRANSFORM
    */
    uint32_t                        validFields;
    /*! Filter mode */
    NvMedia2DStretchFilter          filter;
    /*! Flags to be used when \ref NVMEDIA_2D_BLIT_PARAMS_FLAGS is set */
    uint32_t                        flags;
    /*! Destination transformation when \ref NVMEDIA_2D_BLIT_PARAMS_DST_TRANSFORM is set. */
    NvMediaTransform                dstTransform;
    /*! Color space conversion standard when \ref NVMEDIA_2D_BLIT_PARAMS_COLOR_STD is set */
    NvMediaColorStandard            colorStandard;
} NvMedia2DBlitParameters;

/**
 * Struct for returning additional values from a blit.
 *
 * \ingroup blit
 */
typedef struct
{
    /** Returned CRC value */
    uint32_t crc;
} NvMedia2DBlitParametersOut;

/**
 * \brief Returns the version information for the NvMedia 2D library.
 * \param[in] version A pointer to a \ref NvMediaVersion structure
 *                    filled by the 2D library.
 * \return \ref NvMediaStatus The status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMedia2DGetVersion(
    NvMediaVersion *version
);

/**
 * \brief  An opaque handle representing an NvMedia2D object.
 */
typedef void NvMedia2D;

/**
 * \brief Create a 2D object.
 * \param[in] device The \ref NvMediaDevice device this 2D object will use.
 * \return \ref NvMedia2D The new 2D object's handle or NULL if unsuccessful.
 * \ingroup blit
 */
NvMedia2D *
NvMedia2DCreate(
    NvMediaDevice *device
);

/**
 * \brief Destroy a 2D object.
 * \param[in] i2d The 2D object to be destoryed.
 * \return void
 * \ingroup blit
 */
void
NvMedia2DDestroy(
    NvMedia2D *i2d
);

/**
 * Perform a 2D blit operation with supplementary return values.
 *
 * A blit transfers pixels from a source surface to a destination surface,
 * applying a variety of transformations to the pixel values on the way.
 *
 * The interface aims at making the typical uses of normal pixel copy easy,
 * by not mandating the setting of advanced blit parameters unless they are
 * actually required.
 *
 * Passing in NULL as \a params invokes a standard pixel copy blit without
 * additional transformations. If the dimensions of the source rectangle do
 * not match the dimensions of the destination rectangle, pixels are scaled
 * to fit the destination rectangle. The filtering mode for the scale defaults
 * to NVMEDIA_2D_STRETCH_FILTER_LOW. Additional filtering modes are available
 * by setting the corresponding parameter in NvMedia2DBlitParameters.
 *
 * Passing in NULL as \a srcRect defaults to a source rectangle the size of the
 * full source surface, likewise for \a dstRect and the destination surface.
 *
 * If \a paramsOut is not NULL, the blit operation stores supplementary return
 * values for the blit to the structure pointed to by paramsOut, if applicable.
 * If \a paramsOut is NULL, no supplementatry information is returned.
 *
 * Supplementary values are returned when using blit flag:
 *
 * NVMEDIA_2D_BLIT_FLAG_RETURN_CRC returns a CRC value of blitted pixels.
 *
 * A straight pixel copy between surfaces of the same dimensions (but not
 * necessary the same bit depth or even color format) is issued by:
 *
 * @code
 *      NvMedia2DBlitEx(i2d, dst, NULL, src, NULL, NULL, NULL);
 * @endcode
 *
 * \param[in] i2d Image 2D object.
 * \param[in] dstSurface Destination surface.
 * \param[in] dstRect Destination rectangle.
 * \param[in] srcSurface Source surface.
 * \param[in] srcRect Source rectangle.
 * \param[in] params Parameters.
 * \param[out] paramsOut Returned parameters.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the mandatory pointer is invalid.
 * \ingroup blit
 */
NvMediaStatus
NvMedia2DBlitEx(
    NvMedia2D                       *i2d,
    NvMediaImage                    *dstSurface,
    const NvMediaRect               *dstRect,
    NvMediaImage                    *srcSurface,
    const NvMediaRect               *srcRect,
    const NvMedia2DBlitParameters   *params,
    NvMedia2DBlitParametersOut      *paramsOut
);

/**
 * Copies a plane of a YUV image to another YUV image.
 *
 * The source and the destination must be of the same format.
 *
 * \param[in] dstSurface A pointer to a destination surface.
 * \param[in] dstPlane Destination plane.
 * \param[in] srcSurface A pointer to the source surface.
 * \param[in] srcPlane Source plane.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED
 * \n \ref NVMEDIA_STATUS_ERROR
 * \ingroup blit
 */
NvMediaStatus
NvMedia2DCopyPlane(
    NvMediaImage    *dstSurface,
    uint32_t        dstPlane,
    NvMediaImage    *srcSurface,
    uint32_t        srcPlane
);

/*
 * \defgroup history_nvmedia_2d History
 * Provides change history for the NvMedia 2D API.
 *
 * \section history_nvmedia_2d Version History
 *
 * <b> Version 1.1 </b> Febraury 1, 2016
 * - Initial release
 *
 * <b> Version 1.2 </b> May 11, 2016
 * - Added \ref NvMedia2DCheckVersion API
 *
 * <b> Version 1.3 </b> May 5, 2017
 * - Removed compositing, blending and alpha related defines and structures
 *
 * <b> Version 2.0 </b> May 11, 2017
 * - Deprecated NvMedia2DBlit API
 * - Deprecated NvMedia2DCheckVersion API
 * - Deprecated NvMedia2DColorStandard, NvMedia2DColorRange and
 *   NvMedia2DColorMatrix types
 * - Added \ref NvMedia2DGetVersion API
 *
 * <b> Version 2.1 </b> May 17, 2017
 * - Moved transformation to nvmedia_common.h
 * - Renamed NvMedia2DTransform to \ref NvMediaTransform
 */
/** @} */

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _NVMEDIA_2D_H */
