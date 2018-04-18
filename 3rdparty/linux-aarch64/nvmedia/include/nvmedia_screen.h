/*
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */


/**
 * \file
 * \brief <b> NVIDIA Media Interface: QNX Screen</b>
 *
 * @b Description: This file contains the QNX Screen API.
 */


#ifndef _NVMEDIA_SCREEN_H
#define _NVMEDIA_SCREEN_H

#include <nvmedia_core.h>
#include <nvmedia_surface.h>
#include <nvmedia_video.h>
#include <nvmedia_image.h>
#include <screen/screen.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup api_screen QNX Screen API
 *
 * The QNX Screen API creates video and image screen siblings
 * that mirror the composited windows that QNX Screen produces.
 * Unlike the QNX Screen surface, the screen sibling is compatible
 * with the NvMedia API and samples.
 *
 * For more information about QNX Screen, see
 * <a href="../Graphics Programming Guide/graphics_subsystem_qnx.html">
 * QNX Screen Graphics Subsystem</a>.
 *
 * @ingroup nvmedia_image_top
 * @{
 */

/** \brief Major Version number */
#define NVMEDIA_SCREEN_VERSION_MAJOR    1
/** \brief Minor Version number */
#define NVMEDIA_SCREEN_VERSION_MINOR    0

/**
 * \brief Creates an NvMediaVideoSurface from the given screen buffer.
 * \param[in] device  A pointer to the NvMediaDevice.
 * \param[in] screenSurface The original screen buffer handle.
 * \param[in,out] nvmediaSurface A double-pointer to the pointer to the video
 *                surface to create. If this function returns with the
 *                \c NVMEDIA_STATUS_OK status, the argument contains a pointer
 *                to the video surface.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK            If nvmediaSurface is created
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER If input parameters are invalid
 * \n \ref NVMEDIA_STATUS_ERROR         If other errors occurred
 */
NvMediaStatus
NvxScreenCreateNvMediaVideoSurfaceSibling(
    NvMediaDevice *device,
    screen_buffer_t screenSurface,
    NvMediaVideoSurface **nvmediaSurface
);

/**
 * \brief Destroys a video surface created by NvxScreenCreateNvMediaVideoSurfaceSibling().
 * \param[in] nvmediaSurface A pointer to the video surface to be destroyed.
 */
void NvxScreenDestroyNvMediaVideoSurfaceSibling(
    NvMediaVideoSurface *nvmediaSurface
);

/**
 * \brief Creates an NvMediaImage from the given screen buffer.
 * \param[in] device  A pointer to the NvMediaDevice.
 * \param[in] screenSurface The original screen buffer handle.
 * \param[in,out] image A double-pointer to the pointer to the image
 *                surface to create.
 *                If this function returns with the
 *                \c NVMEDIA_STATUS_OK status, the argument contains a pointer
 *                to the NvMediaImage surface.

 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK            If image is created
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER If input parameters are invalid
 * \n \ref NVMEDIA_STATUS_ERROR         If other errors occurred
 */
NvMediaStatus
NvxScreenCreateNvMediaImageSibling(
    NvMediaDevice *device,
    screen_buffer_t screenSurface,
    NvMediaImage **image
);

/**
 * \brief Destroy a image created by \ref NvxScreenCreateNvMediaImageSibling.
 * \param[in] image The image surface to be destroyed.
 * \return void
 */
void NvxScreenDestroyNvMediaImageSibling(
    NvMediaImage *image
);

/*@}*/

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _NVMEDIA_SCREEN_H */
