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
// Copyright (c) 2015-2018 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef SAMPLES_COMMON_RENDERUTILS_HPP_
#define SAMPLES_COMMON_RENDERUTILS_HPP_

#include <dw/core/Types.h>
#include <dw/renderer/RenderEngine.h>
#include "Checks.hpp"

#include <fstream>

namespace renderutils
{
const dwVector4f colorLightBlue{DW_RENDERER_COLOR_LIGHTBLUE[0],
            DW_RENDERER_COLOR_LIGHTBLUE[1],
            DW_RENDERER_COLOR_LIGHTBLUE[2],
            DW_RENDERER_COLOR_LIGHTBLUE[3]};

const dwVector4f colorYellow{DW_RENDERER_COLOR_YELLOW[0],
            DW_RENDERER_COLOR_YELLOW[1],
            DW_RENDERER_COLOR_YELLOW[2],
            DW_RENDERER_COLOR_YELLOW[3]};

const dwVector4f colorGreen{DW_RENDERER_COLOR_GREEN[0],
            DW_RENDERER_COLOR_GREEN[1],
            DW_RENDERER_COLOR_GREEN[2],
            DW_RENDERER_COLOR_GREEN[3]};

const dwVector4f colorRed{DW_RENDERER_COLOR_RED[0],
            DW_RENDERER_COLOR_RED[1],
            DW_RENDERER_COLOR_RED[2],
            DW_RENDERER_COLOR_RED[3]};

void renderFPS(dwRenderEngineHandle_t renderEngine, const float32_t fps);
}
#endif // SAMPLES_COMMON_RENDERUTILS_HPP_
