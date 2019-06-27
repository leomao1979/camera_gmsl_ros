#include "RenderUtils.hpp"

namespace renderutils {
void renderFPS(dwRenderEngineHandle_t renderEngine, const float32_t fps)
    {
        if (renderEngine != DW_NULL_HANDLE)
        {
            // store previous tile
            uint32_t previousTile = 0;
            CHECK_DW_ERROR(dwRenderEngine_getTile(&previousTile, renderEngine));

            // select default tile
            CHECK_DW_ERROR(dwRenderEngine_setTile(0, renderEngine));

            // store previous state
            dwRenderEngineTileState previousTileState{};
            CHECK_DW_ERROR(dwRenderEngine_getState(&previousTileState, renderEngine));

            // set text render settings
            CHECK_DW_ERROR(dwRenderEngine_setModelView(&DW_IDENTITY_TRANSFORMATION, renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, renderEngine));

            // render
            constexpr uint32_t textBufferSize = 128;
            char fpsText[textBufferSize];
            snprintf(fpsText, textBufferSize, "FPS: %.02f", fps);
            CHECK_DW_ERROR(dwRenderEngine_renderText2D(fpsText, {0.0f, 1.0f}, renderEngine));

            // restore previous settings
            CHECK_DW_ERROR(dwRenderEngine_setState(&previousTileState, renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setTile(previousTile, renderEngine));
        }
    }

}
