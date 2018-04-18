/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/**
 * \file
 * \brief <b> NVIDIA Media Interface: Image Signal Processing</b>
 *
 * @b Description: This file contains the \ref image_isp_api "Image Signal Processing API".
 */

#ifndef _NVMEDIA_ISP_H
#define _NVMEDIA_ISP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_image.h"

/**
 * \defgroup image_isp_api Image Signal Processing
 * \ingroup nvmedia_video_top
 *
 * The Image Signal Processing API encompasses all NvMedia image processing
 * functions that are necessary to produce a processed image from image data
 * captured from an image sensor.
 *
 * @ingroup nvmedia_image_top
 * @{
 */


/** \brief Major Version number */
#define NVMEDIA_ISP_VERSION_MAJOR   1
/** \brief Minor Version number */
#define NVMEDIA_ISP_VERSION_MINOR   4

/**
 * \defgroup image_isp_types Basic ISP Types
 * The Image Signal Processing API provides common ISP processing functions.
 * @ingroup basic_api_top
 *
 * @{
 */

/**
 * \brief Specifies which ISP to use.
 */
typedef enum {
    /** \brief Selects ISP A */
    NVMEDIA_ISP_SELECT_ISP_A,
    /** \brief Selects ISP B */
    NVMEDIA_ISP_SELECT_ISP_B
} NvMediaISPSelect;

enum {
    /** Number of color components */
    NVMEDIA_ISP_COLOR_COMPONENT_NUM = 4
};

enum {
    NVMEDIA_ISP_COLOR_COMPONENT_0 = 0
};
enum {
    NVMEDIA_ISP_COLOR_COMPONENT_1 = 1
};
enum {
    NVMEDIA_ISP_COLOR_COMPONENT_2 = 2
};
enum {
    NVMEDIA_ISP_COLOR_COMPONENT_3 = 3
};
enum {
    NVMEDIA_ISP_COLOR_COMPONENT_MAX = 4
};

/**
 * Specifies color components positions.
 */
enum {
    /** Specifies the top-left pixel position. */
    NVMEDIA_ISP_COLOR_COMPONENT_TL = NVMEDIA_ISP_COLOR_COMPONENT_0,
    /** Specifies the top-rigth pixel position. */
    NVMEDIA_ISP_COLOR_COMPONENT_TR = NVMEDIA_ISP_COLOR_COMPONENT_1,
    /** Specifies the bottom-left pixel position. */
    NVMEDIA_ISP_COLOR_COMPONENT_BL = NVMEDIA_ISP_COLOR_COMPONENT_2,
    /** Specifies the bottom-rigth pixel position. */
    NVMEDIA_ISP_COLOR_COMPONENT_BR = NVMEDIA_ISP_COLOR_COMPONENT_3
};

enum {
    /** Specifies the number of color components in a triplet. */
    NVMEDIA_ISP_COLOR_COMPONENT_TRIPLET_NUM = 3
};


enum {
    NVMEDIA_ISP_HDR_SAMPLE_MAP_NUM = 16
};

enum {
    NVMEDIA_ISP_HIST_RANGE_CFG_NUM = 8
};
enum {
    NVMEDIA_ISP_LAC_ROI_NUM = 4
};
enum {
    NVMEDIA_ISP_AFM_ROI_NUM = 8
};
enum {
    NVMEDIA_ISP_AFM_FILTER_COEFF_NUM = 6
};

/**
  * Defines the ISP color channels.
  */
typedef enum
{
    NVMEDIA_ISP_COLORCHANNEL_TL_R_V = NVMEDIA_ISP_COLOR_COMPONENT_0,
    NVMEDIA_ISP_COLORCHANNEL_TR_G_Y,
    NVMEDIA_ISP_COLORCHANNEL_BL_B_U,
    NVMEDIA_ISP_COLORCHANNEL_BR,
    NVMEDIA_ISP_COLORCHANNEL_LUMINANCE,

    NVMEDIA_ISP_COLORCHANNEL_FORCE32 = 0x7FFFFFFF
} NvMediaISPColorChannel;

/**
 * Defines the ISP pixel format types.
 */
typedef enum {
    /** RGB pixel format */
    NVMEDIA_ISP_PIXELFORMAT_RGB = 0x1,
    /** YUV pixel format */
    NVMEDIA_ISP_PIXELFORMAT_YUV,
    /** Quad pixel format */
    NVMEDIA_ISP_PIXELFORMAT_QUAD
} NvMediaISPPixelFormat;

/**
 * Defines the HDR mode types.
 */
typedef enum
{
    /** Specifies that samples are not distinguished and all are used as short exposure pixels */
    NVMEDIA_ISP_HDR_MODE_NORMAL = 0x1,
    /** Specifies to determine the short exposure pixels */
    NVMEDIA_ISP_HDR_MODE_SHORT,
    /** Specifies to determine the long exposure pixels */
    NVMEDIA_ISP_HDR_MODE_LONG,
    /** Specifies the HDR contains both short and long exposure pixels */
    NVMEDIA_ISP_HDR_MODE_BOTH,
    /** <b>Applies to</b>: code-name Parker: Specifies to separately handle long and short exposure pixels */
    NVMEDIA_ISP_HDR_MODE_SEPARATE,

    NVMEDIA_ISP_HDR_MODE_FORCE32 = 0x7FFFFFFF
} NvMediaISPHdrMode;

/**
 * Holds an integer range.
 */
typedef struct
{
    /**  Lower limit for the range. */
    int low;
    /**  Upper limit for the range. */
    int high;
} NvMediaISPRange;

/**
 * Holds a float range.
 */
typedef struct
{
    /**  Lower limit for the range. */
    float low;
    /**  Upper limit for the range. */
    float high;
} NvMediaISPRangeFloat;

/**
 * Defines a 2-dimensional surface where the surface is
 * determined by the surface height and width in pixels.
 */
typedef struct
{
    /** Holds the width of the surface in pixels. */
    int width;
    /** Holds the height of the surface in pixels. */
    int height;
} NvMediaISPSize;

/**
 * Defines a location on a 2-dimensional object,
 * where the coordinate (0,0) is located at the top-left of the object.  The
 * values of x and y are in pixels.
 */
typedef struct
{
    /** Holds the horizontal location of the point. */
    int x;
    /** Holds the vertical location of the point. */
    int y;
} NvMediaISPPoint;

/**
 * Defines a location on a 2-dimensional object,
 * where the coordinate (0,0) is located at the top-left of the object.  The
 * values of x and y are in pixels.
 */
typedef struct
{
    /** Holds the horizontal location of the point. */
    float x;
    /** Holds the vertical location of the point. */
    float y;
} NvMediaISPFloatPoint;
/** @} <!-- Ends image_isp_types Basic ISP Types --> */

/**
 * \defgroup isp_frame_stats ISP Statistics
 * Defines ISP statistics types, settngs, and functions.
 * @{
 */

/**
 * Defines the ISP statistics selector types.
 */
typedef enum
{
    /** Histogram statistics */
    NVMEDIA_ISP_STATS_HISTOGRAM = 1,
    /** Location and clipped statistics */
    NVMEDIA_ISP_STATS_LAC,
    /** Flicker band statistics */
    NVMEDIA_ISP_STATS_FLICKER_BAND,
    /** Focus metric statistics */
    NVMEDIA_ISP_STATS_FOCUS_METRIC,
    /** Histogram statistics for ISP version 4 */
    NVMEDIA_ISP_STATS_HISTOGRAM_V4,
    /** Location and clipped statistics for ISP version 4 */
    NVMEDIA_ISP_STATS_LAC_V4,
    /** Flicker band statistics for ISP version 4 */
    NVMEDIA_ISP_STATS_FLICKER_BAND_V4,
    /** Focus metric statistics_V4 for ISP version 4 */
    NVMEDIA_ISP_STATS_FOCUS_METRIC_V4,
    /** Histogram statistics for ISP version 5 */
    NVMEDIA_ISP_STATS_HISTOGRAM_V5,
    /** Location and clipped statistics for ISP version 5 */
    NVMEDIA_ISP_STATS_LAC_V5,
    /** Flicker band statistics for ISP version 5 */
    NVMEDIA_ISP_STATS_FLICKER_BAND_V5,
    /** Local tone map statistics for ISP version 5 */
    NVMEDIA_ISP_STATS_LTM_V5,
    /** Outlier rejection statistics for ISP version 5 */
    NVMEDIA_ISP_STATS_OR_V5
} NvMediaISPStats;

/**
 * Holds the settings for the histogram statistics of ISP version 4.
 */
typedef struct
{
    /** Holds a Boolean that enables historgram statistics. */
    NvMediaBool enable;

    /** Holds the pixel format on which a client wants this instance to operate. */
    NvMediaISPPixelFormat pixelFormat;

    /**
     * Range of the pixel values to be considered for each zone. The whole pixel range
     * is divided in to 8 zones.
     * Value 0-2 and 13-15 should not be used. The valid values and the correponding log 2
     * range is specified below.
     * @code
     * 3 : Range = 8
     * 4 : Range = 16
     * 5 : Range = 32
     * 6 : Range = 64
     * 7 : Range = 128
     * 8 : Range = 256
     * 9 : Range = 512
     * 10 : Range = 1024
     * 11 : Range = 2048
     * 12 : Range = 4096
     * @endcode
     */
    unsigned char range[NVMEDIA_ISP_HIST_RANGE_CFG_NUM];

    /**
     * This is the co-effcients for the curve that defines the mapping of input pixel range to
     * the bins. The curve between two knee points are linearly interpolated.
     * Knee[-1] = 0 and Knee[7] = 255(The total number of bins - 1).
     */
    unsigned char knee[NVMEDIA_ISP_HIST_RANGE_CFG_NUM];

    /**
     * Offset to be applied to the input data prior to performing the bin mapping operation.
     */
    float offset;

    /** Window to construct the histogram. */
    NvMediaRect window;

    /**
     * HDR interleave pattern
     * Example:
     * @code
     * 1 1 1 1      1 1 0 0     1 1 0 0
     * 1 1 1 1  OR  1 1 0 0 OR  1 1 0 0
     * 0 0 0 0      1 1 0 0     0 0 1 1
     * 0 0 0 0      1 1 0 0     0 0 1 1
     * @endcode
     */
    unsigned char  hdrSampleMap[NVMEDIA_ISP_HDR_SAMPLE_MAP_NUM];

    /**
     * HDR mode
     */
    NvMediaISPHdrMode   hdrMode;
} NvMediaISPStatsHistogramSettingsV4;

/**
 * Defines the histogram statistics measurement.
 */
typedef struct
{
    /**
     * Holds the number of bins that the histogram of each color component includes.
     * Each color component must have the same number of bins.
     */
    unsigned int numBins;

    /**
     * Holds an array of pointers to the histogram data for different color components.
     * Use the indices based on the color space on which the histogram is
     * gathered.
     * For Bayer, use NVMEDIA_ISP_COLOR_COMPONENT_[TL|TR|BL|BR].
     * For YUV, use NVMEDIA_ISP_COLOR_COMPONENT_[Y|U|V].
     * For RGB, use NVMEDIA_ISP_COLOR_COMPONENT_[R|G|B].
     */
    unsigned int *data[NVMEDIA_ISP_COLOR_COMPONENT_NUM];
} NvMediaISPStatsHistogramMeasurement;

/**
 * Defines the windows used in ISP stats calculations.
 *
 * \code
 *      -------------------------------------------------------------------
 *      |                                                                 |
 *      |      StartOffset                                                |
 *      |     /                                                           |
 *      |     ********        ********        ********  -                 |
 *      |     *      *        *      *        *      *  |                 |
 *      |     *      *        *      *        *      *  |                 |
 *      |     *      *        *      *        *      *  |                 |
 *      |     ********        ********        ********  |                 |
 *      |     |---------------|                         |   \             |
 *      |      HorizontalInterval   VerticalInterval--->|     VerticalNum |
 *      |                                               |   /             |
 *      |     ******** -      ********        ********  -                 |
 *      |     *      * |      *      *        *      *                    |
 *      |     *      * |      *      *        *      *                    |
 *      |     *      * |      *      *        *      *                    |
 *      |     ******** -      ********        ********                    |
 *      |     |------| \                                                  |
 *      |               Size                                              |
 *      |                                                                 |
 *      |                   \     |     /                                 |
 *      |                   HorizontalNum                                 |
 *      |                                                                 |
 *      -------------------------------------------------------------------
 * \endcode
 */
typedef struct
{
    /** Size of each window */
    NvMediaISPSize size;

    /**
     * Number of windows horizontally.
     */
    unsigned int horizontalNum;
    /** Number of windows vertically */
    unsigned int verticalNum;

    /** Distance between the left edges of one window and a horizontally
     *  adjacent window.
     */
    unsigned int horizontalInterval;

    /** Distance between the top edges of one window and a vertically
     *  adjacent window.
     */
    unsigned int verticalInterval;

    /** Position of the top-left pixel in the top-left window. */
    NvMediaISPPoint startOffset;
} NvMediaISPStatsWindows;

/**
 * Defines the settings to use for LAC statistics for ISP version 4.
 *
 */
typedef struct
{
    /** Holds a Boolean that enables LAC statistics. */
    NvMediaBool enable;

    /** Enables each ROI region */
    NvMediaBool ROIEnable[NVMEDIA_ISP_LAC_ROI_NUM];

    /** Indicates if pixels are in Bayer format or triplets (YUV / RGB+Y) */
    NvMediaISPPixelFormat pixelFormat;

    /** Use NVMEDIA_ISP_COLOR_COMPONENT_[R|G|B] for the indices.
     * This will be used to convert RGB to Y when LAC is gathered on RGB.
     * Y = sum of (X + rgbToYOffset[X]) * rgbToYGain[X]) over X = {R, G, B}
     * \n rgbToYOffset range [-1.0, 1.0)
     */
    float rgbToYOffset[NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /** rgbToYGain range [0, 1.0) */
    float rgbToYGain[NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /**
     * The range of each color component to be used in the calculation
     * of the average.  Range [-0.5, 1.5) for RGBY, [-1.0, 1.0) for UV
     */
    NvMediaISPRangeFloat range[NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /**
     * Windows for LAC calculation for each ROI.
     */
    NvMediaISPStatsWindows windows[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * HDR interleave pattern
     * Example:
     * 1 1 1 1      1 1 0 0     1 1 0 0
     * 1 1 1 1  OR  1 1 0 0 OR  1 1 0 0
     * 0 0 0 0      1 1 0 0     0 0 1 1
     * 0 0 0 0      1 1 0 0     0 0 1 1
     */
    unsigned char  hdrSampleMap[NVMEDIA_ISP_HDR_SAMPLE_MAP_NUM];

    /**
     * HDR mode for each ROI. If ROI is not enabled just
     * use the first value in the array and ignore others.
     */
    NvMediaISPHdrMode   hdrMode[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Used to determine the long exposure pixels.
     */
    float   hdrShort;

    /**
     * Used to determine the long exposure pixels.
     */
    float   hdrLong;
} NvMediaISPStatsLacSettingsV4;

/**
 * Holds the LAC statistics measurement for ISP version 4.
 */
typedef struct
{
    /**
     * Holds a Boolean that specifies to ignore the stats value
     * if a ROI is not enabled.
     * If using the values for disabled ROI, it will cause undefined behavior.
     */
    NvMediaBool ROIEnable[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Holds the position of the top-left pixel in the top-left window.
     */
    NvMediaISPPoint startOffset[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Holds the size of each window.
     */
    NvMediaISPSize windowSize[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Holds the number of windows in LAC stats.
     *
     * When the client calls NvIspGetStats(), NumWindows is the size of each
     * array that the pAverage[ROI_ID][COLOR_COMPONENT] and pNumPixels[ROI_ID][COLOR_COMPONENT]
     * pointers point to. It must be >= (NumWindowsH * NumWindowsV) in NvIspStatsLacSettings
     * used to setup LAC. If a particular ROI is disabled ignore the measurement values
     * corresponding to that particlar ROI. Reading values for disabled ROI will give you
     * undefined stats values.
     *
     * The LAC data will be stored in row-major order.
     */
    unsigned int numWindows[NVMEDIA_ISP_LAC_ROI_NUM];

    /** Holds the number of windows in the horizontal direction. */
    unsigned int numWindowsH[NVMEDIA_ISP_LAC_ROI_NUM];

    /** Holds the number of windows in the vertical direction. */
    unsigned int numWindowsV[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * For Bayer, use NVMEDIA_ISP_COLOR_COMPONENT_[TL|TR|BL|BR] for the indices.
     * For YUV, use NVMEDIA_ISP_COLOR_COMPONENT_[Y|U|V].
     * For RGB, use NVMEDIA_ISP_COLOR_COMPONENT_[R|G|B].
     * Data could be negative
     */
    float *average[NVMEDIA_ISP_LAC_ROI_NUM][NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /** Holds the number of pixels per ROI per component. */
    unsigned int *numPixels[NVMEDIA_ISP_LAC_ROI_NUM][NVMEDIA_ISP_COLOR_COMPONENT_NUM];
} NvMediaISPStatsLacMeasurementV4;

/**
 * Holds the flicker band settings for ISP version 4.
 *
 * The flicker band module computes the average brightness of a number of
 * samples of the image.
 */
typedef struct
{
    /** Holds a Boolean that enables flicker band statistics. */
    NvMediaBool enable;

    /**
     * Holds the Windows for the flicker band calculation.
     * The number of horizontal windows must be 1. The height of each window
     * may be rounded due to HW limitation.
     */
    NvMediaISPStatsWindows windows;

    /**
     * Holds the color channel used for flicker band calculation.
     * For the YUV color format, selecting Channel Y or Luminance does the same thing in the hardware.
     */
    NvMediaISPColorChannel colorChannel;

    /**
     * Holds the HDR interleave pattern
     * For example:
     * @code
     * 1 1 1 1      1 1 0 0     1 1 0 0
     * 1 1 1 1  OR  1 1 0 0 OR  1 1 0 0
     * 0 0 0 0      1 1 0 0     0 0 1 1
     * 0 0 0 0      1 1 0 0     0 0 1 1
     * @endcode
     */
    unsigned char hdrSampleMap[NVMEDIA_ISP_HDR_SAMPLE_MAP_NUM];

    /**
     * Holds the HDR mode.
     */
    NvMediaISPHdrMode   hdrMode;
} NvMediaISPStatsFlickerBandSettingsV4;

/**
 * Holds the flicker band statistics measurement.
 */
typedef struct
{
    /**
     * Holds the number of flicker band windows.
     *
     * numWindows is the size of the array that luminance points to.
     */
    unsigned int numWindows;

    /** Holds a pointer to the array of the average luminance value of the samples.
     *  Data could be negative
     */
    int *luminance;
} NvMediaISPStatsFlickerBandMeasurement;

/**
 * Holds the focus metric statistics settings.
 *
 * It calculates the focus metric in each focus window. The stats can only be
 * generated on Bayer data.
 */
typedef struct
{
    /** Enable focus metric statistics */
    NvMediaBool enable;

    /** Whether to enable noise compensate when calculating focus metrics */
    NvMediaBool noiseCompensation;

    /**
     * Application of the noise compensation when noise compensation is
     * enabled. This value will be rounded to the nearest value that the
     * hardware supports.
     */
    float noiseCompensationGain;

    /**
     * Gain applied to the accumulated focus metric. This value will be
     * rounded to the nearest value that the hardware supports.
     * Example of valid values are 1, 0.5, 0.25.
     */
    float metricGain;

    /**
     * Number of the coefficients for each color component.
     */
    unsigned int numCoefficients;

    /**
     * Coefficients of the filter to compute the focus metric for each color
     * component. An example of 9-tap filter: (-1, -2, -1, 2, 4, 2, -1, -2, -1)
     */
    float *coefficient[NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /** Focus metric values lower than this limit will be clamped to zero. */
    float metricLowerLimit;

    /**
     * Maximum value of the input pixels to be used in the calculation of the
     * focus metric.
     */
    float inputThreshold;

    /** Windows for focus metric calculation. */
    NvMediaISPStatsWindows windows;
} NvMediaISPStatsFocusMetricSettings;

/**
 * Holds the focus metric statistics measurement.
 */
typedef struct
{
    /** Holds the position of the top-left pixel in the top-left window. */
    NvMediaISPPoint startOffset;

    /** Holds the size of each window. */
    NvMediaISPSize windowSize;

    /**
     * Holds the size of the array to which pMetric points.
     */
    unsigned int numWindows;

    /** Holds a pointer to the array of the focus metrics of the windows. */
    unsigned int *metric[NVMEDIA_ISP_COLOR_COMPONENT_NUM];
} NvMediaISPStatsFocusMetricMeasurement;

/**
 * Start of statistics defines for ISP version 5
 */

#define NVMEDIA_ISP5_RADTF_CP_COUNT             6
#define NVMEDIA_ISP5_FB_WINDOWS                 256
#define NVMEDIA_ISP5_HIST_TF_KNEE_POINT_COUNT   8
#define NVMEDIA_ISP5_HIST_BIN_COUNT             256
#define NVMEDIA_ISP_LAC_ROI_WIN_NUM             (32 * 32)
#define NVMEDIA_ISP5_LTM_HIST_BIN_COUNT         128
#define NVMEDIA_ISP5_LTM_AVG_WIN_COUNT          8

/**
 * Defines multi-exposure lane selection.
 */
typedef enum
{
    NVMEDIA_ISP_MULTI_EXP_LANE_RV = 0,
    NVMEDIA_ISP_MULTI_EXP_LANE_GY = 1,
    NVMEDIA_ISP_MULTI_EXP_LANE_BU = 2,
} NvMediaIsp5MultiExpLane;

/**
 * Defines radial mask.
 */
typedef struct
{
    /** X coordinate of the center of ellipse in pixels */
    float_t x;
    /** Y coordinate of the center of ellipse in pixels */
    float_t y;
    /** Ellipse matrix transformation coefficients */
    float_t kxx;
    float_t kxy;
    float_t kyx;
    float_t kyy;
} NvMediaIsp5RadialMask;

/**
 * Defines rectangular mask for local tone mapping.
 */
typedef struct
{
    uint16_t top;
    uint16_t bottom;
    uint16_t left;
    uint16_t right;
} NvMediaIsp5RectMask;

/**
 * Defines control point for Cubic Hermite spline. Cubic spline is used
 * in many ISP blocks to interpolate functions.
 *
 * A spline is defined with an array of control points; number of points
 * varies between units.
 */
typedef struct
{
    /** X coordinate of the control point */
    float_t x;
    /** Y coordinate of the control point */
    float_t y;
    /** Slope (tangent) of the interpolated curve at the control point */
    float_t slope;
} NvMediaISP5CubicSplineCtrlPoint;

/**
 * Defines radial transfer function.
 */
typedef struct
{
    NvMediaIsp5RadialMask ellipse;
    NvMediaISP5CubicSplineCtrlPoint tf[NVMEDIA_ISP5_RADTF_CP_COUNT];
} NvMediaIsp5RadialTf;

/**
 * Defines channel selections for histogram for ISP version 5.
 */
typedef enum
{
    NVMEDIA_ISP_HIST_CH_R_V = 0,
    NVMEDIA_ISP_HIST_CH_G_Y = 1,
    NVMEDIA_ISP_HIST_CH_B_U = 2,
    NVMEDIA_ISP_HIST_CH_Y   = 3,
} NvMediaIsp5HistChannel;

/**
 * Defines channel selections for 4th channel histogram for ISP version 5.
 */
typedef enum {
    NVMEDIA_ISP_HIST_CH_MAXORY_MAX = 0,
    NVMEDIA_ISP_HIST_CH_MAXORY_Y   = 3,
} NvMediaIsp5HistChannelMaxOrY;

/**
 * Defines the flicker band settings for ISP version 5.
 *
 * The flicker band module computes the average brightness of a number of
 * samples of the image.
 */
typedef struct
{
    /**
     * Enable the flicker band stats unit.
     */
    NvMediaBool enable;

    /**
     * Select the channel that is used for calculating the stats
     * The "Luminance" channel is calculated as follows
     * - In case of CFA input data, it is set to average of pixels in
     *   a single CFA block, so the value will be 0.25*R + 0.5*G + 0.25*B
     * - In case of RGB data it is set to 0.25*R + 0.625*G + 0.125*B
     * - In case of YUV data, the Y channel is used directly
     */

    NvMediaISPColorChannel chSelect;

    /**
     * Enable the elliptical mask for disabling pixels outside area of interest
     */
    NvMediaBool radialMaskEnable;

    /**
     * Count of flicker band samples to collect per frame
     */
    uint8_t bandCount;

    /**
     * Offset of the first band top line (and first pixel in a line for all bands)
     */
    NvMediaISPPoint offset;

    /**
     * Size of a single band. This must be chosen so that
     * - Both width and height are even and >= 2
     * - Total number of accumulated pixels must be <= 2^18
     */
    NvMediaISPSize bandSize;

    /**
     * Select the range of pixel values to include when calculating the FB statistics
     * In merged multi-exposure HDR image each exposure might have different
     * characteristics (exposure time and timing) that affect FB stats; by setting
     * these limits so that only pixels fro a single exposure are included
     * accuracy of FB stats can be significantly improved.
     */

    NvMediaISPRangeFloat range;
    NvMediaISPRangeFloat chromaRange;

    /**
     * Elliptical mask for selecting pixels included if radialMaskEnable is set to true
     */
    NvMediaIsp5RadialMask radialMask;

} NvMediaISPStatsFlickerBandSettingsV5;

/**
 * Holds the flicker band statistics measurement for isp version 5.
 */
typedef struct
{
    /**
     * Holds the number of flicker band windows.
     *
     * numWindows is the size of the array that luminance points to.
     */
    uint32_t numWindows;

    /** Holds a pointer to the array of the average luminance value of the samples.
     *  Data could be negative
     */
    int32_t luminance[NVMEDIA_ISP5_FB_WINDOWS];
} NvMediaISPStatsFlickerBandMeasurementV5;

/**
 * Defines the settings for the histogram statistics of ISP version 5.
 */
typedef struct
{
    /** Enable the histogram unit  */
    NvMediaBool enable;

    /**
     * Enable a mask for excluding pixels outside specified elliptical area
     */
    NvMediaBool radialMaskEnable;

    /**
     * Enable radial weighting of pixels based on their spatial location. This can be
     * used to e.g. compensate for lens shading if Hist is measured before LS correction,
     * or different area covered by pixels.
     */
    NvMediaBool radialWeigthEnable;

    /**
     * If input is a multi-exposure image, select the exposure used for histogram.
     * Valid values are 0..2
     */
    NvMediaIsp5MultiExpLane laneSelect;

    /**
     * Rectangle used for the histogram
     */
    NvMediaRect window;

    /**
     * Data used for the 4 histogram channels. For channels 0-2, valid values are any of
     * the color channels(R/V, G/Y, B/U) or Y calculated from RGB pixel values.
     * For channel 3 the valid values are either maximum of R,G,B (or U,V in case of YUV
     * input) or calculated Y.
     */
    NvMediaIsp5HistChannel channel0;
    NvMediaIsp5HistChannel channel1;
    NvMediaIsp5HistChannel channel2;
    NvMediaIsp5HistChannelMaxOrY channel3;

    /**
     * Conversion from RGB to Y for calculating the Y channel
     */
    float_t R2YGain;
    float_t G2YGain;
    float_t B2YGain;
    float_t R2YOffset;
    float_t G2YOffset;
    float_t B2YOffset;

    /**
     * Offset to be applied to the input data prior to performing the bin
     * mapping operation
     */
    float_t offset;

    /**
     * Offset to be applied to chroma channels of the input data prior to performing
     * the bin mapping operation
     */
    float_t chromaOffset;

    /**
     * Log2 width of the histogram mapping zones
     */
    uint8_t kneePoints[NVMEDIA_ISP5_HIST_TF_KNEE_POINT_COUNT];

    /**
     * Log2 ranges of the histogram mapping zones
     */
    uint8_t ranges[NVMEDIA_ISP5_HIST_TF_KNEE_POINT_COUNT];

    /**
     * Log2 width of the histogram mapping zones
     */
    uint8_t chromaKneePoints[NVMEDIA_ISP5_HIST_TF_KNEE_POINT_COUNT];

    /**
     * Log2 ranges of the histogram mapping zones
     */
    uint8_t chromaRanges[NVMEDIA_ISP5_HIST_TF_KNEE_POINT_COUNT];

    /**
     * Elliptical mask for selecting pixels included if radialMaskEnable is set to true
     */
    NvMediaIsp5RadialMask radialMask;

    /**
     * Radial transfer function if radialWeigthEnable is set to true
     */
    NvMediaIsp5RadialTf radialWeightTf;

} NvMediaISPStatsHistogramSettingsV5;

/**
 * Holds the histogram statistics measurement for ISP version 5.
 */
typedef struct
{
    /**
     * Array of the histogram data for different color components.
     * Use the indices based on the color space on which the histogram is
     * gathered.
     * For Bayer, use NV_ISP_COLOR_COMPONENT_[TL|TR|BL|BR].
     * For YUV, use NV_ISP_COLOR_COMPONENT_[Y|U|V].
     * For RGB, use NV_ISP_COLOR_COMPONENT_[R|G|B].
     */
    uint32_t histData[NVMEDIA_ISP5_HIST_BIN_COUNT][NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /** The pixel count for excluded pixels for each color components */
    uint32_t excludedCount[NVMEDIA_ISP_COLOR_COMPONENT_NUM];

} NvMediaISPStatsHistogramMeasurementV5;

/**
 * Defines the settings to use for LAC statistics for ISP version 5.
 *
 */
typedef struct
{
    /** Enable the LAC statistics unit */
    NvMediaBool enable;

    /** Enable the individual ROIs */
    NvMediaBool ROIEnable[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Enable a mask for excluding pixels outside specified elliptical area in each ROI
     */
    NvMediaBool radialMaskEnable[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * If input is a multi-exposure image, select the exposure used for LAC statistics.
     * Valid values are
     * 0..2
     */
    NvMediaIsp5MultiExpLane laneSelect;

    /**
     * Conversion from RGB to Y for calculating the Y channel if input is RGB/YUV image
     */
    float_t R2YGain;
    float_t G2YGain;
    float_t B2YGain;
    float_t R2YOffset;
    float_t G2YOffset;
    float_t B2YOffset;

    /**
     * Minimum & maximum value of pixels for TL/R/V, TR/G/Y, BL/B/U and BR/Y channels
     * respectively
     */
    NvMediaISPRangeFloat range[NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /**
     * Definition of the LAC windows for each ROI
     */
    NvMediaISPStatsWindows windows[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Elliptical mask for selecting pixels included if radialMaskEnable is set to true
     */
    NvMediaIsp5RadialMask radialMask;

} NvMediaISPStatsLacSettingsV5;

/**
 * Holds the LAC statistics measurement for ISP version 5.
 */
typedef struct
{
    /** Holds the number of windows in one LAC ROI stats. */
    uint32_t numWindows;

    /** Holds the number of windows in the horizontal direction. */
    uint32_t numWindowsH;

    /** Holds the number of windows in the vertical direction. */
    uint32_t numWindowsV;

    /**
     * For Bayer, use NVMEDIA_ISP_COLOR_COMPONENT_[TL|TR|BL|BR] for the indices.
     * For YUV, use NVMEDIA_ISP_COLOR_COMPONENT_[Y|U|V].
     * For RGB, use NVMEDIA_ISP_COLOR_COMPONENT_[R|G|B].
     * Data could be negative
     */
    float_t average[NVMEDIA_ISP_LAC_ROI_WIN_NUM][NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /** Holds the number of un-clipped pixel count per window per component. */
    uint32_t unClippedCnt[NVMEDIA_ISP_LAC_ROI_WIN_NUM][NVMEDIA_ISP_COLOR_COMPONENT_NUM];

    /** Holds the number of clipped pixel count per window per component. */
    uint32_t clippedCnt[NVMEDIA_ISP_LAC_ROI_WIN_NUM][NVMEDIA_ISP_COLOR_COMPONENT_NUM];

} NvMediaISPStatsLacROIDataV5;

typedef struct
{
    /**
     * Holds a Boolean that specifies to ignore the stats value
     * if a ROI is not enabled.
     * If using the values for disabled ROI, it will cause undefined behavior.
     */
    NvMediaBool ROIEnable[NVMEDIA_ISP_LAC_ROI_NUM];

    /**
     * Holds the data for each ROI in LAC stats.
     *
     * The LAC data will be stored in row-major order.
     */
    NvMediaISPStatsLacROIDataV5 ROIData[NVMEDIA_ISP_LAC_ROI_NUM];

} NvMediaISPStatsLacMeasurementV5;

/**
 * Defines the settings to use for Local tone mapping statistics for ISP version 5
 *
 * Local tone mapping parameters depend on statistics gathered from previous frames
 * by the LTM sub block. It provides a global tone histogram with 128 bins and local
 * average statistics for configurable 8x8 windows. Both of these support rectangular
 * and elliptical masks to block certain image areas from statistics gathering.
 */
typedef struct
{
    /** Enable local tonemapping stats */
    NvMediaBool enable;

    /** Enable rectangular mask to restrict image area used for statistics */
    NvMediaBool rectMaskEnable;

    /** Rectangular mask used to restrict statistics calculation */
    NvMediaIsp5RectMask rectMask;

    /** Enable radial mask to restrict image area used for statistics */
    NvMediaBool radialMaskEnable;

    /** Radial mask used to restrict statistics calculation */
    NvMediaIsp5RadialMask radialMask;

    /**
     * Width of local average statistics window in pixels. Value must be even.
     * Value should be selected so that the 8x8 grid of windows covers whole active
     * image area, i.e. that 8 x StatsLocalAvgWndWidth >= image width.
     * Also, the window should be roughly rectangular.
     */
    uint16_t localAvgWndWidth;

    /**
     * Height of local average statistics window in pixels. Value must be even.
     * See LocalAvgWidth documentation for recommendations on selecting value
     */
    uint16_t localAvgWndHeight;

} NvMediaISPStatsLTMSettingsV5;

/**
 * Holds the LTM statistics measurement for ISP version 5.
 */
typedef struct
{
    /**
     * Count of pixels with tone falling into each histogram bin.
     */
    uint32_t histogram[NVMEDIA_ISP5_LTM_HIST_BIN_COUNT];

    /**
     * Average tone in local average window. The tone values are computed
     * by converting YUV input to tone, taking logarithm from the value and normalizing
     * it so that dynamic range is mapped to range [0.0 .. 1.0].
     * See NvMediaISPStatsLocalToneMapV5 for parameters that control this process.
     *
     * If no pixels contributed to a windows (either because the windows was out of image
     * boundaries, or it was completely excluded by rectangular or elliptical masks)
     * this value is set to zero.
     */
    float_t localAverageTone[NVMEDIA_ISP5_LTM_AVG_WIN_COUNT][NVMEDIA_ISP5_LTM_AVG_WIN_COUNT];

    /**
     * Number of pixels in each local average window that contributed to statistics
     */
    uint32_t nonMaskedPixelCount[NVMEDIA_ISP5_LTM_AVG_WIN_COUNT][NVMEDIA_ISP5_LTM_AVG_WIN_COUNT];

} NvMediaISPStatsLTMMeasurementV5;

/**
 * Holds the OR statistics measurement for ISP version 5.
 */
typedef struct
{
    /** Bad pixel count for pixels corrected upwards within the window */
    uint32_t highInWin;
    /** Bad pixel count for pixels corrected downwards within the window */
    uint32_t lowInWin;
    /** Accumulatd pixel adjustment for pixels corrected upwards within the window */
    uint32_t highMagInWin;
    /** Accumulatd pixel adjustment for pixels corrected downwards within the window */
    uint32_t lowMagInWin;
    /** Bad pixel count for pixels corrected upwards outside the window */
    uint32_t highOutWin;
    /** Bad pixel count for pixels corrected downwards outside the window */
    uint32_t lowOutWin;
    /** Accumulatd pixel adjustment for pixels corrected upwards outside the window */
    uint32_t highMagOutWin;
    /** Accumulatd pixel adjustment for pixels corrected downwards outside the window */
    uint32_t lowMagOutWin;
} NvMediaISPStatsORMeasurementV5;

/** @} <!-- ends isp_frame_stats --> */

/*
 * \defgroup history_isp History
 * Provides change history for the NvMedia ISP API.
 *
 * \section history_isp Version History
 *
 * <b> Version 1.0 </b> July 8, 2014
 * - Initial release
 *
 * <b> Version 1.1 </b> March 13, 2015
 * - Added ISP version 3 support
 *
 * <b> Version 1.2 </b> November 17, 2015
 * - Added ISP version 4 support
 *
 * <b> Version 1.3 </b> March 31, 2017
 * - Removed ISP version 3 support
 *
 * <b> Version 1.4 </b> August 15, 2017
 * - Added ISP version 5 support
 */
/** @} */
#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _NVMEDIA_ISP_H */
