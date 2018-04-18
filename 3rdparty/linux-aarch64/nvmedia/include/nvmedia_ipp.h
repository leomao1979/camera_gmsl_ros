/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/**
 * \file
 * \brief <b> NVIDIA Media Interface: Image Processing Pipeline API </b>
 *
 * @b Description: This file contains the \ref image_ipp_api "Image Processing Pipeline API".
 */

#ifndef _NVMEDIA_IPP_H
#define _NVMEDIA_IPP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_surface.h"
#include "nvmedia_image.h"
#include "nvmedia_icp.h"
#include "nvmedia_isc.h"
#include "nvmedia_isp.h"

/**
 * \defgroup image_ipp_api Image Processing Pipeline (IPP)
 *
 * The NvMedia Image Processing Pipeline (IPP) is a framework that controls all
 * NvMedia processing components. It does the following:
 * @li Manages and synchronizes each of the component threads.
 * @li Provides callbacks for global timestamps and events.
 * @li Supports Embedded Line Information for specific functionality.
 *
 * For information on IPP architecture and functionality,
 * see the "Multimedia Programming" chapter in the <em>Development Guide</em>.
 *
 * @ingroup nvmedia_image_top
 * @{
 */

/**
 * \defgroup image_ipp_types Basic IPP Types
 * The Image Processing Pipeline API provides common IPP processing functions.
 * @ingroup basic_api_top
 * @{
 */

/** \brief Major Version number */
#define NVMEDIA_IPP_VERSION_MAJOR   2u
/** \brief Minor Version number */
#define NVMEDIA_IPP_VERSION_MINOR   12u
/** Version information */
#define NVMEDIA_IPP_VERSION_INFO    (((uint8_t)'N' << 24) | ((uint8_t)'V' << 16) | (NVMEDIA_IPP_VERSION_MAJOR << 8) | NVMEDIA_IPP_VERSION_MINOR)

/**
 * \brief A handle representing IPP manager object.
 */
typedef void NvMediaIPPManager;

/**
 * \hideinitializer
 * \brief Maximum number of IPP pipelines in IPP manager
 */
#define NVMEDIA_MAX_COMPONENTS_PER_PIPELINE 32

/**
 * \hideinitializer
 * \brief Maximum number of IPP components in IPP pipeline
 */
#define NVMEDIA_MAX_PIPELINES_PER_MANAGER   12

/**
 * \brief A handle representing IPP pipeline object
 */
typedef void NvMediaIPPPipeline;

/**
 * \brief A handle representing an IPP component object
 */
typedef void NvMediaIPPComponent;

/**
 * \hideinitializer
 * \brief Specifies the IPP ISP version.
 */
typedef enum {
    /** Specifies ISP version 4. */
    NVMEDIA_IPP_ISP_VERSION_4,
    /** Specifies ISP version 5. */
    NVMEDIA_IPP_ISP_VERSION_5
} NvMediaIPPISPVersion;

/**
 * \hideinitializer
 * \brief Defines IPP component types.
 */
typedef enum {
    /** \hideinitializer Capture component. */
    NVMEDIA_IPP_COMPONENT_ICP = 0,
    /** Image signal processing component */
    NVMEDIA_IPP_COMPONENT_ISP,
    /** Control Algorithm component */
    NVMEDIA_IPP_COMPONENT_ALG,
    /** Sensor control component */
    NVMEDIA_IPP_COMPONENT_ISC,
    /** Image output component */
    NVMEDIA_IPP_COMPONENT_OUTPUT,
    /** File reader */
    NVMEDIA_IPP_COMPONENT_FILE_READER,
    /** CaptureEx component. */
    NVMEDIA_IPP_COMPONENT_ICP_EX,
} NvMediaIPPComponentType;

/** Max number of streams */
#define NVMEDIA_IPP_STREAM_MAX_TYPES    2

/**
 * \brief Holds image information.
 */
typedef struct {
    /** Unique frame ID */
    uint32_t frameId;
    /** Camera ID */
    uint32_t cameraId;
    /** Frame capture time-stamp using global time in microseconds */
    NvMediaGlobalTime frameCaptureGlobalTimeStamp;
    /** Frame sequence number - monotonically increasing frame counter */
    uint32_t frameSequenceNumber;
} NvMediaIPPImageInformation;

/*@} <!-- Ends image_ipp_types Basic IPP types --> */

/** \defgroup nvmedia_ipp_properties IPP Properties
  * Defines IPP Properties
  *
  * @{
  */

/** The maximum size of the local tonemap data. */
#define NVMEDIA_IPP_MAX_LTM_DATA_SIZE       (25 * 1024)

/**
 * Defines Flicker detection/correction modes.
 */
typedef enum
{
    /** Set flicker correction off */
    NVMEDIA_IPP_AE_ANTI_FLICKER_MODE_OFF,
    /** Set flicker correction to 50Hz mode */
    NVMEDIA_IPP_AE_ANTI_FLICKER_MODE_50Hz,
    /** Set flicker correction to 60Hz mode */
    NVMEDIA_IPP_AE_ANTI_FLICKER_MODE_60Hz,
    /** Set flicker correction to auto mode */
    NVMEDIA_IPP_AE_ANTI_FLICKER_MODE_AUTO
} NvMediaIPPAeAntiFlickerMode;

/**
 * Defines detected flicker modes.
 */
typedef enum
{
    /** Reported no flicker detected */
    NVMEDIA_IPP_COMPUTED_ANTI_FLICKER_MODE_NONE,
    /** Reported Illuminant frequency of 50 Hz */
    NVMEDIA_IPP_COMPUTED_ANTI_FLICKER_MODE_50Hz,
    /** Reported Illuminant frequency of 60 Hz */
    NVMEDIA_IPP_COMPUTED_ANTI_FLICKER_MODE_60Hz
} NvMediaIPPComputedAntiFlicker;

/**
 * Holds a 4x4 matrix of floats.
 */
typedef struct
{
    /** 2d array */
    float_t array[4][4];
} NvMediaIPPMathFloatMatrix;

/**
 * Defines the auto exposure current states.
 *
*/
typedef enum
{
    /** AE is off. */
    NVMEDIA_IPP_AE_STATE_INACTIVE,
    /** AE doesn't yet have a good set
     *  of control values for the current scene. */
    NVMEDIA_IPP_AE_STATE_SEARCHING,
    /** AE has a good set of
     *  control values for the current scene. */
    NVMEDIA_IPP_AE_STATE_CONVERGED,
    /** AE has timed out searching for a
     *  good set of values for the current scene. */
    NVMEDIA_IPP_AE_STATE_TIMEOUT
} NvMediaIPPAeState;

/**
 * Defines various auto white balance states.
*/
typedef enum
{
    /** AWB is off. */
    NVMEDIA_IPP_AWB_STATE_INACTIVE,
    /** AWB doesn't yet have a good set
      * of control values for the current scene. */
    NVMEDIA_IPP_AWB_STATE_SEARCHING,
    /** AWB has a good set of
     *  control values for the current scene. */
    NVMEDIA_IPP_AWB_STATE_CONVERGED,
    /** AWB has timed out searching for a
     *  good set of values for the current scene. */
    NVMEDIA_IPP_AWB_STATE_TIMEOUT
} NvMediaIPPAwbState;

/** \defgroup nvmedia_ipp_property_structures IPP Property structures
 * Defines IPP Property structures.
 *
 * @{
 */

/**
 * Defines the control properties associated with the camera.
 * Use these properties to control the settings of ISP and the camera.
 */
typedef struct
{
    /** Manual AE */
    NvMediaBool manualAE;

    /** Sensor exposure time and sensor gain for each sensor exposure modes.
     *  Sensor gain valus within this structure should be within gainRange limits */
    NvMediaISCExposureControl exposureControl;

    /** Manual AWB */
    NvMediaBool manualAWB;

    /** White balance color correction gains */
    NvMediaISCWBGainControl wbGains;

    /** Anti-flicker mode */
    NvMediaIPPAeAntiFlickerMode aeAntiFlickerMode;

    /** ISP digital gain */
    float_t ispDigitalGain;
} NvMediaIPPPropertyControls;

/**
 * Defines the dynamic properties associated with the camera.
 */
typedef struct
{
    /** Auto exposure state */
    NvMediaIPPAeState aeState;

    /** Sensor exposure time and sensor gain for each sensor exposure modes.
     *  Sensor gain valus within this structure should be within sensorGainRange limits */
    NvMediaISCExposureControl exposureControl;

    /** Auto white balance state */
    NvMediaIPPAwbState awbState;

    /** Holds the two set of white balance gain control values
     *  calculated by the plugin Control Algorithm, one to apply
     *  in sensor or before hdr merge and other to apply in ISP
     *  after hdr merge */
    NvMediaISCWBGainControl wbGains[2];

    /** Digital gain applied in ISP */
    float_t ispDigitalGain;

    /** Current scene brightness */
    float_t brightness;

    /** Auto white balance CCT */
    uint32_t awbCCT;

    /** Color correction matrix */
    NvMediaIPPMathFloatMatrix colorCorrectionMatrix;

    /** Computed anti-flicker */
    NvMediaIPPComputedAntiFlicker computedAntiFlicker;
} NvMediaIPPPropertyDynamic;

/**
 * Maximum numbers of Knee points for companding curve.
 */
#define NVMEDIA_IPP_MAX_KNEEPOINTS 24

/**
 * Defines the static properties associated with the camera.
 */
typedef struct
{
    /** Acive array size of sensor excluding embedded lines */
    NvMediaISPSize activeArraySize;

    /** ISP maximum digital gain */
    float_t ispMaxDigitalGain;

    /** Companding Curve: Number of Knee Points*/
     unsigned int numKneePoints;

     /** Companding Curve: Knee Points*/
     NvMediaPoint kneePoints[NVMEDIA_IPP_MAX_KNEEPOINTS];

    /** Holds a pointer to the camera module name and
      *  camera-specific configuration string. */
    NvMediaISCModuleConfig *moduleConfig;

} NvMediaIPPPropertyStatic;

/**
 * Defines IPP pipeline property types.
*/
typedef enum {
    /**
     * Specifies to use only embedded data statistics.
     * This property takes the NvMediaBool data type.
     * Possible values are:
     * \n \ref NVMEDIA_TRUE
     * \n \ref NVMEDIA_FALSE (Default)
     * If set, the pipeline uses only the embedded data stats.
     * This property must be set when there is no ISP component.
     */
    NVMEDIA_IPP_PIPELINE_PROPERTY_ONLY_EMB_STATS = 0,
    /**
     * Specifies to use trigger-based capture.
     * This property requires an NvMediaBool data type.
     * Possible values are:
     * \n \ref NVMEDIA_TRUE
     * \n \ref NVMEDIA_FALSE (Default)
     * If set, the pipeline is configured for trigger based capture.
     * \ref NvMediaIPPPipelineSingleCapture does the trigger.
     */
    NVMEDIA_IPP_PIPELINE_PROPERTY_TRIGGER_BASED_CAPTURE,
    /**
     * Specifies the settings delay (in number of frames)
     * from the time of programming the sensor.
     * This property requires an uint32_t data type.
     * Supported range is [0,10]. Default value is 2.
     * This property is used only when sensor does not have any embedded lines.
     */
    NVMEDIA_IPP_PIPELINE_PROPERTY_SETTINGS_DELAY,
} NvMediaIPPPipelinePropertyType;

/**
 * Holds the IPP pipeline property.
*/
typedef struct {
    /** Holds the IPP pipeline property type. */
    NvMediaIPPPipelinePropertyType type;
    /** Holds a pointer to the property specific data. */
    void *value;
} NvMediaIPPPipelineProperty;

/**
 * \brief Sets the IPP pipeline properties.
 * \param[in] ippPipeline The IPP pipeline
 * \param[in] numProperties Number of entries in the properties list
 * \param[in] properties List of IPP pipeline properties
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED if the property type is not supported.
 */
NvMediaStatus
NvMediaIPPPipelineSetProperties(
    NvMediaIPPPipeline *ippPipeline,
    uint32_t numProperties,
    NvMediaIPPPipelineProperty *properties);

/** @} <!-- Ends nvmedia_ipp_property_structures IPP Property structures --> */
/** @} <!-- Ends nvmedia_ipp_properties IPP Properties --> */

/**
 * \defgroup version_info_ipp_api IPP Version Information
 *
 * Provides version information for the NvMedia IPP library.
 * @{
 */

/**
 * \brief Holds version information for the NvMedia IPP library.
 */
typedef struct {
    /*! Library version information */
    NvMediaVersion libVersion;
} NvMediaIPPVersionInfo;

/**
 * \brief Returns the version information for the NvMedia IPP library.
 * \param[in] versionInfo Pointer to a \ref NvMediaIPPVersionInfo structure
 *                        to be filled by the function.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMediaIPPGetVersionInfo(
    NvMediaIPPVersionInfo *versionInfo
);

/** @} <!-- Ends version_info_ipp_api IPP Version Information sub-group -> */

/**
 * \defgroup ipp_manager_creation IPP Manager
 * Defines IPP manager-related structures and functions.
 * @{
 */


/**
 * \brief Allocates an Image Processing Pipeline manager object.
 * \param[in] versionInfo Set it to NVMEDIA_IPP_VERSION_INFO
 * \param[in] device The already created \ref NvMediaDevice.
 * \return \ref NvMediaIPPManager The new IPP manager's handle or NULL if unsuccessful.
 */
NvMediaIPPManager *
NvMediaIPPManagerCreate(
    uint32_t versionInfo,
    NvMediaDevice *device
);

/**
 * \brief Destroys an IPP manager object.
 * \param[in] ippManager The IPP manager object to destroy.
 * \return void
 */
void
NvMediaIPPManagerDestroy(
    NvMediaIPPManager *ippManager
);

/**
 * \brief Defines the global time callback function prototype. The client must create a function
 *  with the same function signature.
 * \param clientContext The client context that was passed to \ref NvMediaIPPManagerSetTimeSource
 * \param timeValue A pointer to location where the callback writes the absolute global time.
 *                 The timeValue is a 64-bit number measured in microseconds.
 */
typedef NvMediaStatus NvMediaIPPGetAbsoluteGlobalTime(
        void *clientContext,
        NvMediaGlobalTime *timeValue);

/**
 * \brief Sets the callback function for image time-stamping.
 * \param[in] ippManager The IPP manager object.
 * \param[in] clientContext Context of the caller application. If not needed set it NULL.
 * \param[in] getAbsoluteGlobalTime A function pointer pointing to a function that returns the absolute global time.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPManagerSetTimeSource(
    NvMediaIPPManager *ippManager,
    void *clientContext,
    NvMediaIPPGetAbsoluteGlobalTime getAbsoluteGlobalTime
);
/*@} <!-- Ends ipp_manager_creation IPP Manager --> */

/**
 * \defgroup ipp_pipeline_creation IPP Pipeline
 * Defines IPP pipeline-related structures and functions.
 * @{
 */

/**
 * \brief Allocates an IPP pipeline object.
 * \param[in] ippManager The \ref NvMediaIPPManager.
 * \return \ref NvMediaIPPPipeline The new IPP pipeline's handle or NULL if unsuccessful.
 */
NvMediaIPPPipeline *
NvMediaIPPPipelineCreate(
    NvMediaIPPManager *ippManager
);

/**
 * \brief Destroys an IPP pipeline object.
 * \param[in] ippPipeline The IPP pipeline object to destroy.
 * \return void
 */
void
NvMediaIPPPipelineDestroy(
    NvMediaIPPPipeline *ippPipeline
);

/**
 * \brief Triggers the pipeline to do the single capture.
 * Triggers are queued if called multiple times.
 * Pipeline must be configured for trigger based capture.
 * \param[in] ippPipeline The IPP pipeline.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPPipelineSingleCapture(
    NvMediaIPPPipeline *ippPipeline
);

/**
 * \brief Defines event types.
 */
typedef enum {
    /** Event Info: End of file */
    NVMEDIA_IPP_EVENT_INFO_EOF,
    /** Event Info: Component processed one frame */
    NVMEDIA_IPP_EVENT_INFO_PROCESSING_DONE,
    /** Event Info: One frame captured */
    NVMEDIA_IPP_EVENT_INFO_FRAME_CAPTURE,
    /** Event Warning: Capture frame drop */
    NVMEDIA_IPP_EVENT_WARNING_CAPTURE_FRAME_DROP,
    /** Event Error: Internal failure */
    NVMEDIA_IPP_EVENT_ERROR_INTERNAL_FAILURE,
    /** Event Error: I2C transmission failure */
    NVMEDIA_IPP_EVENT_ERROR_I2C_TRANSMISSION_FAILURE,
    /** Event Warning: CSI frame discontinuity */
    NVMEDIA_IPP_EVENT_WARNING_CSI_FRAME_DISCONTINUITY,
    /** Event Error: CSI input stream error */
    NVMEDIA_IPP_EVENT_ERROR_CSI_INPUT_STREAM_FAILURE
} NvMediaIPPEventType;

/**
 * \brief Holds additional event information.
 */
typedef struct {
    /** Holds information about the image associated with the event. */
    NvMediaIPPImageInformation imageInformation;
    /**
     * Holds capture error information.
     * This field is populated only for following event type:
     * - \ref NVMEDIA_IPP_EVENT_ERROR_CSI_INPUT_STREAM_FAILURE
     */
    NvMediaICPErrorInfo captureErrorInfo;
} NvMediaIPPEventData;

/**
 * \brief Event callback function prototype. The client must create a function
 *  with the same function signature.
 * \param clientContext The client context that was passed to \ref NvMediaIPPManagerSetEventCallback.
 * \param componentType The component that is reporting the event
 * \param ippComponent A pointer to the IPP component handle.
 * \param eventType The event type
 * \param eventData Additional event information data
 */
typedef void NvMediaIPPEventCallback(
        void *clientContext,
        NvMediaIPPComponentType componentType,
        NvMediaIPPComponent *ippComponent,
        NvMediaIPPEventType eventType,
        NvMediaIPPEventData *eventData);

/**
 * \brief Sets a callback function for IPP events.
 * \param[in] ippManager The \ref NvMediaIPPManager.
 * \param[in] clientContext The context of the client. If not needed set it to NULL.
 * \param[in] eventCallback An event callback function pointer. This function is going
 * to be called when an event happens in the IPP pipeline.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPManagerSetEventCallback(
    NvMediaIPPManager *ippManager,
    void *clientContext,
    NvMediaIPPEventCallback eventCallback
);

/**
 * \brief Starts the IPP pipeline. This creates all the threads and starts all
 * attached components in a pipeline.
 * \param[in] ippPipeline The IPP pipeline to start.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPPipelineStart(
    NvMediaIPPPipeline *ippPipeline
);

/**
 * \brief Stops the IPP pipeline. This destroys all the threads in all components
 * in a pipeline.
 * \param[in] ippPipeline The IPP pipeline to stop.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPPipelineStop(
    NvMediaIPPPipeline *ippPipeline
);

/**
 * \brief Applies control properties to the pipeline
 * \param[in] ippPipeline The IPP pipeline
 * \param[in] controlProperties A client allocated and filled structure
 * of control properties
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPPipelineApplyControlProperties(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPPropertyControls *controlProperties
);

/**
 * \brief Gets static properties for the pipeline.
 * \param[in] ippPipeline The IPP pipeline
 * \param[out] staticProperties A client allocated structure
 *  to be filled with the static properties of the pipeline.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPPipelineGetStaticProperties(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPPropertyStatic *staticProperties
);

/**
 * \brief Gets default controls properties for the pipeline.
 * \param[in] ippPipeline The IPP pipeline
 * \param[out] defaultControlsProperties A client allocated structure
 *  to be filled with the default controls properties of the pipeline.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPPipelineGetDefaultControlsProperties(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPPropertyControls *defaultControlsProperties
);

/*@} <!-- Ends ipp_pipeline_creation IPP Pipeline --> */

/**
 * \defgroup ipp_plugin_algorithm IPP Control Algorithm Plugin
 * Defines IPP Control Algorithm Plugin related structures and functions.
 * @{
 */

/**
 * \brief Holds an opaque handle representing a Control Algorithm plugin.
 */
typedef void NvMediaIPPPlugin;

/**
 * Holds stream data for Control Algorithm plugin input.
 */
typedef struct
{
    /** Specify if the statistics are enabled or not */
    NvMediaBool enabled;
    union {
        struct {
            /** Histogram statistics */
            NvMediaISPStatsHistogramMeasurement *histogramStats[2];
            /** LAC statistics */
            NvMediaISPStatsLacMeasurementV4 *lacStats[2];
            /** Flicker band statistics */
            NvMediaISPStatsFlickerBandMeasurement *flickerBandStats;
        } v4;
        struct {
            /** Histogram statistics */
            NvMediaISPStatsHistogramMeasurementV5 *histogramStats[2];
            /** LAC statistics */
            NvMediaISPStatsLacMeasurementV5 *lacStats[2];
            /** Flicker band statistics */
            NvMediaISPStatsFlickerBandMeasurementV5 *flickerBandStats;
        } v5;
    };
} NvMediaIPPPluginInputStreamData;

/**
 * Holds Control Algorithm plugin input parameters.
 */
typedef struct
{
    /** Image information */
    NvMediaIPPImageInformation imageInfo;
    /** Controls properties that determine the operation mode of the plugin Control Algorithm */
    NvMediaIPPPropertyControls *controlsProperties;
    /** Exposure control values associated with the currently processed image */
    NvMediaISCExposureControl exposureControl;
    /** Holds the two set of white balance gain control values, one
     *  associated with sensor or before hdr merge and other associated
     *  with ISP after hdr merge */
    NvMediaISCWBGainControl whiteBalanceGainControl[2];
    /** Top embedded data of current image */
    NvMediaISCEmbeddedDataBuffer topEmbeddedData;
    /** Bottom embedded data of current image */
    NvMediaISCEmbeddedDataBuffer bottomEmbeddedData;
    /** An array of NvMediaIPPPluginInputStreamData. */
    NvMediaIPPPluginInputStreamData streamData[NVMEDIA_IPP_STREAM_MAX_TYPES];
    /** First run flag */
    NvMediaBool firstRun;
    /** Scene brightness */
     float_t brightness;
} NvMediaIPPPluginInput;

/**
 * Holds stream-specific settings of Control Algorithm plugin output.
 */
typedef struct
{
    union {
        struct {
            /** Histogram settings valid flags.
              * Set to NVMEDIA_TRUE if the settings are required */
            NvMediaBool histogramSettingsValid[2];
            /** Histogram settings.
              * Settings are applied if the histogramSettingsValid is set to NVMEDIA_TRUE */
            NvMediaISPStatsHistogramSettingsV4 histogramSettings[2];
            /** LAC settings valid flags.
              * Set to NVMEDIA_TRUE if the settings are required */
            NvMediaBool lacSettingsValid[2];
            /** LAC settings.
              * Settings are applied if the lacSettingsValid is set to NVMEDIA_TRUE */
            NvMediaISPStatsLacSettingsV4 lacSettings[2];
            /** Flicker band settings valid flags.
              * Set to NVMEDIA_TRUE if the settings are required */
            NvMediaBool flickerBandSettingsValid;
            /** Flicker band settings.
              * Settings are applied if flickerBandSettingsValid is set to NVMEDIA_TRUE */
            NvMediaISPStatsFlickerBandSettingsV4 flickerBandSettings;
        } v4;
        struct {
            /** Histogram settings valid flags.
              * Set to NVMEDIA_TRUE if the settings are required */
            NvMediaBool histogramSettingsValid[2];
            /** Histogram settings.
              * Settings are applied if the histogramSettingsValid is set to NVMEDIA_TRUE */
            NvMediaISPStatsHistogramSettingsV5 histogramSettings[2];
            /** LAC settings valid flags.
              * Set to NVMEDIA_TRUE if the settings are required */
            NvMediaBool lacSettingsValid[2];
            /** LAC settings.
              * Settings are applied if the lacSettingsValid is set to NVMEDIA_TRUE */
            NvMediaISPStatsLacSettingsV5 lacSettings[2];
            /** Flicker band settings valid flags.
              * Set to NVMEDIA_TRUE if the settings are required */
            NvMediaBool flickerBandSettingsValid;
            /** Flicker band settings.
              * Settings are applied if flickerBandSettingsValid is set to NVMEDIA_TRUE */
            NvMediaISPStatsFlickerBandSettingsV5 flickerBandSettings;
        } v5;
    };
} NvMediaIPPPluginOutputStreamSettings;

/**
 * Maximum numbers of exposure sets for bracketed exposure.
 */
#define NVMEDIA_IPP_MAX_EXPOSURE_SETS 8

/**
 * Holds the Control Algorithm plugin output parameters for bracketed exposure.
 */
typedef struct
{
    /** Holds flag to use either bracketed or continuos exposure */
    NvMediaBool useBracketedExp;
    /** Holds the auto exposure state. */
    NvMediaIPPAeState aeState;
    /** Holds the numbers of sets. */
    uint32_t numExposureControl;
    /** Holds the exposure control values calculated by the plugin Control Algorithm. */
    NvMediaISCExposureControl exposureControl[NVMEDIA_IPP_MAX_EXPOSURE_SETS];
    /** Holds the auto white balance state. */
    NvMediaIPPAwbState awbState;
    /** Holds the two set of white balance gain control values
     *  calculated by the plugin Control Algorithm, one to apply
     *  in sensor or before hdr merge and other to apply in ISP
     *  after hdr merge */
    NvMediaISCWBGainControl whiteBalanceGainControl[2];
    /** Holds a color correction matrix for use with sRGB output
     *  type. */
    NvMediaIPPMathFloatMatrix colorCorrectionMatrix;
    /** Holds a color correction matrix for use with rec2020 output
     *  type. */
    NvMediaIPPMathFloatMatrix colorCorrectionsMatrixRec2020;
    /** Holds ISP digital gain calculated by the plugin control
     *  algorithm */
    float_t ispDigitalGain;
    /** Holds an array of \ref NvMediaIPPPluginOutputStreamSettings. */
    NvMediaIPPPluginOutputStreamSettings streamSettings[NVMEDIA_IPP_STREAM_MAX_TYPES];
    /** Holds CCT estimated by Plugin  */
    float_t awbCCT;
} NvMediaIPPPluginOutputEx;

/**
 * \brief Gets the sensor attribute function prototype.
 * \param[in] parentControlAlgorithmHandle The handle that was passed during the create plugin call
 * \param[in] type Sensor attribute type.
 * \param[in] size Size of the attribute.
 * \param[out] attribute Sensor attribute value.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \li \ref NVMEDIA_STATUS_OK
 * \li \ref NVMEDIA_STATUS_BAD_PARAMETER if the input parameters are not valid.
 * \li \ref NVMEDIA_STATUS_NOT_SUPPORTED if the functionality is not supported or
 * the attribute is not supported by the device driver.
 * \li \ref NVMEDIA_STATUS_ERROR if other error occurred.
 */
typedef NvMediaStatus NvMediaIPPGetSensorAttr(
        NvMediaIPPComponent *parentControlAlgorithmHandle,
        NvMediaISCSensorAttrType type,
        uint32_t size,
        void *attribute);

/**
 * Holds the Control Algorithm plugin support functions descriptor.
 * This structure is filled by the
 * parent Control Algorithm. The plugin driver must make a copy of this structure and use
 * the function pointers from this structure to call the support functions.
 */
typedef struct
{
    /** Holds a function pointer to get the sensor attribute. */
    NvMediaIPPGetSensorAttr *getSensorAttribute;
} NvMediaIPPPluginSupportFuncs;

/**
 * \brief Plugin Control Algorithm Create callback function prototype. The client must
 *  create a function with the same function signature. This function is going to be called
 *  when the IPP Control Algorithm component is created.
 * \param[in] parentControlAlgorithmHandle A handle representing the parent Control Algorithm
 *  component. The plugin Control Algorithm must store this handle internally. This handle is
 *  needed to call any support function.
 * \param[in] supportFunctions A list of function pointers that the plugin Control Algorithm can
 *  call. This structure is filled by the parent Control Algorithm. The plugin driver must
 *  make a copy of this structure and use the function pointers to call the support functions.
 * \param[in] staticProperties Static properties associated with the camera.
 * \param[in] clientContext Client context passed in the Control Algorithm configuration structure.
 * \param[out] pluginHandle The plugin Control Algorithm's handle
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
typedef NvMediaStatus NvMediaIPPluginCreateFunc(
        NvMediaIPPComponent *parentControlAlgorithmHandle,
        NvMediaIPPPluginSupportFuncs *supportFunctions,
        NvMediaIPPPropertyStatic *staticProperties,
        void *clientContext,
        NvMediaIPPPlugin **pluginHandle,
        NvMediaIPPISPVersion ispVersion);

/**
 * \brief Plugin Control Algorithm Destroy callback function prototype. The client must
 * create a function with the same function signature. This function is going to be called
 * when the IPP Control Algorithm component is destroyed.
 * \param[in] pluginHandle The plugin Control Algorithm's handle
 * \return void.
 */
typedef void NvMediaIPPPluginDestroyFunc(
        NvMediaIPPPlugin *pluginHandle);

/**
 * \brief Defines the Plugin Control Algorithm Process callback function prototype for bracketed
 * exposure. The client must create a function with the same function signature.
 * This function is called when the IPP Control Algorithm component is
 * processing the statistics information for an image.
 * \param[in] pluginHandle The plugin Control Algorithm handle.
 * \param[in] pluginInput The input parameters for plugin Control Algorithm.
 * \param[out] pluginOutput The output parameters that the plugin Control Algorithm
 *  is to generate.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
typedef NvMediaStatus NvMediaIPPPluginProcessExFunc(
        NvMediaIPPPlugin *pluginHandle,
        NvMediaIPPPluginInput *pluginInput,
        NvMediaIPPPluginOutputEx *pluginOutput);
/**
 * Holds the Control Algorithm plugin functions descriptor. This structure must be filled by the client
 * and passed to the Control Algorithm component as a configuration parameter.
 */
typedef struct
{
    /** Plugin Control Algorithm Create function pointer */
    NvMediaIPPluginCreateFunc *createFunc;
    /** Plugin Control Algorithm Destroy function pointer */
    NvMediaIPPPluginDestroyFunc *destroyFunc;
    /** Plugin Control Algorithm Process function pointer */
    NvMediaIPPPluginProcessExFunc *processExFunc;
} NvMediaIPPPluginFuncs;

/*@} <!-- Ends ipp_plugin_algorithm IPP Control Algorithm Plugin --> */

/**
 * \defgroup ipp_component IPP Component
 * Defines IPP component-related structures and functions.
 * @{
 */

/**
 * \hideinitializer
 * \brief Defines IPP port types.
 */
typedef enum {
    /** \hideinitializer Image port 1 */
    NVMEDIA_IPP_PORT_IMAGE_1,
    /** Image port 2 */
    NVMEDIA_IPP_PORT_IMAGE_2,
    /** Statistics port 1 */
    NVMEDIA_IPP_PORT_STATS_1,
    /** Sensor control port 1 */
    NVMEDIA_IPP_PORT_SENSOR_CONTROL_1,
    /** Capture port of aggregate images */
    NVMEDIA_IPP_PORT_IMAGE_CAPTURE_AGGREGATE
} NvMediaIPPPortType;

/**
 * \brief Holds new buffer pool parameters for initializing an IPP component.
 *
 */

typedef struct {
    /*! Port type associated with the pool */
    NvMediaIPPPortType portType;
    /*! Number of pool buffer elements */
    uint32_t poolBuffersNum;
    /*! Image width */
    uint32_t width;
    /*! Image height */
    uint32_t height;
    /*! Image surface type */
    NvMediaSurfaceType surfaceType;
    /*! Image surface allocation attributes (\ref NvMediaSurfAllocAttr) */
    NvMediaSurfAllocAttr surfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
    /*! number of surface allocation attributes */
    uint32_t numSurfAllocAttrs;
    /*! Images count */
    uint32_t imagesCount;
} NvMediaIPPBufferPoolParamsNew;

/**
 * \brief Holds image group buffer pool parameters for initializing an IPP component.
 *
 */
typedef struct {
    /*! Port type associated with the pool */
    NvMediaIPPPortType portType;
    /*! Number of pool buffer elements */
    uint32_t poolBuffersNum;
    /*! An array of surface type & allocation attributes for allocating \ref NvMediaImageGroup */
    struct {
        /*! Image surface type */
        NvMediaSurfaceType surfaceType;
        /*! Image surface allocation attributes (\ref NvMediaSurfAllocAttr) */
        NvMediaSurfAllocAttr surfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
        /*! Number of surface allocation attributes */
        uint32_t numSurfAllocAttrs;
        /*! Boolean to specify if the top embedded is valid. */
        NvMediaBool topEmbeddedDataValid;
        /*! Boolean to specify if the bottom embedded is valid. */
        NvMediaBool bottomEmbeddedDataValid;
    } imageConfig[NVMEDIA_MAX_IMAGE_GROUP_SIZE];
    /*! Number of image configs */
    uint32_t numImageConfigs;
} NvMediaIPPBufferPoolParamsImgGrp;

/**
 * \brief Holds configuration for an ICP component.
 */
typedef struct {
    /*! Holds the capture settings. */
    NvMediaICPSettings *settings;
    /*! Holds the sibling images per captured frame.
     *  A value of 0 indicates the non-aggregated case. */
    uint32_t siblingsNum;
} NvMediaIPPIcpComponentConfig;

/**
 * \brief File-reader image-read callback prototype.
 *  The client must create a function with the same function
 *  signature. The IPP File Reader component calls this function
 *  when it is ready to read an image into the framework.
 *  The component passes a pointer to an image that holds the
 *  entire image/frame.
 * \param[in] clientContext   Pointer to the client's context.
 * \param[in] imageGroup   Pointer to the image group that will be read
 *       in the function.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
typedef NvMediaStatus (*NvMediaIPPImgGrpReadCallback)(
        void *clientContext,
        NvMediaImageGroup *imageGroup
);

/**
 * \brief Holds configuration information for a file reader component.
 */
typedef struct {
    /*! Holds the context of the client, if not set to NULL. */
    void *clientContext;
    /*! Holds a callback for accepting an image from NvMedia IPP client. */
    NvMediaIPPImgGrpReadCallback imageGroupReadCallback;
    /*! Holds the sibling images per frame.
     *  A value of 0 indicates the non-aggregated case. */
    uint32_t siblingsNum;
} NvMediaIPPFileReaderComponentConfig;

/**
 * \brief Holds configuration information for an ISC component.
 */
typedef struct {
    NvMediaISCDevice           *iscSensorDevice;
} NvMediaIPPIscComponentConfig;

/**
 * \hideinitializer
 * \brief Defines IPP ISP attribute flags.
 */
typedef enum {
    /*! Indicates a NON-HDR ISP pipeline must be set in the ISP component. */
    NVMEDIA_IPP_ISP_MODE_NONHDR = (1 << 0),
    /*! Indicates the single ISP pipeline mode is enabled. */
    NVMEDIA_IPP_ISP_SINGLE_PIPELINE_MODE = (1 << 1),
    /*! Indicates the second ISP output mode 1 is selected */
    NVMEDIA_IPP_ISP_OUTPUT2_MODE_1 = (1 << 3),
    /*! Indicates the second ISP output mode 2 is selected */
    NVMEDIA_IPP_ISP_OUTPUT2_MODE_2 = (1 << 4)
} NvMediaIPPIspAttrFlags;

/**
 * \brief Holds configuration information for an ISP component.
 */
typedef struct {
    /*! ISP select */
    NvMediaISPSelect ispSelect;
    /*! Holds the ISP-setting attribute flag, which specifies bit-wise OR`ed flags
        defined NvMediaIPPIspAttrFlags enum. */
    uint32_t ispSettingAttr;
} NvMediaIPPIspComponentConfig;

/**
 * \brief Holds configuration information for a Control Algorithm component.
 */
typedef struct {
    /*! Image width */
    uint32_t width;
    /*! Image height */
    uint32_t height;
    /*! Image raw pixel order */
    NvMediaRawPixelOrder pixelOrder;
    /*! Image bits per pixel */
    NvMediaBitsPerPixel bitsPerPixel;
    /*! Plugin Control Algorithm functions. Set to NULL if no plugin is required */
    NvMediaIPPPluginFuncs *pluginFuncs;
    /* Client context passed to plugin Control Algorithm */
    void *clientContext;
    /*! ISC sensor device handle to be used to get sensor properties */
    NvMediaISCDevice *iscSensorDevice;
} NvMediaIPPControlAlgorithmComponentConfig;

/**
 * \brief Defines metadata types.
 */
typedef enum {
    /*! Image information. Data corresponds to \ref NvMediaIPPImageInformation. */
    NVMEDIA_IPP_METADATA_IMAGE_INFO,
    /*! Control properties. Data corresponds to \ref NvMediaIPPPropertyControls. */
    NVMEDIA_IPP_METADATA_CONTROL_PROPERTIES,
    /*! Dynamic properties. Data corresponds to \ref NvMediaIPPPropertyDynamic */
    NVMEDIA_IPP_METADATA_DYNAMIC_PROPERTIES,
    /*! Embedded data. Data corresponds to \ref NvMediaISCEmbeddedData.
        The top and bottom embedded lines will not provided. Use
        \ref NVMEDIA_IPP_METADATA_EMBEDDED_DATA_TOP and
        \ref NVMEDIA_IPP_METADATA_EMBEDDED_DATA_BOTTOM */
    NVMEDIA_IPP_METADATA_EMBEDDED_DATA_ISC,
    /*! The top embedded line whose size and base register are defined by
       size and baseRegAddress of \ref NvMediaISCEmbeddedDataBuffer.  */
    NVMEDIA_IPP_METADATA_EMBEDDED_DATA_TOP,
    /*! The bottom embedded line whose size and base register are defined by
       size and baseRegAddress of \ref NvMediaISCEmbeddedDataBuffer.  */
    NVMEDIA_IPP_METADATA_EMBEDDED_DATA_BOTTOM,
    /*! Local Tone Map data. */
    NVMEDIA_IPP_METADATA_LTM_DATA,
    /*! Number of metadata types */
    NVMEDIA_IPP_METADATA_MAX_TYPES
} NvMediaIPPMetadataType;

/**
 * \brief Gets the size of the specified metadata type.
 * \param[in] metadata The buffer holding metadata.
 * \param[in] type Type of the requested metadata.
 * \return uint32_t. The size.
 */
uint32_t
NvMediaIPPMetadataGetSize(
    void *metadata,
    NvMediaIPPMetadataType type
);

/**
 * \brief Gets the data of the specified metadata type.
 * \param[in] metadata The buffer holding the metadata.
 * \param[in] type Type of the requested metadata.
 * \param[out] buffer Destionation buffer.
 * \param[in] size The size of the requested type.
 * \return \ref NvMediaStatus. \ref NVMEDIA_STATUS_OK or \ref NVMEDIA_STATUS_ERROR
 * if the size does not match the returned size from
 * \ref NvMediaIPPMetadataGetSize().
 */
NvMediaStatus
NvMediaIPPMetadataGet(
    void *metadata,
    NvMediaIPPMetadataType type,
    void *buffer,
    uint32_t size);

/**
 * \brief Gets the address of the data of the specified metadata type.
 * \param[in] metadata The buffer holding the metadata.
 * \param[in] type Type of the requested metadata.
 * \param[out] buffer The pointer inside the metadata. This pointer is filled by
 *  this function.
 * \param[out] size Pointer to the size of the requested type filled by this
 * function.
 * \return \ref NvMediaStatus. \ref NVMEDIA_STATUS_OK or \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPMetadataGetAddress(
    void *metadata,
    NvMediaIPPMetadataType type,
    void **buffer,
    uint32_t *size);

/**
 * \brief Creates an IPP component object.
 * \param[in] ippPipeline The NvMedia IPP pipeline the component belongs to.
 * \param[in] componentType Type of the component. Currently the following types are supported:
 *  - \ref NVMEDIA_IPP_COMPONENT_ICP
 *  - \ref NVMEDIA_IPP_COMPONENT_ICP_EX
 *  - \ref NVMEDIA_IPP_COMPONENT_ISP
 *  - \ref NVMEDIA_IPP_COMPONENT_ALG
 *  - \ref NVMEDIA_IPP_COMPONENT_ISC
 *  - \ref NVMEDIA_IPP_COMPONENT_OUTPUT
 *  - \ref NVMEDIA_IPP_COMPONENT_FILE_READER
 * \param[in] bufferPools A NULL terminated list of pointers ponting to new buffer pool parameters.
 * \param[in] componentConfig Component specific configuration.
 * \return \ref NvMediaIPPComponent The new IPP component's handle or NULL if unsuccessful.
 */
NvMediaIPPComponent *
NvMediaIPPComponentCreateNew(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPComponentType componentType,
    NvMediaIPPBufferPoolParamsNew **bufferPools,
    void *componentConfig
);

/**
 * \brief Creates an IPP component object.
 * \param[in] ippPipeline The NvMedia IPP pipeline the component belongs to.
 * \param[in] componentType Type of the component. Currently the following types are supported:
 *  - \ref NVMEDIA_IPP_COMPONENT_ICP_EX
 *  - \ref NVMEDIA_IPP_COMPONENT_FILE_READER
 * \param[in] bufferPools A NULL terminated list of pointers ponting to image group buffer pool parameters.
 * \param[in] componentConfig Component specific configuration.
 * \return \ref NvMediaIPPComponent The new IPP component's handle or NULL if unsuccessful.
 */
NvMediaIPPComponent *
NvMediaIPPComponentCreateImgGrp(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPComponentType componentType,
    NvMediaIPPBufferPoolParamsImgGrp **bufferPools,
    void *componentConfig
);

/**
 * \brief Adds an IPP component to pipeline.
 * \param[in] ippPipeline The NvMedia IPP pipeline the component will be added to.
 * \param[in] ippComponent The IPP component's handle.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPComponentAddToPipeline(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPComponent *ippComponent);

/**
 * \brief Attaches an IPP source component to a destination component.
 * \param[in] ippPipeline The NvMedia IPP pipeline the components belongs to.
 * \param[in] srcComponent Source component.
 * \param[in] dstComponent Destination component.
 * \param[in] portType Specifies which port of source component is attached to the destination.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPComponentAttach(
    NvMediaIPPPipeline *ippPipeline,
    NvMediaIPPComponent *srcComponent,
    NvMediaIPPComponent *dstComponent,
    NvMediaIPPPortType portType
);

/**
 * \brief Holds a handle representing an IPP component output object.
 */
typedef struct {
    NvMediaImage *image;
    /** Metadata bufffer */
    void *metadata;
    /** Metadata size */
    uint32_t metadataSize;
} NvMediaIPPComponentOutput;

/**
 * \brief Gets output from a component. Only the \ref NVMEDIA_IPP_COMPONENT_OUTPUT
 * supports this functionaly.
 * \param[in] component Component handle
 * \param[in] millisecondTimeout Time-out in milliseconds.
 * Use \ref NVMEDIA_IMAGE_TIMEOUT_INFINITE for infinite timeout.
 * \param[out] output Output structure filed by the component.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_TIMED_OUT If the output is not received within millisecondTimeout time
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPComponentGetOutput(
    NvMediaIPPComponent *component,
    uint32_t millisecondTimeout,
    NvMediaIPPComponentOutput *output
);

/**
 * \brief Returns output to a component. Only the \ref NVMEDIA_IPP_COMPONENT_OUTPUT
 * supports this functionaly. This function must be called for each output structure
 * received by \ref NvMediaIPPComponentGetOutput.
 * \param[in] component Component handle.
 * \param[in] output Output structure to be returned.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPComponentReturnOutput(
    NvMediaIPPComponent *component,
    NvMediaIPPComponentOutput *output
);

/**
 * \brief Holds a handle representing an IPP component output object.
 */
typedef struct {
    NvMediaImageGroup imageGroup;
    /** Metadata bufffer */
    void *metadata;
    /** Metadata size */
    uint32_t metadataSize;
} NvMediaIPPComponentOutputImgGrp;

/**
 * \brief Gets image group output from a component. Only the \ref NVMEDIA_IPP_COMPONENT_OUTPUT
 * supports this functionaly.
 * \param[in] component Component handle
 * \param[in] millisecondTimeout Time-out in milliseconds.
 * Use \ref NVMEDIA_IMAGE_TIMEOUT_INFINITE for infinite timeout.
 * \param[out] output Output structure filed by the component.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_TIMED_OUT If the output is not received within millisecondTimeout time
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPComponentGetOutputImgGrp(
    NvMediaIPPComponent *component,
    uint32_t millisecondTimeout,
    NvMediaIPPComponentOutputImgGrp *output
);

/**
 * \brief Returns image group output to a component. Only the \ref NVMEDIA_IPP_COMPONENT_OUTPUT
 * supports this functionaly. This function must be called for each output structure
 * received by \ref NvMediaIPPComponentGetOutput.
 * \param[in] component Component handle.
 * \param[in] output Output structure to be returned.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIPPComponentReturnOutputImgGrp(
    NvMediaIPPComponent *component,
    NvMediaIPPComponentOutputImgGrp *output
);

/**
 * \brief Destroys an IPP component object.
 * \param[in] ippComponent The IPP component object to destroy.
 * \return void
 */
void
NvMediaIPPComponentDestroy(
    NvMediaIPPComponent *ippComponent
);
/** @} <!-- Ends ipp_component IPP Component --> */

/*
 * \defgroup history_ipp History
 * Provides change history for the NvMedia Image Processing Pipeline API.
 *
 * \section history_ipp Version History
 *
 * <b> Version 1.0 </b> July 8, 2014
 * - Initial release
 *
 * <b> Version 1.1 </b> November 4, 2014
 * - Added IPP control, static and dynamic properties
 *
 * <b> Version 1.2 </b> January 15, 2015
 * - Added fileLoopBackCount to FileReader ComponentConfig
 *
 * <b> Version 1.3 </b> January 16, 2015
 * - Added plugin Control Algorithm related functions and structures
 * - Renamed Camera Driver to Control Algorithm
 * - Added version information to IPP manager create API
 *
 * <b> Version 1.4 </b> January 28, 2015
 * - Added metadata to NvMediaIPPComponentOutput
 *
 * <b> Version 1.5 </b> February 27, 2015
 * - Fixed a typo in NvMediaIPPCompandingControl
 * - Changed pointers in NvMediaIPPPluginOutput to the actual types.
 *
 * <b> Version 1.6 </b> March 25, 2015
 * - Removed list of sensor modes from NvMediaIPPPropertyStatic
 *
 * <b> Version 1.7 </b> March 26, 2015
 * - Added global time-stamp to info strcuture
 * - Changed the \ref NvMediaIPPGetAbsoluteGlobalTime to use 64-bit
 *   microsecond based time-stamp.
 *
 * <b> Version 1.8 </b> March 30, 2015
 * - Added NVMEDIA_IPP_PORT_IMAGE_CAPTURE_X capture ports
 *
 * <b> Version 1.9 </b> April 14, 2015
 * - Added \ref NvMediaIPPStreamType.
 * - Changed \ref NvMediaIPPPluginInput to have separate stats data for
 *   different streams.
 * - Changed \ref NvMediaIPPPluginOutput to have separate settings for different
 *   streams.
 *
 * <b> Version 1.10 </b> April 23, 2015
 * - Added reserved member to \ref NvMediaIPPLensShadingControl structure to resolve
 *   C and C++ structure size differences.
 *
 * <b> Version 1.11 </b> April 29, 2015
 * - Removed dynamicProperties from \ref NvMediaIPPPluginOutput structure
 * - Added aeLock, aeState, awbLock & awbState in \ref NvMediaIPPPluginOutput structure
 *
 * <b> Version 1.12 </b> May 26, 2015
 * - Updated \ref NvMediaIPPEventCallback callback API
 *
 * <b> Version 1.13 </b> June 16, 2015
 * - Added port type NVMEDIA_IPP_PORT_IMAGE_CAPTURE_AGGREGATE
 * - Added metadataFileName to \ref NvMediaIPPFileWriterComponentConfig
 *
 * <b> Version 1.14 </b> July 17, 2015
 * - Added iscSensorDevice to \ref NvMediaIPPFileReaderComponentConfig,
 *   \ref NvMediaIPPIcpComponentConfig and \ref NvMediaIPPControlAlgorithmComponentConfig.
 *
 * <b> Version 1.15 </b> September 4, 2015
 * - Added sensorMode to \ref NvMediaIPPExposureControl
 *
 * <b> Version 1.16 </b> September 4, 2015
 * - Added NVMEDIA_IPP_EVENT_WARNING_CAPTURE_RECOVERY to \ref NvMediaIPPEventType.
 *
 * <b> Version 1.17 </b> December 8, 2015
 * - Removed ISP control functions.
 * - Added ISP version 4 support.
 *
 * <b> Version 1.18 </b> January 7, 2016
 * - Added ChannelGainRatio (Gain ratio between exposure channels).
 *
 * <b> Version 1.19 </b> January 27, 2016
 * - Added \ref NvMediaIPPPropertyStatic to \ref NvMediaIPPPluginInput.
 *
 * <b> Version 1.20 </b> Jan 27, 2016
 * - Added firstRun flag to to \ref NvMediaIPPPluginInput.
 *
 * <b> Version 1.21 </b> March 11, 2016
 * - Increased maximum IPP pipelines in IPP manager to 12.
 *
 * <b> Version 1.22 </b> March 28, 2016
 * - Added NVMEDIA_IPP_EVENT_ERROR_I2C_TRANSMISSION_FAILURE event type.
 *
 * <b> Version 1.23 </b> March 29, 2016
 * - Added \ref NVMEDIA_IPP_MAX_LTM_DATA_SIZE define for max LTM data size.
 *
 * <b> Version 1.24 </b> May 3, 2016
 * - Added \ref NvMediaIPPIspAttrFlags to \ref NvMediaIPPIspComponentConfig.
 *
 * <b> Version 1.25 </b> May 4, 2016
 * - Added ModuleConfig to static properties
 *
 * <b> Version 1.26 </b> May 11, 2016
 * - Changed \ref NvMediaIPPImageInformation frameSequenceNumber type to unsigned int
 *
 * <b> Version 1.27 </b> May 23, 2016
 * - Added NVMEDIA_IPP_EVENT_WARNING_CSI_DISCONTINUITY to \ref NvMediaIPPEventType.
 *
 * <b> Version 1.28 </b> June 16, 2016
 * - Added NVMEDIA_IPP_COMPONENT_CAPTURE_EX component for virtual channels capture support.
 *
 * <b> Version 1.29 </b> June 24, 2016
 * - Added \ref NvMediaIPPPipelineSetProperties new API to set pipeline properties.
 *
 * <b> Version 1.30 </b> June 29, 2016
 * - Added \ref NvMediaIPPPluginProcessExFunc new plugin process function for bracketed exposure.
 *
 * <b> Version 1.31 </b> July 18, 2016
 * - Added support for event based capture \ref NvMediaIPPPipelineSingleCapture.
 *
 * <b> Version 1.32 </b> September 12, 2016
 * - Added \ref NVMEDIA_IPP_PIPELINE_PROPERTY_SETTINGS_DELAY property to support
 *   sensor without embedded lines
 *
 * <b> Version 1.33 </b> October 10, 2016
 * - Added new support function for getting sensor attributes \ref NvMediaIPPGetSensorAttr
 *
 * <b> Version 1.34 </b> Febraury 3, 2017
 * - Added \ref NVMEDIA_IPP_METADATA_EMBEDDED_DATA_ISC for ISC embedded data inside
 *   \ref NvMediaIPPMetadataType
 *
 * <b> Version 1.35 </b> March 10, 2017
 * - Added capture error info in \ref NvMediaIPPEventData
 * - Added new event type for CSI input stream error in \ref NvMediaIPPEventType
 *
 * <b> Version 1.36 </b> March 31, 2017
 * - Removed ISP stats version 3 support from \ref NvMediaIPPPluginOutputStreamSettings
 *   and \ref NvMediaIPPPluginInputStreamData.
 *
 * <b> Version 1.37 </b> April 13, 2017
 * - Fixed violation MISRA-C rule 10.1 in NVMEDIA_IPP_VERSION_INFO macro.
 *
 * <b> Version 2.00 </b> April 27, 2017
 * - Removed PRE_PROCESSING, POST_PROCESSING & DISPLAY components
 * - Removed ipaDevice argument from \ref NvMediaIPPManagerCreate
 * - Removed iscSensorDevice & registerImageBuffersWithIPA from CAPTURE & FILE_READER
     component config
 * - Removed iscRootDevice, iscAggregatorDevice & iscSerializerDevice from ISC component config
 * - Removed ispSettingsFile & registerImageBuffersWithIPA from ISP component config
 * - Removed logging callback \ref NvMediaIPPPluginSupportFuncs
 * - Removed parseConfigurationFunc from \ref NvMediaIPPPluginFuncs
 * - Removed IPP session related APIs. Use \ref NvMediaIPPPipelineApplyControlProperties
     to apply control properties
 * - Removed unused data structures & enums
 * - Removed captureIntent, aeExposureCompensation, requestId from \ref NvMediaIPPPropertyControls
 * - Removed exposureTimeRange from \ref NvMediaIPPPropertyControls use GetSensorAttr API
     to get the exposure time range
 * - Removed companding control from \ref NvMediaIPPPluginInput
 * - Removed lens shading from \ref NvMediaIPPPluginOutput & \ref NvMediaIPPPluginOutputEx
 * - Removed ISP stats version 3 support from \ref NvMediaIPPPluginOutputStreamSettings
     and \ref NvMediaIPPPluginInputStreamData.
 *
 * <b> Version 2.01 </b> May 5, 2017
 * - Removed NvMediaICPInterfaceFormat from \ref NvMediaIPPIcpComponentConfig
 *
 * <b> Version 2.02 </b> April 27, 2017
 * - Replaced NvMediaIPPExposureControl with NvMediaISCExposureControl
     - ISP digital gain is added in impacted plugin structures
     - Sensor mode & hdrRatio have been removed
 * - Replaced NvMediaIPPWBGainControl use NvMediaISCWBGainControl
 * - Removed NvMediaIPPSensorExposureMode use NvMediaISCExposureMode
 * - Removed notion of human & machine vision streams
 * - Removed NvMediaIPPCameraSensorMode
     - ActiveArraySize is added in NvMediaIPPPropertyStatic
     - For frame rate use GetSensorAttr API
     - Suface type have been removed
 * - Removed AE & AWB lock flags
 * - Removed valid flag for color correction
 * - Removed requestId from NvMediaIPPPropertyDynamic
 * - Removed sensorCFA from NvMediaIPPPropertyStatic
 * - Removed exposure time range & sensor analog gain range &
     channelGainRaio from NvMediaIPPPropertyStatic use GetSensorAttr API
 *
 * <b> Version 2.03 </b> May 15, 2017
 * - Added deprecated warning message for \ref NvMediaIPPComponentCreate
 *
 * <b> Version 2.04 </b> May 23, 2017
 * - Replaced NvMediaIPPEmbeddedDataInformation with \ref NvMediaISCEmbeddedDataBuffer
 * - Removed frameCaptureTimeStamp from NvMediaIPPImageInformation
 * - Removed AE & AWB modes and replaced with NvMediaBool
 * - Removed unused event types.
 * - Removed NVMEDIA_IPP_PORT_IMAGE_CAPTURE_* ports use Image ports
 * - Renamed NVMEDIA_IPP_COMPONENT_CAPTURE to NVMEDIA_IPP_COMPONENT_ICP
 * - Renamed NVMEDIA_IPP_COMPONENT_CONTROL_ALGORITHM to NVMEDIA_IPP_COMPONENT_ALG
 * - Renamed NVMEDIA_IPP_COMPONENT_SENSOR_CONTROL to NVMEDIA_IPP_COMPONENT_ISC
 * - Renamed NVMEDIA_IPP_COMPONENT_CAPTURE_EX to NVMEDIA_IPP_COMPONENT_ICP
 * - Changed \ref NvMediaIPPPluginInputStreamData & \ref NvMediaIPPPluginOutputStreamSettings
     to have union for all stats
 *
 * <b> Version 2.05 </b> May 17, 2017
 * - Added companding params in \ref NvMediaIPPPropertyStatic
 * - Added brightness value in \ref NvMediaIPPPluginInputEx
 * - Added awbCCT value in \ref NvMediaIPPPluginOutputEx
 * - Renamed CurrentSceneLux to brightness in \ref NvMediaIPPPropertyDynamic
 * - Changed whiteBalanceGainControl to be an array of two in \ref NvMediaIPPPluginOutputEx
 * - Removed \ref NvMediaIPPPluginOutput struct and processFunc in \ref NvMediaIPPPluginFuncs
 * - Added useBracketedExp flag in \ref NvMediaIPPPluginInputEx
 *
 * <b> Version 2.06 </b> June 27, 2017
 * - Added file read callback functions
 *
 * <b> Version 2.07 </b> July 28, 2017
 * - Removed deprecated file writer component
 *
 * <b> Version 2.08 </b> Aug 17, 2017
 * - Add ISP Version 5 Stats structs to Plugin Interface
 * - Update NvMediaIPPISPVersion enum to add NVMEDIA_IPP_ISP_VERSION_5
 *
 * <b> Version 2.09 </b> August 25, 2017
 * - Add new APIs to support \ref NvMediaImageGroup.
 *
 * <b> Version 2.10 </b> September 05, 2017
 * - Add colorCorrectionsMatrixRec2020 in \ref NvMediaIPPPluginOutputEx.
 *
 * <b> Version 2.11 </b> September 07, 2017
 * - Add new attribute flags in \ref NvMediaIPPIspAttrFlags.
 *
 * <b> Version 2.12 </b> September 12, 2017
 * - Deprecated \ref NvMediaIPPBufferPoolParams, \ref NvMediaIPPComponentCreate
 *
 */
/*@}*/

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _NVMEDIA_IPP_H */
