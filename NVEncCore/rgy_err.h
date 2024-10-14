// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_ERR_H__
#define __RGY_ERR_H__

#include "rgy_version.h"
#include "rgy_tchar.h"
#if ENCODER_QSV
#pragma warning (push)
#pragma warning (disable: 4201) //C4201: 非標準の拡張機能が使用されています: 無名の構造体または共用体です。
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include "mfxdefs.h"
#pragma warning(pop)
#endif
#if ENCODER_NVENC || CUFILTERS
#pragma warning (push)
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#if ENCODER_NVENC
#include "nvEncodeAPI.h"
#endif // #if ENCODER_NVENC
#include "cuda.h"
#include "cuda_runtime.h"
#if ENABLE_NVVFX
#include "nvCVStatus.h"
#endif //#if ENABLE_NVVFX
#include "nppdefs.h"
#pragma warning(pop)
#endif
#if ENCODER_VCEENC
#include "core/Result.h"
#endif
#if ENABLE_VULKAN
#include "rgy_vulkan.h"
#endif
#if ENCODER_MPP
#include "mpp_err.h"
#include "rga/im2d.hpp"
#endif //#if ENCODER_MPP

#include <algorithm>

enum RGY_ERR {
    RGY_ERR_NONE                        = 0,
    RGY_ERR_UNKNOWN                     = -1,
    RGY_ERR_NULL_PTR                    = -2,
    RGY_ERR_UNSUPPORTED                 = -3,
    RGY_ERR_MEMORY_ALLOC                = -4,
    RGY_ERR_NOT_ENOUGH_BUFFER           = -5,
    RGY_ERR_INVALID_HANDLE              = -6,
    RGY_ERR_LOCK_MEMORY                 = -7,
    RGY_ERR_NOT_INITIALIZED             = -8,
    RGY_ERR_NOT_FOUND                   = -9,
    RGY_ERR_MORE_DATA                   = -10,
    RGY_ERR_MORE_SURFACE                = -11,
    RGY_ERR_ABORTED                     = -12,
    RGY_ERR_DEVICE_LOST                 = -13,
    RGY_ERR_INCOMPATIBLE_VIDEO_PARAM    = -14,
    RGY_ERR_INVALID_VIDEO_PARAM         = -15,
    RGY_ERR_UNDEFINED_BEHAVIOR          = -16,
    RGY_ERR_DEVICE_FAILED               = -17,
    RGY_ERR_MORE_BITSTREAM              = -18,
    RGY_ERR_INCOMPATIBLE_AUDIO_PARAM    = -19,
    RGY_ERR_INVALID_AUDIO_PARAM         = -20,
    RGY_ERR_GPU_HANG                    = -21,
    RGY_ERR_REALLOC_SURFACE             = -22,
    RGY_ERR_ACCESS_DENIED               = -23,
    RGY_ERR_INVALID_PARAM               = -24,
    RGY_ERR_OUT_OF_RANGE                = -25,
    RGY_ERR_ALREADY_INITIALIZED         = -26,
    RGY_ERR_INVALID_FORMAT              = -27,
    RGY_ERR_WRONG_STATE                 = -28,
    RGY_ERR_FILE_OPEN                   = -29,
    RGY_ERR_INPUT_FULL                  = -30,
    RGY_ERR_INVALID_CODEC               = -31,
    RGY_ERR_INVALID_DATA_TYPE           = -32,
    RGY_ERR_INVALID_RESOLUTION          = -33,
    RGY_ERR_NO_DEVICE                   = -34,
    RGY_ERR_INVALID_DEVICE              = -35,
    RGY_ERR_INVALID_CALL                = -36,
    RGY_ERR_INVALID_VERSION             = -37,
    RGY_ERR_MAP_FAILED                  = -38,
    RGY_ERR_CUDA                        = -39,
    RGY_ERR_RUN_PROCESS                 = -40,
    RGY_ERR_NONE_PARTIAL_OUTPUT         = -41,

    //OpenCL
    RGY_ERR_INVALID_PLATFORM                = -50,
    RGY_ERR_INVALID_DEVICE_TYPE             = -51,
    RGY_ERR_INVALID_CONTEXT                 = -52,
    RGY_ERR_INVALID_QUEUE_PROPERTIES        = -53,
    RGY_ERR_INVALID_COMMAND_QUEUE           = -54,
    RGY_ERR_DEVICE_NOT_FOUND                = -55,
    RGY_ERR_DEVICE_NOT_AVAILABLE            = -56,
    RGY_ERR_COMPILER_NOT_AVAILABLE          = -57,
    RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE   = -58,
    RGY_ERR_OUT_OF_RESOURCES                = -59,
    RGY_ERR_OUT_OF_HOST_MEMORY              = -60,
    RGY_ERR_PROFILING_INFO_NOT_AVAILABLE    = -61,
    RGY_ERR_MEM_COPY_OVERLAP                = -62,
    RGY_ERR_IMAGE_FORMAT_MISMATCH           = -63,
    RGY_ERR_IMAGE_FORMAT_NOT_SUPPORTED      = -64,
    RGY_ERR_BUILD_PROGRAM_FAILURE           = -65,
    RGY_ERR_MAP_FAILURE                     = -66,

    RGY_ERR_INVALID_HOST_PTR                = -68,
    RGY_ERR_INVALID_MEM_OBJECT              = -69,
    RGY_ERR_INVALID_IMAGE_FORMAT_DESCRIPTOR = -70,
    RGY_ERR_INVALID_IMAGE_SIZE              = -71,
    RGY_ERR_INVALID_SAMPLER                 = -72,
    RGY_ERR_INVALID_BINARY                  = -73,
    RGY_ERR_INVALID_BUILD_OPTIONS           = -74,
    RGY_ERR_INVALID_PROGRAM                 = -75,
    RGY_ERR_INVALID_PROGRAM_EXECUTABLE      = -76,
    RGY_ERR_INVALID_KERNEL_NAME             = -77,
    RGY_ERR_INVALID_KERNEL_DEFINITION       = -78,
    RGY_ERR_INVALID_KERNEL                  = -79,
    RGY_ERR_INVALID_ARG_INDEX               = -80,
    RGY_ERR_INVALID_ARG_VALUE               = -81,
    RGY_ERR_INVALID_ARG_SIZE                = -82,
    RGY_ERR_INVALID_KERNEL_ARGS             = -83,
    RGY_ERR_INVALID_WORK_DIMENSION          = -84,
    RGY_ERR_INVALID_WORK_GROUP_SIZE         = -85,
    RGY_ERR_INVALID_WORK_ITEM_SIZE          = -86,
    RGY_ERR_INVALID_GLOBAL_OFFSET           = -87,
    RGY_ERR_INVALID_EVENT_WAIT_LIST         = -88,
    RGY_ERR_INVALID_EVENT                   = -89,
    RGY_ERR_INVALID_OPERATION               = -90,
    RGY_ERR_INVALID_GL_OBJECT               = -91,
    RGY_ERR_INVALID_BUFFER_SIZE             = -92,
    RGY_ERR_INVALID_MIP_LEVEL               = -93,
    RGY_ERR_INVALID_GLOBAL_WORK_SIZE        = -94,
    RGY_ERR_COMPILE_PROGRAM_FAILURE         = -95,
    RGY_ERR_OPENCL_CRUSH                    = -96,

    //Vulkan
    RGY_ERR_VK_NOT_READY                       =  -97,
    RGY_ERR_VK_TIMEOUT                         =  -98,
    RGY_ERR_VK_EVENT_SET                       =  -99,
    RGY_ERR_VK_EVENT_RESET                     = -100,
    RGY_ERR_VK_INCOMPLETE                      = -101,
    RGY_ERR_VK_OUT_OF_HOST_MEMORY              = -102,
    RGY_ERR_VK_OUT_OF_DEVICE_MEMORY            = -103,
    RGY_ERR_VK_INITIALIZATION_FAILED           = -104,
    RGY_ERR_VK_DEVICE_LOST                     = -105,
    RGY_ERR_VK_MEMORY_MAP_FAILED               = -106,
    RGY_ERR_VK_LAYER_NOT_PRESENT               = -107,
    RGY_ERR_VK_EXTENSION_NOT_PRESENT           = -108,
    RGY_ERR_VK_FEATURE_NOT_PRESENT             = -109,
    RGY_ERR_VK_INCOMPATIBLE_DRIVER             = -110,
    RGY_ERR_VK_TOO_MANY_OBJECTS                = -111,
    RGY_ERR_VK_FORMAT_NOT_SUPPORTED            = -112,
    RGY_ERR_VK_FRAGMENTED_POOL                 = -113,
    RGY_ERR_VK_UNKNOWN                         = -114,
    RGY_ERR_VK_OUT_OF_POOL_MEMORY              = -115,
    RGY_ERR_VK_INVALID_EXTERNAL_HANDLE         = -116,
    RGY_ERR_VK_FRAGMENTATION                   = -117,
    RGY_ERR_VK_INVALID_OPAQUE_CAPTURE_ADDRESS  = -118,
    RGY_ERR_VK_SURFACE_LOST_KHR                = -119,
    RGY_ERR_VK_NATIVE_WINDOW_IN_USE_KHR        = -120,
    RGY_ERR_VK__SUBOPTIMAL_KHR                 = -121,
    RGY_ERR_VK_OUT_OF_DATE_KHR                 = -122,
    RGY_ERR_VK_INCOMPATIBLE_DISPLAY_KHR        = -123,
    RGY_ERR_VK_VALIDATION_FAILED_EXT           = -124,
    RGY_ERR_VK_INVALID_SHADER_NV               = -125,
    RGY_ERR_VK_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT = -126,
    RGY_ERR_VK_NOT_PERMITTED_EXT                   = -127,
    RGY_ERR_VK_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT = -128,
    RGY_VK_THREAD_IDLE_KHR                         = -129,
    RGY_VK_THREAD_DONE_KHR                         = -130,
    RGY_VK_OPERATION_DEFERRED_KHR                  = -131,
    RGY_VK_OPERATION_NOT_DEFERRED_KHR              = -132,
    RGY_VK_PIPELINE_COMPILE_REQUIRED_EXT           = -133,
    RGY_ERR_VK_OUT_OF_POOL_MEMORY_KHR              = -134,
    RGY_ERR_VK_INVALID_EXTERNAL_HANDLE_KHR         = -135,
    RGY_ERR_VK_FRAGMENTATION_EXT                   = -136,
    RGY_ERR_VK_INVALID_DEVICE_ADDRESS_EXT          = -137,
    RGY_ERR_VK_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR  = -138,
    RGY_ERR_VK_PIPELINE_COMPILE_REQUIRED_EXT       = -139,

    //NvCV_Status
    RGY_ERR_NVCV_OFFSET                = -200,    //!< The procedure returned successfully.
    RGY_ERR_NVCV_GENERAL               = RGY_ERR_NVCV_OFFSET -1,   //!< An otherwise unspecified error has occurred.
    RGY_ERR_NVCV_UNIMPLEMENTED         = RGY_ERR_NVCV_OFFSET -2,   //!< The requested feature is not yet implemented.
    RGY_ERR_NVCV_MEMORY                = RGY_ERR_NVCV_OFFSET -3,   //!< There is not enough memory for the requested operation.
    RGY_ERR_NVCV_EFFECT                = RGY_ERR_NVCV_OFFSET -4,   //!< An invalid effect handle has been supplied.
    RGY_ERR_NVCV_SELECTOR              = RGY_ERR_NVCV_OFFSET -5,   //!< The given parameter selector is not valid in this effect filter.
    RGY_ERR_NVCV_BUFFER                = RGY_ERR_NVCV_OFFSET -6,   //!< An image buffer has not been specified.
    RGY_ERR_NVCV_PARAMETER             = RGY_ERR_NVCV_OFFSET -7,   //!< An invalid parameter value has been supplied for this effect+selector.
    RGY_ERR_NVCV_MISMATCH              = RGY_ERR_NVCV_OFFSET -8,   //!< Some parameters are not appropriately matched.
    RGY_ERR_NVCV_PIXELFORMAT           = RGY_ERR_NVCV_OFFSET -9,   //!< The specified pixel format is not accommodated.
    RGY_ERR_NVCV_MODEL                 = RGY_ERR_NVCV_OFFSET -10,  //!< Error while loading the TRT model.
    RGY_ERR_NVCV_LIBRARY               = RGY_ERR_NVCV_OFFSET -11,  //!< Error loading the dynamic library.
    RGY_ERR_NVCV_INITIALIZATION        = RGY_ERR_NVCV_OFFSET -12,  //!< The effect has not been properly initialized.
    RGY_ERR_NVCV_FILE                  = RGY_ERR_NVCV_OFFSET -13,  //!< The file could not be found.
    RGY_ERR_NVCV_FEATURENOTFOUND       = RGY_ERR_NVCV_OFFSET -14,  //!< The requested feature was not found
    RGY_ERR_NVCV_MISSINGINPUT          = RGY_ERR_NVCV_OFFSET -15,  //!< A required parameter was not set
    RGY_ERR_NVCV_RESOLUTION            = RGY_ERR_NVCV_OFFSET -16,  //!< The specified image resolution is not supported.
    RGY_ERR_NVCV_UNSUPPORTEDGPU        = RGY_ERR_NVCV_OFFSET -17,  //!< The GPU is not supported
    RGY_ERR_NVCV_WRONGGPU              = RGY_ERR_NVCV_OFFSET -18,  //!< The current GPU is not the one selected.
    RGY_ERR_NVCV_UNSUPPORTEDDRIVER     = RGY_ERR_NVCV_OFFSET -19,  //!< The currently installed graphics driver is not supported
    RGY_ERR_NVCV_MODELDEPENDENCIES     = RGY_ERR_NVCV_OFFSET -20,  //!< There is no model with dependencies that match this system
    RGY_ERR_NVCV_PARSE                 = RGY_ERR_NVCV_OFFSET -21,  //!< There has been a parsing or syntax error while reading a file
    RGY_ERR_NVCV_MODELSUBSTITUTION     = RGY_ERR_NVCV_OFFSET -22,  //!< The specified model does not exist and has been substituted.
    RGY_ERR_NVCV_READ                  = RGY_ERR_NVCV_OFFSET -23,  //!< An error occurred while reading a file.
    RGY_ERR_NVCV_WRITE                 = RGY_ERR_NVCV_OFFSET -24,  //!< An error occurred while writing a file.
    RGY_ERR_NVCV_PARAMREADONLY         = RGY_ERR_NVCV_OFFSET -25,  //!< The selected parameter is read-only.
    RGY_ERR_NVCV_TRT_ENQUEUE           = RGY_ERR_NVCV_OFFSET -26,  //!< TensorRT enqueue failed.
    RGY_ERR_NVCV_TRT_BINDINGS          = RGY_ERR_NVCV_OFFSET -27,  //!< Unexpected TensorRT bindings.
    RGY_ERR_NVCV_TRT_CONTEXT           = RGY_ERR_NVCV_OFFSET -28,  //!< An error occurred while creating a TensorRT context.
    RGY_ERR_NVCV_TRT_INFER             = RGY_ERR_NVCV_OFFSET -29,  ///< The was a problem creating the inference engine.
    RGY_ERR_NVCV_TRT_ENGINE            = RGY_ERR_NVCV_OFFSET -30,  ///< There was a problem deserializing the inference runtime engine.
    RGY_ERR_NVCV_NPP                   = RGY_ERR_NVCV_OFFSET -31,  //!< An error has occurred in the NPP library.
    RGY_ERR_NVCV_CONFIG                = RGY_ERR_NVCV_OFFSET -32,  //!< No suitable model exists for the specified parameter configuration.
    RGY_ERR_NVCV_TOOSMALL              = RGY_ERR_NVCV_OFFSET -33,  //!< A supplied parameter or buffer is not large enough.
    RGY_ERR_NVCV_TOOBIG                = RGY_ERR_NVCV_OFFSET -34,  //!< A supplied parameter is too big.
    RGY_ERR_NVCV_WRONGSIZE             = RGY_ERR_NVCV_OFFSET -35,  //!< A supplied parameter is not the expected size.
    RGY_ERR_NVCV_OBJECTNOTFOUND        = RGY_ERR_NVCV_OFFSET -36,  //!< The specified object was not found.
    RGY_ERR_NVCV_SINGULAR              = RGY_ERR_NVCV_OFFSET -37,  //!< A mathematical singularity has been encountered.
    RGY_ERR_NVCV_NOTHINGRENDERED       = RGY_ERR_NVCV_OFFSET -38,  //!< Nothing was rendered in the specified region.
    RGY_ERR_NVCV_CONVERGENCE           = RGY_ERR_NVCV_OFFSET -39,  //!< An iteration did not converge satisfactorily.

    RGY_ERR_NVCV_OPENGL                = RGY_ERR_NVCV_OFFSET -40,  //!< An OpenGL error has occurred.
    RGY_ERR_NVCV_DIRECT3D              = RGY_ERR_NVCV_OFFSET -41,  //!< A Direct3D error has occurred.

    RGY_ERR_NVCV_CUDA_BASE             = RGY_ERR_NVCV_OFFSET -50,  //!< CUDA errors are offset from this value.
    RGY_ERR_NVCV_CUDA_VALUE            = RGY_ERR_NVCV_OFFSET -51,  //!< A CUDA parameter is not within the acceptable range.
    RGY_ERR_NVCV_CUDA_MEMORY           = RGY_ERR_NVCV_OFFSET -52,  //!< There is not enough CUDA memory for the requested operation.
    RGY_ERR_NVCV_CUDA_PITCH            = RGY_ERR_NVCV_OFFSET -53,  //!< A CUDA pitch is not within the acceptable range.
    RGY_ERR_NVCV_CUDA_INIT             = RGY_ERR_NVCV_OFFSET -54,  //!< The CUDA driver and runtime could not be initialized.
    RGY_ERR_NVCV_CUDA_LAUNCH           = RGY_ERR_NVCV_OFFSET -55,  //!< The CUDA kernel launch has failed.
    RGY_ERR_NVCV_CUDA_KERNEL           = RGY_ERR_NVCV_OFFSET -56,  //!< No suitable kernel image is available for the device.
    RGY_ERR_NVCV_CUDA_DRIVER           = RGY_ERR_NVCV_OFFSET -57,  //!< The installed NVIDIA CUDA driver is older than the CUDA runtime library.
    RGY_ERR_NVCV_CUDA_UNSUPPORTED      = RGY_ERR_NVCV_OFFSET -58,  //!< The CUDA operation is not supported on the current system or device.
    RGY_ERR_NVCV_CUDA_ILLEGAL_ADDRESS  = RGY_ERR_NVCV_OFFSET -59,  //!< CUDA tried to load or store on an invalid memory address.
    RGY_ERR_NVCV_CUDA                  = RGY_ERR_NVCV_OFFSET -60, //!< An otherwise unspecified CUDA error has been reported.

    //mpp
    RGY_ERR_MPP_NOK = -300,
    RGY_ERR_MPP_ERR_UNKNOW          = RGY_ERR_MPP_NOK -  1,
    RGY_ERR_MPP_ERR_NULL_PTR        = RGY_ERR_MPP_NOK -  2,
    RGY_ERR_MPP_ERR_MALLOC          = RGY_ERR_MPP_NOK -  3,
    RGY_ERR_MPP_ERR_OPEN_FILE       = RGY_ERR_MPP_NOK -  4,
    RGY_ERR_MPP_ERR_VALUE           = RGY_ERR_MPP_NOK -  5,
    RGY_ERR_MPP_ERR_READ_BIT        = RGY_ERR_MPP_NOK -  6,
    RGY_ERR_MPP_ERR_TIMEOUT         = RGY_ERR_MPP_NOK -  7,
    RGY_ERR_MPP_ERR_PERM            = RGY_ERR_MPP_NOK -  8,
    RGY_ERR_MPP_ERR_BASE            = RGY_ERR_MPP_NOK -  9,
    RGY_ERR_MPP_ERR_LIST_STREAM     = RGY_ERR_MPP_NOK - 10,
    RGY_ERR_MPP_ERR_INIT            = RGY_ERR_MPP_NOK - 11,
    RGY_ERR_MPP_ERR_VPU_CODEC_INIT  = RGY_ERR_MPP_NOK - 12,
    RGY_ERR_MPP_ERR_STREAM          = RGY_ERR_MPP_NOK - 13,
    RGY_ERR_MPP_ERR_FATAL_THREAD    = RGY_ERR_MPP_NOK - 14,
    RGY_ERR_MPP_ERR_NOMEM           = RGY_ERR_MPP_NOK - 15,
    RGY_ERR_MPP_ERR_PROTOL          = RGY_ERR_MPP_NOK - 16,
    RGY_ERR_MPP_FAIL_SPLIT_FRAME    = RGY_ERR_MPP_NOK - 17,
    RGY_ERR_MPP_ERR_VPUHW           = RGY_ERR_MPP_NOK - 18,
    RGY_ERR_MPP_EOS_STREAM_REACHED  = RGY_ERR_MPP_NOK - 19,
    RGY_ERR_MPP_ERR_BUFFER_FULL     = RGY_ERR_MPP_NOK - 20,
    RGY_ERR_MPP_ERR_DISPLAY_FULL    = RGY_ERR_MPP_NOK - 21,

    //im2d
    RGY_ERR_IM_STATUS_NOT_SUPPORTED = RGY_ERR_UNSUPPORTED,
    RGY_ERR_IM_STATUS_OUT_OF_MEMORY = RGY_ERR_NULL_PTR,
    RGY_ERR_IM_STATUS_INVALID_PARAM = RGY_ERR_INVALID_PARAM,
    RGY_ERR_IM_STATUS_ILLEGAL_PARAM = RGY_ERR_INVALID_PARAM,
    RGY_ERR_IM_STATUS_FAILED        = RGY_ERR_DEVICE_FAILED,

    RGY_WRN_IN_EXECUTION                = 1,
    RGY_WRN_DEVICE_BUSY                 = 2,
    RGY_WRN_VIDEO_PARAM_CHANGED         = 3,
    RGY_WRN_PARTIAL_ACCELERATION        = 4,
    RGY_WRN_INCOMPATIBLE_VIDEO_PARAM    = 5,
    RGY_WRN_VALUE_NOT_CHANGED           = 6,
    RGY_WRN_OUT_OF_RANGE                = 7,
    RGY_WRN_FILTER_SKIPPED              = 10,
    RGY_WRN_INCOMPATIBLE_AUDIO_PARAM    = 11,

    RGY_PRINT_OPTION_DONE = 20,
    RGY_PRINT_OPTION_ERR  = 21,

    RGY_ERR_INVALID_COLOR_FORMAT = -100,

    RGY_ERR_MORE_DATA_SUBMIT_TASK       = -10000,

    // CUDA
    //  grep '=' cudaErrors.txt | awk '{print "RGY_ERR_",$1,$2,"RGY_ERR_CUDA_OFFSET -",$3}'
    RGY_ERR_CUDA_OFFSET = -20000,
    RGY_ERR_cudaSuccess = RGY_ERR_NONE,
    RGY_ERR_cudaErrorInvalidValue = RGY_ERR_CUDA_OFFSET - 1,
    RGY_ERR_cudaErrorMemoryAllocation = RGY_ERR_CUDA_OFFSET - 2,
    RGY_ERR_cudaErrorInitializationError = RGY_ERR_CUDA_OFFSET - 3,
    RGY_ERR_cudaErrorCudartUnloading = RGY_ERR_CUDA_OFFSET - 4,
    RGY_ERR_cudaErrorProfilerDisabled = RGY_ERR_CUDA_OFFSET - 5,
    RGY_ERR_cudaErrorProfilerNotInitialized = RGY_ERR_CUDA_OFFSET - 6,
    RGY_ERR_cudaErrorProfilerAlreadyStarted = RGY_ERR_CUDA_OFFSET - 7,
    RGY_ERR_cudaErrorProfilerAlreadyStopped = RGY_ERR_CUDA_OFFSET - 8,
    RGY_ERR_cudaErrorInvalidConfiguration = RGY_ERR_CUDA_OFFSET - 9,
    RGY_ERR_cudaErrorInvalidPitchValue = RGY_ERR_CUDA_OFFSET - 12,
    RGY_ERR_cudaErrorInvalidSymbol = RGY_ERR_CUDA_OFFSET - 13,
    RGY_ERR_cudaErrorInvalidHostPointer = RGY_ERR_CUDA_OFFSET - 16,
    RGY_ERR_cudaErrorInvalidDevicePointer = RGY_ERR_CUDA_OFFSET - 17,
    RGY_ERR_cudaErrorInvalidTexture = RGY_ERR_CUDA_OFFSET - 18,
    RGY_ERR_cudaErrorInvalidTextureBinding = RGY_ERR_CUDA_OFFSET - 19,
    RGY_ERR_cudaErrorInvalidChannelDescriptor = RGY_ERR_CUDA_OFFSET - 20,
    RGY_ERR_cudaErrorInvalidMemcpyDirection = RGY_ERR_CUDA_OFFSET - 21,
    RGY_ERR_cudaErrorAddressOfConstant = RGY_ERR_CUDA_OFFSET - 22,
    RGY_ERR_cudaErrorTextureFetchFailed = RGY_ERR_CUDA_OFFSET - 23,
    RGY_ERR_cudaErrorTextureNotBound = RGY_ERR_CUDA_OFFSET - 24,
    RGY_ERR_cudaErrorSynchronizationError = RGY_ERR_CUDA_OFFSET - 25,
    RGY_ERR_cudaErrorInvalidFilterSetting = RGY_ERR_CUDA_OFFSET - 26,
    RGY_ERR_cudaErrorInvalidNormSetting = RGY_ERR_CUDA_OFFSET - 27,
    RGY_ERR_cudaErrorMixedDeviceExecution = RGY_ERR_CUDA_OFFSET - 28,
    RGY_ERR_cudaErrorNotYetImplemented = RGY_ERR_CUDA_OFFSET - 31,
    RGY_ERR_cudaErrorMemoryValueTooLarge = RGY_ERR_CUDA_OFFSET - 32,
    RGY_ERR_cudaErrorStubLibrary = RGY_ERR_CUDA_OFFSET - 34,
    RGY_ERR_cudaErrorInsufficientDriver = RGY_ERR_CUDA_OFFSET - 35,
    RGY_ERR_cudaErrorCallRequiresNewerDriver = RGY_ERR_CUDA_OFFSET - 36,
    RGY_ERR_cudaErrorInvalidSurface = RGY_ERR_CUDA_OFFSET - 37,
    RGY_ERR_cudaErrorDuplicateVariableName = RGY_ERR_CUDA_OFFSET - 43,
    RGY_ERR_cudaErrorDuplicateTextureName = RGY_ERR_CUDA_OFFSET - 44,
    RGY_ERR_cudaErrorDuplicateSurfaceName = RGY_ERR_CUDA_OFFSET - 45,
    RGY_ERR_cudaErrorDevicesUnavailable = RGY_ERR_CUDA_OFFSET - 46,
    RGY_ERR_cudaErrorIncompatibleDriverContext = RGY_ERR_CUDA_OFFSET - 49,
    RGY_ERR_cudaErrorMissingConfiguration = RGY_ERR_CUDA_OFFSET - 52,
    RGY_ERR_cudaErrorPriorLaunchFailure = RGY_ERR_CUDA_OFFSET - 53,
    RGY_ERR_cudaErrorLaunchMaxDepthExceeded = RGY_ERR_CUDA_OFFSET - 65,
    RGY_ERR_cudaErrorLaunchFileScopedTex = RGY_ERR_CUDA_OFFSET - 66,
    RGY_ERR_cudaErrorLaunchFileScopedSurf = RGY_ERR_CUDA_OFFSET - 67,
    RGY_ERR_cudaErrorSyncDepthExceeded = RGY_ERR_CUDA_OFFSET - 68,
    RGY_ERR_cudaErrorLaunchPendingCountExceeded = RGY_ERR_CUDA_OFFSET - 69,
    RGY_ERR_cudaErrorInvalidDeviceFunction = RGY_ERR_CUDA_OFFSET - 98,
    RGY_ERR_cudaErrorNoDevice = RGY_ERR_CUDA_OFFSET - 100,
    RGY_ERR_cudaErrorInvalidDevice = RGY_ERR_CUDA_OFFSET - 101,
    RGY_ERR_cudaErrorDeviceNotLicensed = RGY_ERR_CUDA_OFFSET - 102,
    RGY_ERR_cudaErrorSoftwareValidityNotEstablished = RGY_ERR_CUDA_OFFSET - 103,
    RGY_ERR_cudaErrorStartupFailure = RGY_ERR_CUDA_OFFSET - 127,
    RGY_ERR_cudaErrorInvalidKernelImage = RGY_ERR_CUDA_OFFSET - 200,
    RGY_ERR_cudaErrorDeviceUninitialized = RGY_ERR_CUDA_OFFSET - 201,
    RGY_ERR_cudaErrorMapBufferObjectFailed = RGY_ERR_CUDA_OFFSET - 205,
    RGY_ERR_cudaErrorUnmapBufferObjectFailed = RGY_ERR_CUDA_OFFSET - 206,
    RGY_ERR_cudaErrorArrayIsMapped = RGY_ERR_CUDA_OFFSET - 207,
    RGY_ERR_cudaErrorAlreadyMapped = RGY_ERR_CUDA_OFFSET - 208,
    RGY_ERR_cudaErrorNoKernelImageForDevice = RGY_ERR_CUDA_OFFSET - 209,
    RGY_ERR_cudaErrorAlreadyAcquired = RGY_ERR_CUDA_OFFSET - 210,
    RGY_ERR_cudaErrorNotMapped = RGY_ERR_CUDA_OFFSET - 211,
    RGY_ERR_cudaErrorNotMappedAsArray = RGY_ERR_CUDA_OFFSET - 212,
    RGY_ERR_cudaErrorNotMappedAsPointer = RGY_ERR_CUDA_OFFSET - 213,
    RGY_ERR_cudaErrorECCUncorrectable = RGY_ERR_CUDA_OFFSET - 214,
    RGY_ERR_cudaErrorUnsupportedLimit = RGY_ERR_CUDA_OFFSET - 215,
    RGY_ERR_cudaErrorDeviceAlreadyInUse = RGY_ERR_CUDA_OFFSET - 216,
    RGY_ERR_cudaErrorPeerAccessUnsupported = RGY_ERR_CUDA_OFFSET - 217,
    RGY_ERR_cudaErrorInvalidPtx = RGY_ERR_CUDA_OFFSET - 218,
    RGY_ERR_cudaErrorInvalidGraphicsContext = RGY_ERR_CUDA_OFFSET - 219,
    RGY_ERR_cudaErrorNvlinkUncorrectable = RGY_ERR_CUDA_OFFSET - 220,
    RGY_ERR_cudaErrorJitCompilerNotFound = RGY_ERR_CUDA_OFFSET - 221,
    RGY_ERR_cudaErrorUnsupportedPtxVersion = RGY_ERR_CUDA_OFFSET - 222,
    RGY_ERR_cudaErrorJitCompilationDisabled = RGY_ERR_CUDA_OFFSET - 223,
    RGY_ERR_cudaErrorUnsupportedExecAffinity = RGY_ERR_CUDA_OFFSET - 224,
    RGY_ERR_cudaErrorInvalidSource = RGY_ERR_CUDA_OFFSET - 300,
    RGY_ERR_cudaErrorFileNotFound = RGY_ERR_CUDA_OFFSET - 301,
    RGY_ERR_cudaErrorSharedObjectSymbolNotFound = RGY_ERR_CUDA_OFFSET - 302,
    RGY_ERR_cudaErrorSharedObjectInitFailed = RGY_ERR_CUDA_OFFSET - 303,
    RGY_ERR_cudaErrorOperatingSystem = RGY_ERR_CUDA_OFFSET - 304,
    RGY_ERR_cudaErrorInvalidResourceHandle = RGY_ERR_CUDA_OFFSET - 400,
    RGY_ERR_cudaErrorIllegalState = RGY_ERR_CUDA_OFFSET - 401,
    RGY_ERR_cudaErrorSymbolNotFound = RGY_ERR_CUDA_OFFSET - 500,
    RGY_ERR_cudaErrorNotReady = RGY_ERR_CUDA_OFFSET - 600,
    RGY_ERR_cudaErrorIllegalAddress = RGY_ERR_CUDA_OFFSET - 700,
    RGY_ERR_cudaErrorLaunchOutOfResources = RGY_ERR_CUDA_OFFSET - 701,
    RGY_ERR_cudaErrorLaunchTimeout = RGY_ERR_CUDA_OFFSET - 702,
    RGY_ERR_cudaErrorLaunchIncompatibleTexturing = RGY_ERR_CUDA_OFFSET - 703,
    RGY_ERR_cudaErrorPeerAccessAlreadyEnabled = RGY_ERR_CUDA_OFFSET - 704,
    RGY_ERR_cudaErrorPeerAccessNotEnabled = RGY_ERR_CUDA_OFFSET - 705,
    RGY_ERR_cudaErrorSetOnActiveProcess = RGY_ERR_CUDA_OFFSET - 708,
    RGY_ERR_cudaErrorContextIsDestroyed = RGY_ERR_CUDA_OFFSET - 709,
    RGY_ERR_cudaErrorAssert = RGY_ERR_CUDA_OFFSET - 710,
    RGY_ERR_cudaErrorTooManyPeers = RGY_ERR_CUDA_OFFSET - 711,
    RGY_ERR_cudaErrorHostMemoryAlreadyRegistered = RGY_ERR_CUDA_OFFSET - 712,
    RGY_ERR_cudaErrorHostMemoryNotRegistered = RGY_ERR_CUDA_OFFSET - 713,
    RGY_ERR_cudaErrorHardwareStackError = RGY_ERR_CUDA_OFFSET - 714,
    RGY_ERR_cudaErrorIllegalInstruction = RGY_ERR_CUDA_OFFSET - 715,
    RGY_ERR_cudaErrorMisalignedAddress = RGY_ERR_CUDA_OFFSET - 716,
    RGY_ERR_cudaErrorInvalidAddressSpace = RGY_ERR_CUDA_OFFSET - 717,
    RGY_ERR_cudaErrorInvalidPc = RGY_ERR_CUDA_OFFSET - 718,
    RGY_ERR_cudaErrorLaunchFailure = RGY_ERR_CUDA_OFFSET - 719,
    RGY_ERR_cudaErrorCooperativeLaunchTooLarge = RGY_ERR_CUDA_OFFSET - 720,
    RGY_ERR_cudaErrorNotPermitted = RGY_ERR_CUDA_OFFSET - 800,
    RGY_ERR_cudaErrorNotSupported = RGY_ERR_CUDA_OFFSET - 801,
    RGY_ERR_cudaErrorSystemNotReady = RGY_ERR_CUDA_OFFSET - 802,
    RGY_ERR_cudaErrorSystemDriverMismatch = RGY_ERR_CUDA_OFFSET - 803,
    RGY_ERR_cudaErrorCompatNotSupportedOnDevice = RGY_ERR_CUDA_OFFSET - 804,
    RGY_ERR_cudaErrorMpsConnectionFailed = RGY_ERR_CUDA_OFFSET - 805,
    RGY_ERR_cudaErrorMpsRpcFailure = RGY_ERR_CUDA_OFFSET - 806,
    RGY_ERR_cudaErrorMpsServerNotReady = RGY_ERR_CUDA_OFFSET - 807,
    RGY_ERR_cudaErrorMpsMaxClientsReached = RGY_ERR_CUDA_OFFSET - 808,
    RGY_ERR_cudaErrorMpsMaxConnectionsReached = RGY_ERR_CUDA_OFFSET - 809,
    RGY_ERR_cudaErrorStreamCaptureUnsupported = RGY_ERR_CUDA_OFFSET - 900,
    RGY_ERR_cudaErrorStreamCaptureInvalidated = RGY_ERR_CUDA_OFFSET - 901,
    RGY_ERR_cudaErrorStreamCaptureMerge = RGY_ERR_CUDA_OFFSET - 902,
    RGY_ERR_cudaErrorStreamCaptureUnmatched = RGY_ERR_CUDA_OFFSET - 903,
    RGY_ERR_cudaErrorStreamCaptureUnjoined = RGY_ERR_CUDA_OFFSET - 904,
    RGY_ERR_cudaErrorStreamCaptureIsolation = RGY_ERR_CUDA_OFFSET - 905,
    RGY_ERR_cudaErrorStreamCaptureImplicit = RGY_ERR_CUDA_OFFSET - 906,
    RGY_ERR_cudaErrorCapturedEvent = RGY_ERR_CUDA_OFFSET - 907,
    RGY_ERR_cudaErrorStreamCaptureWrongThread = RGY_ERR_CUDA_OFFSET - 908,
    RGY_ERR_cudaErrorTimeout = RGY_ERR_CUDA_OFFSET - 909,
    RGY_ERR_cudaErrorGraphExecUpdateFailure = RGY_ERR_CUDA_OFFSET - 910,
    RGY_ERR_cudaErrorExternalDevice = RGY_ERR_CUDA_OFFSET - 911,
    RGY_ERR_cudaErrorUnknown = RGY_ERR_CUDA_OFFSET - 999,
    RGY_ERR_cudaErrorApiFailureBase = RGY_ERR_CUDA_OFFSET - 10000,

    // CUDA Driver Error
    // grep '=' cudaErrors.txt | awk '{print "RGY_ERR_",$1,$2,"RGY_ERR_CUDA_DRIVER_OFFSET -",$3}'
    RGY_ERR_CUDA_DRIVER_OFFSET = -30000,
    RGY_ERR_CUDA_SUCCESS = RGY_ERR_NONE,
    RGY_ERR_CUDA_ERROR_INVALID_VALUE = RGY_ERR_CUDA_DRIVER_OFFSET - 1,
    RGY_ERR_CUDA_ERROR_OUT_OF_MEMORY = RGY_ERR_CUDA_DRIVER_OFFSET - 2,
    RGY_ERR_CUDA_ERROR_NOT_INITIALIZED = RGY_ERR_CUDA_DRIVER_OFFSET - 3,
    RGY_ERR_CUDA_ERROR_DEINITIALIZED = RGY_ERR_CUDA_DRIVER_OFFSET - 4,
    RGY_ERR_CUDA_ERROR_PROFILER_DISABLED = RGY_ERR_CUDA_DRIVER_OFFSET - 5,
    RGY_ERR_CUDA_ERROR_PROFILER_NOT_INITIALIZED = RGY_ERR_CUDA_DRIVER_OFFSET - 6,
    RGY_ERR_CUDA_ERROR_PROFILER_ALREADY_STARTED = RGY_ERR_CUDA_DRIVER_OFFSET - 7,
    RGY_ERR_CUDA_ERROR_PROFILER_ALREADY_STOPPED = RGY_ERR_CUDA_DRIVER_OFFSET - 8,
    RGY_ERR_CUDA_ERROR_STUB_LIBRARY = RGY_ERR_CUDA_DRIVER_OFFSET - 34,
    RGY_ERR_CUDA_ERROR_DEVICE_UNAVAILABLE = RGY_ERR_CUDA_DRIVER_OFFSET - 46,
    RGY_ERR_CUDA_ERROR_NO_DEVICE = RGY_ERR_CUDA_DRIVER_OFFSET - 100,
    RGY_ERR_CUDA_ERROR_INVALID_DEVICE = RGY_ERR_CUDA_DRIVER_OFFSET - 101,
    RGY_ERR_CUDA_ERROR_DEVICE_NOT_LICENSED = RGY_ERR_CUDA_DRIVER_OFFSET - 102,
    RGY_ERR_CUDA_ERROR_INVALID_IMAGE = RGY_ERR_CUDA_DRIVER_OFFSET - 200,
    RGY_ERR_CUDA_ERROR_INVALID_CONTEXT = RGY_ERR_CUDA_DRIVER_OFFSET - 201,
    RGY_ERR_CUDA_ERROR_CONTEXT_ALREADY_CURRENT = RGY_ERR_CUDA_DRIVER_OFFSET - 202,
    RGY_ERR_CUDA_ERROR_MAP_FAILED = RGY_ERR_CUDA_DRIVER_OFFSET - 205,
    RGY_ERR_CUDA_ERROR_UNMAP_FAILED = RGY_ERR_CUDA_DRIVER_OFFSET - 206,
    RGY_ERR_CUDA_ERROR_ARRAY_IS_MAPPED = RGY_ERR_CUDA_DRIVER_OFFSET - 207,
    RGY_ERR_CUDA_ERROR_ALREADY_MAPPED = RGY_ERR_CUDA_DRIVER_OFFSET - 208,
    RGY_ERR_CUDA_ERROR_NO_BINARY_FOR_GPU = RGY_ERR_CUDA_DRIVER_OFFSET - 209,
    RGY_ERR_CUDA_ERROR_ALREADY_ACQUIRED = RGY_ERR_CUDA_DRIVER_OFFSET - 210,
    RGY_ERR_CUDA_ERROR_NOT_MAPPED = RGY_ERR_CUDA_DRIVER_OFFSET - 211,
    RGY_ERR_CUDA_ERROR_NOT_MAPPED_AS_ARRAY = RGY_ERR_CUDA_DRIVER_OFFSET - 212,
    RGY_ERR_CUDA_ERROR_NOT_MAPPED_AS_POINTER = RGY_ERR_CUDA_DRIVER_OFFSET - 213,
    RGY_ERR_CUDA_ERROR_ECC_UNCORRECTABLE = RGY_ERR_CUDA_DRIVER_OFFSET - 214,
    RGY_ERR_CUDA_ERROR_UNSUPPORTED_LIMIT = RGY_ERR_CUDA_DRIVER_OFFSET - 215,
    RGY_ERR_CUDA_ERROR_CONTEXT_ALREADY_IN_USE = RGY_ERR_CUDA_DRIVER_OFFSET - 216,
    RGY_ERR_CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = RGY_ERR_CUDA_DRIVER_OFFSET - 217,
    RGY_ERR_CUDA_ERROR_INVALID_PTX = RGY_ERR_CUDA_DRIVER_OFFSET - 218,
    RGY_ERR_CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = RGY_ERR_CUDA_DRIVER_OFFSET - 219,
    RGY_ERR_CUDA_ERROR_NVLINK_UNCORRECTABLE = RGY_ERR_CUDA_DRIVER_OFFSET - 220,
    RGY_ERR_CUDA_ERROR_JIT_COMPILER_NOT_FOUND = RGY_ERR_CUDA_DRIVER_OFFSET - 221,
    RGY_ERR_CUDA_ERROR_UNSUPPORTED_PTX_VERSION = RGY_ERR_CUDA_DRIVER_OFFSET - 222,
    RGY_ERR_CUDA_ERROR_JIT_COMPILATION_DISABLED = RGY_ERR_CUDA_DRIVER_OFFSET - 223,
    RGY_ERR_CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = RGY_ERR_CUDA_DRIVER_OFFSET - 224,
    RGY_ERR_CUDA_ERROR_INVALID_SOURCE = RGY_ERR_CUDA_DRIVER_OFFSET - 300,
    RGY_ERR_CUDA_ERROR_FILE_NOT_FOUND = RGY_ERR_CUDA_DRIVER_OFFSET - 301,
    RGY_ERR_CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = RGY_ERR_CUDA_DRIVER_OFFSET - 302,
    RGY_ERR_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = RGY_ERR_CUDA_DRIVER_OFFSET - 303,
    RGY_ERR_CUDA_ERROR_OPERATING_SYSTEM = RGY_ERR_CUDA_DRIVER_OFFSET - 304,
    RGY_ERR_CUDA_ERROR_INVALID_HANDLE = RGY_ERR_CUDA_DRIVER_OFFSET - 400,
    RGY_ERR_CUDA_ERROR_ILLEGAL_STATE = RGY_ERR_CUDA_DRIVER_OFFSET - 401,
    RGY_ERR_CUDA_ERROR_NOT_FOUND = RGY_ERR_CUDA_DRIVER_OFFSET - 500,
    RGY_ERR_CUDA_ERROR_NOT_READY = RGY_ERR_CUDA_DRIVER_OFFSET - 600,
    RGY_ERR_CUDA_ERROR_ILLEGAL_ADDRESS = RGY_ERR_CUDA_DRIVER_OFFSET - 700,
    RGY_ERR_CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = RGY_ERR_CUDA_DRIVER_OFFSET - 701,
    RGY_ERR_CUDA_ERROR_LAUNCH_TIMEOUT = RGY_ERR_CUDA_DRIVER_OFFSET - 702,
    RGY_ERR_CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = RGY_ERR_CUDA_DRIVER_OFFSET - 703,
    RGY_ERR_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = RGY_ERR_CUDA_DRIVER_OFFSET - 704,
    RGY_ERR_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = RGY_ERR_CUDA_DRIVER_OFFSET - 705,
    RGY_ERR_CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = RGY_ERR_CUDA_DRIVER_OFFSET - 708,
    RGY_ERR_CUDA_ERROR_CONTEXT_IS_DESTROYED = RGY_ERR_CUDA_DRIVER_OFFSET - 709,
    RGY_ERR_CUDA_ERROR_ASSERT = RGY_ERR_CUDA_DRIVER_OFFSET - 710,
    RGY_ERR_CUDA_ERROR_TOO_MANY_PEERS = RGY_ERR_CUDA_DRIVER_OFFSET - 711,
    RGY_ERR_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = RGY_ERR_CUDA_DRIVER_OFFSET - 712,
    RGY_ERR_CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = RGY_ERR_CUDA_DRIVER_OFFSET - 713,
    RGY_ERR_CUDA_ERROR_HARDWARE_STACK_ERROR = RGY_ERR_CUDA_DRIVER_OFFSET - 714,
    RGY_ERR_CUDA_ERROR_ILLEGAL_INSTRUCTION = RGY_ERR_CUDA_DRIVER_OFFSET - 715,
    RGY_ERR_CUDA_ERROR_MISALIGNED_ADDRESS = RGY_ERR_CUDA_DRIVER_OFFSET - 716,
    RGY_ERR_CUDA_ERROR_INVALID_ADDRESS_SPACE = RGY_ERR_CUDA_DRIVER_OFFSET - 717,
    RGY_ERR_CUDA_ERROR_INVALID_PC = RGY_ERR_CUDA_DRIVER_OFFSET - 718,
    RGY_ERR_CUDA_ERROR_LAUNCH_FAILED = RGY_ERR_CUDA_DRIVER_OFFSET - 719,
    RGY_ERR_CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = RGY_ERR_CUDA_DRIVER_OFFSET - 720,
    RGY_ERR_CUDA_ERROR_NOT_PERMITTED = RGY_ERR_CUDA_DRIVER_OFFSET - 800,
    RGY_ERR_CUDA_ERROR_NOT_SUPPORTED = RGY_ERR_CUDA_DRIVER_OFFSET - 801,
    RGY_ERR_CUDA_ERROR_SYSTEM_NOT_READY = RGY_ERR_CUDA_DRIVER_OFFSET - 802,
    RGY_ERR_CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = RGY_ERR_CUDA_DRIVER_OFFSET - 803,
    RGY_ERR_CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = RGY_ERR_CUDA_DRIVER_OFFSET - 804,
    RGY_ERR_CUDA_ERROR_MPS_CONNECTION_FAILED = RGY_ERR_CUDA_DRIVER_OFFSET - 805,
    RGY_ERR_CUDA_ERROR_MPS_RPC_FAILURE = RGY_ERR_CUDA_DRIVER_OFFSET - 806,
    RGY_ERR_CUDA_ERROR_MPS_SERVER_NOT_READY = RGY_ERR_CUDA_DRIVER_OFFSET - 807,
    RGY_ERR_CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = RGY_ERR_CUDA_DRIVER_OFFSET - 808,
    RGY_ERR_CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = RGY_ERR_CUDA_DRIVER_OFFSET - 809,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = RGY_ERR_CUDA_DRIVER_OFFSET - 900,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = RGY_ERR_CUDA_DRIVER_OFFSET - 901,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_MERGE = RGY_ERR_CUDA_DRIVER_OFFSET - 902,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = RGY_ERR_CUDA_DRIVER_OFFSET - 903,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_UNJOINED = RGY_ERR_CUDA_DRIVER_OFFSET - 904,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_ISOLATION = RGY_ERR_CUDA_DRIVER_OFFSET - 905,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = RGY_ERR_CUDA_DRIVER_OFFSET - 906,
    RGY_ERR_CUDA_ERROR_CAPTURED_EVENT = RGY_ERR_CUDA_DRIVER_OFFSET - 907,
    RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = RGY_ERR_CUDA_DRIVER_OFFSET - 908,
    RGY_ERR_CUDA_ERROR_TIMEOUT = RGY_ERR_CUDA_DRIVER_OFFSET - 909,
    RGY_ERR_CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = RGY_ERR_CUDA_DRIVER_OFFSET - 910,
    RGY_ERR_CUDA_ERROR_EXTERNAL_DEVICE = RGY_ERR_CUDA_DRIVER_OFFSET - 911,
    RGY_ERR_CUDA_ERROR_UNKNOWN = RGY_ERR_CUDA_DRIVER_OFFSET - 999,

    // npp error
    RGY_ERR_NPP_OFFSET = -40000,
    RGY_ERR_NPP_NOT_SUPPORTED_MODE_ERROR = RGY_ERR_NPP_OFFSET  -9999,
    RGY_ERR_NPP_INVALID_HOST_POINTER_ERROR = RGY_ERR_NPP_OFFSET  -1032,
    RGY_ERR_NPP_INVALID_DEVICE_POINTER_ERROR = RGY_ERR_NPP_OFFSET  -1031,
    RGY_ERR_NPP_LUT_PALETTE_BITSIZE_ERROR = RGY_ERR_NPP_OFFSET  -1030,
    RGY_ERR_NPP_ZC_MODE_NOT_SUPPORTED_ERROR = RGY_ERR_NPP_OFFSET  -1028,
    RGY_ERR_NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY = RGY_ERR_NPP_OFFSET  -1027,
    RGY_ERR_NPP_TEXTURE_BIND_ERROR = RGY_ERR_NPP_OFFSET  -1024,
    RGY_ERR_NPP_WRONG_INTERSECTION_ROI_ERROR = RGY_ERR_NPP_OFFSET  -1020,
    RGY_ERR_NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR = RGY_ERR_NPP_OFFSET  -1006,
    RGY_ERR_NPP_MEMFREE_ERROR = RGY_ERR_NPP_OFFSET  -1005,
    RGY_ERR_NPP_MEMSET_ERROR = RGY_ERR_NPP_OFFSET  -1004,
    RGY_ERR_NPP_MEMCPY_ERROR = RGY_ERR_NPP_OFFSET  -1003,
    RGY_ERR_NPP_ALIGNMENT_ERROR = RGY_ERR_NPP_OFFSET  -1002,
    RGY_ERR_NPP_CUDA_KERNEL_EXECUTION_ERROR = RGY_ERR_NPP_OFFSET  -1000,
    RGY_ERR_NPP_ROUND_MODE_NOT_SUPPORTED_ERROR = RGY_ERR_NPP_OFFSET  -213,
    RGY_ERR_NPP_QUALITY_INDEX_ERROR = RGY_ERR_NPP_OFFSET  -210,
    RGY_ERR_NPP_RESIZE_NO_OPERATION_ERROR = RGY_ERR_NPP_OFFSET  -201,
    RGY_ERR_NPP_OVERFLOW_ERROR = RGY_ERR_NPP_OFFSET  -109,
    RGY_ERR_NPP_NOT_EVEN_STEP_ERROR = RGY_ERR_NPP_OFFSET  -108,
    RGY_ERR_NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR = RGY_ERR_NPP_OFFSET  -107,
    RGY_ERR_NPP_LUT_NUMBER_OF_LEVELS_ERROR = RGY_ERR_NPP_OFFSET  -106,
    RGY_ERR_NPP_CORRUPTED_DATA_ERROR = RGY_ERR_NPP_OFFSET  -61,
    RGY_ERR_NPP_CHANNEL_ORDER_ERROR = RGY_ERR_NPP_OFFSET  -60,
    RGY_ERR_NPP_ZERO_MASK_VALUE_ERROR = RGY_ERR_NPP_OFFSET  -59,
    RGY_ERR_NPP_QUADRANGLE_ERROR = RGY_ERR_NPP_OFFSET  -58,
    RGY_ERR_NPP_RECTANGLE_ERROR = RGY_ERR_NPP_OFFSET  -57,
    RGY_ERR_NPP_COEFFICIENT_ERROR = RGY_ERR_NPP_OFFSET  -56,
    RGY_ERR_NPP_NUMBER_OF_CHANNELS_ERROR = RGY_ERR_NPP_OFFSET  -53,
    RGY_ERR_NPP_COI_ERROR = RGY_ERR_NPP_OFFSET  -52,
    RGY_ERR_NPP_DIVISOR_ERROR = RGY_ERR_NPP_OFFSET  -51,
    RGY_ERR_NPP_CHANNEL_ERROR = RGY_ERR_NPP_OFFSET  -47,
    RGY_ERR_NPP_STRIDE_ERROR = RGY_ERR_NPP_OFFSET  -37,
    RGY_ERR_NPP_ANCHOR_ERROR = RGY_ERR_NPP_OFFSET  -34,
    RGY_ERR_NPP_MASK_SIZE_ERROR = RGY_ERR_NPP_OFFSET  -33,
    RGY_ERR_NPP_RESIZE_FACTOR_ERROR = RGY_ERR_NPP_OFFSET  -23,
    RGY_ERR_NPP_INTERPOLATION_ERROR = RGY_ERR_NPP_OFFSET  -22,
    RGY_ERR_NPP_MIRROR_FLIP_ERROR = RGY_ERR_NPP_OFFSET  -21,
    RGY_ERR_NPP_MOMENT_00_ZERO_ERROR = RGY_ERR_NPP_OFFSET  -20,
    RGY_ERR_NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR = RGY_ERR_NPP_OFFSET  -19,
    RGY_ERR_NPP_THRESHOLD_ERROR = RGY_ERR_NPP_OFFSET  -18,
    RGY_ERR_NPP_CONTEXT_MATCH_ERROR = RGY_ERR_NPP_OFFSET  -17,
    RGY_ERR_NPP_FFT_FLAG_ERROR = RGY_ERR_NPP_OFFSET  -16,
    RGY_ERR_NPP_FFT_ORDER_ERROR = RGY_ERR_NPP_OFFSET  -15,
    RGY_ERR_NPP_STEP_ERROR = RGY_ERR_NPP_OFFSET  -14,
    RGY_ERR_NPP_SCALE_RANGE_ERROR = RGY_ERR_NPP_OFFSET  -13,
    RGY_ERR_NPP_DATA_TYPE_ERROR = RGY_ERR_NPP_OFFSET  -12,
    RGY_ERR_NPP_OUT_OFF_RANGE_ERROR = RGY_ERR_NPP_OFFSET  -11,
    RGY_ERR_NPP_DIVIDE_BY_ZERO_ERROR = RGY_ERR_NPP_OFFSET  -10,
    RGY_ERR_NPP_MEMORY_ALLOCATION_ERR = RGY_ERR_NPP_OFFSET  -9,
    RGY_ERR_NPP_NULL_POINTER_ERROR = RGY_ERR_NPP_OFFSET  -8,
    RGY_ERR_NPP_RANGE_ERROR = RGY_ERR_NPP_OFFSET  -7,
    RGY_ERR_NPP_SIZE_ERROR = RGY_ERR_NPP_OFFSET  -6,
    RGY_ERR_NPP_BAD_ARGUMENT_ERROR = RGY_ERR_NPP_OFFSET  -5,
    RGY_ERR_NPP_NO_MEMORY_ERROR = RGY_ERR_NPP_OFFSET  -4,
    RGY_ERR_NPP_NOT_IMPLEMENTED_ERROR = RGY_ERR_NPP_OFFSET  -3,
    RGY_ERR_NPP_ERROR = RGY_ERR_NPP_OFFSET  -2,
    RGY_ERR_NPP_ERROR_RESERVED = RGY_ERR_NPP_OFFSET  -1,
    RGY_ERR_NPP_NO_ERROR = RGY_ERR_NONE,
    RGY_ERR_NPP_SUCCESS = RGY_ERR_NONE,
    //RGY_ERR_NPP_NO_OPERATION_WARNING = RGY_ERR_NPP_OFFSET  1,
    //RGY_ERR_NPP_DIVIDE_BY_ZERO_WARNING = RGY_ERR_NPP_OFFSET  6,
    //RGY_ERR_NPP_AFFINE_QUAD_INCORRECT_WARNING = RGY_ERR_NPP_OFFSET  28,
    //RGY_ERR_NPP_WRONG_INTERSECTION_ROI_WARNING = RGY_ERR_NPP_OFFSET  29,
    //RGY_ERR_NPP_WRONG_INTERSECTION_QUAD_WARNING = RGY_ERR_NPP_OFFSET  30,
    //RGY_ERR_NPP_DOUBLE_SIZE_WARNING = RGY_ERR_NPP_OFFSET  35,
    //RGY_ERR_NPP_MISALIGNED_DST_ROI_WARNING = RGY_ERR_NPP_OFFSET  10000,

    //nvoffruc error
    RGY_ERR_NvOFFRUC_OFFSET = -50000,
    RGY_ERR_NvOFFRUC_NvOFFRUC_NOT_SUPPORTED = RGY_ERR_NvOFFRUC_OFFSET-1,
    RGY_ERR_NvOFFRUC_INVALID_PTR = RGY_ERR_NvOFFRUC_OFFSET - 2,
    RGY_ERR_NvOFFRUC_INVALID_PARAM = RGY_ERR_NvOFFRUC_OFFSET - 3,
    RGY_ERR_NvOFFRUC_INVALID_HANDLE = RGY_ERR_NvOFFRUC_OFFSET - 4,
    RGY_ERR_NvOFFRUC_OUT_OF_SYSTEM_MEMORY = RGY_ERR_NvOFFRUC_OFFSET - 5,
    RGY_ERR_NvOFFRUC_OUT_OF_VIDEO_MEMORY = RGY_ERR_NvOFFRUC_OFFSET - 6,
    RGY_ERR_NvOFFRUC_OPENCV_NOT_AVAILABLE = RGY_ERR_NvOFFRUC_OFFSET - 7,
    RGY_ERR_NvOFFRUC_UNIMPLEMENTED = RGY_ERR_NvOFFRUC_OFFSET - 8,
    RGY_ERR_NvOFFRUC_OF_FAILURE = RGY_ERR_NvOFFRUC_OFFSET - 9,
    RGY_ERR_NvOFFRUC_DUPLICATE_RESOURCE = RGY_ERR_NvOFFRUC_OFFSET - 10,
    RGY_ERR_NvOFFRUC_UNREGISTERED_RESOURCE = RGY_ERR_NvOFFRUC_OFFSET - 11,
    RGY_ERR_NvOFFRUC_INCORRECT_API_SEQUENCE = RGY_ERR_NvOFFRUC_OFFSET - 12,
    RGY_ERR_NvOFFRUC_WRITE_TODISK_FAILED = RGY_ERR_NvOFFRUC_OFFSET - 13,
    RGY_ERR_NvOFFRUC_PIPELINE_EXECUTION_FAILURE = RGY_ERR_NvOFFRUC_OFFSET - 14,
    RGY_ERR_NvOFFRUC_SYNC_WRITE_FAILED = RGY_ERR_NvOFFRUC_OFFSET - 15,
    RGY_ERR_NvOFFRUC_GENERIC = RGY_ERR_NvOFFRUC_OFFSET - 16,

    // NVSDK_NGX error
    RGY_ERR_NVSDK_NGX_Fail = -60000,
    RGY_ERR_NVSDK_NGX_FeatureNotSupported = -60001,
    RGY_ERR_NVSDK_NGX_PlatformError = -60002,
    RGY_ERR_NVSDK_NGX_FeatureAlreadyExists = -60003,
    RGY_ERR_NVSDK_NGX_FeatureNotFound = -60004,
    RGY_ERR_NVSDK_NGX_InvalidParameter = -60005,
    RGY_ERR_NVSDK_NGX_ScratchBufferTooSmall = -60006,
    RGY_ERR_NVSDK_NGX_NotInitialized = -60007,
    RGY_ERR_NVSDK_NGX_UnsupportedInputFormat = -60008,
    RGY_ERR_NVSDK_NGX_RWFlagMissing = -60009,
    RGY_ERR_NVSDK_NGX_MissingInput = -60010,
    RGY_ERR_NVSDK_NGX_UnableToInitializeFeature = -60011,
    RGY_ERR_NVSDK_NGX_OutOfDate = -60012,
    RGY_ERR_NVSDK_NGX_OutOfGPUMemory = -60013,
    RGY_ERR_NVSDK_NGX_UnsupportedFormat = -60014,
    RGY_ERR_NVSDK_NGX_UnableToWriteToAppDataPath = -60015,
    RGY_ERR_NVSDK_NGX_UnsupportedParameter = -60016,
    RGY_ERR_NVSDK_NGX_Denied = -60017,
    RGY_ERR_NVSDK_NGX_NotImplemented = -60018,
};

#if ENCODER_QSV
mfxStatus err_to_mfx(RGY_ERR err);
RGY_ERR err_to_rgy(mfxStatus err);
#endif //#if ENCODER_QSV

#if ENCODER_NVENC || CUFILTERS
#if ENCODER_NVENC
NVENCSTATUS err_to_nv(RGY_ERR err);
RGY_ERR err_to_rgy(NVENCSTATUS err);
#endif
#if ENABLE_NVVFX
NvCV_Status err_to_nvcv(RGY_ERR err);
RGY_ERR err_to_rgy(NvCV_Status err);
#endif //#if ENABLE_NVVFX

cudaError err_to_cuda(RGY_ERR err);
RGY_ERR err_to_rgy(cudaError err);

CUresult err_to_cuda_driver(RGY_ERR err);
RGY_ERR err_to_rgy(CUresult err);

NppStatus err_to_npp(RGY_ERR err);
RGY_ERR err_to_rgy(NppStatus err);
#endif //#if ENCODER_NVENC

#if ENCODER_VCEENC
AMF_RESULT err_to_amf(RGY_ERR err);
RGY_ERR err_to_rgy(AMF_RESULT err);
#endif //#if ENCODER_VCEENC

#if ENABLE_VULKAN
VkResult err_to_vk(RGY_ERR err);
RGY_ERR err_to_rgy(VkResult err);
#endif //#if ENABLE_VULKAN

#if ENCODER_MPP
MPP_RET err_to_mpp(RGY_ERR err);
RGY_ERR err_to_rgy(MPP_RET err);
IM_STATUS err_to_im2d(RGY_ERR err);
RGY_ERR err_to_rgy(IM_STATUS err);
#endif //#if ENCODER_MPP


const TCHAR *get_err_mes(RGY_ERR sts);

#endif //__RGY_ERR_H__
