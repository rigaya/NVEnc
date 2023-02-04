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
#if ENCODER_NVENC
#pragma warning (push)
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include "nvEncodeAPI.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvCVStatus.h"
#pragma warning(pop)
#endif
#if ENCODER_VCEENC
#include "core/Result.h"
#include "rgy_vulkan.h"
#endif

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
    RGY_NVCV_ERR_OFFSET                = -200,    //!< The procedure returned successfully.
    RGY_NVCV_ERR_GENERAL               = RGY_NVCV_ERR_OFFSET -1,   //!< An otherwise unspecified error has occurred.
    RGY_NVCV_ERR_UNIMPLEMENTED         = RGY_NVCV_ERR_OFFSET -2,   //!< The requested feature is not yet implemented.
    RGY_NVCV_ERR_MEMORY                = RGY_NVCV_ERR_OFFSET -3,   //!< There is not enough memory for the requested operation.
    RGY_NVCV_ERR_EFFECT                = RGY_NVCV_ERR_OFFSET -4,   //!< An invalid effect handle has been supplied.
    RGY_NVCV_ERR_SELECTOR              = RGY_NVCV_ERR_OFFSET -5,   //!< The given parameter selector is not valid in this effect filter.
    RGY_NVCV_ERR_BUFFER                = RGY_NVCV_ERR_OFFSET -6,   //!< An image buffer has not been specified.
    RGY_NVCV_ERR_PARAMETER             = RGY_NVCV_ERR_OFFSET -7,   //!< An invalid parameter value has been supplied for this effect+selector.
    RGY_NVCV_ERR_MISMATCH              = RGY_NVCV_ERR_OFFSET -8,   //!< Some parameters are not appropriately matched.
    RGY_NVCV_ERR_PIXELFORMAT           = RGY_NVCV_ERR_OFFSET -9,   //!< The specified pixel format is not accommodated.
    RGY_NVCV_ERR_MODEL                 = RGY_NVCV_ERR_OFFSET -10,  //!< Error while loading the TRT model.
    RGY_NVCV_ERR_LIBRARY               = RGY_NVCV_ERR_OFFSET -11,  //!< Error loading the dynamic library.
    RGY_NVCV_ERR_INITIALIZATION        = RGY_NVCV_ERR_OFFSET -12,  //!< The effect has not been properly initialized.
    RGY_NVCV_ERR_FILE                  = RGY_NVCV_ERR_OFFSET -13,  //!< The file could not be found.
    RGY_NVCV_ERR_FEATURENOTFOUND       = RGY_NVCV_ERR_OFFSET -14,  //!< The requested feature was not found
    RGY_NVCV_ERR_MISSINGINPUT          = RGY_NVCV_ERR_OFFSET -15,  //!< A required parameter was not set
    RGY_NVCV_ERR_RESOLUTION            = RGY_NVCV_ERR_OFFSET -16,  //!< The specified image resolution is not supported.
    RGY_NVCV_ERR_UNSUPPORTEDGPU        = RGY_NVCV_ERR_OFFSET -17,  //!< The GPU is not supported
    RGY_NVCV_ERR_WRONGGPU              = RGY_NVCV_ERR_OFFSET -18,  //!< The current GPU is not the one selected.
    RGY_NVCV_ERR_UNSUPPORTEDDRIVER     = RGY_NVCV_ERR_OFFSET -19,  //!< The currently installed graphics driver is not supported
    RGY_NVCV_ERR_MODELDEPENDENCIES     = RGY_NVCV_ERR_OFFSET -20,  //!< There is no model with dependencies that match this system
    RGY_NVCV_ERR_PARSE                 = RGY_NVCV_ERR_OFFSET -21,  //!< There has been a parsing or syntax error while reading a file
    RGY_NVCV_ERR_MODELSUBSTITUTION     = RGY_NVCV_ERR_OFFSET -22,  //!< The specified model does not exist and has been substituted.
    RGY_NVCV_ERR_READ                  = RGY_NVCV_ERR_OFFSET -23,  //!< An error occurred while reading a file.
    RGY_NVCV_ERR_WRITE                 = RGY_NVCV_ERR_OFFSET -24,  //!< An error occurred while writing a file.
    RGY_NVCV_ERR_PARAMREADONLY         = RGY_NVCV_ERR_OFFSET -25,  //!< The selected parameter is read-only.
    RGY_NVCV_ERR_TRT_ENQUEUE           = RGY_NVCV_ERR_OFFSET -26,  //!< TensorRT enqueue failed.
    RGY_NVCV_ERR_TRT_BINDINGS          = RGY_NVCV_ERR_OFFSET -27,  //!< Unexpected TensorRT bindings.
    RGY_NVCV_ERR_TRT_CONTEXT           = RGY_NVCV_ERR_OFFSET -28,  //!< An error occurred while creating a TensorRT context.
    RGY_NVCV_ERR_TRT_INFER             = RGY_NVCV_ERR_OFFSET -29,  ///< The was a problem creating the inference engine.
    RGY_NVCV_ERR_TRT_ENGINE            = RGY_NVCV_ERR_OFFSET -30,  ///< There was a problem deserializing the inference runtime engine.
    RGY_NVCV_ERR_NPP                   = RGY_NVCV_ERR_OFFSET -31,  //!< An error has occurred in the NPP library.
    RGY_NVCV_ERR_CONFIG                = RGY_NVCV_ERR_OFFSET -32,  //!< No suitable model exists for the specified parameter configuration.
    RGY_NVCV_ERR_TOOSMALL              = RGY_NVCV_ERR_OFFSET -33,  //!< A supplied parameter or buffer is not large enough.
    RGY_NVCV_ERR_TOOBIG                = RGY_NVCV_ERR_OFFSET -34,  //!< A supplied parameter is too big.
    RGY_NVCV_ERR_WRONGSIZE             = RGY_NVCV_ERR_OFFSET -35,  //!< A supplied parameter is not the expected size.
    RGY_NVCV_ERR_OBJECTNOTFOUND        = RGY_NVCV_ERR_OFFSET -36,  //!< The specified object was not found.
    RGY_NVCV_ERR_SINGULAR              = RGY_NVCV_ERR_OFFSET -37,  //!< A mathematical singularity has been encountered.
    RGY_NVCV_ERR_NOTHINGRENDERED       = RGY_NVCV_ERR_OFFSET -38,  //!< Nothing was rendered in the specified region.
    RGY_NVCV_ERR_CONVERGENCE           = RGY_NVCV_ERR_OFFSET -39,  //!< An iteration did not converge satisfactorily.

    RGY_NVCV_ERR_OPENGL                = RGY_NVCV_ERR_OFFSET -40,  //!< An OpenGL error has occurred.
    RGY_NVCV_ERR_DIRECT3D              = RGY_NVCV_ERR_OFFSET -41,  //!< A Direct3D error has occurred.

    RGY_NVCV_ERR_CUDA_BASE             = RGY_NVCV_ERR_OFFSET -50,  //!< CUDA errors are offset from this value.
    RGY_NVCV_ERR_CUDA_VALUE            = RGY_NVCV_ERR_OFFSET -51,  //!< A CUDA parameter is not within the acceptable range.
    RGY_NVCV_ERR_CUDA_MEMORY           = RGY_NVCV_ERR_OFFSET -52,  //!< There is not enough CUDA memory for the requested operation.
    RGY_NVCV_ERR_CUDA_PITCH            = RGY_NVCV_ERR_OFFSET -53,  //!< A CUDA pitch is not within the acceptable range.
    RGY_NVCV_ERR_CUDA_INIT             = RGY_NVCV_ERR_OFFSET -54,  //!< The CUDA driver and runtime could not be initialized.
    RGY_NVCV_ERR_CUDA_LAUNCH           = RGY_NVCV_ERR_OFFSET -55,  //!< The CUDA kernel launch has failed.
    RGY_NVCV_ERR_CUDA_KERNEL           = RGY_NVCV_ERR_OFFSET -56,  //!< No suitable kernel image is available for the device.
    RGY_NVCV_ERR_CUDA_DRIVER           = RGY_NVCV_ERR_OFFSET -57,  //!< The installed NVIDIA CUDA driver is older than the CUDA runtime library.
    RGY_NVCV_ERR_CUDA_UNSUPPORTED      = RGY_NVCV_ERR_OFFSET -58,  //!< The CUDA operation is not supported on the current system or device.
    RGY_NVCV_ERR_CUDA_ILLEGAL_ADDRESS  = RGY_NVCV_ERR_OFFSET -59,  //!< CUDA tried to load or store on an invalid memory address.
    RGY_NVCV_ERR_CUDA                  = RGY_NVCV_ERR_OFFSET -60, //!< An otherwise unspecified CUDA error has been reported.

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
};

#if ENCODER_QSV
mfxStatus err_to_mfx(RGY_ERR err);
RGY_ERR err_to_rgy(mfxStatus err);
#endif //#if ENCODER_QSV

#if ENCODER_NVENC
NVENCSTATUS err_to_nv(RGY_ERR err);
RGY_ERR err_to_rgy(NVENCSTATUS err);
NvCV_Status err_to_nvcv(RGY_ERR err);
RGY_ERR err_to_rgy(NvCV_Status err);

cudaError_t err_to_cuda(RGY_ERR err);
RGY_ERR err_to_rgy(cudaError_t err);
#endif //#if ENCODER_NVENC

#if ENCODER_VCEENC
AMF_RESULT err_to_amf(RGY_ERR err);
RGY_ERR err_to_rgy(AMF_RESULT err);

#if ENABLE_VULKAN
VkResult err_to_vk(RGY_ERR err);
RGY_ERR err_to_rgy(VkResult err);
#endif //#if ENABLE_VULKAN
#endif //#if ENCODER_VCEENC

const TCHAR *get_err_mes(RGY_ERR sts);

#endif //__RGY_ERR_H__
