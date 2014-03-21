/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


#ifndef _DRVAPI_ERROR_STRING_H_
#define _DRVAPI_ERROR_STRING_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef  __cuda_cuda_h__ // check to see if CUDA_H is included above

// Error Code string definitions here
typedef struct
{
    char const *error_string;
    int  error_id;
} s_CudaErrorStr;

/**
 * Error codes
 */
static s_CudaErrorStr sCudaDrvErrorString[] =
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    { "CUDA_SUCCESS", 0 },

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    { "CUDA_ERROR_INVALID_VALUE", 1 },

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    { "CUDA_ERROR_OUT_OF_MEMORY", 2 },

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    { "CUDA_ERROR_NOT_INITIALIZED", 3 },

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    { "CUDA_ERROR_DEINITIALIZED", 4 },

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
    */
    { "CUDA_ERROR_PROFILER_DISABLED", 5 },
    /**
     * This indicates profiling has not been initialized for this context.
     * Call cuProfilerInitialize() to resolve this.
    */
    { "CUDA_ERROR_PROFILER_NOT_INITIALIZED", 6 },
    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
    */
    { "CUDA_ERROR_PROFILER_ALREADY_STARTED", 7 },
    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
    */
    { "CUDA_ERROR_PROFILER_ALREADY_STOPPED", 8 },
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    { "CUDA_ERROR_NO_DEVICE (no CUDA-capable devices were detected)", 100 },

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    { "CUDA_ERROR_INVALID_DEVICE (device specified is not a valid CUDA device)", 101 },


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    { "CUDA_ERROR_INVALID_IMAGE", 200 },

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    { "CUDA_ERROR_INVALID_CONTEXT", 201 },

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    { "CUDA_ERROR_CONTEXT_ALREADY_CURRENT", 202 },

    /**
     * This indicates that a map or register operation has failed.
     */
    { "CUDA_ERROR_MAP_FAILED", 205 },

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    { "CUDA_ERROR_UNMAP_FAILED", 206 },

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    { "CUDA_ERROR_ARRAY_IS_MAPPED", 207 },

    /**
     * This indicates that the resource is already mapped.
     */
    { "CUDA_ERROR_ALREADY_MAPPED", 208 },

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    { "CUDA_ERROR_NO_BINARY_FOR_GPU", 209 },

    /**
     * This indicates that a resource has already been acquired.
     */
    { "CUDA_ERROR_ALREADY_ACQUIRED", 210 },

    /**
     * This indicates that a resource is not mapped.
     */
    { "CUDA_ERROR_NOT_MAPPED", 211 },

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    { "CUDA_ERROR_NOT_MAPPED_AS_ARRAY", 212 },

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    { "CUDA_ERROR_NOT_MAPPED_AS_POINTER", 213 },

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    { "CUDA_ERROR_ECC_UNCORRECTABLE", 214 },

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    { "CUDA_ERROR_UNSUPPORTED_LIMIT", 215 },

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    { "CUDA_ERROR_CONTEXT_ALREADY_IN_USE", 216 },

    /**
     * This indicates that the device kernel source is invalid.
     */
    { "CUDA_ERROR_INVALID_SOURCE", 300 },

    /**
     * This indicates that the file specified was not found.
     */
    { "CUDA_ERROR_FILE_NOT_FOUND", 301 },

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    { "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302 },

    /**
     * This indicates that initialization of a shared object failed.
     */
    { "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303 },

    /**
     * This indicates that an OS call failed.
     */
    { "CUDA_ERROR_OPERATING_SYSTEM", 304 },


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    { "CUDA_ERROR_INVALID_HANDLE", 400 },


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names }, and surface names.
     */
    { "CUDA_ERROR_NOT_FOUND", 500 },


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    { "CUDA_ERROR_NOT_READY", 600 },


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used }, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_FAILED", 700 },

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    { "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701 },

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_TIMEOUT", 702 },

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    { "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703 },

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    { "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704 },

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     */
    { "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705 },

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    { "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708 },

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy }, or is a primary context which
     * has not yet been initialized.
     */
    { "CUDA_ERROR_CONTEXT_IS_DESTROYED", 709 },

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_ASSERT", 710 },

    /**
     * This indicates that an unknown internal error has occurred.
     */
    { "CUDA_ERROR_UNKNOWN", 999 },
    { NULL, -1 }
};

// This is just a linear search through the array, since the error_id's are not
// always ocurring consecutively
inline const char *getCudaDrvErrorString(CUresult error_id)
{
    int index = 0;

    while (sCudaDrvErrorString[index].error_id != error_id &&
           sCudaDrvErrorString[index].error_id != -1)
    {
        index++;
    }

    if (sCudaDrvErrorString[index].error_id == error_id)
        return (const char *)sCudaDrvErrorString[index].error_string;
    else
        return (const char *)"CUDA_ERROR not found!";
}

#endif // __cuda_cuda_h__


#endif
