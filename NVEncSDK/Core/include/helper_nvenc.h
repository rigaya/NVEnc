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

////////////////////////////////////////////////////////////////////////////////
// These are NVENC Helper functions for initialization and error checking

#ifndef HELPER_NVENC_H
#define HELPER_NVENC_H

#include <nvEncodeAPI.h>

static const char *_nvencGetErrorEnum(NVENCSTATUS error)
{
    switch (error)
    {
        case NV_ENC_SUCCESS:
            return "NVENC Success";

        case NV_ENC_ERR_NO_ENCODE_DEVICE:
            return "NVENC No Available Encoding Device";

        case NV_ENC_ERR_UNSUPPORTED_DEVICE:
            return "NVENC that devices pass by the client is not supported";

        case NV_ENC_ERR_INVALID_ENCODERDEVICE:
            return "NVENC this indicates that the encoder device supplied by the client is not valid";

        case NV_ENC_ERR_INVALID_DEVICE:
            return "NVENC this indicates that device passed to the API call is invalid";

        case NV_ENC_ERR_DEVICE_NOT_EXIST:
            return "NVENC This indicates that device passed to the API call is no longer available and needs to be reinitialized.";

        case NV_ENC_ERR_INVALID_PTR:
            return "NVENC one or more of the pointers passed to the API call is invalid.";

        case NV_ENC_ERR_INVALID_EVENT:
            return "NVENC indicates that completion event passed in ::NvEncEncodePicture() call is invalid.";

        case NV_ENC_ERR_INVALID_PARAM:
            return "NVENC indicates that one or more of the parameter passed to the API call is invalid.";

        case NV_ENC_ERR_INVALID_CALL:
            return "NVENC indicates that an API call was made in wrong sequence/order.";

        case NV_ENC_ERR_OUT_OF_MEMORY:
            return "NVENC indicates that the API call failed because it was unable to allocate enough memory to perform the requested operation.";

        case NV_ENC_ERR_ENCODER_NOT_INITIALIZED:
            return "NVENC indicates that the encoder has not been initialized with::NvEncInitializeEncoder() or that initialization has failed.";

        case NV_ENC_ERR_UNSUPPORTED_PARAM:
            return "NVENC  that an unsupported parameter was passed by the client.";

        case NV_ENC_ERR_LOCK_BUSY:
            return "NVENC indicates that the ::NvEncLockBitstream() failed to lock the output  buffer.";

        case NV_ENC_ERR_NOT_ENOUGH_BUFFER:
            return "NVENC indicates that the size of the user buffer passed by the client is insufficient for the requested operation";

        case NV_ENC_ERR_INVALID_VERSION:
            return "NVENC indicates that an invalid struct version was used by the client";

        case NV_ENC_ERR_MAP_FAILED:
            return "NVENC NvEncMapInputResource() API failed to map the client provided input resource.";

        case NV_ENC_ERR_NEED_MORE_INPUT:
            return "NVENC HW encode driver requires more input buffers to produce an output bitstream";

        case NV_ENC_ERR_ENCODER_BUSY:
            return "NVENC HW encoder is busy encoding and is unable to encode the input. The client should call ::NvEncEncodePicture() again after few milliseconds.";

        case NV_ENC_ERR_EVENT_NOT_REGISTERD:
            return "NVENC completion event passed in ::NvEncEncodePicture() API has not been registered with encoder driver using ::NvEncRegisterAsyncEvent()";

        case NV_ENC_ERR_GENERIC:
            return "NVENC unknown internal error";

        case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY:
            return "NVENC Feature not available for current license key type";

        case NV_ENC_ERR_UNIMPLEMENTED:
            return "NVENC Feature has not been implemented yet.";

        case NV_ENC_ERR_RESOURCE_REGISTER_FAILED:
            return "NVENC NvEncRegisterResource failed to register resource.";

        case NV_ENC_ERR_RESOURCE_NOT_REGISTERED:
            return "NVENC Client is attempting to unregister resource that hasn't been registered.";

        case NV_ENC_ERR_RESOURCE_NOT_MAPPED:
            return "NVENC Client is attempting to unmap resource that hasn't been mapped.";
    }

    return "<unknown>";
}

template< typename T >
void check_error(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "NVENC error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _nvencGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkNVENCErrors(val)           check_error ( (val), #val, __FILE__, __LINE__ )

#endif
