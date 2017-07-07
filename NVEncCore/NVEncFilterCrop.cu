// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <map>
#include <array>
#include <cstdint>
#include "NVEncFilter.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

union RGY_CSP_2 {
    struct {
        RGY_CSP a, b;
    } csp;
    uint64_t i;

    RGY_CSP_2() {

    };
    RGY_CSP_2(RGY_CSP _a, RGY_CSP _b) {
        csp.a = _a;
        csp.b = _b;
    };
};

#define BIT_DEPTH_CONV(x) (TypeOut)((out_bit_depth == in_bit_depth) \
    ? (x) \
    : ((out_bit_depth > in_bit_depth) \
        ? ((x) << (out_bit_depth - in_bit_depth)) \
        : ((x) >> (in_bit_depth - out_bit_depth))))

#define BIT_DEPTH_CONV_FLOAT(x) (TypeOut)((out_bit_depth == in_bit_depth) \
    ? (x) \
    : ((out_bit_depth > in_bit_depth) \
        ? ((x) * (float)(1 << (out_bit_depth - in_bit_depth))) \
        : ((x) * (float)(1.0f / (1 << (in_bit_depth - out_bit_depth))))))

#define BIT_DEPTH_CONV_AVG(a, b) (TypeOut)((out_bit_depth == in_bit_depth + 1) \
    ? ((int)(a) + (int)(b) + 1) \
    : ((out_bit_depth > in_bit_depth + 1) \
        ? (((int)(a) + (int)(b) + 1) << (out_bit_depth - in_bit_depth - 1)) \
        : (((int)(a) + (int)(b) + (1 << (in_bit_depth - out_bit_depth))) >> (in_bit_depth + 1 - out_bit_depth))))

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_nv12_nv12(TypeOut *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const TypeIn *__restrict__ pSrc, const int srcPitch, const int srcHeight, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        //Y
        int idst = y * dstPitch + x;
        int isrc = (y + offsetY) * srcPitch + x + offsetX;
        pDst[idst] = pSrc[isrc];
        if (y < (dstHeight >> 1)) {
            //UV
            idst += dstHeight * dstPitch;
            isrc += (srcHeight - (offsetY >> 1)) * srcPitch;
            pDst[idst] = pSrc[isrc];
        }
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_y(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        //Y
        int idst = y * dstPitch + x * sizeof(TypeOut);
        int isrc = (y + offsetY) * srcPitch + (x + offsetX) * sizeof(TypeIn);
        const TypeIn *ptr_src = (const TypeIn *)(pSrc + isrc);
        TypeOut *ptr_dst = (TypeOut *)(pDst + idst);
        ptr_dst[0] = BIT_DEPTH_CONV(ptr_src[0]);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_y(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(64, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    kernel_crop_y<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        (uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        (uint8_t *)pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(64, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    kernel_crop_y<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height >> 1,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        pInputFrame->pitch, pCrop->e.left, pCrop->e.up >> 1);
}

NVENCSTATUS NVEncFilterCspCrop::convertYBitDepth(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
#define CONV_DEPTH_TO_FROM(to, from) ((to) << 8 | (from))
    static const std::map<int, void (*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> crop_y_list = {
        { CONV_DEPTH_TO_FROM(16,  8), crop_y<uint16_t, 16, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM(14,  8), crop_y<uint16_t, 14, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM(12,  8), crop_y<uint16_t, 12, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM(10,  8), crop_y<uint16_t, 10, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM( 9,  8), crop_y<uint16_t,  9, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM( 8, 16), crop_y<uint8_t,   8, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM( 8, 14), crop_y<uint8_t,   8, uint16_t, 14> },
        { CONV_DEPTH_TO_FROM( 8, 12), crop_y<uint8_t,   8, uint16_t, 12> },
        { CONV_DEPTH_TO_FROM( 8, 10), crop_y<uint8_t,   8, uint16_t, 10> },
        { CONV_DEPTH_TO_FROM( 8,  9), crop_y<uint8_t,   8, uint16_t,  9> },
        { CONV_DEPTH_TO_FROM(14, 16), crop_y<uint16_t, 14, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM(12, 16), crop_y<uint16_t, 12, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM(10, 16), crop_y<uint16_t, 10, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM( 9, 16), crop_y<uint16_t,  9, uint16_t, 16> },
    };
    const auto bit_depth_conv = CONV_DEPTH_TO_FROM(RGY_CSP_BIT_DEPTH[pOutputFrame->csp], RGY_CSP_BIT_DEPTH[pInputFrame->csp]);
    if (crop_y_list.count(bit_depth_conv) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
#undef CONV_DEPTH_TO_FROM
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    crop_y_list.at(bit_depth_conv)(pOutputFrame, pInputFrame, &pCropParam->crop);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at convertYBitDepth(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv12_yv12(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut); //YV12
        int isrc = (uv_y + (offsetY >> 1)) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //NV12
        const TypeIn *ptr_src = (const TypeIn *)(pSrc  + isrc);
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + idst);
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + idst);
        ptr_dst_u[0] = BIT_DEPTH_CONV(ptr_src[0]);
        ptr_dst_v[0] = BIT_DEPTH_CONV(ptr_src[1]);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv12_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2;
    const uint8_t *ptrC = (const uint8_t  *)pInputFrame->ptr + pInputFrame->pitch  * pInputFrame->height;
    kernel_crop_uv_nv12_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrC, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv12_yuv444_p(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcWidth, const int srcHeight, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = (uv_y << 1) * dstPitch + (uv_x << 1) * sizeof(TypeOut); //YUV444
        int isrc = (uv_y + (offsetY >> 1)) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //NV12
        const TypeIn *ptr_src = (const TypeIn *)(pSrc  + isrc);
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + idst);
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + idst);
        const int y0_offset  = (uv_y > 0) ? -1 * srcPitch : 0;
        const int y2_offset  = (uv_y+1 < (srcHeight >> 1)) ? srcPitch : 0;
        const int next_pixel = (uv_x+1 < (dstWidth >> 1)) ? 2 : 0;
        const int u_y0x0 = ptr_src[y0_offset+0];
        const int v_y0x0 = ptr_src[y0_offset+1];
        const int u_y0x1 = (ptr_src[y0_offset+next_pixel+0] + u_y0x0 + 1) >> 1;
        const int v_y0x1 = (ptr_src[y0_offset+next_pixel+1] + v_y0x0 + 1) >> 1;
        const int u_y1x0 = ptr_src[0];
        const int v_y1x0 = ptr_src[1];
        const int u_y1x1 = (ptr_src[next_pixel+0] + u_y1x0 + 1) >> 1;
        const int v_y1x1 = (ptr_src[next_pixel+1] + v_y1x0 + 1) >> 1;
        const int u_y2x0 = ptr_src[y2_offset+0];
        const int v_y2x0 = ptr_src[y2_offset+1];
        const int u_y2x1 = (ptr_src[y2_offset+next_pixel+0] + u_y2x0 + 1) >> 1;
        const int v_y2x1 = (ptr_src[y2_offset+next_pixel+1] + v_y2x0 + 1) >> 1;

#define BIT_DEPTH_CONV_a3_b1_rsh2(a, b) (TypeOut)((out_bit_depth == in_bit_depth + 2) \
    ? (3 * (a) + (b) + 2) \
    : ((out_bit_depth > in_bit_depth + 2) \
        ? ((3 * (a) + (b) + 2) << (out_bit_depth - in_bit_depth - 2)) \
        : ((3 * (a) + (b) + 2) >> (in_bit_depth + 2 - out_bit_depth))))

        ptr_dst_u[0] = BIT_DEPTH_CONV_a3_b1_rsh2(u_y1x0, u_y0x0);
        ptr_dst_v[0] = BIT_DEPTH_CONV_a3_b1_rsh2(v_y1x0, v_y0x0);
        ptr_dst_u[1] = BIT_DEPTH_CONV_a3_b1_rsh2(u_y1x1, u_y0x1);
        ptr_dst_v[1] = BIT_DEPTH_CONV_a3_b1_rsh2(v_y1x1, v_y0x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = BIT_DEPTH_CONV_a3_b1_rsh2(u_y1x0, u_y2x0);
        ptr_dst_v[0] = BIT_DEPTH_CONV_a3_b1_rsh2(v_y1x0, v_y2x0);
        ptr_dst_u[1] = BIT_DEPTH_CONV_a3_b1_rsh2(u_y1x1, u_y2x1);
        ptr_dst_v[1] = BIT_DEPTH_CONV_a3_b1_rsh2(v_y1x1, v_y2x1);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv12_yuv444_i(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcWidth, const int srcHeight, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = (uv_y << 1) * dstPitch + (uv_x << 1) * sizeof(TypeOut); //YUV444
        int isrc = (uv_y + (offsetY >> 1)) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //NV12
        const TypeIn *ptr_src = (const TypeIn *)(pSrc  + isrc);
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + idst);
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + idst);
        const int y0_offset  = (uv_y - 1 > 0) ? -2 * srcPitch : 0;
        const int y1_offset  = (uv_y > 0)     ? -1 * srcPitch : srcPitch;
        const int y3_offset  = (uv_y+1 < (srcHeight >> 1)) ? srcPitch     : y1_offset;
        const int y4_offset  = (uv_y+2 < (srcHeight >> 1)) ? srcPitch * 2 : 0;
        const int y5_offset  = (uv_y+3 < (srcHeight >> 1)) ? srcPitch * 3 : y3_offset;
        const int next_pixel = (uv_x+1 < (dstWidth >> 1)) ? 2 : 0;
        const int u_y0x0 = ptr_src[y0_offset+0];
        const int v_y0x0 = ptr_src[y0_offset+1];
        const int u_y0x1 = (ptr_src[y0_offset+next_pixel+0] + u_y0x0 + 1) >> 1;
        const int v_y0x1 = (ptr_src[y0_offset+next_pixel+1] + v_y0x0 + 1) >> 1;
        const int u_y1x0 = ptr_src[y1_offset+0];
        const int v_y1x0 = ptr_src[y1_offset+1];
        const int u_y1x1 = (ptr_src[y1_offset+next_pixel+0] + u_y1x0 + 1) >> 1;
        const int v_y1x1 = (ptr_src[y1_offset+next_pixel+1] + v_y1x0 + 1) >> 1;
        const int u_y2x0 = ptr_src[0];
        const int v_y2x0 = ptr_src[1];
        const int u_y2x1 = (ptr_src[next_pixel+0] + u_y2x0 + 1) >> 1;
        const int v_y2x1 = (ptr_src[next_pixel+1] + v_y2x0 + 1) >> 1;
        const int u_y3x0 = ptr_src[y3_offset+0];
        const int v_y3x0 = ptr_src[y3_offset+1];
        const int u_y3x1 = (ptr_src[y3_offset+next_pixel+0] + u_y3x0 + 1) >> 1;
        const int v_y3x1 = (ptr_src[y3_offset+next_pixel+1] + v_y3x0 + 1) >> 1;
        const int u_y4x0 = ptr_src[y4_offset+0];
        const int v_y4x0 = ptr_src[y4_offset+1];
        const int u_y4x1 = (ptr_src[y4_offset+next_pixel+0] + u_y4x0 + 1) >> 1;
        const int v_y4x1 = (ptr_src[y4_offset+next_pixel+1] + v_y4x0 + 1) >> 1;
        const int u_y5x0 = ptr_src[y5_offset+0];
        const int v_y5x0 = ptr_src[y5_offset+1];
        const int u_y5x1 = (ptr_src[y5_offset+next_pixel+0] + u_y5x0 + 1) >> 1;
        const int v_y5x1 = (ptr_src[y5_offset+next_pixel+1] + v_y5x0 + 1) >> 1;

#define BIT_DEPTH_CONV_ia_jb_rsh3(i, a, j, b) (TypeOut)((out_bit_depth == in_bit_depth + 3) \
    ? ((i) * (a) + (j) * (b) + 4) \
    : ((out_bit_depth > in_bit_depth + 2) \
        ? (((i) * (a) + (j) * (b) + 4) << (out_bit_depth - in_bit_depth - 3)) \
        : (((i) * (a) + (j) * (b) + 4) >> (in_bit_depth + 3 - out_bit_depth))))

        ptr_dst_u[0] = BIT_DEPTH_CONV_ia_jb_rsh3(1, u_y0x0, 7, u_y2x0);
        ptr_dst_v[0] = BIT_DEPTH_CONV_ia_jb_rsh3(1, v_y0x0, 7, v_y2x0);
        ptr_dst_u[1] = BIT_DEPTH_CONV_ia_jb_rsh3(1, u_y0x1, 7, u_y2x1);
        ptr_dst_v[1] = BIT_DEPTH_CONV_ia_jb_rsh3(1, v_y0x1, 7, v_y2x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = BIT_DEPTH_CONV_ia_jb_rsh3(3, u_y1x0, 5, u_y3x0);
        ptr_dst_v[0] = BIT_DEPTH_CONV_ia_jb_rsh3(3, v_y1x0, 5, v_y3x0);
        ptr_dst_u[1] = BIT_DEPTH_CONV_ia_jb_rsh3(3, u_y1x1, 5, u_y3x1);
        ptr_dst_v[1] = BIT_DEPTH_CONV_ia_jb_rsh3(3, v_y1x1, 5, v_y3x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = BIT_DEPTH_CONV_ia_jb_rsh3(5, u_y2x0, 3, u_y4x0);
        ptr_dst_v[0] = BIT_DEPTH_CONV_ia_jb_rsh3(5, v_y2x0, 3, v_y4x0);
        ptr_dst_u[1] = BIT_DEPTH_CONV_ia_jb_rsh3(5, u_y2x1, 3, u_y4x1);
        ptr_dst_v[1] = BIT_DEPTH_CONV_ia_jb_rsh3(5, v_y2x1, 3, v_y4x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = BIT_DEPTH_CONV_ia_jb_rsh3(7, u_y3x0, 1, u_y5x0);
        ptr_dst_v[0] = BIT_DEPTH_CONV_ia_jb_rsh3(7, v_y3x0, 1, v_y5x0);
        ptr_dst_u[1] = BIT_DEPTH_CONV_ia_jb_rsh3(7, u_y3x1, 1, u_y5x1);
        ptr_dst_v[1] = BIT_DEPTH_CONV_ia_jb_rsh3(7, v_y3x1, 1, v_y5x1);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv12_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2;
    const uint8_t *ptrC = (const uint8_t  *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    if (pInputFrame->interlaced) {
        kernel_crop_uv_nv12_yuv444_i<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
            ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
            ptrC, pInputFrame->pitch, pInputFrame->width, pInputFrame->height, pCrop->e.left, pCrop->e.up);
    } else {
        kernel_crop_uv_nv12_yuv444_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
            ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
            ptrC, pInputFrame->pitch, pInputFrame->width, pInputFrame->height, pCrop->e.left, pCrop->e.up);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_yv12_nv12(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + (uv_x << 1) * sizeof(TypeOut); //NV12
        int isrc = (uv_y + (offsetY >> 1)) * srcPitch + (uv_x + (offsetX >> 1)) * sizeof(TypeIn); //YV12
        const TypeIn *ptr_src_u = (const TypeIn *)(pSrcU + isrc);
        const TypeIn *ptr_src_v = (const TypeIn *)(pSrcV + isrc);
        TypeOut *ptr_dst = (TypeOut *)(pDst + idst);
        ptr_dst[0] = BIT_DEPTH_CONV(ptr_src_u[0]);
        ptr_dst[1] = BIT_DEPTH_CONV(ptr_src_v[0]);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_yv12_nv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrC = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    const uint8_t *ptrU = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    const uint8_t *ptrV = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2;
    kernel_crop_uv_yv12_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrC, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrU, ptrV, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv16_yuv444(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    //横方向拡張
    int uv_x_nv16 = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y      = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x_nv16 < (dstWidth >> 1) && uv_y < dstHeight) {
        int idst = uv_y * dstPitch + (uv_x_nv16 << 1) * sizeof(TypeOut); //YUV444
        int isrc_x = (uv_x_nv16 + (offsetX >> 1));
        int isrc = (uv_y + offsetY) * srcPitch + isrc_x * 2 * sizeof(TypeIn); //NV16
        struct TypeIn2 { TypeIn a, b; };
        const TypeIn2 *ptr_src = (const TypeIn2 *)(pSrc + isrc); //u,v 2要素ロード
        TypeIn2 src0 = ptr_src[0];
        TypeIn2 src1 = (isrc_x + 1 < (dstWidth >> 1)) ? ptr_src[1] : src0; //隣のu,v 2要素ロード
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + idst);
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + idst);
        struct TypeOut2 { TypeOut a, b; };
        TypeOut2 dst_u = { BIT_DEPTH_CONV(src0.a), BIT_DEPTH_CONV_AVG(src0.a, src1.b) };
        TypeOut2 dst_v = { BIT_DEPTH_CONV(src0.b), BIT_DEPTH_CONV_AVG(src0.b, src1.b) };
        *(TypeOut2 *)ptr_dst_u = dst_u;
        *(TypeOut2 *)ptr_dst_v = dst_v;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv16_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2;
    const uint8_t *ptrC = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    kernel_crop_uv_nv16_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrC, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv16_yv12_p(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut); //yv12
        int isrc_y_nv16 = (uv_y << 1) + offsetY;
        int isrc = isrc_y_nv16 * srcPitch + (uv_x + (offsetX >> 1)) * 2 * sizeof(TypeIn); //nv16

        struct TypeIn2 { TypeIn a, b; };
        const TypeIn2 *ptr_src_0 = (const TypeIn2 *)(pSrc + isrc +        0);
        const TypeIn2 *ptr_src_1 = (const TypeIn2 *)(pSrc + isrc + srcPitch);
        TypeIn2 src0 = *ptr_src_0; //u,v 2要素ロード
        TypeIn2 src1 = *ptr_src_1; //下のu,v 2要素ロード

        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + idst);
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + idst);

        ptr_dst_u[0] = BIT_DEPTH_CONV_AVG(src0.a, src1.a);
        ptr_dst_v[0] = BIT_DEPTH_CONV_AVG(src0.b, src1.b);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv16_yv12_p(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2;
    const uint8_t *ptrC = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    kernel_crop_uv_nv16_yv12_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrC, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv16_nv12_p(uint8_t *__restrict__ pDstC, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut) * 2; //nv12
        int isrc_y_nv16 = (uv_y << 1) + offsetY;
        int isrc = isrc_y_nv16 * srcPitch + (uv_x + (offsetX >> 1)) * sizeof(TypeIn) * 2; //nv16

        struct TypeIn2 { TypeIn a, b; };
        const TypeIn2 *ptr_src_0 = (const TypeIn2 *)(pSrc + isrc +        0); 
        const TypeIn2 *ptr_src_1 = (const TypeIn2 *)(pSrc + isrc + srcPitch);
        TypeIn2 src0 = *ptr_src_0; //u,v 2要素ロード
        TypeIn2 src1 = *ptr_src_1; //下のu,v 2要素ロード

        struct TypeOut2 { TypeOut a, b; };
        TypeOut2 dst = { BIT_DEPTH_CONV_AVG(src0.a, src1.a), BIT_DEPTH_CONV_AVG(src0.b, src1.b) };

        *(TypeOut2 *)(pDstC + idst) = dst;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv16_nv12_p(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrDstC = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    const uint8_t *ptrSrcC = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    kernel_crop_uv_nv16_nv12_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrDstC, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrSrcC, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

static __device__ float3 rgb_2_yuv(float3 rgb) {
    float3 yuv;
    yuv.x =  0.257f * rgb.x + 0.504f * rgb.y + 0.098f * rgb.z +  16.0f;
    yuv.y = -0.148f * rgb.x - 0.291f * rgb.y + 0.439f * rgb.z + 128.0f;
    yuv.z =  0.439f * rgb.x - 0.368f * rgb.y - 0.071f * rgb.z + 128.0f;
    return yuv;
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
static __device__ void rgb3_2_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const TypeIn *ptr) {
    float3 rgb;
    rgb.z = (float)ptr[0];
    rgb.y = (float)ptr[1];
    rgb.x = (float)ptr[2];
    float3 yuv = rgb_2_yuv(rgb);

    TypeOut *ptr_dst_y = (TypeOut *)(pDstY);
    TypeOut *ptr_dst_u = (TypeOut *)(pDstU);
    TypeOut *ptr_dst_v = (TypeOut *)(pDstV);
    ptr_dst_y[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv.x) + 0.5f);
    ptr_dst_u[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv.y) + 0.5f);
    ptr_dst_v[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv.z) + 0.5f);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_rgb3_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数分立てる
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //ブロック内のスレッドは32pixelx4
    //読み込み量は 96要素x4
    __shared__ TypeIn ptrSharedIn[32 * 3 * 4];
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    const int shared_read = blockDim.x * 3 / 4;
    if (x < dstWidth && y < dstHeight && threadIdx.x < shared_read) {
        const int isrc = (y + offsetY) * srcPitch + (blockIdx.x * sizeof(TypeIn) * blockDim.x * 3) + (threadIdx.x * sizeof(TypeIn4)) + offsetX * 3;
        TypeIn4 *ptrSharedIn4 = (TypeIn4 *)ptrSharedIn;
        ptrSharedIn4[threadIdx.x + threadIdx.y * shared_read] = *((TypeIn4 *)(pSrc + isrc));
    }
    __syncthreads();
    if (x < dstWidth && y < dstHeight) {
        const int isrc = (threadIdx.y * blockDim.x * 3 + threadIdx.x * 3) * sizeof(TypeIn);
        const int idst = y * dstPitch + x * sizeof(TypeOut);
        rgb3_2_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth>(pDstY + idst, pDstU + idst, pDstV + idst, ptrSharedIn + isrc);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb3_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    uint8_t *ptrY = (uint8_t *)pOutputFrame->ptr;
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2;
    kernel_crop_rgb3_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrY, ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
static __device__ void rgb4_2_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const uint8_t *ptr) {
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    TypeIn4 input = *(TypeIn4 *)ptr;
    float3 rgb;
    rgb.z = (float)input.x;
    rgb.y = (float)input.y;
    rgb.x = (float)input.z;
    float3 yuv = rgb_2_yuv(rgb);

    TypeOut *ptr_dst_y = (TypeOut *)(pDstY);
    TypeOut *ptr_dst_u = (TypeOut *)(pDstU);
    TypeOut *ptr_dst_v = (TypeOut *)(pDstV);
    ptr_dst_y[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv.x) + 0.5f);
    ptr_dst_u[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv.y) + 0.5f);
    ptr_dst_v[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv.z) + 0.5f);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_rgb4_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数分立てる
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        const int isrc = y * srcPitch + x * 4 * sizeof(TypeIn);
        const int idst = y * dstPitch + x * sizeof(TypeOut);
        rgb4_2_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth>(pDstY + idst, pDstU + idst, pDstV + idst, pSrc + isrc);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb4_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    uint8_t *ptrY = (uint8_t *)pOutputFrame->ptr;
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2;
    kernel_crop_rgb4_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrY, ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_rgb3_yv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/2
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    //ブロック内のスレッドは32x4
    //読み込み量は 64pixel(192要素) x 8
    __shared__ TypeIn ptrSharedIn[32 * 3 * 4 * 4];
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    const int shared_read = blockDim.x * 3 / 4;
    if (x * 2 < dstWidth && y * 2 < dstHeight && threadIdx.x < shared_read) {
        int isrc = (blockIdx.y * blockDim.y * 2 + threadIdx.y + offsetY) * srcPitch + (blockIdx.x * sizeof(TypeIn) * blockDim.x * 6) + (threadIdx.x * sizeof(TypeIn4)) + offsetX * 3;
        TypeIn4 *ptrSharedIn4 = (TypeIn4 *)ptrSharedIn;
        ptrSharedIn4[(threadIdx.y + 0) * shared_read * 2 + threadIdx.x +           0] = *((TypeIn4 *)(pSrc + isrc +                                      0));
        ptrSharedIn4[(threadIdx.y + 0) * shared_read * 2 + threadIdx.x + shared_read] = *((TypeIn4 *)(pSrc + isrc +                         blockDim.x * 3));
        ptrSharedIn4[(threadIdx.y + 4) * shared_read * 2 + threadIdx.x +           0] = *((TypeIn4 *)(pSrc + isrc + srcPitch * blockDim.y +              0));
        ptrSharedIn4[(threadIdx.y + 4) * shared_read * 2 + threadIdx.x + shared_read] = *((TypeIn4 *)(pSrc + isrc + srcPitch * blockDim.y + blockDim.x * 3));
    }
    __syncthreads();
    if (x * 2 < dstWidth && y * 2 < dstHeight) {
        float3 yuv00 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv01 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv10 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 1) * blockDim.x * 6);
        float3 yuv11 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 1) * blockDim.x * 6);

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + y * dstPitch + x * sizeof(TypeOut));
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + y * dstPitch + x * sizeof(TypeOut));
        ptr_dst_y00[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv00.x) + 0.5f);
        ptr_dst_y01[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv01.x) + 0.5f);
        ptr_dst_y10[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv10.x) + 0.5f);
        ptr_dst_y11[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv11.x) + 0.5f);
        ptr_dst_u[0]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f) + 0.5f);
        ptr_dst_v[0]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f) + 0.5f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb3_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    uint8_t *ptrDstY = (uint8_t *)pOutputFrame->ptr;
    uint8_t *ptrDstU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrDstV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2;
    kernel_crop_rgb3_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrDstY, ptrDstU, ptrDstV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeIn>
static __device__ float3 rgb3_2_yuv(const TypeIn *ptr) {
    float3 rgb;
    rgb.z = (float)ptr[0];
    rgb.y = (float)ptr[1];
    rgb.x = (float)ptr[2];
    return rgb_2_yuv(rgb);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_rgb3_nv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstC,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/2
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
                                                   //ブロック内のスレッドは32x4
                                                   //読み込み量は 96x4だが、128x4とっておく
    __shared__ TypeIn ptrSharedIn[32 * 3 * 4 * 4];
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    const int shared_read = blockDim.x * 3 / 4;
    if (x * 2 < dstWidth && y * 2 < dstHeight && threadIdx.x < shared_read) {
        int isrc = (blockIdx.y * blockDim.y * 2 + threadIdx.y + offsetY) * srcPitch + (blockIdx.x * sizeof(TypeIn) * blockDim.x * 6) + (threadIdx.x * sizeof(TypeIn4)) + offsetX * 3;
        TypeIn4 *ptrSharedIn4 = (TypeIn4 *)ptrSharedIn;
        ptrSharedIn4[(threadIdx.y + 0) * shared_read * 2 + threadIdx.x +           0] = *((TypeIn4 *)(pSrc + isrc +                                      0));
        ptrSharedIn4[(threadIdx.y + 0) * shared_read * 2 + threadIdx.x + shared_read] = *((TypeIn4 *)(pSrc + isrc +                         blockDim.x * 3));
        ptrSharedIn4[(threadIdx.y + 4) * shared_read * 2 + threadIdx.x +           0] = *((TypeIn4 *)(pSrc + isrc + srcPitch * blockDim.y +              0));
        ptrSharedIn4[(threadIdx.y + 4) * shared_read * 2 + threadIdx.x + shared_read] = *((TypeIn4 *)(pSrc + isrc + srcPitch * blockDim.y + blockDim.x * 3));
    }
    __syncthreads();
    if (x * 2 < dstWidth && y * 2 < dstHeight) {
        float3 yuv00 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv01 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv10 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 1) * blockDim.x * 6);
        float3 yuv11 = rgb3_2_yuv(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 1) * blockDim.x * 6);

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_c = (TypeOut *)(pDstC + y * dstPitch + x * 2 * sizeof(TypeOut));
        ptr_dst_y00[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv00.x) + 0.5f);
        ptr_dst_y01[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv01.x) + 0.5f);
        ptr_dst_y10[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv10.x) + 0.5f);
        ptr_dst_y11[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv11.x) + 0.5f);
        ptr_dst_c[0]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f) + 0.5f);
        ptr_dst_c[1]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f) + 0.5f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb3_nv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    uint8_t *ptrDstY = (uint8_t *)pOutputFrame->ptr;
    uint8_t *ptrDstC = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    kernel_crop_rgb3_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrDstY, ptrDstC, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeIn>
static __device__ float3 rgb4_2_yuv(const TypeIn *ptr) {
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    TypeIn4 input = *(TypeIn4 *)ptr;
    float3 rgb;
    rgb.x = (float)input.z;
    rgb.y = (float)input.y;
    rgb.z = (float)input.x;
    return rgb_2_yuv(rgb);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_rgb4_yv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/2
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    if (x * 2 < dstWidth && y * 2 < dstHeight) {
        float3 yuv00 = rgb4_2_yuv(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv01 = rgb4_2_yuv(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));
        float3 yuv10 = rgb4_2_yuv(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv11 = rgb4_2_yuv(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + y * dstPitch + x * sizeof(TypeOut));
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + y * dstPitch + x * sizeof(TypeOut));
        ptr_dst_y00[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv00.x) + 0.5f);
        ptr_dst_y01[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv01.x) + 0.5f);
        ptr_dst_y10[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv10.x) + 0.5f);
        ptr_dst_y11[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv11.x) + 0.5f);
        ptr_dst_u[0]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f) + 0.5f);
        ptr_dst_v[0]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f) + 0.5f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb4_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    uint8_t *ptrDstY = (uint8_t *)pOutputFrame->ptr;
    uint8_t *ptrDstU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrDstV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2;
    kernel_crop_rgb4_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrDstY, ptrDstU, ptrDstV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_rgb4_nv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstC,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/2
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    if (x * 2 < dstWidth && y * 2 < dstHeight) {
        float3 yuv00 = rgb4_2_yuv(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv01 = rgb4_2_yuv(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));
        float3 yuv10 = rgb4_2_yuv(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv11 = rgb4_2_yuv(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_c = (TypeOut *)(pDstC + y * dstPitch + x * 2 * sizeof(TypeOut));
        ptr_dst_y00[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv00.x) + 0.5f);
        ptr_dst_y01[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv01.x) + 0.5f);
        ptr_dst_y10[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv10.x) + 0.5f);
        ptr_dst_y11[0] = (TypeOut)(BIT_DEPTH_CONV_FLOAT(yuv11.x) + 0.5f);
        ptr_dst_c[0]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f) + 0.5f);
        ptr_dst_c[1]   = (TypeOut)(BIT_DEPTH_CONV_FLOAT((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f) + 0.5f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb4_nv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    uint8_t *ptrDstY = (uint8_t *)pOutputFrame->ptr;
    uint8_t *ptrDstC = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    kernel_crop_rgb4_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrDstY, ptrDstC, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

NVENCSTATUS NVEncFilterCspCrop::convertCspFromNV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto frameOutInfoEx = getFrameInfoExtra(pOutputFrame);
    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        auto cudaerr = cudaMemcpy2DAsync((uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch,
            (uint8_t *)pInputFrame->ptr + pCropParam->crop.e.left + pCropParam->crop.e.up * pInputFrame->pitch,
            pInputFrame->pitch,
            frameOutInfoEx.width_byte, pOutputFrame->height, cudaMemcpyDeviceToDevice);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromNV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame);
        if (ret != NV_ENC_SUCCESS) {
            return ret;
        }
    }

    //UV
    static const std::map<uint64_t, void (*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> convert_from_nv12_list = {
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12   ).i,   crop_uv_nv12_yv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_16).i,   crop_uv_nv12_yv12<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_14).i,   crop_uv_nv12_yv12<uint16_t, 14, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_12).i,   crop_uv_nv12_yv12<uint16_t, 12, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_10).i,   crop_uv_nv12_yv12<uint16_t, 10, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_09).i,   crop_uv_nv12_yv12<uint16_t,  9, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YV12_16).i,   crop_uv_nv12_yv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YV12_14).i,   crop_uv_nv12_yv12<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YV12_12).i,   crop_uv_nv12_yv12<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YV12_10).i,   crop_uv_nv12_yv12<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YV12_09).i,   crop_uv_nv12_yv12<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YUV444   ).i, crop_uv_nv12_yuv444<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YUV444_16).i, crop_uv_nv12_yuv444<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YUV444_14).i, crop_uv_nv12_yuv444<uint16_t, 14, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YUV444_12).i, crop_uv_nv12_yuv444<uint16_t, 12, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YUV444_10).i, crop_uv_nv12_yuv444<uint16_t, 10, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YUV444_09).i, crop_uv_nv12_yuv444<uint16_t,  9, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YUV444   ).i, crop_uv_nv12_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YUV444_16).i, crop_uv_nv12_yuv444<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YUV444_14).i, crop_uv_nv12_yuv444<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YUV444_12).i, crop_uv_nv12_yuv444<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YUV444_10).i, crop_uv_nv12_yuv444<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YUV444_09).i, crop_uv_nv12_yuv444<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_P010     ).i, crop_uv<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_NV12     ).i, crop_uv<uint8_t,   8, uint16_t, 16> },
    };
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (convert_from_nv12_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    convert_from_nv12_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at convert_from_nv12_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}
NVENCSTATUS NVEncFilterCspCrop::convertCspFromYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto frameOutInfoEx = getFrameInfoExtra(&pCropParam->frameOut);
    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        auto cudaerr = cudaMemcpy2DAsync((uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch,
            (uint8_t *)pInputFrame->ptr + pCropParam->crop.e.left + pCropParam->crop.e.up * pInputFrame->pitch,
            pInputFrame->pitch,
            frameOutInfoEx.width_byte, pCropParam->frameOut.height, cudaMemcpyDeviceToDevice);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromYV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame);
        if (ret != NV_ENC_SUCCESS) {
            return ret;
        }
    }

    //UV
    static const std::map<uint64_t, void (*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> crop_uv_yv12_nv12_list = {
        { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_NV12).i, crop_uv_yv12_nv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YV12_09, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t,  9, uint16_t, 16> },
    };
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (crop_uv_yv12_nv12_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    crop_uv_yv12_nv12_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at crop_uv_nv12_yv12_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;

}

NVENCSTATUS NVEncFilterCspCrop::convertCspFromNV16(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto frameOutInfoEx = getFrameInfoExtra(pOutputFrame);
    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        auto cudaerr = cudaMemcpy2DAsync((uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch,
            (uint8_t *)pInputFrame->ptr + pCropParam->crop.e.left + pCropParam->crop.e.up * pInputFrame->pitch,
            pInputFrame->pitch,
            frameOutInfoEx.width_byte, pOutputFrame->height, cudaMemcpyDeviceToDevice);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromNV16(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame);
        if (ret != NV_ENC_SUCCESS) {
            return ret;
        }
    }

    //UV
    static const std::map<uint64_t, void(*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> convert_from_nv16_list = {
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12).i,      crop_uv_nv16_yv12_p<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_16).i,   crop_uv_nv16_yv12_p<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_14).i,   crop_uv_nv16_yv12_p<uint16_t, 14, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_12).i,   crop_uv_nv16_yv12_p<uint16_t, 12, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_10).i,   crop_uv_nv16_yv12_p<uint16_t, 10, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_09).i,   crop_uv_nv16_yv12_p<uint16_t,  9, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YV12_16).i,   crop_uv_nv16_yv12_p<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YV12_14).i,   crop_uv_nv16_yv12_p<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YV12_12).i,   crop_uv_nv16_yv12_p<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YV12_10).i,   crop_uv_nv16_yv12_p<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YV12_09).i,   crop_uv_nv16_yv12_p<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_NV12).i,      crop_uv_nv16_nv12_p<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_P010).i,      crop_uv_nv16_nv12_p<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_NV12).i,      crop_uv_nv16_nv12_p<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_P010).i,      crop_uv_nv16_nv12_p<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YUV444).i,    crop_uv_nv16_yuv444<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YUV444_16).i, crop_uv_nv16_yuv444<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YUV444_14).i, crop_uv_nv16_yuv444<uint16_t, 14, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YUV444_12).i, crop_uv_nv16_yuv444<uint16_t, 12, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YUV444_10).i, crop_uv_nv16_yuv444<uint16_t, 10, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YUV444_09).i, crop_uv_nv16_yuv444<uint16_t,  9, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YUV444).i,    crop_uv_nv16_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YUV444_16).i, crop_uv_nv16_yuv444<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YUV444_14).i, crop_uv_nv16_yuv444<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YUV444_12).i, crop_uv_nv16_yuv444<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YUV444_10).i, crop_uv_nv16_yuv444<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YUV444_09).i, crop_uv_nv16_yuv444<uint16_t,  9, uint16_t, 16> },
    };
    if (pInputFrame->interlaced && RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported interlaced csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (convert_from_nv16_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    convert_from_nv16_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at convert_from_nv16_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}
NVENCSTATUS NVEncFilterCspCrop::convertCspFromRGB(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto frameOutInfoEx = getFrameInfoExtra(&pCropParam->frameOut);

    static const std::map<uint64_t, void(*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> convert_from_rgb_list = {
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444).i,    crop_rgb3_yuv444<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_09).i, crop_rgb3_yuv444<uint8_t,   8, uint16_t,  9> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_10).i, crop_rgb3_yuv444<uint8_t,   8, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_12).i, crop_rgb3_yuv444<uint8_t,   8, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_14).i, crop_rgb3_yuv444<uint8_t,   8, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_16).i, crop_rgb3_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12).i,      crop_rgb3_yv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_09).i,   crop_rgb3_yv12<uint8_t,   8, uint16_t,  9> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_10).i,   crop_rgb3_yv12<uint8_t,   8, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_12).i,   crop_rgb3_yv12<uint8_t,   8, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_14).i,   crop_rgb3_yv12<uint8_t,   8, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_16).i,   crop_rgb3_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_NV12).i,      crop_rgb3_nv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_P010).i,      crop_rgb3_nv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444).i,    crop_rgb4_yuv444<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_09).i, crop_rgb4_yuv444<uint8_t,   8, uint16_t,  9> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_10).i, crop_rgb4_yuv444<uint8_t,   8, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_12).i, crop_rgb4_yuv444<uint8_t,   8, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_14).i, crop_rgb4_yuv444<uint8_t,   8, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_16).i, crop_rgb4_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12).i,      crop_rgb4_yv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_09).i,   crop_rgb4_yv12<uint8_t,   8, uint16_t,  9> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_10).i,   crop_rgb4_yv12<uint8_t,   8, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_12).i,   crop_rgb4_yv12<uint8_t,   8, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_14).i,   crop_rgb4_yv12<uint8_t,   8, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_16).i,   crop_rgb4_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_NV12).i,      crop_rgb4_nv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_P010).i,      crop_rgb4_nv12<uint8_t,   8, uint16_t, 16> },
    };
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (convert_from_rgb_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    convert_from_rgb_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at convert_from_rgb_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncFilterCspCrop::convertCspFromYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
    return NV_ENC_ERR_UNIMPLEMENTED;
}

NVEncFilterCspCrop::NVEncFilterCspCrop() {
    m_sFilterName = _T("copy/cspconv/crop");
}

NVEncFilterCspCrop::~NVEncFilterCspCrop() {
    close();
}

NVENCSTATUS NVEncFilterCspCrop::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //フィルタ名の調整
    m_sFilterName = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_sFilterName += _T("crop");
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_sFilterName += (m_sFilterName.length()) ? _T("/cspconv") : _T("cspconv");
    }
    if (m_sFilterName.length() == 0) {
        const auto memcpyKind = getCudaMemcpyKind(pParam->frameIn.deivce_mem, pParam->frameOut.deivce_mem);
        m_sFilterName += getCudaMemcpyKindStr(memcpyKind);
    }
    //パラメータチェック
    for (int i = 0; i < _countof(pCropParam->crop.c); i++) {
        if ((pCropParam->crop.c[i] & 1) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("crop should be divided by 2.\n"));
            return NV_ENC_ERR_INVALID_PARAM;
        }
    }
    pCropParam->frameOut.height = pCropParam->frameIn.height - pCropParam->crop.e.bottom - pCropParam->crop.e.up;
    pCropParam->frameOut.width = pCropParam->frameIn.width - pCropParam->crop.e.left - pCropParam->crop.e.right;
    if (pCropParam->frameOut.height <= 0 || pCropParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("crop size is too big.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pCropParam->frameOut, 2);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pCropParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    //フィルタ情報の調整
    m_sFilterInfo = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_sFilterInfo += strsprintf(_T("crop: %d,%d,%d,%d"), pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->crop.e.right, pCropParam->crop.e.bottom);
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_sFilterInfo += (m_sFilterInfo.length()) ? _T("/cspconv") : _T("cspconv");
        m_sFilterInfo += strsprintf(_T("(%s -> %s)"), RGY_CSP_NAMES[pCropParam->frameIn.csp], RGY_CSP_NAMES[pCropParam->frameOut.csp]);
    }
    if (m_sFilterInfo.length() == 0) {
        m_sFilterInfo += getCudaMemcpyKindStr(pCropParam->frameIn.deivce_mem, pCropParam->frameOut.deivce_mem);
    }

    m_pParam = pCropParam;
    return sts;
}

NVENCSTATUS NVEncFilterCspCrop::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    if (pInputFrame == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    ppOutputFrames[0]->interlaced = pInputFrame->interlaced;
    if (m_pParam->frameOut.csp == m_pParam->frameIn.csp) {
        auto cudaMemcpyErrMes = [&](cudaError_t cudaerr, const TCHAR *mes) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (filter(%s)): %s.\n"),
                mes, RGY_CSP_NAMES[pInputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        };
#if 1
        const auto frameOutInfoEx = getFrameInfoExtra(ppOutputFrames[0]);
        if (!cropEnabled(pCropParam->crop)) {
            //cropがなければ、一度に転送可能
            auto cudaerr = cudaMemcpy2DAsync((uint8_t *)ppOutputFrames[0]->ptr, ppOutputFrames[0]->pitch,
                (uint8_t *)pInputFrame->ptr, pInputFrame->pitch,
                frameOutInfoEx.width_byte, frameOutInfoEx.height_total, memcpyKind);
            if (cudaerr != cudaSuccess) {
                cudaMemcpyErrMes(cudaerr, _T("cudaMemcpy2DAsyncAll"));
                return NV_ENC_ERR_INVALID_CALL;
            };
        } else {
            if (pCropParam->frameOut.csp == RGY_CSP_NV12) {
                cudaError_t cudaerr;
                //Y
                cudaerr = cudaMemcpy2DAsync((uint8_t *)ppOutputFrames[0]->ptr, ppOutputFrames[0]->pitch,
                    (uint8_t *)pInputFrame->ptr + pCropParam->crop.e.left + pCropParam->crop.e.up * pInputFrame->pitch,
                    pInputFrame->pitch,
                    frameOutInfoEx.width_byte, pCropParam->frameOut.height, memcpyKind);
                if (cudaerr != cudaSuccess) {
                    cudaMemcpyErrMes(cudaerr, _T("cudaMemcpy2DAsync_Y"));
                    return NV_ENC_ERR_INVALID_CALL;
                };
                //UV
                cudaerr = cudaMemcpy2DAsync((uint8_t *)ppOutputFrames[0]->ptr + ppOutputFrames[0]->pitch * ppOutputFrames[0]->height, ppOutputFrames[0]->pitch,
                    (uint8_t *)pInputFrame->ptr
                    + pInputFrame->height * pInputFrame->pitch
                    + pCropParam->crop.e.left + (pCropParam->crop.e.up >> 1) * pInputFrame->pitch,
                    pInputFrame->pitch,
                    frameOutInfoEx.width_byte, pCropParam->frameOut.height >> 1, memcpyKind);
                if (cudaerr != cudaSuccess) {
                    cudaMemcpyErrMes(cudaerr, _T("cudaMemcpy2DAsync_UV"));
                    return NV_ENC_ERR_INVALID_CALL;
                };
            } else {
                AddMessage(RGY_LOG_ERROR, _T("unsupported output csp with crop.\n"));
                return NV_ENC_ERR_UNIMPLEMENTED;
            }
        }
#else
        if (pCropParam->frameOut.csp == RGY_CSP_NV12) {
            dim3 blockSize(32, 4);
            dim3 gridSize(divCeil(pCropParam->frameOut.width, blockSize.x), divCeil(pCropParam->frameOut.height, blockSize.y));
            kernel_crop_nv12_nv12<uint8_t><<<gridSize, blockSize>>>((uint8_t *)ppOutputFrames[0]->ptr, (uint8_t *)pInputFrame->ptr, pInputFrame->pitch);
        } else {
            AddMessage(RGY_LOG_ERROR, _T("unsupported output csp.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
#endif
    } else if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("converting csp while copying from host to device is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    } else {
        //色空間変換
        static const auto supportedCspNV12   = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
        static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
        static const auto supportedCspNV16   = make_array<RGY_CSP>(RGY_CSP_NV16, RGY_CSP_P210);
        static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
        static const auto supportedCspRGB    = make_array<RGY_CSP>(RGY_CSP_RGB24, RGY_CSP_RGB32);
        if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameIn.csp) != supportedCspNV12.end()) {
            sts = convertCspFromNV12(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameIn.csp) != supportedCspYV12.end()) {
            sts = convertCspFromYV12(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspNV16.begin(), supportedCspNV16.end(), pCropParam->frameIn.csp) != supportedCspNV16.end()) {
            sts = convertCspFromNV16(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameIn.csp) != supportedCspYUV444.end()) {
            sts = convertCspFromYUV444(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspRGB.begin(), supportedCspRGB.end(), pCropParam->frameIn.csp) != supportedCspRGB.end()) {
            sts = convertCspFromRGB(ppOutputFrames[0], pInputFrame);
        } else {
            AddMessage(RGY_LOG_ERROR, _T("converting csp from %s is not supported.\n"), RGY_CSP_NAMES[pCropParam->frameIn.csp]);
            sts = NV_ENC_ERR_UNIMPLEMENTED;
        }
    }
    return sts;
}

void NVEncFilterCspCrop::close() {
    m_pFrameBuf.clear();
}
