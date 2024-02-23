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

#if 0
RGY_ERR copyPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    const int pix_size = RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8 ? 2 : 1;
    const auto memkind = getCudaMemcpyKind(pInputFrame->mem_type, pOutputFrame->mem_type);;
    auto cudaerr = cudaMemcpy2DAsync((uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch,
        (uint8_t *)pInputFrame->ptr, pInputFrame->pitch,
        pInputFrame->width * pix_size, pInputFrame->height, memkind, stream);
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR copyFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    auto err = copyPlane(&planeOutputY, &planeInputY, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyPlane(&planeOutputU, &planeInputU, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyPlane(&planeOutputV, &planeInputV, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return err;
}
#endif

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

#define BIT_DEPTH_CONV(x) \
    conv_bit_depth<in_bit_depth, out_bit_depth, 0>(x)

#define BIT_DEPTH_CONV_FLOAT(x) (TypeOut)((out_bit_depth == in_bit_depth) \
    ? (x) \
    : ((out_bit_depth > in_bit_depth) \
        ? ((x) * (float)(1 << (out_bit_depth - in_bit_depth))) \
        : ((x) * (float)(1.0f / (1 << (in_bit_depth - out_bit_depth))))))

#define BIT_DEPTH_CONV_AVG(a, b) \
    conv_bit_depth<in_bit_depth, out_bit_depth, 1>((int)a + (int)b)

#define BIT_DEPTH_CONV_3x1_AVG(a, b) \
    conv_bit_depth<in_bit_depth, out_bit_depth, 2>((int)a * 3 + (int)b)

#define BIT_DEPTH_CONV_ia_jb_rsh3(i, a, j, b) \
    conv_bit_depth<in_bit_depth, out_bit_depth, 3>((int)a * i + (int)b * j)

bool isAligned(const RGYFrameInfo& plane, const size_t align) {
    static_assert(sizeof(plane.ptr[0]) == sizeof(size_t), "size mismatch");
    return (((size_t)plane.ptr[0]) & (align - 1)) == 0 && (plane.pitch[0] & (align - 1)) == 0;
}
bool isAlignedRGB(const RGYFrameInfo *plane, const size_t align) {
    return isAligned(getPlane(plane, RGY_PLANE_R), align)
        && isAligned(getPlane(plane, RGY_PLANE_G), align)
        && isAligned(getPlane(plane, RGY_PLANE_B), align);
}
bool isAlignedYUV444(const RGYFrameInfo *plane, const size_t align) {
    return isAligned(getPlane(plane, RGY_PLANE_Y), align)
        && isAligned(getPlane(plane, RGY_PLANE_U), align)
        && isAligned(getPlane(plane, RGY_PLANE_V), align);
}
bool isAlignedYV12(const RGYFrameInfo *plane, const size_t align) {
    return isAlignedYUV444(plane, align);
}
bool isAlignedNV12(const RGYFrameInfo *plane, const size_t align) {
    return isAligned(getPlane(plane, RGY_PLANE_Y), align)
        && isAligned(getPlane(plane, RGY_PLANE_C), align);
}
bool cropEnabled(const sInputCrop *crop) {
    return crop && cropEnabled(*crop);
}

template<typename Type, typename Type2, bool aligned>
static __device__ Type2 kernel_crop_load2(const void *ptr) {
    static_assert(sizeof(Type2) == sizeof(Type) * 2, "size mismatch");
    static_assert(sizeof(Type2::x) == sizeof(Type), "size mismatch");
    Type2 val;
    if (aligned) {
        val = ((const Type2 *)ptr)[0];
    } else {
        const Type *ptr_scalar = (const Type *)ptr;
        val.x = ptr_scalar[0];
        val.y = ptr_scalar[1];
    }
    return val;
}

template<typename Type, typename Type4, bool aligned>
static __device__ Type4 kernel_crop_load4(const void *ptr) {
    static_assert(sizeof(Type4) == sizeof(Type) * 4, "size mismatch");
    static_assert(sizeof(Type4::x) == sizeof(Type), "size mismatch");
    Type4 val;
    if (aligned) {
        val = ((const Type4 *)ptr)[0];
    } else {
        const Type *ptr_scalar = (const Type *)ptr;
        val.x = ptr_scalar[0];
        val.y = ptr_scalar[1];
        val.z = ptr_scalar[2];
        val.w = ptr_scalar[3];
    }
    return val;
}

template<typename Type, typename Type2, bool aligned>
static __device__ void kernel_crop_store2(Type2 *ptr, const Type2 val) {
    static_assert(sizeof(Type2) == sizeof(Type) * 2, "size mismatch");
    static_assert(sizeof(Type2::x) == sizeof(Type), "size mismatch");
    if (aligned) {
        ptr[0] = val;
    } else {
        Type *ptr_scalar = (Type *)ptr;
        ptr_scalar[0] = val.x;
        ptr_scalar[1] = val.y;
    }
}

template<typename Type, typename Type4, bool aligned>
static __device__ void kernel_crop_store4(Type4 *ptr, const Type4 val) {
    static_assert(sizeof(Type4) == sizeof(Type) * 4, "size mismatch");
    static_assert(sizeof(Type4::x) == sizeof(Type), "size mismatch");
    if (aligned) {
        ptr[0] = val;
    } else {
        Type *ptr_scalar = (Type *)ptr;
        ptr_scalar[0] = val.x;
        ptr_scalar[1] = val.y;
        ptr_scalar[2] = val.z;
        ptr_scalar[3] = val.w;
    }
}

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
        ptr_dst[0] = (TypeOut)BIT_DEPTH_CONV(ptr_src[0]);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(64, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    kernel_crop_y<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    auto outPlaneC = getPlane(pOutputFrame, RGY_PLANE_C);
    const auto inPlaneC = getPlane(pInputFrame, RGY_PLANE_C);
    dim3 blockSize(64, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(outPlaneC.height, blockSize.y));
    kernel_crop_y<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)outPlaneC.ptr[0], outPlaneC.pitch[0], pOutputFrame->width, outPlaneC.height,
        (const uint8_t *)inPlaneC.ptr[0], inPlaneC.pitch[0], pCrop->e.left, pCrop->e.up >> 1);
}

RGY_ERR NVEncFilterCspCrop::convertYBitDepth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
#define CONV_DEPTH_TO_FROM(to, from) ((to) << 8 | (from))
    static const std::map<int, void (*)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream)> crop_y_list = {
        { CONV_DEPTH_TO_FROM(16,  8), crop_y<uint16_t, 16, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM(14,  8), crop_y<uint16_t, 14, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM(12,  8), crop_y<uint16_t, 12, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM(10,  8), crop_y<uint16_t, 10, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM( 9,  8), crop_y<uint16_t,  9, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM( 8,  8), crop_y<uint8_t,   8, uint8_t,   8> },
        { CONV_DEPTH_TO_FROM( 8, 16), crop_y<uint8_t,   8, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM( 8, 14), crop_y<uint8_t,   8, uint16_t, 14> },
        { CONV_DEPTH_TO_FROM( 8, 12), crop_y<uint8_t,   8, uint16_t, 12> },
        { CONV_DEPTH_TO_FROM( 8, 10), crop_y<uint8_t,   8, uint16_t, 10> },
        { CONV_DEPTH_TO_FROM( 8,  9), crop_y<uint8_t,   8, uint16_t,  9> },
        { CONV_DEPTH_TO_FROM(16, 16), crop_y<uint16_t, 16, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM(14, 16), crop_y<uint16_t, 14, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM(12, 16), crop_y<uint16_t, 12, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM(10, 16), crop_y<uint16_t, 10, uint16_t, 16> },
        { CONV_DEPTH_TO_FROM( 9, 16), crop_y<uint16_t,  9, uint16_t, 16> },
    };
    const auto bit_depth_conv = CONV_DEPTH_TO_FROM(RGY_CSP_BIT_DEPTH[pOutputFrame->csp], RGY_CSP_BIT_DEPTH[pInputFrame->csp]);
    if (crop_y_list.count(bit_depth_conv) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
#undef CONV_DEPTH_TO_FROM
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    crop_y_list.at(bit_depth_conv)(pOutputFrame, pInputFrame, &pCropParam->crop, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at convertYBitDepth(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
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
        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV(ptr_src[0]);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV(ptr_src[1]);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv12_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    auto outPlaneU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto outPlaneV = getPlane(pOutputFrame, RGY_PLANE_V);
    const auto inPlaneC = getPlane(pInputFrame, RGY_PLANE_C);
    kernel_crop_uv_nv12_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
        outPlaneU.ptr[0], outPlaneV.ptr[0], outPlaneU.pitch[0], pOutputFrame->width, pOutputFrame->height,
        inPlaneC.ptr[0], inPlaneC.pitch[0], pCrop->e.left, pCrop->e.up);
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
        const int y0_offset  = (uv_y > 0) ? -1 * srcPitch / sizeof(TypeIn) : 0;
        const int y2_offset  = ((uv_y+1) < (srcHeight >> 1)) ? srcPitch / sizeof(TypeIn) : 0;
        const int next_pixel = ((uv_x+1) < (srcWidth >> 1)) ? 2 : 0;
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

        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(u_y1x0, u_y0x0);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(v_y1x0, v_y0x0);
        ptr_dst_u[1] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(u_y1x1, u_y0x1);
        ptr_dst_v[1] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(v_y1x1, v_y0x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(u_y1x0, u_y2x0);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(v_y1x0, v_y2x0);
        ptr_dst_u[1] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(u_y1x1, u_y2x1);
        ptr_dst_v[1] = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(v_y1x1, v_y2x1);
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
        const int y0_offset  = (uv_y - 1 > 0) ? -2 * srcPitch / sizeof(TypeIn) : 0;
        const int y1_offset  = (uv_y > 0)     ? -1 * srcPitch / sizeof(TypeIn) : srcPitch / sizeof(TypeIn);
        const int y3_offset  = (uv_y+1 < (srcHeight >> 1)) ? srcPitch / sizeof(TypeIn)     : y1_offset;
        const int y4_offset  = (uv_y+2 < (srcHeight >> 1)) ? srcPitch / sizeof(TypeIn) * 2 : 0;
        const int y5_offset  = (uv_y+3 < (srcHeight >> 1)) ? srcPitch / sizeof(TypeIn) * 3 : y3_offset;
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

        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(1, u_y0x0, 7, u_y2x0);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(1, v_y0x0, 7, v_y2x0);
        ptr_dst_u[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(1, u_y0x1, 7, u_y2x1);
        ptr_dst_v[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(1, v_y0x1, 7, v_y2x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(3, u_y1x0, 5, u_y3x0);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(3, v_y1x0, 5, v_y3x0);
        ptr_dst_u[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(3, u_y1x1, 5, u_y3x1);
        ptr_dst_v[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(3, v_y1x1, 5, v_y3x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(5, u_y2x0, 3, u_y4x0);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(5, v_y2x0, 3, v_y4x0);
        ptr_dst_u[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(5, u_y2x1, 3, u_y4x1);
        ptr_dst_v[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(5, v_y2x1, 3, v_y4x1);
        ptr_dst_u = (TypeOut *)((uint8_t *)ptr_dst_u + dstPitch);
        ptr_dst_v = (TypeOut *)((uint8_t *)ptr_dst_v + dstPitch);
        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(7, u_y3x0, 1, u_y5x0);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(7, v_y3x0, 1, v_y5x0);
        ptr_dst_u[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(7, u_y3x1, 1, u_y5x1);
        ptr_dst_v[1] = (TypeOut)BIT_DEPTH_CONV_ia_jb_rsh3(7, v_y3x1, 1, v_y5x1);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv12_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    const auto planeInputC = getPlane(pInputFrame, RGY_PLANE_C);
    if (interlaced(*pInputFrame)) {
        kernel_crop_uv_nv12_yuv444_i<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputU.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputC.ptr[0], planeInputC.pitch[0], pInputFrame->width, pInputFrame->height, pCrop->e.left, pCrop->e.up);
    } else {
        kernel_crop_uv_nv12_yuv444_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputU.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputC.ptr[0], planeInputC.pitch[0], pInputFrame->width, pInputFrame->height, pCrop->e.left, pCrop->e.up);
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
        ptr_dst[0] = (TypeOut)BIT_DEPTH_CONV(ptr_src_u[0]);
        ptr_dst[1] = (TypeOut)BIT_DEPTH_CONV(ptr_src_v[0]);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_yv12_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    auto outPlaneC = getPlane(pOutputFrame, RGY_PLANE_C);
    const auto inPlaneU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto inPlaneV = getPlane(pInputFrame, RGY_PLANE_V);
    kernel_crop_uv_yv12_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
        outPlaneC.ptr[0], outPlaneC.pitch[0], pOutputFrame->width, pOutputFrame->height,
        inPlaneU.ptr[0], inPlaneV.ptr[0], inPlaneU.pitch[0], pCrop->e.left, pCrop->e.up);
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
        TypeOut2 dst_u = { (TypeOut)BIT_DEPTH_CONV(src0.a), (TypeOut)BIT_DEPTH_CONV_AVG(src0.a, src1.a) };
        TypeOut2 dst_v = { (TypeOut)BIT_DEPTH_CONV(src0.b), (TypeOut)BIT_DEPTH_CONV_AVG(src0.b, src1.b) };
        *(TypeOut2 *)ptr_dst_u = dst_u;
        *(TypeOut2 *)ptr_dst_v = dst_v;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv16_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    const auto planeInputC = getPlane(pInputFrame, RGY_PLANE_C);
    kernel_crop_uv_nv16_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
        planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputU.pitch[0], pOutputFrame->width, pOutputFrame->height,
        planeInputC.ptr[0], planeInputC.pitch[0], pCrop->e.left, pCrop->e.up);
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

        ptr_dst_u[0] = (TypeOut)BIT_DEPTH_CONV_AVG(src0.a, src1.a);
        ptr_dst_v[0] = (TypeOut)BIT_DEPTH_CONV_AVG(src0.b, src1.b);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv16_yv12_p(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    if (interlaced(*pInputFrame)) {
        fprintf(stderr, "interlaced yuv422 -> yuv420 is not supported.\n");
        exit(1);
    } else {
        dim3 blockSize(32, 4);
        dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        const auto planeInputC = getPlane(pInputFrame, RGY_PLANE_C);
        kernel_crop_uv_nv16_yv12_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputU.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputC.ptr[0], planeInputC.pitch[0], pCrop->e.left, pCrop->e.up);
    }
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
        TypeOut2 dst = { (TypeOut)BIT_DEPTH_CONV_AVG(src0.a, src1.a), (TypeOut)BIT_DEPTH_CONV_AVG(src0.b, src1.b) };

        *(TypeOut2 *)(pDstC + idst) = dst;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv16_nv12_p(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    if (interlaced(*pInputFrame)) {
        fprintf(stderr, "interlaced yuv422 -> yuv420 is not supported.\n");
        exit(1);
    } else {
        dim3 blockSize(32, 4);
        dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
        auto planeOutputC = getPlane(pOutputFrame, RGY_PLANE_C);
        const auto planeInputC = getPlane(pInputFrame, RGY_PLANE_C);
        kernel_crop_uv_nv16_nv12_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputC.ptr[0], planeOutputC.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputC.ptr[0], planeInputC.pitch[0], pCrop->e.left, pCrop->e.up);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_yuv444_nv12_p(uint8_t *__restrict__ pDstC, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut) * 2; //nv12
        int isrc = ((uv_y << 1) + offsetY) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //yuv444

        const TypeIn src_u0 = *(const TypeIn *)(pSrcU + isrc +        0);
        const TypeIn src_u1 = *(const TypeIn *)(pSrcU + isrc + srcPitch);
        const TypeIn src_v0 = *(const TypeIn *)(pSrcV + isrc +        0);
        const TypeIn src_v1 = *(const TypeIn *)(pSrcV + isrc + srcPitch);

        struct TypeOut2 { TypeOut a, b; };
        TypeOut2 dst = { (TypeOut)BIT_DEPTH_CONV_AVG(src_u0, src_u1), (TypeOut)BIT_DEPTH_CONV_AVG(src_v0, src_v1) };
        *(TypeOut2 *)(pDstC + idst) = dst;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_yuv444_nv12_i(uint8_t *__restrict__ pDstC, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 2)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut) * 2; //nv12
        int isrc = ((uv_y << 2) + offsetY) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //yuv444

        const TypeIn src_u0 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 0);
        const TypeIn src_u1 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 1);
        const TypeIn src_u2 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 2);
        const TypeIn src_u3 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 3);
        const TypeIn src_v0 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 0);
        const TypeIn src_v1 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 1);
        const TypeIn src_v2 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 2);
        const TypeIn src_v3 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 3);

        struct TypeOut2 { TypeOut a, b; };
        TypeOut2 dst0 = { (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_u0, src_u2), (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_v0, src_v2) };
        TypeOut2 dst1 = { (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_u3, src_u1), (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_v3, src_v1) };

        *(TypeOut2 *)(pDstC + idst +        0) = dst0;
        *(TypeOut2 *)(pDstC + idst + dstPitch) = dst1;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_yuv444_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputC = getPlane(pOutputFrame, RGY_PLANE_U);
    dim3 blockSize(32, 4);
    if (interlaced(*pInputFrame)) {
        dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 2), divCeil(pOutputFrame->height, blockSize.y * 4));
        kernel_crop_uv_yuv444_nv12_i<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputC.ptr[0], planeOutputC.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputU.ptr[0], planeInputV.ptr[0], planeInputU.pitch[0], pCrop->e.left, pCrop->e.up);
    } else {
        dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 2), divCeil(pOutputFrame->height, blockSize.y * 2));
        kernel_crop_uv_yuv444_nv12_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputC.ptr[0], planeOutputC.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputU.ptr[0], planeInputV.ptr[0], planeInputU.pitch[0], pCrop->e.left, pCrop->e.up);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_yuv444_yv12_p(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut); //yv12
        int isrc = ((uv_y << 1) + offsetY) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //yuv444

        const TypeIn src_u0 = *(const TypeIn *)(pSrcU + isrc +        0);
        const TypeIn src_u1 = *(const TypeIn *)(pSrcU + isrc + srcPitch);
        const TypeIn src_v0 = *(const TypeIn *)(pSrcV + isrc +        0);
        const TypeIn src_v1 = *(const TypeIn *)(pSrcV + isrc + srcPitch);

        *(TypeOut *)(pDstU + idst) = (TypeOut)BIT_DEPTH_CONV_AVG(src_u0, src_u1);
        *(TypeOut *)(pDstV + idst) = (TypeOut)BIT_DEPTH_CONV_AVG(src_v0, src_v1);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_yuv444_yv12_i(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut); //yv12
        int isrc = ((uv_y << 1) + offsetY) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //yuv444

        const TypeIn src_u0 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 0);
        const TypeIn src_u1 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 1);
        const TypeIn src_u2 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 2);
        const TypeIn src_u3 = *(const TypeIn *)(pSrcU + isrc + srcPitch * 3);
        const TypeIn src_v0 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 0);
        const TypeIn src_v1 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 1);
        const TypeIn src_v2 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 2);
        const TypeIn src_v3 = *(const TypeIn *)(pSrcV + isrc + srcPitch * 3);

        *(TypeOut *)(pDstU + idst +        0) = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_u0, src_u2);
        *(TypeOut *)(pDstV + idst +        0) = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_v0, src_v2);
        *(TypeOut *)(pDstU + idst + dstPitch) = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_u3, src_u1);
        *(TypeOut *)(pDstV + idst + dstPitch) = (TypeOut)BIT_DEPTH_CONV_3x1_AVG(src_v3, src_v1);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_yuv444_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    dim3 blockSize(32, 4);
    if (interlaced(*pInputFrame)) {
        dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 2), divCeil(pOutputFrame->height, blockSize.y * 4));
        kernel_crop_uv_yuv444_yv12_i<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputU.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputU.ptr[0], planeInputV.ptr[0], planeInputU.pitch[0], pCrop->e.left, pCrop->e.up);
    } else {
        dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 2), divCeil(pOutputFrame->height, blockSize.y * 2));
        kernel_crop_uv_yuv444_yv12_p<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize, 0, stream >>>(
            planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputU.pitch[0], pOutputFrame->width, pOutputFrame->height,
            planeInputU.ptr[0], planeInputV.ptr[0], planeInputU.pitch[0], pCrop->e.left, pCrop->e.up);
    }
}


static __device__ __inline__
float mat_det(const float mat[3][3]) {
    const float determinant =
        + mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
        - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
        + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    return determinant;
}

static __device__ __inline__
float mat_det2(float a00, float a01, float a10, float a11) {
    return a00 * a11 - a01 * a10;
}

static __device__ __inline__
void mat_inv(float rmat[3][3], const float mat[3][3]) {
    const float invdet = 1.0f / mat_det(mat);

    rmat[0][0] = mat_det2(mat[1][1], mat[1][2], mat[2][1], mat[2][2]) * invdet;
    rmat[0][1] = mat_det2(mat[0][2], mat[0][1], mat[2][2], mat[2][1]) * invdet;
    rmat[0][2] = mat_det2(mat[0][1], mat[0][2], mat[1][1], mat[1][2]) * invdet;
    rmat[1][0] = mat_det2(mat[1][2], mat[1][0], mat[2][2], mat[2][0]) * invdet;
    rmat[1][1] = mat_det2(mat[0][0], mat[0][2], mat[2][0], mat[2][2]) * invdet;
    rmat[1][2] = mat_det2(mat[0][2], mat[0][0], mat[1][2], mat[1][0]) * invdet;
    rmat[2][0] = mat_det2(mat[1][0], mat[1][1], mat[2][0], mat[2][1]) * invdet;
    rmat[2][1] = mat_det2(mat[0][1], mat[0][0], mat[2][1], mat[2][0]) * invdet;
    rmat[2][2] = mat_det2(mat[0][0], mat[0][1], mat[1][0], mat[1][1]) * invdet;
}

static __device__ __inline__
void genMatrix(float mat[3][3], const float r, const float b) {
    const float g = 1.0f - (r + b);
    const float u = 0.5f / (1.0f - b);
    const float v = 0.5f / (1.0f - r);
    mat[0][0] =                r; mat[0][1] =      g; mat[0][2] =             b;
    mat[1][0] =           -r * u; mat[1][1] = -g * u; mat[1][2] = (1.0f - b) * u;
    mat[2][0] =   (1.0f - r) * v; mat[2][1] = -g * v; mat[2][2] =        -b * v;
}

template<CspMatrix matrix>
static __device__ __inline__ void getMatrix(float mat[3][3]) {
    switch (matrix) {
    //case RGY_MATRIX_YCGCO:
    //    return mat3x3(
    //     0.25, 0.5,  0.25,
    //    -0.25, 0.5, -0.25,
    //      0.5, 0.0,  -0.5);
    //case RGY_MATRIX_2100_LMS:
    //    return mat3x3(
    //        1688.0 / 4096.0, 2146.0 / 4096.0,  262.0 / 4096.0,
    //         683.0 / 4096.0, 2951.0 / 4096.0,  462.0 / 4096.0,
    //          99.0 / 4096.0,  309.0 / 4096.0, 3688.0 / 4096.0);
    case RGY_MATRIX_RGB:       genMatrix(mat, 0.0f,       0.0f); break;
    case RGY_MATRIX_BT709:     genMatrix(mat, 0.2126f, 0.0722f); break;
    case RGY_MATRIX_FCC:       genMatrix(mat, 0.3f,      0.11f); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:   genMatrix(mat, 0.299f,   0.114f); break;
    case RGY_MATRIX_ST240_M:   genMatrix(mat, 0.212f,   0.087f); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: genMatrix(mat, 0.2627f, 0.0593f); break;
    default:                   return;
    }
}

template<CspMatrix matrix>
static __device__ float3 yuv_2_rgb(float3 yuv) {
    float mattmp[3][3];
    getMatrix<matrix>(mattmp);
    float mat[3][3];
    mat_inv(mat, mattmp);

    float3 rgb;
    rgb.x = mat[0][0] * yuv.x + mat[0][1] * yuv.y + mat[0][2] * yuv.z;
    rgb.y = mat[1][0] * yuv.x + mat[1][1] * yuv.y + mat[1][2] * yuv.z;
    rgb.z = mat[2][0] * yuv.x + mat[2][1] * yuv.y + mat[2][2] * yuv.z;
    return rgb;
}

template<CspMatrix matrix>
static __device__ float3 rgb_2_yuv(float3 rgb) {
    float mat[3][3];
    getMatrix<matrix>(mat);
    float3 yuv;
    yuv.x = mat[0][0] * rgb.x + mat[0][1] * rgb.y + mat[0][2] * rgb.z;
    yuv.y = mat[1][0] * rgb.x + mat[1][1] * rgb.y + mat[1][2] * rgb.z;
    yuv.z = mat[2][0] * rgb.x + mat[2][1] * rgb.y + mat[2][2] * rgb.z;
    return yuv;
}

template<typename TypeOut, int out_bit_depth>
static __device__ TypeOut scaleRGBFloatToPix(float x) {
    if (out_bit_depth == 32) {
        return x;
    }
    const float range = (float)((1ll << out_bit_depth) - 1);
    return (TypeOut)clamp(x * range + 0.5f, 0.0f, (float)(1ll << (out_bit_depth)) - 0.5f);
}

template<typename TypeOut, int out_bit_depth>
static __device__ TypeOut scaleYFloatToPix(float x) {
    if (out_bit_depth == 32) {
        return x;
    }
    const float range = (float)(219 << (out_bit_depth - 8));
    const float offset = (float)(16 << (out_bit_depth - 8));
    return (TypeOut)clamp(x * range + offset + 0.5f, 0.0f, (float)(1ll << (out_bit_depth)) - 0.5f);
}

template<typename TypeOut, int out_bit_depth>
static __device__ TypeOut scaleUVFloatToPix(float x) {
    if (out_bit_depth == 32) {
        return x;
    }
    const float range = (float)(224 << (out_bit_depth - 8));
    const float offset = (float)(1 << (out_bit_depth - 1));
    return (TypeOut)clamp(x * range + offset + 0.5f, 0.0f, (float)(1ll << (out_bit_depth)) - 0.5f);
}

template<typename TypeIn, int in_bit_depth>
static __device__ float scaleRGBPixToFloat(TypeIn x) {
    if (in_bit_depth == 32) {
        return x;
    }
    const float range = (float)((1ll << in_bit_depth) - 1);
    const float range_inv = 1.0f / range;
    return clamp((float)x * range_inv, 0.0f, 1.0f);
}

template<typename TypeIn, int in_bit_depth>
static __device__ float scaleYPixToFloat(TypeIn x) {
    if (in_bit_depth == 32) {
        return x;
    }
    const float range = (float)(219 << (in_bit_depth - 8));
    const float offset = (float)(16 << (in_bit_depth - 8));
    const float range_inv = 1.0f / range;
    const float offset_inv = -offset * (1.0f / range);
    return clamp((float)x * range_inv + offset_inv, 0.0f, 1.0f);
}

template<typename TypeIn, int in_bit_depth>
static __device__ float scaleUVPixToFloat(TypeIn x) {
    if (in_bit_depth == 32) {
        return x;
    }
    const float range = (float)(224 << (in_bit_depth - 8));
    const float offset = (float)(1 << (in_bit_depth - 1));
    const float range_inv = 1.0f / range;
    const float offset_inv = -offset * (1.0f / range);
    return clamp((float)x * range_inv + offset_inv, -0.5f, 0.5f);
}

template<typename TypeIn, int bit_depth>
static __device__ float3 make_float_yuv3(TypeIn y, TypeIn u, TypeIn v) {
    if (bit_depth == 32) {
        return make_float3(y, u, v);
    }
    return make_float3(
        scaleYPixToFloat<TypeIn, bit_depth>(y),
        scaleUVPixToFloat<TypeIn, bit_depth>(u),
        scaleUVPixToFloat<TypeIn, bit_depth>(v));
}

template<typename TypeIn, int bit_depth>
static __device__ float3 make_float_rgb3(TypeIn r, TypeIn g, TypeIn b) {
    if (bit_depth == 32) {
        return make_float3(r, g, b);
    }
    return make_float3(
        scaleRGBPixToFloat<TypeIn, bit_depth>(r),
        scaleRGBPixToFloat<TypeIn, bit_depth>(g),
        scaleRGBPixToFloat<TypeIn, bit_depth>(b));
}


template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
static __device__ void rgb3_2_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const TypeIn *ptr) {
    float3 rgb = make_float_rgb3<TypeIn, in_bit_depth>(ptr[0], ptr[1], ptr[2]);
    float3 yuv = rgb_2_yuv<matrix>(rgb);

    TypeOut *ptr_dst_y = (TypeOut *)(pDstY);
    TypeOut *ptr_dst_u = (TypeOut *)(pDstU);
    TypeOut *ptr_dst_v = (TypeOut *)(pDstV);
    ptr_dst_y[0] = (TypeOut)(scaleYFloatToPix<TypeOut, out_bit_depth>(yuv.x) + 0.5f);
    ptr_dst_u[0] = (TypeOut)(scaleUVFloatToPix<TypeOut, out_bit_depth>(yuv.y) + 0.5f);
    ptr_dst_v[0] = (TypeOut)(scaleUVFloatToPix<TypeOut, out_bit_depth>(yuv.z) + 0.5f);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
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
        rgb3_2_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix>(pDstY + idst, pDstU + idst, pDstV + idst, ptrSharedIn + isrc);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb3_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    kernel_crop_rgb3_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputY.pitch[0], pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb3_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb3_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb3_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb3_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
static __device__ void rgb4_2_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV, const uint8_t *ptr) {
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    TypeIn4 input = *(TypeIn4 *)ptr;
    float3 yuv = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(input.x, input.y, input.z));

    TypeOut *ptr_dst_y = (TypeOut *)(pDstY);
    TypeOut *ptr_dst_u = (TypeOut *)(pDstU);
    TypeOut *ptr_dst_v = (TypeOut *)(pDstV);
    ptr_dst_y[0] = (TypeOut)(scaleYFloatToPix<TypeOut, out_bit_depth> (yuv.x) + 0.5f);
    ptr_dst_u[0] = (TypeOut)(scaleUVFloatToPix<TypeOut, out_bit_depth>(yuv.y) + 0.5f);
    ptr_dst_v[0] = (TypeOut)(scaleUVFloatToPix<TypeOut, out_bit_depth>(yuv.z) + 0.5f);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
__global__ void kernel_crop_rgb4_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数分立てる
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        const int isrc = y * srcPitch + x * 4 * sizeof(TypeIn);
        const int idst = y * dstPitch + x * sizeof(TypeOut);
        rgb4_2_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix>(pDstY + idst, pDstU + idst, pDstV + idst, pSrc + isrc);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb4_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    kernel_crop_rgb4_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputY.pitch[0], pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb4_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb4_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb4_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb4_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
__global__ void kernel_crop_rgb_yuv444(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcR, const uint8_t *__restrict__ pSrcG, const uint8_t *__restrict__ pSrcB,
    const int srcPitch, const int offsetX, const int offsetY) {
    const int PIX_PER_THREAD = 4;
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * PIX_PER_THREAD; //4pixel分ロードする
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    struct __align__(sizeof(TypeIn) * 4) TypeIn4 {
        TypeIn x, y, z, w;
    };
    struct __align__(sizeof(TypeOut) * 4) TypeOut4 {
        TypeOut x, y, z, w;
    };
    if (x + PIX_PER_THREAD - 1 < dstWidth && y < dstHeight) {
        TypeIn4 srcR = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcR + y * srcPitch + x * sizeof(TypeIn));
        TypeIn4 srcG = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcG + y * srcPitch + x * sizeof(TypeIn));
        TypeIn4 srcB = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcB + y * srcPitch + x * sizeof(TypeIn));

        float3 pix0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(srcR.x, srcG.x, srcB.x));
        float3 pix1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(srcR.y, srcG.y, srcB.y));
        float3 pix2 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(srcR.z, srcG.z, srcB.z));
        float3 pix3 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(srcR.w, srcG.w, srcB.w));

        TypeOut4 dstY, dstU, dstV;
        dstY.x = scaleYFloatToPix<TypeOut, out_bit_depth>(pix0.x); dstU.x = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix0.y); dstV.x = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix0.z);
        dstY.y = scaleYFloatToPix<TypeOut, out_bit_depth>(pix1.x); dstU.y = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix1.y); dstV.y = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix1.z);
        dstY.z = scaleYFloatToPix<TypeOut, out_bit_depth>(pix2.x); dstU.z = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix2.y); dstV.z = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix2.z);
        dstY.w = scaleYFloatToPix<TypeOut, out_bit_depth>(pix3.x); dstU.w = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix3.y); dstV.w = scaleUVFloatToPix<TypeOut, out_bit_depth>(pix3.z);

        TypeOut4 *ptrDstY = (TypeOut4 *)(pDstY + y * dstPitch + x * sizeof(TypeOut));
        TypeOut4 *ptrDstU = (TypeOut4 *)(pDstU + y * dstPitch + x * sizeof(TypeOut));
        TypeOut4 *ptrDstV = (TypeOut4 *)(pDstV + y * dstPitch + x * sizeof(TypeOut));

        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptrDstY, dstY);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptrDstU, dstU);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptrDstV, dstV);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
void crop_rgb_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputR = getPlane(pInputFrame, RGY_PLANE_R);
    const auto planeInputG = getPlane(pInputFrame, RGY_PLANE_G);
    const auto planeInputB = getPlane(pInputFrame, RGY_PLANE_B);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 4), divCeil(pOutputFrame->height, blockSize.y));
    kernel_crop_rgb_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, aligned, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputY.pitch[0], planeOutputY.width, planeOutputY.height,
        planeInputR.ptr[0], planeInputG.ptr[0], planeInputB.ptr[0], planeInputR.pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb_yuv444_a(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const bool aligned = isAlignedYUV444(pOutputFrame, sizeof(TypeOut) * 4) && isAlignedRGB(pInputFrame, sizeof(TypeIn) * 4) && !cropEnabled(pCrop);
    (aligned) ? crop_rgb_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, true, matrix>(pOutputFrame, pInputFrame, pCrop, stream)
                : crop_rgb_yuv444<TypeOut, out_bit_depth, TypeIn, in_bit_depth, false, matrix>(pOutputFrame, pInputFrame, pCrop, stream);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb_yuv444_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb_yuv444_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb_yuv444_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
__global__ void kernel_crop_yuv444_rgb(
    uint8_t *__restrict__ pDstR, uint8_t *__restrict__ pDstG, uint8_t *__restrict__ pDstB,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcY, const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV,
    const int srcPitch, const int offsetX, const int offsetY) {
    const int PIX_PER_THREAD = 4;
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * PIX_PER_THREAD; //4pixel分ロードする
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    struct __align__(sizeof(TypeIn) * 4) TypeIn4 {
        TypeIn x, y, z, w;
    };
    struct __align__(sizeof(TypeOut) * 4) TypeOut4 {
        TypeOut x, y, z, w;
    };
    if (x + PIX_PER_THREAD - 1 < dstWidth && y < dstHeight) {
        TypeIn4 srcY = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcY + y * srcPitch + x * sizeof(TypeIn));
        TypeIn4 srcU = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcU + y * srcPitch + x * sizeof(TypeIn));
        TypeIn4 srcV = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcV + y * srcPitch + x * sizeof(TypeIn));

        float3 pix0 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY.x, srcU.x, srcV.x));
        float3 pix1 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY.y, srcU.y, srcV.y));
        float3 pix2 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY.z, srcU.z, srcV.z));
        float3 pix3 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY.w, srcU.w, srcV.w));

        TypeOut4 dstR, dstG, dstB;
        dstR.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix0.x); dstG.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix0.y); dstB.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix0.z);
        dstR.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix1.x); dstG.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix1.y); dstB.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix1.z);
        dstR.z = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix2.x); dstG.z = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix2.y); dstB.z = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix2.z);
        dstR.w = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix3.x); dstG.w = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix3.y); dstB.w = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix3.z);

        TypeOut4 *ptrDstR = (TypeOut4 *)(pDstR + y * dstPitch + x * sizeof(TypeOut));
        TypeOut4 *ptrDstG = (TypeOut4 *)(pDstG + y * dstPitch + x * sizeof(TypeOut));
        TypeOut4 *ptrDstB = (TypeOut4 *)(pDstB + y * dstPitch + x * sizeof(TypeOut));

        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptrDstR, dstR);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptrDstG, dstG);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptrDstB, dstB);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
void crop_yuv444_rgb(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputR = getPlane(pOutputFrame, RGY_PLANE_R);
    auto planeOutputG = getPlane(pOutputFrame, RGY_PLANE_G);
    auto planeOutputB = getPlane(pOutputFrame, RGY_PLANE_B);

    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 4), divCeil(pOutputFrame->height, blockSize.y));
    kernel_crop_yuv444_rgb<TypeOut, out_bit_depth, TypeIn, in_bit_depth, aligned, matrix> << <gridSize, blockSize, 0, stream >> > (
        planeOutputR.ptr[0], planeOutputG.ptr[0], planeOutputB.ptr[0], planeOutputR.pitch[0], planeOutputR.width, planeOutputR.height,
        planeInputY.ptr[0], planeInputU.ptr[0], planeInputV.ptr[0], planeInputY.pitch[0],
        pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_yuv444_rgb_a(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const bool aligned = isAlignedRGB(pOutputFrame, sizeof(TypeOut) * 4) && isAlignedYUV444(pInputFrame, sizeof(TypeIn) * 4) && !cropEnabled(pCrop);
    (aligned) ? crop_yuv444_rgb<TypeOut, out_bit_depth, TypeIn, in_bit_depth, true, matrix>(pOutputFrame, pInputFrame, pCrop, stream)
                : crop_yuv444_rgb<TypeOut, out_bit_depth, TypeIn, in_bit_depth, false, matrix>(pOutputFrame, pInputFrame, pCrop, stream);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_yuv444_rgb(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_yuv444_rgb_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_yuv444_rgb_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_yuv444_rgb_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeIn, int in_bit_depth, CspMatrix matrix>
static __device__ float3 rgb3_2_yuv(const TypeIn *ptr) {
    float3 rgb = make_float_rgb3<TypeIn, in_bit_depth>(ptr[0], ptr[1], ptr[2]);
    return rgb_2_yuv<matrix>(rgb);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
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
        float3 yuv00 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv01 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv10 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 1) * blockDim.x * 6);
        float3 yuv11 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 1) * blockDim.x * 6);

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + y * dstPitch + x * sizeof(TypeOut));
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + y * dstPitch + x * sizeof(TypeOut));
        ptr_dst_y00[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv00.x);
        ptr_dst_y01[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv01.x);
        ptr_dst_y10[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv10.x);
        ptr_dst_y11[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv11.x);
        ptr_dst_u[0]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f);
        ptr_dst_v[0]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb3_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    kernel_crop_rgb3_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputY.pitch[0], pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb3_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb3_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb3_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb3_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
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
        float3 yuv00 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv01 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 0) * blockDim.x * 6);
        float3 yuv10 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 0 + (threadIdx.y * 2 + 1) * blockDim.x * 6);
        float3 yuv11 = rgb3_2_yuv<TypeIn, in_bit_depth, matrix>(ptrSharedIn + threadIdx.x * 6 + 3 + (threadIdx.y * 2 + 1) * blockDim.x * 6);

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_c = (TypeOut *)(pDstC + y * dstPitch + x * 2 * sizeof(TypeOut));
        ptr_dst_y00[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv00.x);
        ptr_dst_y01[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv01.x);
        ptr_dst_y10[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv10.x);
        ptr_dst_y11[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv11.x);
        ptr_dst_c[0]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f);
        ptr_dst_c[1]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb3_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputC = getPlane(pOutputFrame, RGY_PLANE_C);
    kernel_crop_rgb3_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputC.ptr[0], planeOutputY.pitch[0], pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb3_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb3_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb3_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb3_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeIn, int in_bit_depth, CspMatrix matrix>
static __device__ float3 rgb4_2_yuv(const TypeIn *ptr) {
    struct TypeIn4 {
        TypeIn x, y, z, w;
    };
    TypeIn4 input = *(TypeIn4 *)ptr;
    float3 rgb = make_float_rgb3<TypeIn, in_bit_depth>(input.z, input.y, input.x);
    return rgb_2_yuv<matrix>(rgb);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
__global__ void kernel_crop_rgb4_yv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/2
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    if (x * 2 < dstWidth && y * 2 < dstHeight) {
        float3 yuv00 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv01 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));
        float3 yuv10 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv11 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + y * dstPitch + x * sizeof(TypeOut));
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + y * dstPitch + x * sizeof(TypeOut));
        ptr_dst_y00[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv00.x);
        ptr_dst_y01[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv01.x);
        ptr_dst_y10[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv10.x);
        ptr_dst_y11[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv11.x);
        ptr_dst_u[0]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f);
        ptr_dst_v[0]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb4_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    kernel_crop_rgb4_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputY.pitch[0], pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb4_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb4_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb4_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb4_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
__global__ void kernel_crop_rgb4_nv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstC,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/2
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    if (x * 2 < dstWidth && y * 2 < dstHeight) {
        float3 yuv00 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv01 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 0) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));
        float3 yuv10 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 0) * 4 * sizeof(TypeIn));
        float3 yuv11 = rgb4_2_yuv<TypeIn, in_bit_depth, matrix>(pSrc + (y * 2 + 1) * srcPitch + (x * 2 + 1) * 4 * sizeof(TypeIn));

        TypeOut *ptr_dst_y00 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y01 = (TypeOut *)(pDstY + ((y * 2 + 0) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_y10 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 0) * sizeof(TypeOut));
        TypeOut *ptr_dst_y11 = (TypeOut *)(pDstY + ((y * 2 + 1) * dstPitch) + (x * 2 + 1) * sizeof(TypeOut));
        TypeOut *ptr_dst_c = (TypeOut *)(pDstC + y * dstPitch + x * 2 * sizeof(TypeOut));
        ptr_dst_y00[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv00.x);
        ptr_dst_y01[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv01.x);
        ptr_dst_y10[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv10.x);
        ptr_dst_y11[0] = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv11.x);
        ptr_dst_c[0]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.y + yuv01.y + yuv10.y + yuv11.y) * 0.25f);
        ptr_dst_c[1]   = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv00.z + yuv01.z + yuv10.z + yuv11.z) * 0.25f);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb4_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width / 2, blockSize.x), divCeil(pOutputFrame->height / 2, blockSize.y));
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputC = getPlane(pOutputFrame, RGY_PLANE_C);
    kernel_crop_rgb4_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputC.ptr[0], planeOutputY.pitch[0], pOutputFrame->width, pOutputFrame->height,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb4_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb4_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb4_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb4_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
__global__ void kernel_crop_rgb_yv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcR, const uint8_t *__restrict__ pSrcG, const uint8_t *__restrict__ pSrcB,
    const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/4
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    struct __align__(sizeof(TypeIn) * 4) TypeIn4 {
        TypeIn x, y, z, w;
    };
    struct __align__(sizeof(TypeOut) * 4) TypeOut4 {
        TypeOut x, y, z, w;
    };
    struct __align__(sizeof(TypeOut) * 2) TypeOut2 {
        TypeOut x, y;
    };
    if (((x*4 + 3) & ~3) < dstWidth && y * 2 < dstHeight) {
        TypeIn4 r_x0123_y0 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcR + (y * 2 + 0) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 r_x0123_y1 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcR + (y * 2 + 1) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 g_x0123_y0 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcG + (y * 2 + 0) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 g_x0123_y1 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcG + (y * 2 + 1) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 b_x0123_y0 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcB + (y * 2 + 0) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 b_x0123_y1 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcB + (y * 2 + 1) * srcPitch + x * 4 * sizeof(TypeIn));

        float3 yuv_x0_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.x, g_x0123_y0.x, b_x0123_y0.x));
        float3 yuv_x1_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.y, g_x0123_y0.y, b_x0123_y0.y));
        float3 yuv_x2_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.z, g_x0123_y0.z, b_x0123_y0.z));
        float3 yuv_x3_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.w, g_x0123_y0.w, b_x0123_y0.w));
        float3 yuv_x0_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.x, g_x0123_y1.x, b_x0123_y1.x));
        float3 yuv_x1_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.y, g_x0123_y1.y, b_x0123_y1.y));
        float3 yuv_x2_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.z, g_x0123_y1.z, b_x0123_y1.z));
        float3 yuv_x3_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.w, g_x0123_y1.w, b_x0123_y1.w));

        TypeOut4 *ptr_dst_y0 = (TypeOut4 *)(pDstY + ((y * 2 + 0) * dstPitch) + x * 4 * sizeof(TypeOut));
        TypeOut4 *ptr_dst_y1 = (TypeOut4 *)(pDstY + ((y * 2 + 1) * dstPitch) + x * 4 * sizeof(TypeOut));
        TypeOut4 dstY0, dstY1;
        dstY0.x = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x0_y0.x); dstY1.x = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x0_y1.x);
        dstY0.y = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x1_y0.x); dstY1.y = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x1_y1.x);
        dstY0.z = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x2_y0.x); dstY1.z = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x2_y1.x);
        dstY0.w = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x3_y0.x); dstY1.w = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x3_y1.x);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptr_dst_y0, dstY0);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptr_dst_y1, dstY1);

        TypeOut2 *ptr_dst_u = (TypeOut2 *)(pDstU + y * dstPitch + x * 2 * sizeof(TypeOut));
        TypeOut2 *ptr_dst_v = (TypeOut2 *)(pDstV + y * dstPitch + x * 2 * sizeof(TypeOut));
        TypeOut2 dstU, dstV;
        dstU.x = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x0_y0.y + yuv_x1_y0.y + yuv_x0_y1.y + yuv_x1_y1.y) * 0.25f);
        dstU.y = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x2_y0.y + yuv_x3_y0.y + yuv_x0_y1.y + yuv_x2_y1.y) * 0.25f);
        dstV.x = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x0_y0.z + yuv_x1_y0.z + yuv_x0_y1.z + yuv_x1_y1.z) * 0.25f);
        dstV.y = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x2_y0.z + yuv_x3_y0.z + yuv_x0_y1.z + yuv_x2_y1.z) * 0.25f);
        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptr_dst_u, dstU);
        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptr_dst_v, dstV);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
void crop_rgb_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputR = getPlane(pInputFrame, RGY_PLANE_R);
    const auto planeInputG = getPlane(pInputFrame, RGY_PLANE_G);
    const auto planeInputB = getPlane(pInputFrame, RGY_PLANE_B);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 4), divCeil(pOutputFrame->height, blockSize.y * 2));
    kernel_crop_rgb_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, aligned, matrix><<<gridSize, blockSize, 0, stream >>>(
        planeOutputY.ptr[0], planeOutputU.ptr[0], planeOutputV.ptr[0], planeOutputY.pitch[0], planeOutputY.width, planeOutputY.height,
        planeInputR.ptr[0], planeInputG.ptr[0], planeInputB.ptr[0], planeInputR.pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb_yv12_a(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const bool aligned = isAlignedYV12(pOutputFrame, sizeof(TypeOut) * 4) && isAlignedRGB(pInputFrame, sizeof(TypeIn) * 4) && !cropEnabled(pCrop);
    (aligned) ? crop_rgb_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, true, matrix>(pOutputFrame, pInputFrame, pCrop, stream)
                : crop_rgb_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, false, matrix>(pOutputFrame, pInputFrame, pCrop, stream);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb_yv12_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb_yv12_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb_yv12_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
__global__ void kernel_crop_yv12_rgb(
    uint8_t *__restrict__ pDstR, uint8_t *__restrict__ pDstG, uint8_t *__restrict__ pDstB,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcY, const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV,
    const int srcPitch, const int offsetX, const int offsetY) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2; //2pixel分ロードする
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2; //2pixel分ロードする
    struct __align__(sizeof(TypeIn) * 2) TypeIn2 {
        TypeIn x, y;
    };
    struct __align__(sizeof(TypeOut) * 2) TypeOut2 {
        TypeOut x, y;
    };
    if (x < dstWidth && y < dstHeight) {
        TypeIn2 srcY0 = kernel_crop_load2<TypeIn, TypeIn2, aligned>(pSrcY + (y+0) * srcPitch +  x     * sizeof(TypeIn));
        TypeIn2 srcY1 = kernel_crop_load2<TypeIn, TypeIn2, aligned>(pSrcY + (y+1) * srcPitch +  x     * sizeof(TypeIn));
        TypeIn srcU0  = *(TypeIn *)(pSrcU + (y>>1) * srcPitch + (x>>1) * sizeof(TypeIn));
        TypeIn srcV0  = *(TypeIn *)(pSrcV + (y>>1) * srcPitch + (x>>1) * sizeof(TypeIn));

        TypeIn srcY00 = srcY0.x;
        TypeIn srcY01 = srcY0.y;
        TypeIn srcY10 = srcY1.x;
        TypeIn srcY11 = srcY1.y;

        TypeIn srcU00 = srcU0;
        TypeIn srcV00 = srcV0;
        TypeIn srcU01 = (x + 2 < dstWidth) ? *(TypeIn *)(pSrcU + (y >> 1) * srcPitch + ((x >> 1) + 1) * sizeof(TypeIn)) : srcU00;
        TypeIn srcV01 = (x + 2 < dstWidth) ? *(TypeIn *)(pSrcV + (y >> 1) * srcPitch + ((x >> 1) + 1) * sizeof(TypeIn)) : srcV00;
        TypeIn srcU001 = (srcU00 + srcU01 + 1) >> 1;
        TypeIn srcV001 = (srcV00 + srcV01 + 1) >> 1;

        float3 pix00 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY00, srcU00,  srcV00));
        float3 pix01 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY01, srcU001, srcV001));
        float3 pix10 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY10, srcU00,  srcV00));
        float3 pix11 = yuv_2_rgb<matrix>(make_float_yuv3<TypeIn, in_bit_depth>(srcY11, srcU001, srcV001));

        TypeOut2 dstR0, dstR1, dstG0, dstG1, dstB0, dstB1;
        dstR0.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix00.x); dstG0.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix00.y); dstB0.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix00.z);
        dstR0.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix01.x); dstG0.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix01.y); dstB0.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix01.z);
        dstR1.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix10.x); dstG1.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix10.y); dstB1.x = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix10.z);
        dstR1.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix11.x); dstG1.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix11.y); dstB1.y = scaleRGBFloatToPix<TypeOut, out_bit_depth>(pix11.z);

        TypeOut2 *ptrDstR = (TypeOut2 *)(pDstR + y * dstPitch + x * sizeof(TypeOut));
        TypeOut2 *ptrDstG = (TypeOut2 *)(pDstG + y * dstPitch + x * sizeof(TypeOut));
        TypeOut2 *ptrDstB = (TypeOut2 *)(pDstB + y * dstPitch + x * sizeof(TypeOut));

        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptrDstR, dstR0);
        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptrDstG, dstG0);
        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptrDstB, dstB0);

        ptrDstR = (TypeOut2 *)(pDstR + (y+1) * dstPitch + x * sizeof(TypeOut));
        ptrDstG = (TypeOut2 *)(pDstG + (y+1) * dstPitch + x * sizeof(TypeOut));
        ptrDstB = (TypeOut2 *)(pDstB + (y+1) * dstPitch + x * sizeof(TypeOut));

        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptrDstR, dstR1);
        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptrDstG, dstG1);
        kernel_crop_store2<TypeOut, TypeOut2, aligned>(ptrDstB, dstB1);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
void crop_yv12_rgb(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputR = getPlane(pOutputFrame, RGY_PLANE_R);
    auto planeOutputG = getPlane(pOutputFrame, RGY_PLANE_G);
    auto planeOutputB = getPlane(pOutputFrame, RGY_PLANE_B);

    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 2), divCeil(pOutputFrame->height, blockSize.y * 2));
    kernel_crop_yv12_rgb<TypeOut, out_bit_depth, TypeIn, in_bit_depth, aligned, matrix> << <gridSize, blockSize, 0, stream >> > (
        planeOutputR.ptr[0], planeOutputG.ptr[0], planeOutputB.ptr[0], planeOutputR.pitch[0], planeOutputR.width, planeOutputR.height,
        planeInputY.ptr[0], planeInputU.ptr[0], planeInputV.ptr[0], planeInputY.pitch[0],
        pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_yv12_rgb_a(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const bool aligned = isAlignedRGB(pOutputFrame, sizeof(TypeOut)*2) && isAlignedYV12(pInputFrame, sizeof(TypeIn) * 2) && !cropEnabled(pCrop);
    (aligned) ? crop_yv12_rgb<TypeOut, out_bit_depth, TypeIn, in_bit_depth, true, matrix>(pOutputFrame, pInputFrame, pCrop, stream)
                : crop_yv12_rgb<TypeOut, out_bit_depth, TypeIn, in_bit_depth, false, matrix>(pOutputFrame, pInputFrame, pCrop, stream);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_yv12_rgb(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_yv12_rgb_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_yv12_rgb_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_yv12_rgb_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
__global__ void kernel_crop_rgb_nv12(uint8_t *__restrict__ pDstY, uint8_t *__restrict__ pDstC,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcR, const uint8_t *__restrict__ pSrcG, const uint8_t *__restrict__ pSrcB,
    const int srcPitch, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //スレッドはpixel数/4
    int y = blockIdx.y * blockDim.y + threadIdx.y; //スレッドはpixel数/2
    struct __align__(sizeof(TypeIn) * 4) TypeIn4 {
        TypeIn x, y, z, w;
    };
    struct __align__(sizeof(TypeOut) * 4) TypeOut4 {
        TypeOut x, y, z, w;
    };
    if (((x*4 + 3) & ~3) < dstWidth && y * 2 < dstHeight) {
        TypeIn4 r_x0123_y0 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcR + (y * 2 + 0) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 r_x0123_y1 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcR + (y * 2 + 1) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 g_x0123_y0 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcG + (y * 2 + 0) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 g_x0123_y1 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcG + (y * 2 + 1) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 b_x0123_y0 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcB + (y * 2 + 0) * srcPitch + x * 4 * sizeof(TypeIn));
        TypeIn4 b_x0123_y1 = kernel_crop_load4<TypeIn, TypeIn4, aligned>(pSrcB + (y * 2 + 1) * srcPitch + x * 4 * sizeof(TypeIn));

        float3 yuv_x0_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.x, g_x0123_y0.x, b_x0123_y0.x));
        float3 yuv_x1_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.y, g_x0123_y0.y, b_x0123_y0.y));
        float3 yuv_x2_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.z, g_x0123_y0.z, b_x0123_y0.z));
        float3 yuv_x3_y0 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y0.w, g_x0123_y0.w, b_x0123_y0.w));
        float3 yuv_x0_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.x, g_x0123_y1.x, b_x0123_y1.x));
        float3 yuv_x1_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.y, g_x0123_y1.y, b_x0123_y1.y));
        float3 yuv_x2_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.z, g_x0123_y1.z, b_x0123_y1.z));
        float3 yuv_x3_y1 = rgb_2_yuv<matrix>(make_float_rgb3<TypeIn, in_bit_depth>(r_x0123_y1.w, g_x0123_y1.w, b_x0123_y1.w));

        TypeOut4 *ptr_dst_y0 = (TypeOut4 *)(pDstY + ((y * 2 + 0) * dstPitch) + x * 4 * sizeof(TypeOut));
        TypeOut4 *ptr_dst_y1 = (TypeOut4 *)(pDstY + ((y * 2 + 1) * dstPitch) + x * 4 * sizeof(TypeOut));
        TypeOut4 dstY0, dstY1;
        dstY0.x = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x0_y0.x); dstY1.x = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x0_y1.x);
        dstY0.y = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x1_y0.x); dstY1.y = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x1_y1.x);
        dstY0.z = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x2_y0.x); dstY1.z = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x2_y1.x);
        dstY0.w = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x3_y0.x); dstY1.w = scaleYFloatToPix<TypeOut, out_bit_depth>(yuv_x3_y1.x);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptr_dst_y0, dstY0);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptr_dst_y1, dstY1);

        TypeOut4 *ptr_dst_c = (TypeOut4 *)(pDstC + y * dstPitch + x * 4 * sizeof(TypeOut));
        TypeOut4 dstC;
        dstC.x = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x0_y0.y + yuv_x1_y0.y + yuv_x0_y1.y + yuv_x1_y1.y) * 0.25f);
        dstC.y = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x0_y0.z + yuv_x1_y0.z + yuv_x0_y1.z + yuv_x1_y1.z) * 0.25f);
        dstC.z = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x2_y0.y + yuv_x3_y0.y + yuv_x0_y1.y + yuv_x2_y1.y) * 0.25f);
        dstC.w = scaleUVFloatToPix<TypeOut, out_bit_depth>((yuv_x2_y0.z + yuv_x3_y0.z + yuv_x0_y1.z + yuv_x2_y1.z) * 0.25f);
        kernel_crop_store4<TypeOut, TypeOut4, aligned>(ptr_dst_c, dstC);
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, bool aligned, CspMatrix matrix>
void crop_rgb_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const auto planeInputR = getPlane(pInputFrame, RGY_PLANE_R);
    const auto planeInputG = getPlane(pInputFrame, RGY_PLANE_G);
    const auto planeInputB = getPlane(pInputFrame, RGY_PLANE_B);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputC = getPlane(pOutputFrame, RGY_PLANE_U);
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x * 4), divCeil(pOutputFrame->height, blockSize.y * 2));
    kernel_crop_rgb_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, aligned, matrix><<<gridSize, blockSize, 0, stream>>>(
        planeOutputY.ptr[0], planeOutputC.ptr[0], planeOutputY.pitch[0], planeOutputY.width, planeOutputY.height,
        planeInputR.ptr[0], planeInputG.ptr[0], planeInputB.ptr[0], planeInputR.pitch[0], pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth, CspMatrix matrix>
void crop_rgb_nv12_a(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream) {
    const bool aligned = isAlignedNV12(pOutputFrame, sizeof(TypeOut) * 4) && isAlignedRGB(pInputFrame, sizeof(TypeIn) * 4) && !cropEnabled(pCrop);
    (aligned) ? crop_rgb_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, true, matrix>(pOutputFrame, pInputFrame, pCrop, stream)
              : crop_rgb_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth, false, matrix>(pOutputFrame, pInputFrame, pCrop, stream);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_rgb_nv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, const CspMatrix matrix, cudaStream_t stream) {
    switch (matrix) {
    case RGY_MATRIX_BT709:     crop_rgb_nv12_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT709>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL: crop_rgb_nv12_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_BT2020_NCL>(pOutputFrame, pInputFrame, pCrop, stream); break;
    case RGY_MATRIX_BT470_BG:
    case RGY_MATRIX_ST170_M:
    default:                   crop_rgb_nv12_a<TypeOut, out_bit_depth, TypeIn, in_bit_depth, RGY_MATRIX_ST170_M>(pOutputFrame, pInputFrame, pCrop, stream); break;
    }
}

RGY_ERR NVEncFilterCspCrop::convertCspFromNV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        const int pixsize = RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8 ? 2 : 1;
        auto sts = copyPlaneAsync(pOutputFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromNV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        CUDA_DEBUG_SYNC_ERR;
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        CUDA_DEBUG_SYNC_ERR;
    }

    //UV
    static const std::map<uint64_t, void (*)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream)> convert_from_nv12_list = {
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12   ).i,   crop_uv_nv12_yv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_16).i,   crop_uv_nv12_yv12<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_14).i,   crop_uv_nv12_yv12<uint16_t, 14, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_12).i,   crop_uv_nv12_yv12<uint16_t, 12, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_10).i,   crop_uv_nv12_yv12<uint16_t, 10, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV12, RGY_CSP_YV12_09).i,   crop_uv_nv12_yv12<uint16_t,  9, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P010, RGY_CSP_YV12   ).i,   crop_uv_nv12_yv12<uint8_t,   8, uint16_t, 16> },
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
        return RGY_ERR_UNSUPPORTED;
    }
    convert_from_nv12_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at convert_from_nv12_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
        return sts;
    }
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}
RGY_ERR NVEncFilterCspCrop::convertCspFromYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB) {
        static const std::map<uint64_t, decltype(&crop_yv12_rgb<uint8_t, 8, uint8_t, 8>)> convert_from_yv12_to_rgb_list = {
            { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_RGB    ).i, crop_yv12_rgb<uint8_t,   8, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_RGB    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_RGB    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_RGB    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_RGB    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_GBR    ).i, crop_yv12_rgb<uint8_t,   8, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_GBR    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_GBR    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_GBR    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_GBR    ).i, crop_yv12_rgb<uint8_t,   8, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_RGB_16 ).i, crop_yv12_rgb<uint16_t, 16, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_RGB_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_RGB_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_RGB_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_RGB_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_BGR_16 ).i, crop_yv12_rgb<uint16_t, 16, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_BGR_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_BGR_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_BGR_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_BGR_16 ).i, crop_yv12_rgb<uint16_t, 16, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_RGB_F32).i, crop_yv12_rgb<float,    32, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_RGB_F32).i, crop_yv12_rgb<float,    32, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_RGB_F32).i, crop_yv12_rgb<float,    32, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_RGB_F32).i, crop_yv12_rgb<float,    32, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_RGB_F32).i, crop_yv12_rgb<float,    32, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_BGR_F32).i, crop_yv12_rgb<float,    32, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_BGR_F32).i, crop_yv12_rgb<float,    32, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_BGR_F32).i, crop_yv12_rgb<float,    32, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_BGR_F32).i, crop_yv12_rgb<float,    32, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_BGR_F32).i, crop_yv12_rgb<float,    32, uint16_t, 10> }
        };
        if (interlaced(*pInputFrame)) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported interlaced csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
        if (convert_from_yv12_to_rgb_list.count(cspconv.i) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        convert_from_yv12_to_rgb_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, pCropParam->matrix, stream);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto sts = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at convert_from_yv12_to_rgb_list(%s -> %s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        return RGY_ERR_NONE;
    }

    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        auto sts = copyPlaneAsync(pOutputFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromYV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        CUDA_DEBUG_SYNC_ERR;
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        CUDA_DEBUG_SYNC_ERR;
    }
    // YV12 - UV
    static const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pOutputFrame->csp) != supportedCspYV12.end()) {
        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        convertYBitDepth(&planeOutputU, &planeInputU, stream);
        convertYBitDepth(&planeOutputV, &planeInputV, stream);
        return RGY_ERR_NONE;
    }
    // NV12 - UV
    static const std::map<uint64_t, void (*)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream)> crop_uv_yv12_nv12_list = {
        { RGY_CSP_2(RGY_CSP_YV12,    RGY_CSP_NV12).i, crop_uv_yv12_nv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_YV12_16, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YV12_14, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_YV12_12, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_YV12_10, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_YV12_09, RGY_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t,  9> },
    };
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (crop_uv_yv12_nv12_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    crop_uv_yv12_nv12_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at crop_uv_nv12_yv12_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
        return sts;
    }
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;

}

RGY_ERR NVEncFilterCspCrop::convertCspFromNV16(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        auto sts = copyPlaneAsync(pOutputFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromNV16(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        CUDA_DEBUG_SYNC_ERR;
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        CUDA_DEBUG_SYNC_ERR;
    }

    //UV
    static const std::map<uint64_t, void(*)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream)> convert_from_nv16_list = {
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12).i,      crop_uv_nv16_yv12_p<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_16).i,   crop_uv_nv16_yv12_p<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_14).i,   crop_uv_nv16_yv12_p<uint16_t, 14, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_12).i,   crop_uv_nv16_yv12_p<uint16_t, 12, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_10).i,   crop_uv_nv16_yv12_p<uint16_t, 10, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_NV16, RGY_CSP_YV12_09).i,   crop_uv_nv16_yv12_p<uint16_t,  9, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_P210, RGY_CSP_YV12).i,      crop_uv_nv16_yv12_p<uint8_t,   8, uint16_t, 16> },
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
    if (interlaced(*pInputFrame) && RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported interlaced csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (convert_from_nv16_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    convert_from_nv16_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at convert_from_nv16_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
        return sts;
    }
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterCspCrop::convertCspFromYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB) {
        static const std::map<uint64_t, decltype(&crop_yuv444_rgb<uint8_t, 8, uint8_t, 8>)> convert_from_yuv444_to_rgb_list = {
            { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_RGB    ).i, crop_yuv444_rgb<uint8_t,   8, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_RGB    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_RGB    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_RGB    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_RGB    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_GBR    ).i, crop_yuv444_rgb<uint8_t,   8, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_GBR    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_GBR    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_GBR    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_GBR    ).i, crop_yuv444_rgb<uint8_t,   8, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_RGB_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_RGB_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_RGB_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_RGB_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_RGB_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_BGR_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_BGR_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_BGR_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_BGR_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_BGR_16 ).i, crop_yuv444_rgb<uint16_t, 16, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_RGB_F32).i, crop_yuv444_rgb<float,    32, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_RGB_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_RGB_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_RGB_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_RGB_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 10> },
            { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_BGR_F32).i, crop_yuv444_rgb<float,    32, uint8_t,   8> },
            { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_BGR_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 16> },
            { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_BGR_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 14> },
            { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_BGR_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 12> },
            { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_BGR_F32).i, crop_yuv444_rgb<float,    32, uint16_t, 10> }
        };
        const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
        if (convert_from_yuv444_to_rgb_list.count(cspconv.i) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        convert_from_yuv444_to_rgb_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, pCropParam->matrix, stream);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto sts = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at convert_from_yuv444_to_rgb_list(%s -> %s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        return RGY_ERR_NONE;
    }
    //Y
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] == RGY_CSP_BIT_DEPTH[pOutputFrame->csp]) {
        auto sts = copyPlaneAsync(pOutputFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync (convertCspFromYUV444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        CUDA_DEBUG_SYNC_ERR;
    } else {
        auto ret = convertYBitDepth(pOutputFrame, pInputFrame, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        CUDA_DEBUG_SYNC_ERR;
    }

    //UV
    static const std::map<uint64_t, void(*)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream)> convert_from_yuv444_list = {
        { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_NV12).i, crop_uv_yuv444_nv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_P010).i, crop_uv_yuv444_nv12<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_YUV444_09, RGY_CSP_NV12).i, crop_uv_yuv444_nv12<uint8_t,   8, uint16_t,  9> },
        { RGY_CSP_2(RGY_CSP_YUV444_09, RGY_CSP_P010).i, crop_uv_yuv444_nv12<uint16_t, 16, uint16_t,  9> },
        { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_NV12).i, crop_uv_yuv444_nv12<uint8_t,   8, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_YUV444_10, RGY_CSP_P010).i, crop_uv_yuv444_nv12<uint16_t, 16, uint16_t, 10> },
        { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_NV12).i, crop_uv_yuv444_nv12<uint8_t,   8, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_YUV444_12, RGY_CSP_P010).i, crop_uv_yuv444_nv12<uint16_t, 16, uint16_t, 12> },
        { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_NV12).i, crop_uv_yuv444_nv12<uint8_t,   8, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_YUV444_14, RGY_CSP_P010).i, crop_uv_yuv444_nv12<uint16_t, 16, uint16_t, 14> },
        { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_NV12).i, crop_uv_yuv444_nv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_P010).i, crop_uv_yuv444_nv12<uint16_t, 16, uint16_t, 16> },

        { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_YV12).i,    crop_uv_yuv444_yv12<uint8_t,   8, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_YUV444,    RGY_CSP_YV12_16).i, crop_uv_yuv444_yv12<uint16_t, 16, uint8_t,   8> },
        { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_YV12).i,    crop_uv_yuv444_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_YUV444_16, RGY_CSP_YV12_16).i, crop_uv_yuv444_yv12<uint16_t, 16, uint16_t, 16> }
    };
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (convert_from_yuv444_list.count(cspconv.i) != 0) {
        convert_from_yuv444_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, stream);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto sts = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at convert_from_yuv444(%s -> %s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
            return sts;
        }
        CUDA_DEBUG_SYNC_ERR;
        return RGY_ERR_NONE;
    }
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pOutputFrame->csp) != supportedCspYUV444.end()) {
        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        convertYBitDepth(&planeOutputU, &planeInputU, stream);
        convertYBitDepth(&planeOutputV, &planeInputV, stream);
        CUDA_DEBUG_SYNC_ERR;
        return RGY_ERR_NONE;
    }
    AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterCspCrop::convertCspFromRGB(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<uint64_t, decltype(&crop_rgb3_yuv444<uint8_t, 8, uint8_t, 8 >)> convert_from_rgb_list = {
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444).i,    crop_rgb3_yuv444<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_09).i, crop_rgb3_yuv444<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_10).i, crop_rgb3_yuv444<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_12).i, crop_rgb3_yuv444<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_14).i, crop_rgb3_yuv444<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YUV444_16).i, crop_rgb3_yuv444<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12).i,      crop_rgb3_yv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_09).i,   crop_rgb3_yv12<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_10).i,   crop_rgb3_yv12<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_12).i,   crop_rgb3_yv12<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_14).i,   crop_rgb3_yv12<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_YV12_16).i,   crop_rgb3_yv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_NV12).i,      crop_rgb3_nv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB24, RGY_CSP_P010).i,      crop_rgb3_nv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444).i,    crop_rgb4_yuv444<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_09).i, crop_rgb4_yuv444<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_10).i, crop_rgb4_yuv444<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_12).i, crop_rgb4_yuv444<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_14).i, crop_rgb4_yuv444<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YUV444_16).i, crop_rgb4_yuv444<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12).i,      crop_rgb4_yv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_09).i,   crop_rgb4_yv12<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_10).i,   crop_rgb4_yv12<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_12).i,   crop_rgb4_yv12<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_14).i,   crop_rgb4_yv12<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_YV12_16).i,   crop_rgb4_yv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_NV12).i,      crop_rgb4_nv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB32, RGY_CSP_P010).i,      crop_rgb4_nv12<uint16_t, 16, uint8_t, 8> },

        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_RGB, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint8_t, 8> },

        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_RGB_16, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint16_t, 16> },

        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint16_t, 16> },
        { RGY_CSP_2(RGY_CSP_BGR_16, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint16_t, 16> },

        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_RGB_F32, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, float, 32> },

        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, float, 32> },
        { RGY_CSP_2(RGY_CSP_BGR_F32, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, float, 32> },

        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444).i,    crop_rgb_yuv444<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_09).i, crop_rgb_yuv444<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_10).i, crop_rgb_yuv444<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_12).i, crop_rgb_yuv444<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_14).i, crop_rgb_yuv444<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YUV444_16).i, crop_rgb_yuv444<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12).i,      crop_rgb_yv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_09).i,   crop_rgb_yv12<uint16_t,  9, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_10).i,   crop_rgb_yv12<uint16_t, 10, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_12).i,   crop_rgb_yv12<uint16_t, 12, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_14).i,   crop_rgb_yv12<uint16_t, 14, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_YV12_16).i,   crop_rgb_yv12<uint16_t, 16, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_NV12).i,      crop_rgb_nv12<uint8_t,   8, uint8_t, 8> },
        { RGY_CSP_2(RGY_CSP_GBR, RGY_CSP_P010).i,      crop_rgb_nv12<uint16_t, 16, uint8_t, 8> },
    };
    const auto cspconv = RGY_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (convert_from_rgb_list.count(cspconv.i) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    convert_from_rgb_list.at(cspconv.i)(pOutputFrame, pInputFrame, &pCropParam->crop, pCropParam->matrix, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at convert_from_rgb_list(%s -> %s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(sts));
        return sts;
    }
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}

NVEncFilterCspCrop::NVEncFilterCspCrop() {
    m_name = _T("copy/cspconv/crop");
}

NVEncFilterCspCrop::~NVEncFilterCspCrop() {
    close();
}

RGY_ERR NVEncFilterCspCrop::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //フィルタ名の調整
    m_name = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_name += _T("crop");
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_name += (m_name.length()) ? _T("/cspconv") : _T("cspconv");
    }
    if (m_name.length() == 0) {
        const auto memcpyKind = getCudaMemcpyKind(pParam->frameIn.mem_type, pParam->frameOut.mem_type);
        m_name += getCudaMemcpyKindStr(memcpyKind);
    }
    //パラメータチェック
    for (int i = 0; i < _countof(pCropParam->crop.c); i++) {
        if ((pCropParam->crop.c[i] & 1) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("crop should be divided by 2 (%d,%d,%d,%d).\n"), pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->crop.e.right, pCropParam->crop.e.bottom);
            return RGY_ERR_INVALID_PARAM;
        }
    }
    //yuv422->yuv420のインタレ対応の変換はないので、yuv444を経由するようにする
    if (RGY_CSP_CHROMA_FORMAT[pCropParam->frameIn.csp] == RGY_CHROMAFMT_YUV422
        && RGY_CSP_CHROMA_FORMAT[pCropParam->frameOut.csp] == RGY_CHROMAFMT_YUV420
        && (pCropParam->frameIn.picstruct & RGY_PICSTRUCT_INTERLACED) != 0) {
        pCropParam->frameOut.csp = (RGY_CSP_BIT_DEPTH[pCropParam->frameOut.csp] > 8) ? RGY_CSP_YUV444_16 : RGY_CSP_YUV444;
    }
    pCropParam->frameOut.picstruct = pCropParam->frameIn.picstruct;
    pCropParam->frameOut.height = pCropParam->frameIn.height - pCropParam->crop.e.bottom - pCropParam->crop.e.up;
    pCropParam->frameOut.width = pCropParam->frameIn.width - pCropParam->crop.e.left - pCropParam->crop.e.right;
    if (pCropParam->frameOut.height <= 0 || pCropParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("crop size is too big.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = AllocFrameBuf(pCropParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pCropParam->frameOut.pitch[0] = m_frameBuf[0]->frame.pitch[0];
    }

    //フィルタ情報の調整
    setFilterInfo(pCropParam->print());
    m_param = pCropParam;
    return sts;
}

tstring NVEncFilterParamCrop::print() const {
    tstring filterInfo;
    if (cropEnabled(crop)) {
        filterInfo += strsprintf(_T("crop: %d,%d,%d,%d"), crop.e.left, crop.e.up, crop.e.right, crop.e.bottom);
    }
    if (frameOut.csp != frameIn.csp) {
        filterInfo += (filterInfo.length()) ? _T("/cspconv") : _T("cspconv");
        filterInfo += strsprintf(_T("(%s -> %s)"), RGY_CSP_NAMES[frameIn.csp], RGY_CSP_NAMES[frameOut.csp]);
        if (   RGY_CSP_CHROMA_FORMAT[frameIn.csp]  == RGY_CHROMAFMT_RGB
            || RGY_CSP_CHROMA_FORMAT[frameOut.csp] == RGY_CHROMAFMT_RGB) {
            if (   matrix == RGY_MATRIX_BT709
                || matrix == RGY_MATRIX_BT470_BG
                || matrix == RGY_MATRIX_ST170_M
                || matrix == RGY_MATRIX_BT2020_NCL
                || matrix == RGY_MATRIX_BT2020_CL) {
                //filterInfo += strsprintf(_T(" [%s]"), get_cx_desc(list_colormatrix, matrix));
            }
        }
    }
    if (filterInfo.length() == 0) {
        filterInfo += getCudaMemcpyKindStr(frameIn.mem_type, frameOut.mem_type);
    }
    return filterInfo;
}

RGY_ERR NVEncFilterCspCrop::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (m_param->frameOut.csp == m_param->frameIn.csp) {
        auto cudaMemcpyErrMes = [&](RGY_ERR sts, const TCHAR *mes) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (filter(%s)): %s.\n"),
                mes, RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        };
#if 1
        if (!cropEnabled(pCropParam->crop)) {
            //cropがなければ、一度に転送可能
            sts = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);;
            if (sts != cudaSuccess) {
                return cudaMemcpyErrMes(sts, _T("cudaMemcpy2DAsyncAll"));
            };
            CUDA_DEBUG_SYNC_ERR;
        } else {
            if (pCropParam->frameOut.csp == RGY_CSP_NV12) {
                const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
                const auto planeInputC = getPlane(pInputFrame, RGY_PLANE_C);
                auto planeOutputY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
                auto planeOutputC = getPlane(ppOutputFrames[0], RGY_PLANE_C);
                cudaError_t cudaerr;
                //Y
                cudaerr = cudaMemcpy2DAsync((uint8_t *)planeOutputY.ptr[0], planeOutputY.pitch[0],
                    (uint8_t *)planeInputY.ptr[0] + pCropParam->crop.e.left + pCropParam->crop.e.up * planeInputY.pitch[0],
                    planeInputY.pitch[0],
                    planeInputY.width * bytesPerPix(planeInputY.csp), pCropParam->frameOut.height, memcpyKind, stream);
                if (cudaerr != cudaSuccess) {
                    return cudaMemcpyErrMes(err_to_rgy(cudaerr), _T("cudaMemcpy2DAsync_Y"));
                };
                CUDA_DEBUG_SYNC_ERR;
                //UV
                cudaerr = cudaMemcpy2DAsync((uint8_t *)planeOutputC.ptr[0], planeOutputC.pitch[0],
                    (uint8_t *)planeInputC.ptr[0]
                    + pCropParam->crop.e.left + (pCropParam->crop.e.up >> 1) * planeInputC.pitch[0],
                    pInputFrame->pitch[0],
                    planeInputC.width * bytesPerPix(planeInputC.csp), pCropParam->frameOut.height >> 1, memcpyKind, stream);
                if (cudaerr != cudaSuccess) {
                    return cudaMemcpyErrMes(err_to_rgy(cudaerr), _T("cudaMemcpy2DAsync_UV"));
                };
                CUDA_DEBUG_SYNC_ERR;
            } else {
                AddMessage(RGY_LOG_ERROR, _T("unsupported output csp with crop.\n"));
                return RGY_ERR_UNSUPPORTED;
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
        return RGY_ERR_UNSUPPORTED;
    } else {
        //色空間変換
        static const auto supportedCspNV12   = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
        static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
        static const auto supportedCspNV16   = make_array<RGY_CSP>(RGY_CSP_NV16, RGY_CSP_P210);
        static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
        static const auto supportedCspRGB    = make_array<RGY_CSP>(RGY_CSP_RGB24, RGY_CSP_RGB32, RGY_CSP_RGB, RGY_CSP_GBR, RGY_CSP_RGB_16, RGY_CSP_BGR_16, RGY_CSP_RGB_F32, RGY_CSP_BGR_F32);
        if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameIn.csp) != supportedCspNV12.end()) {
            sts = convertCspFromNV12(ppOutputFrames[0], pInputFrame, stream);
        } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameIn.csp) != supportedCspYV12.end()) {
            sts = convertCspFromYV12(ppOutputFrames[0], pInputFrame, stream);
        } else if (std::find(supportedCspNV16.begin(), supportedCspNV16.end(), pCropParam->frameIn.csp) != supportedCspNV16.end()) {
            sts = convertCspFromNV16(ppOutputFrames[0], pInputFrame, stream);
        } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameIn.csp) != supportedCspYUV444.end()) {
            sts = convertCspFromYUV444(ppOutputFrames[0], pInputFrame, stream);
        } else if (std::find(supportedCspRGB.begin(), supportedCspRGB.end(), pCropParam->frameIn.csp) != supportedCspRGB.end()) {
            sts = convertCspFromRGB(ppOutputFrames[0], pInputFrame, stream);
        } else {
            AddMessage(RGY_LOG_ERROR, _T("converting csp from %s is not supported.\n"), RGY_CSP_NAMES[pCropParam->frameIn.csp]);
            sts = RGY_ERR_UNSUPPORTED;
        }
    }
    return sts;
}

void NVEncFilterCspCrop::close() {
    m_frameBuf.clear();
}
