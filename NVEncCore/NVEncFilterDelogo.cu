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
#include "NVEncFilterDelogo.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rgy_cuda_util_kernel.h"

#define DELOGO_DEBUG_CUDA 0

template<typename Type>
void debug_out_csv(const CUFrameBuf *ptr_mask_nr, const TCHAR *filename) {
#if DELOGO_DEBUG_CUDA
#if _DEBUG
    std::ofstream file;
    file.open(filename, std::ios::out);

    const int size = ptr_mask_nr->frame.pitch * ptr_mask_nr->frame.height;
    std::vector<char> buffer(size);
    auto cudaerr = cudaMemcpy(buffer.data(), ptr_mask_nr->frame.ptr, size, cudaMemcpyDeviceToHost);
    if (cudaerr != cudaSuccess) {
        _ftprintf(stderr, _T("cudaMemcpy : %s\n"), get_err_mes(sts));
    }
    for (int j = 0; j < ptr_mask_nr->frame.height; j++) {
        for (int i = 0; i < ptr_mask_nr->frame.width; i++) {
            auto ptr = (Type *)&buffer[j * ptr_mask_nr->frame.pitch + i * sizeof(Type)];
            file << (int)(ptr[0]) << ",";
        }
        file << std::endl;
    }
    file.close();
#endif //#if _DEBUG
#endif //#if DELOGO_DEBUG_CUDA
}

template<typename Type, int bit_depth, bool target_y>
__inline__ __device__
float delogo_yc48(Type pixel, const int16x2_t logo_data, const float logo_depth_mul_fade) {
    const float nv12_2_yc48_mul = (target_y) ? 1197.0f / (1<<(bit_depth-2)) : 4681.0f / (1<<bit_depth);
    const float nv12_2_yc48_sub = (target_y) ? 299.0f : 599332.0f / 256.0f;

    //ロゴ情報取り出し
    float logo_dp = (float)logo_data.x;
    float logo    = (float)logo_data.y;

    logo_dp = (logo_dp * logo_depth_mul_fade) * (1.0f / (float)(128 * LOGO_FADE_MAX));
    //0での除算回避
    if (logo_dp == LOGO_MAX_DP) {
        logo_dp -= 1.0f;
    }

    //nv12->yc48
    float pixel_yc48 = (float)pixel * nv12_2_yc48_mul - nv12_2_yc48_sub;

    //ロゴ除去
    return (pixel_yc48 * (float)LOGO_MAX_DP - logo * logo_dp + ((float)LOGO_MAX_DP - logo_dp) * 0.5f) * __frcp_rn((float)LOGO_MAX_DP - logo_dp);
}

template<typename Type, int bit_depth, bool target_y>
__inline__ __device__
Type delogo(Type pixel, const int16x2_t logo_data, const float logo_depth_mul_fade) {
    //ロゴ除去
    float yc = delogo_yc48<Type, bit_depth, target_y>(pixel, logo_data, logo_depth_mul_fade);

    const float yc48_2_nv12_mul = (target_y) ?   219.0f / (1<<(20-bit_depth)) :    14.0f / (1<<(16-bit_depth));
    const float yc48_2_nv12_add = (target_y) ? 65919.0f / (1<<(20-bit_depth)) : 32900.0f / (1<<(16-bit_depth));
    return (Type)clamp((yc * yc48_2_nv12_mul + yc48_2_nv12_add + 0.5f), 0.0f, (float)(1<<bit_depth)-0.1f);
}

template<typename Type, int bit_depth, bool target_y>
__inline__ __device__
Type logo_add(Type pixel, const int16x2_t logo_data, const float logo_depth_mul_fade) {
    const float nv12_2_yc48_mul = (target_y) ? 1197.0f / (1<<(bit_depth-2)) : 4681.0f / (1<<bit_depth);
    const float nv12_2_yc48_sub = (target_y) ? 299.0f : 599332.0f / 256.0f;
    const float yc48_2_nv12_mul = (target_y) ?   219.0f / (1<<(20-bit_depth)) :    14.0f / (1<<(16-bit_depth));
    const float yc48_2_nv12_add = (target_y) ? 65919.0f / (1<<(20-bit_depth)) : 32900.0f / (1<<(16-bit_depth));

    //ロゴ情報取り出し
    float logo_dp = (float)logo_data.x;
    float logo    = (float)logo_data.y;

    logo_dp = (logo_dp * logo_depth_mul_fade) * (1.0f / (float)(128 * LOGO_FADE_MAX));

    //nv12->yc48
    float pixel_yc48 = (float)pixel * nv12_2_yc48_mul - nv12_2_yc48_sub;

    //ロゴ付加
    float yc = (pixel_yc48 * ((float)LOGO_MAX_DP - logo_dp) + logo * logo_dp) * (1.0f / (float)LOGO_MAX_DP);

    return (Type)clamp((yc * yc48_2_nv12_mul + yc48_2_nv12_add + 0.5f), 0.0f, (float)(1<<bit_depth)-0.1f);
}

template<typename Type, int bit_depth, bool target_y>
__global__ void kernel_delogo(
    const uint8_t *__restrict__ pFrame, const int frame_pitch, const int width, const int height,
    const uint8_t *__restrict__ pLogo, const int logo_pitch, const int logo_x, const int logo_y, const int logo_width, const int logo_height, const float logo_depth_mul_fade) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const int16x2_t logo_data = *(int16x2_t *)(&pLogo[y * logo_pitch + x * sizeof(int16x2_t)]);

        //画素データ取り出し
        pFrame += (y + logo_y) * frame_pitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(Type *)pFrame;
        Type ret = delogo<Type, bit_depth, target_y>(pixel_yuv, logo_data, logo_depth_mul_fade);

        *(Type *)pFrame = ret;
    }
}

template<typename Type, int bit_depth, bool target_y>
__global__ void kernel_logo_add(
    const uint8_t *__restrict__ pFrame, const int frame_pitch, const int width, const int height,
    const uint8_t *__restrict__ pLogo, const int logo_pitch, const int logo_x, const int logo_y, const int logo_width, const int logo_height, const float logo_depth_mul_fade) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const int16x2_t logo_data = *(int16x2_t *)(&pLogo[y * logo_pitch + x * sizeof(int16x2_t)]);

        //画素データ取り出し
        pFrame += (y + logo_y) * frame_pitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(Type *)pFrame;
        Type ret = logo_add<Type, bit_depth, target_y>(pixel_yuv, logo_data, logo_depth_mul_fade);

        *(Type *)pFrame = ret;
    }
}

template<typename Type, int bit_depth, bool target_y>
__global__ void kernel_delogo_add_multi(
    uint8_t *__restrict__ pFrame, const int framePitch, const int width, const int height,
    uint8_t *__restrict__ pLogo, const int logo_pitch, const int logo_width, const int logo_height,
    const int logo_block_width, const int logo_block_height,
    const int logo_block_padding_x, const int logo_block_padding_y,
    const int block_count_x,
    float *__restrict__ pBlockDepth, const float logo_fade) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    int by = z / block_count_x;
    int bx = z - block_count_x * by;
    int logo_x = bx * logo_block_width  + logo_block_padding_x;
    int logo_y = by * logo_block_height + logo_block_padding_y;
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const int16x2_t logo_data = *(int16x2_t *)(&pLogo[y * logo_pitch + x * sizeof(int16x2_t)]);

        //画素データ取り出し
        pFrame += (y + logo_y) * framePitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(Type *)pFrame;
        Type ret = logo_add<Type, bit_depth, target_y>(pixel_yuv, logo_data, pBlockDepth[blockIdx.z] * logo_fade);
        *(Type *)pFrame = ret;
    }
}

template<typename Type, int bit_depth, bool target_y>
void run_delogo(RGYFrameInfo *pFrame, const ProcessDataDelogo *pDelego, const int target_yuv, const int mode, const float fade) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pDelego->width, blockSize.x), divCeil(pDelego->height, blockSize.y));
    auto plane = getPlane(pFrame, (target_yuv == LOGO__Y) ? RGY_PLANE_Y : ((target_yuv == LOGO__V) ? RGY_PLANE_V : RGY_PLANE_U));
    if (mode == DELOGO_MODE_ADD) {
        kernel_logo_add<Type, bit_depth, target_y><<<gridSize, blockSize>>>(
            plane.ptr[0],
            plane.pitch[0],
            pFrame->width,
            plane.height,
            (uint8_t *)pDelego->pDevLogo->frame.ptr[0], pDelego->pDevLogo->frame.pitch[0],
            pDelego->i_start, pDelego->j_start, pDelego->width, pDelego->height, (float)pDelego->depth * fade);
    } else if (mode == DELOGO_MODE_ADD_MULTI) {
        const ProcessDataDelogo *pDelogoY = &pDelego[LOGO__Y - target_yuv];
        auto logo_multi_data = get_logo_multi_data(pDelogoY->width, pDelogoY->height, pFrame->width, pFrame->height);
        int logo_block_padding_x = LOGO_MULTI_PADDING;
        int logo_block_padding_y = LOGO_MULTI_PADDING;
        switch (target_yuv) {
        case LOGO__U:
        case LOGO__V:
            logo_block_padding_x >>= 1;
            logo_block_padding_y >>= 1;
            logo_multi_data.block_width >>= 1;
            logo_multi_data.block_height >>= 1;
            logo_multi_data.block_offset_x >>= 1;
            logo_multi_data.block_offset_y >>= 1;
            break;
        case LOGO_UV:
            logo_block_padding_y >>= 1;
            logo_multi_data.block_height >>= 1;
            logo_multi_data.block_offset_y >>= 1;
            break;
        case LOGO__Y:
        default:
            break;
        }
        gridSize.z = logo_multi_data.block_x * logo_multi_data.block_y;
        kernel_delogo_add_multi<Type, bit_depth, target_y> << <gridSize, blockSize >> > (
            plane.ptr[0],
            plane.pitch[0],
            pFrame->width,
            plane.height,
            (uint8_t *)pDelego->pDevLogo->frame.ptr[0], pDelego->pDevLogo->frame.pitch[0],
            pDelego->width, pDelego->height, logo_multi_data.block_width, logo_multi_data.block_height,
            logo_block_padding_x + logo_multi_data.block_offset_x, logo_block_padding_y + logo_multi_data.block_offset_y, logo_multi_data.block_x,
            (float *)pDelego->pBlockDepth->ptr, (float)pDelego->fade);
    } else {
        kernel_delogo<Type, bit_depth, target_y><<<gridSize, blockSize>>>(
            plane.ptr[0],
            plane.pitch[0],
            pFrame->width,
            plane.height,
            (uint8_t *)pDelego->pDevLogo->frame.ptr[0], pDelego->pDevLogo->frame.pitch[0],
            pDelego->i_start, pDelego->j_start, pDelego->width, pDelego->height, (float)pDelego->depth * fade);
    }
}

RGY_ERR NVEncFilterDelogo::delogoY(RGYFrameInfo *pFrame, const float fade) {
    //Y
    static const std::map<RGY_CSP, decltype(&run_delogo<uint8_t, 8, true>)> delogo_y_list ={
        { RGY_CSP_YV12,      run_delogo<uint8_t,   8, true> },
        { RGY_CSP_YV12_16,   run_delogo<uint16_t, 16, true> },
        { RGY_CSP_YV12_14,   run_delogo<uint16_t, 14, true> },
        { RGY_CSP_YV12_12,   run_delogo<uint16_t, 12, true> },
        { RGY_CSP_YV12_10,   run_delogo<uint16_t, 10, true> },
        { RGY_CSP_YV12_09,   run_delogo<uint16_t,  9, true> },
        { RGY_CSP_NV12,      run_delogo<uint8_t,   8, true> },
        { RGY_CSP_P010,      run_delogo<uint16_t, 16, true> },
        { RGY_CSP_YUV444,    run_delogo<uint8_t,   8, true> },
        { RGY_CSP_YUV444_16, run_delogo<uint16_t, 16, true> },
        { RGY_CSP_YUV444_14, run_delogo<uint16_t, 14, true> },
        { RGY_CSP_YUV444_12, run_delogo<uint16_t, 12, true> },
        { RGY_CSP_YUV444_10, run_delogo<uint16_t, 10, true> },
        { RGY_CSP_YUV444_09, run_delogo<uint16_t,  9, true> },
    };
    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    delogo_y_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO__Y], LOGO__Y, pDelogoParam->delogo.mode, fade);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at delogo_y_list(%s): %s.\n"),
            RGY_CSP_NAMES[pFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDelogo::delogoUV(RGYFrameInfo *pFrame, float fade) {
    const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    //const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    //UV
    static const std::map<RGY_CSP, decltype(&run_delogo<uint8_t, 8, false>)> delogo_uv_list ={
        { RGY_CSP_YV12,    run_delogo<uint8_t,   8, false> },
        { RGY_CSP_YV12_16, run_delogo<uint16_t, 16, false> },
        { RGY_CSP_YV12_14, run_delogo<uint16_t, 14, false> },
        { RGY_CSP_YV12_12, run_delogo<uint16_t, 12, false> },
        { RGY_CSP_YV12_10, run_delogo<uint16_t, 10, false> },
        { RGY_CSP_YV12_09, run_delogo<uint16_t,  9, false> },
        { RGY_CSP_NV12,    run_delogo<uint8_t,   8, false> },
        { RGY_CSP_P010,    run_delogo<uint16_t, 16, false> },
    };
    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (delogo_uv_list.count(pFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for delogo: %s.\n"), RGY_CSP_NAMES[pFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pFrame->csp) != supportedCspYV12.end()) {
        //YV12
        delogo_uv_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO__U], LOGO__U, pDelogoParam->delogo.mode, fade);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto sts = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
                RGY_CSP_NAMES[pFrame->csp],
                get_err_mes(sts));
            return sts;
        }
        delogo_uv_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO__V], LOGO__V, pDelogoParam->delogo.mode, fade);
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto sts = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
                RGY_CSP_NAMES[pFrame->csp],
                get_err_mes(sts));
            return sts;
        }
    } else {
        //NV12
        delogo_uv_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO_UV], LOGO_UV, pDelogoParam->delogo.mode, fade);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto sts = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
                RGY_CSP_NAMES[pFrame->csp],
                get_err_mes(sts));
            return sts;
        }
    }
    return RGY_ERR_NONE;

}

template<typename Type> __device__ constexpr Type type_min();
template<typename Type> __device__ constexpr Type type_max();
template<> __device__ constexpr char type_min<char>() { return INT8_MIN; }
template<> __device__ constexpr char type_max<char>() { return INT8_MAX; }
template<> __device__ constexpr uint8_t type_min<uint8_t>() { return 0; }
template<> __device__ constexpr uint8_t type_max<uint8_t>() { return UINT8_MAX; }
template<> __device__ constexpr short type_min<short>() { return INT16_MIN; }
template<> __device__ constexpr short type_max<short>() { return INT16_MAX; }
template<> __device__ constexpr uint16_t type_min<uint16_t>() { return 0; }
template<> __device__ constexpr uint16_t type_max<uint16_t>() { return UINT16_MAX; }
template<> __device__ constexpr int type_min<int>() { return INT_MIN; }
template<> __device__ constexpr int type_max<int>() { return INT_MAX; }

template<typename TypeSrc, typename TypeDst, int bit_depth, bool target_y>
__global__ void kernel_delogo_multi_fade(
    uint8_t *__restrict__ pFrameDst, const int dst_pitch, const int dst_size,
    const uint8_t *__restrict__ pFrameSrc, const int frame_pitch, const int frame_width, const int frame_size,
    const uint8_t *__restrict__ pLogo, const int logo_pitch, const int logo_width, const int logo_height,
    const float *__restrict__ pFadeDepth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x < logo_width && y < logo_height) {
        //ロゴ情報取り出し
        const int16x2_t logo_data = *(int16x2_t *)(&pLogo[y * logo_pitch + x * sizeof(int16x2_t)]);

        //画素データ取り出し
        pFrameSrc += z * frame_size + y * frame_pitch + x * sizeof(TypeSrc);
        pFrameDst += z * dst_size   + y * dst_pitch   + x * sizeof(TypeDst);
        TypeSrc pixel_yuv = *(TypeSrc *)pFrameSrc;
        float ret = delogo_yc48<TypeSrc, bit_depth, target_y>(pixel_yuv, logo_data, pFadeDepth[z]);

        *(TypeDst *)pFrameDst = (TypeDst)clamp((int)(ret + 0.5f), type_min<TypeDst>(), type_max<TypeDst>());
    }
}

#define DELOGO_SHARED_X DELOGO_BLOCK_X
#define DELOGO_SHARED_Y (16)
#define S_IDX(x,y,z) ((z)*(DELOGO_SHARED_Y*DELOGO_SHARED_X) + ((y)&(DELOGO_SHARED_Y-1))*DELOGO_SHARED_X + (x))

struct float4x2 {
    float4 a, b;
};

struct int16x2x4 {
    int16x2_t x, y, z, w;
};

template<typename Type4>
__inline__ __device__
Type4 type4_set1(int val) {
    Type4 t;
    t.x = val;
    t.y = val;
    t.z = val;
    t.w = val;
    return t;
}

__inline__ __device__
void type4_to_array(short *ptr, short4 val) {
    ptr[0] = val.x;
    ptr[1] = val.y;
    ptr[2] = val.z;
    ptr[3] = val.w;
}

__inline__ __device__
void type4_to_array(float *ptr, uchar4 val) {
    ptr[0] = val.x;
    ptr[1] = val.y;
    ptr[2] = val.z;
    ptr[3] = val.w;
}

__inline__ __device__
void type4_to_array(float *ptr, float4 val) {
    ptr[0] = val.x;
    ptr[1] = val.y;
    ptr[2] = val.z;
    ptr[3] = val.w;
}

__inline__ __device__
void type4_to_array(float *ptr, short4 val) {
    ptr[0] = (float)val.x;
    ptr[1] = (float)val.y;
    ptr[2] = (float)val.z;
    ptr[3] = (float)val.w;
}

__inline__ __device__
void type4_to_array(float *ptr, ushort4 val) {
    ptr[0] = val.x;
    ptr[1] = val.y;
    ptr[2] = val.z;
    ptr[3] = val.w;
}

__inline__ __device__
short4 array_to_short4(short *ptr) {
    short4 t;
    t.x = ptr[0];
    t.y = ptr[1];
    t.z = ptr[2];
    t.w = ptr[3];
    return t;
}

__inline__ __device__
short4 array_to_short4(float *ptr) {
    short4 t;
    t.x = (short)ptr[0];
    t.y = (short)ptr[1];
    t.z = (short)ptr[2];
    t.w = (short)ptr[3];
    return t;
}

__inline__ __device__
float4 array_to_float4(float *ptr) {
    float4 t;
    t.x = ptr[0];
    t.y = ptr[1];
    t.z = ptr[2];
    t.w = ptr[3];
    return t;
}

__inline__ __device__
float calc_prewitt_val(float sum_h, float sum_v) {
    return sqrtf(sum_h * sum_h + sum_v * sum_v);
}

__inline__ __device__
float4 calc_prewitt_val(float4 sum_h, float4 sum_v) {
    float4 ret;
    ret.x = calc_prewitt_val(sum_h.x, sum_v.x);
    ret.y = calc_prewitt_val(sum_h.y, sum_v.y);
    ret.z = calc_prewitt_val(sum_h.z, sum_v.z);
    ret.w = calc_prewitt_val(sum_h.w, sum_v.w);
    return ret;
}

__inline__ __device__
float4 add_float4(float4 a, float4 b) {
    float4 ret;
    ret.x = a.x + b.x;
    ret.y = a.y + b.y;
    ret.z = a.z + b.z;
    ret.w = a.w + b.w;
    return ret;
}

template<typename TypeSrc4, typename TypeDst4>
__inline__ __device__
TypeDst4 vec_cast4(TypeSrc4 a) {
    TypeDst4 ret;
    ret.x = (decltype(TypeDst4::x))a.x;
    ret.y = (decltype(TypeDst4::y))a.y;
    ret.z = (decltype(TypeDst4::z))a.z;
    ret.w = (decltype(TypeDst4::w))a.w;
    return ret;
}

template<>
__inline__ __device__
short4 vec_cast4(float4 a) {
    short4 ret;
    ret.x = (short)clamp(a.x, INT16_MIN, INT16_MAX);
    ret.y = (short)clamp(a.y, INT16_MIN, INT16_MAX);
    ret.z = (short)clamp(a.z, INT16_MIN, INT16_MAX);
    ret.w = (short)clamp(a.w, INT16_MIN, INT16_MAX);
    return ret;
}

template<>
__inline__ __device__
float4 vec_cast4(int16x2x4 a) {
    float4 ret;
    ret.x = (float)a.x.x;
    ret.y = (float)a.y.x;
    ret.z = (float)a.z.x;
    ret.w = (float)a.w.x;
    return ret;
}

template<typename Type4, typename TypeMask4>
__inline__ __device__
Type4 apply_mask(Type4 a, Type4 b, TypeMask4 mask, int mask_threshold) {
    Type4 c;
    c.x = (mask.x > mask_threshold) ? b.x : a.x;
    c.y = (mask.y > mask_threshold) ? b.y : a.y;
    c.z = (mask.z > mask_threshold) ? b.z : a.z;
    c.w = (mask.w > mask_threshold) ? b.w : a.w;
    return c;
}

template<typename TypeSrc4, int range>
__inline__ __device__
float4 symmetery_pixel_x(
    const uint8_t *__restrict__ ptr_src, //Type4
    const int src_pitch,
    const int imgx, const int imgy, const int logo_w, const int logo_h,
    const float *__restrict__ kernel_x
) {
    const TypeSrc4 *pTy4Src = (const TypeSrc4 *)(ptr_src
        + clamp(imgy, 0, logo_h-1) * src_pitch
        + min(imgx, (logo_w>>2)-1) * sizeof(TypeSrc4));

    const TypeSrc4 t1 = pTy4Src[0];
    const TypeSrc4 t0 = (imgx > 0) ? pTy4Src[-1] : type4_set1<TypeSrc4>(t1.x);
    const TypeSrc4 t2 = (imgx+1 < (logo_w>>2)) ? pTy4Src[1] : type4_set1<TypeSrc4>(t1.w);

    float src[12];
    type4_to_array(&src[0], t0);
    type4_to_array(&src[4], t1);
    type4_to_array(&src[8], t2);

    float kernel_sum = 0.0f;
    float dst[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (int ir = -range; ir <= range; ir++) {
        const float kernel_val = kernel_x[range+ir];
        #pragma unroll
        for (int ie = 0; ie < 4; ie++) {
            dst[ie] += src[4+ie+ir] * kernel_val;
        }
        kernel_sum += kernel_val;
    }
    float kernel_sum_inv = __frcp_rn(kernel_sum);
    dst[0] *= kernel_sum_inv;
    dst[1] *= kernel_sum_inv;
    dst[2] *= kernel_sum_inv;
    dst[3] *= kernel_sum_inv;
    return array_to_float4(dst);
}

template<int range>
__inline__ __device__
float4 symmetery_pixel_y(
    const float4 *__restrict__ ptr_src, int ly, const float *__restrict__ kernel_y
) {
    float kernel_sum = 0.0f;
    float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (int ir = -range; ir <= range; ir++) {
        const float kernel_val = kernel_y[range+ir];
        const float4 src = ptr_src[S_IDX(0,ly+ir,0)];
        sum[0] += src.x * kernel_val;
        sum[1] += src.y * kernel_val;
        sum[2] += src.z * kernel_val;
        sum[3] += src.w * kernel_val;
        kernel_sum += kernel_val;
    }
    float kernel_sum_inv = __frcp_rn(kernel_sum);
    sum[0] *= kernel_sum_inv;
    sum[1] *= kernel_sum_inv;
    sum[2] *= kernel_sum_inv;
    sum[3] *= kernel_sum_inv;
    return array_to_float4(sum);
}

template<typename Type4, typename TypeMask4, int range>
__global__ void kernel_proc_symmetry(
    uint8_t *__restrict__ ptr_dst, const int dst_pitch, const int dst_size, //Type4
    const uint8_t *__restrict__ ptr_src, const int src_pitch, const int src_size, //Type4
    const int logo_w, const int logo_h,
    const uint8_t *__restrict__ ptr_mask, const int mask_pitch, //TypeMask4
    const int mask_threshold,
    const float *__restrict__ kernel_x,
    const float *__restrict__ kernel_y
) {
    static_assert(range <= sizeof(Type4) / sizeof(decltype(Type4::x)), "range <= sizeof(Type4) / sizeof(decltype(Type4::x))");
    static_assert(DELOGO_BLOCK_Y >= range * 2, "BLOCK_Y >= range * 2");
    static_assert(DELOGO_BLOCK_Y + range * 2 <= DELOGO_SHARED_Y, "BLOCK_Y >= range * 2");
    __shared__ float4 shared[DELOGO_SHARED_X * DELOGO_SHARED_Y];
    const int lx = threadIdx.x; //スレッド数=DELOGO_BLOCK_X
    int ly = threadIdx.y; //スレッド数=BLOCK_Y
    const int gidy = blockIdx.y; //グループID
    const int imgx = blockIdx.x * DELOGO_BLOCK_X /*blockDim.x*/ + threadIdx.x;
    int imgy = (gidy * DELOGO_BLOCK_LOOP_Y * DELOGO_BLOCK_Y + ly);

    ptr_src += blockIdx.z * src_size;
    ptr_dst += blockIdx.z * dst_size + imgy * dst_pitch + imgx * sizeof(Type4);
    ptr_mask += imgy * mask_pitch + imgx * sizeof(TypeMask4);

    //先読み
    if (ly < range * 2) {
        shared[S_IDX(lx, ly, 0)] = symmetery_pixel_x<Type4, range>(ptr_src, src_pitch, imgx, imgy-range, logo_w, logo_h, kernel_x);
    }

    for (int iloop = 0; iloop < DELOGO_BLOCK_LOOP_Y; iloop++,
        ly += DELOGO_BLOCK_Y, imgy += DELOGO_BLOCK_Y, ptr_dst += DELOGO_BLOCK_Y * dst_pitch, ptr_mask += DELOGO_BLOCK_Y * mask_pitch
        ) {
        shared[S_IDX(lx, ly+range*2, 0)] = symmetery_pixel_x<Type4, range>(ptr_src, src_pitch, imgx, imgy+range, logo_w, logo_h, kernel_x);
        __syncthreads();

        float4 ret = symmetery_pixel_y<range>(&shared[S_IDX(lx,0,0)], ly+range, kernel_y);
        __syncthreads();
        if (imgx < (logo_w>>2) && imgy < logo_h) {
            const Type4 src = *(Type4 *)&ptr_src[imgy * src_pitch + imgx * sizeof(Type4)];
            *(Type4 *)ptr_dst = apply_mask<Type4, TypeMask4>(
                vec_cast4<Type4, Type4>(src), vec_cast4<float4, Type4>(ret),
                *(const TypeMask4 *)ptr_mask, mask_threshold);
        }
    }
}

template<typename TypeSrc4, int range>
__inline__ __device__
TypeSrc4 erosion_pixel_x(
    const uint8_t *__restrict__ ptr_src, const int src_pitch,
    const int imgx, const int imgy, const int logo_w, const int logo_h,
    const int threshold
) {
    const TypeSrc4 *pTy4Src = (const TypeSrc4 *)(ptr_src
        + clamp(imgy, 0, logo_h-1) * src_pitch
        + min(imgx, (logo_w>>2)-1) * sizeof(TypeSrc4));

    const TypeSrc4 t1 = pTy4Src[0];
    const TypeSrc4 t0 = (imgx > 0) ? pTy4Src[-1] : type4_set1<TypeSrc4>(t1.x);
    const TypeSrc4 t2 = (imgx+1 < (logo_w>>2)) ? pTy4Src[1] : type4_set1<TypeSrc4>(t1.w);

    decltype(TypeSrc4::x) src[12];
    type4_to_array(&src[0], t0);
    type4_to_array(&src[4], t1);
    type4_to_array(&src[8], t2);

    decltype(TypeSrc4::x) dst[4] = { 0, 0, 0, 0 };
    #pragma unroll
    for (int ir = -range; ir <= range; ir++) {
        #pragma unroll
        for (int ie = 0; ie < 4; ie++) {
            if (src[4+ie+ir] > threshold) {
                dst[ie] = threshold+1;
            }
        }
    }
    return array_to_short4(dst);
}

template<typename TypeSrc4, int range>
__inline__ __device__
TypeSrc4 erosion_pixel_y(
    const TypeSrc4 *__restrict__ ptr_src, int ly, const int threshold
) {
    float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (int ir = -range; ir <= range; ir++) {
        const TypeSrc4 src = ptr_src[S_IDX(0, ly+ir, 0)];
        if (src.x > threshold) sum[0] = threshold + 1;
        if (src.y > threshold) sum[1] = threshold + 1;
        if (src.z > threshold) sum[2] = threshold + 1;
        if (src.w > threshold) sum[3] = threshold + 1;
    }
    return array_to_short4(sum);
}

//処理単位: 4要素/thread
//ブロック構成: DELOGO_BLOCK_X * (DELOGO_BLOCK_LOOP_Y * DELOGO_BLOCK_Y)
template<typename TypeMask4, int range>
__global__ void kernel_erosion(
    uint8_t *__restrict__ ptr_mask_dst, //TypeMask4
    const int mask_pitch, const int logo_w, const int logo_h,
    const uint8_t *__restrict__ ptr_mask_src, //TypeMask4
    const int mask_threshold
) {
    static_assert(range <= sizeof(TypeMask4) / sizeof(decltype(TypeMask4::x)), "range <= sizeof(Type4) / sizeof(Type)");
    static_assert(DELOGO_BLOCK_Y >= range * 2, "BLOCK_Y >= range * 2");
    static_assert(DELOGO_BLOCK_Y + range * 2 <= DELOGO_SHARED_Y, "BLOCK_Y >= range * 2");
    __shared__ TypeMask4 shared[DELOGO_SHARED_X * DELOGO_SHARED_Y];
    const int lx = threadIdx.x; //スレッド数=DELOGO_BLOCK_X
    int ly = threadIdx.y; //スレッド数=BLOCK_Y
    const int gidy = blockIdx.y; //グループID
    const int imgx = blockIdx.x * DELOGO_BLOCK_X /*blockDim.x*/ + threadIdx.x;
    int imgy = (gidy * DELOGO_BLOCK_LOOP_Y * DELOGO_BLOCK_Y + ly);

    ptr_mask_dst += imgy * mask_pitch + imgx * sizeof(TypeMask4);

    //先読み
    if (ly < range * 2) {
        shared[S_IDX(lx, ly, 0)] = erosion_pixel_x<TypeMask4, range>(ptr_mask_src, mask_pitch, imgx, imgy-range, logo_w, logo_h, mask_threshold);
    }

    for (int iloop = 0; iloop < DELOGO_BLOCK_LOOP_Y; iloop++,
        ly += DELOGO_BLOCK_Y, imgy += DELOGO_BLOCK_Y, ptr_mask_dst += DELOGO_BLOCK_Y * mask_pitch
        ) {
        shared[S_IDX(lx, ly+range*2, 0)] = erosion_pixel_x<TypeMask4, range>(ptr_mask_src, mask_pitch, imgx, imgy+range, logo_w, logo_h, mask_threshold);
        __syncthreads();

        TypeMask4 ret = erosion_pixel_y<TypeMask4, range>(&shared[S_IDX(lx, 0, 0)], ly+range, mask_threshold);
        __syncthreads();
        if (imgx < (logo_w>>2) && imgy < logo_h) {
            *(TypeMask4 *)ptr_mask_dst = ret;
        }
    }
}

template<typename TypeSrc4, int range>
__inline__ __device__
float4x2 prewitt_pixel_x(
    const uint8_t *__restrict__ ptr_src, const int src_frame_pitch,
    const int imgx, const int imgy, const int logo_w, const int logo_h
) {
    const TypeSrc4 *pTy4Src = (const TypeSrc4 *)(ptr_src
        + clamp(imgy, 0, logo_h-1) * src_frame_pitch
        + min(imgx, (logo_w>>2)-1) * sizeof(TypeSrc4));

    float4 t1 = vec_cast4<TypeSrc4, float4>(pTy4Src[0]);
    float4 t0 = (imgx > 0) ? vec_cast4<TypeSrc4, float4>(pTy4Src[-1]) : type4_set1<float4>(t1.x);
    float4 t2 = (imgx+1 < (logo_w>>2)) ? vec_cast4<TypeSrc4, float4>(pTy4Src[1]) : type4_set1<float4>(t1.w);

    float src[12];
    type4_to_array(&src[0], t0);
    type4_to_array(&src[4], t1);
    type4_to_array(&src[8], t2);

    float dst[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (int ir = -range; ir < 0; ir++) {
        #pragma unroll
        for (int ie = 0; ie < 4; ie++) {
            dst[ie+0] -= src[4+ie+ir];
            dst[ie+4] += src[4+ie+ir];
        }
    }
    #pragma unroll
    for (int ie = 0; ie < 4; ie++) {
        dst[ie+4] += src[4+ie];
    }
    #pragma unroll
    for (int ir = 1; ir <= range; ir++) {
        #pragma unroll
        for (int ie = 0; ie < 4; ie++) {
            dst[ie+0] += src[4+ie+ir];
            dst[ie+4] += src[4+ie+ir];
        }
    }
    float4x2 ret;
    ret.a = array_to_float4(&dst[0]);
    ret.b = array_to_float4(&dst[4]);
    return ret;
}

template<int range>
__inline__ __device__
float4 sum_pixel_y(
    const float4 *__restrict__ ptr_src, int ly
) {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int ir = -range; ir <= range; ir++) {
        float4 src = ptr_src[S_IDX(0, ly+ir, 0)];
        sum.x += src.x;
        sum.y += src.y;
        sum.z += src.z;
        sum.w += src.w;
    }
    return sum;
}

template<int range>
__inline__ __device__
float4 prewitt_pixel_y(const float4 *__restrict__ ptr_src, int ly) {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int ir = -range; ir < 0; ir++) {
        float4 src = ptr_src[S_IDX(0, ly+ir, 0)];
        sum.x -= src.x;
        sum.y -= src.y;
        sum.z -= src.z;
        sum.w -= src.w;
    }
    #pragma unroll
    for (int ir = 1; ir <= range; ir++) {
        float4 src = ptr_src[S_IDX(0, ly+ir, 0)];
        sum.x += src.x;
        sum.y += src.y;
        sum.z += src.z;
        sum.w += src.w;
    }
    return sum;
}

//処理単位: 4要素/thread
//ブロック構成: DELOGO_BLOCK_X * (DELOGO_BLOCK_LOOP_Y * DELOGO_BLOCK_Y)
template<typename TypeSrc4, typename TypeMask4, int range, bool store_pixel_result, bool eval, bool use_mask, bool count_valid_mask>
__global__ void kernel_proc_prewitt(
    uint8_t *__restrict__ ptr_dst, //TypeMask4
    float *__restrict__ ptr_temp_eval,
    int *__restrict__ ptr_temp_valid_mask_count,
    const uint8_t *__restrict__ ptr_src, //TypeSrc4
    const int src_frame_pitch,
    const uint8_t *__restrict__ ptr_mask,  //TypeMask4
    const int mask_pitch, const int logo_w, const int logo_h, const int mask_size,
    const int mask_threshold
) {
    static_assert(range <= sizeof(TypeMask4) / sizeof(decltype(TypeMask4::x)), "range <= sizeof(Type4) / sizeof(Type)");
    static_assert(DELOGO_BLOCK_Y >= range * 2, "BLOCK_Y >= range * 2");
    static_assert(DELOGO_BLOCK_Y + range * 2 <= DELOGO_SHARED_Y, "BLOCK_Y >= range * 2");
    __shared__ float4 shared[DELOGO_SHARED_X * DELOGO_SHARED_Y * 2];
    const int lx = threadIdx.x; //スレッド数=DELOGO_BLOCK_X
    int ly = threadIdx.y; //スレッド数=BLOCK_Y
    const int gidy = blockIdx.y; //グループID
    const int imgx = blockIdx.x * DELOGO_BLOCK_X /*blockDim.x*/ + threadIdx.x;
    int imgy = (gidy * DELOGO_BLOCK_LOOP_Y * DELOGO_BLOCK_Y + ly);

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    int valid_mask_count = 0;

    ptr_src += blockIdx.z * src_frame_pitch * logo_h;
    ptr_dst += blockIdx.z * mask_size + imgy * mask_pitch + imgx * sizeof(TypeMask4);
    ptr_mask += imgy * mask_pitch + imgx * sizeof(TypeMask4);

    //先読み
    if (ly < range * 2) {
        float4x2 ret = prewitt_pixel_x<TypeSrc4, range>(ptr_src, src_frame_pitch, imgx, imgy-range, logo_w, logo_h);
        shared[S_IDX(lx, ly, 0)] = ret.a;
        shared[S_IDX(lx, ly, 1)] = ret.b;
    }

    for (int iloop = 0; iloop <= DELOGO_BLOCK_LOOP_Y; iloop++,
        ly += DELOGO_BLOCK_Y, imgy += DELOGO_BLOCK_Y, ptr_dst += DELOGO_BLOCK_Y * mask_pitch, ptr_mask += DELOGO_BLOCK_Y * mask_pitch
        ) {
        {
            float4x2 ret = prewitt_pixel_x<TypeSrc4, range>(ptr_src, src_frame_pitch, imgx, imgy+range, logo_w, logo_h);
            shared[S_IDX(lx, ly+range*2, 0)] = ret.a;
            shared[S_IDX(lx, ly+range*2, 1)] = ret.b;
        }
        __syncthreads();

        const float4 ret_h = sum_pixel_y<range>(    &shared[S_IDX(lx, 0, 0)], ly+range);
        const float4 ret_v = prewitt_pixel_y<range>(&shared[S_IDX(lx, 0, 1)], ly+range);
        __syncthreads();
        const float4 ret = calc_prewitt_val(ret_h, ret_v);
        if (imgx < (logo_w>>2) && imgy < logo_h) {
            const TypeMask4 ret_masked = (use_mask)
                ? apply_mask<TypeMask4, TypeMask4>(
                    type4_set1<TypeMask4>(0), vec_cast4<float4, TypeMask4>(ret),
                    *(TypeMask4 *)ptr_mask, mask_threshold)
                : vec_cast4<float4, TypeMask4>(ret);
            if (store_pixel_result) {
                *(TypeMask4 *)ptr_dst = ret_masked;
            }
            sum = add_float4(sum, vec_cast4<TypeMask4, float4>(ret_masked));
            if (count_valid_mask) {
                if (ret_masked.x > mask_threshold) valid_mask_count++;
                if (ret_masked.y > mask_threshold) valid_mask_count++;
                if (ret_masked.z > mask_threshold) valid_mask_count++;
                if (ret_masked.w > mask_threshold) valid_mask_count++;
            }
        }
    }

    if (eval) {
        float tmp = 0.0f;
        tmp  = sum.x + sum.y;
        tmp += sum.z + sum.w;

        tmp = block_sum<decltype(tmp), DELOGO_BLOCK_X, DELOGO_BLOCK_Y>(tmp, (float *)&shared);

        const int lid = threadIdx.y * DELOGO_BLOCK_X + threadIdx.x;
        if (lid == 0) {
            const int gid = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
            ptr_temp_eval[gid] = tmp;
        }
    }
    if (count_valid_mask) {
        valid_mask_count = block_sum<decltype(valid_mask_count), DELOGO_BLOCK_X, DELOGO_BLOCK_Y>(valid_mask_count, (int *)&shared);

        const int lid = threadIdx.y * DELOGO_BLOCK_X + threadIdx.x;
        if (lid == 0) {
            const int gid = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
            ptr_temp_valid_mask_count[gid] = valid_mask_count;
        }
    }
}

template<typename Type>
__inline__ __device__
int adjust_mask_analyze(int *each_fade_count, int& valid_mask_count, int& whole_min_result,
    const int min_fade, const Type min_result, const Type max_result) {
    if (max_result > 0 && (min_result * __frcp_rn((float)max_result)) < 0.7f) {
        atomicAdd(&each_fade_count[min_fade], 1);
        whole_min_result = min(whole_min_result, (int)min_result);
        valid_mask_count++;
        return min_fade;
    }
    return -2;
}

//処理単位: 4要素/thread
//ブロック構成: DELOGO_BLOCK_X * DELOGO_BLOCK_Y
template<typename TypeIdx4, typename TypeMask4>
__global__ void kernel_create_adjust_mask1(
    uint8_t *__restrict__ ptr_dst_min_fade_idx, //TypeIdx4: dst_pitch
    int dst_pitch,
    int *__restrict__ ptr_temp_each_fade_count,
    int2 *__restrict__ ptr_temp_whole_min_result_valid_mask_count,
    const int mask_pitch, const int logo_w, const int logo_h, const int mask_size,
    const uint8_t *__restrict__ ptr_mask, //TypeMask4: mask_pitch
    const uint8_t *__restrict__ ptr_eval_result, //TypeMask4: mask_pitch
    const int mask_threshold
) {
    const int lx = threadIdx.x; //スレッド数=DELOGO_BLOCK_X
    const int ly = threadIdx.y; //スレッド数=BLOCK_Y
    const int lid = ly * DELOGO_BLOCK_X + lx;
    const int gidy = blockIdx.y; //グループID
    const int imgx = blockIdx.x * DELOGO_BLOCK_X /*blockDim.x*/ + threadIdx.x;
    const int imgy = (gidy * DELOGO_BLOCK_Y + ly);

    __shared__ int each_fade_count[DELOGO_PRE_DIV_COUNT+1];
    //まずはブロックごとにsharedメモリを使って加算
    //sharedメモリを初期化
    if (lid <= DELOGO_PRE_DIV_COUNT) {
        each_fade_count[lid] = 0;
    }
    int valid_mask_count = 0;
    int whole_min_result = 0;

    ptr_dst_min_fade_idx += imgy * dst_pitch  + imgx * sizeof(TypeIdx4);
    ptr_mask             += imgy * mask_pitch + imgx * sizeof(TypeMask4);
    ptr_eval_result      += imgy * mask_pitch + imgx * sizeof(TypeMask4);

    TypeIdx4 ret = type4_set1<TypeIdx4>(-1);
    if (imgx < (logo_w>>2)) {
        if (2 <= imgy && imgy < logo_h-2) {
            int4 min_fade        = type4_set1<int4>(-1);
            TypeMask4 min_result = type4_set1<TypeMask4>(type_max<decltype(TypeMask4::x)>());
            TypeMask4 max_result = type4_set1<TypeMask4>(0);

            #pragma unroll
            for (int i = 0; i <= DELOGO_PRE_DIV_COUNT; i++, ptr_eval_result += mask_size) {
                const TypeMask4 evals = *(const TypeMask4 *)ptr_eval_result;
                if (evals.x < min_result.x) { min_result.x = evals.x; min_fade.x = i; }
                if (evals.y < min_result.y) { min_result.y = evals.y; min_fade.y = i; }
                if (evals.z < min_result.z) { min_result.z = evals.z; min_fade.z = i; }
                if (evals.w < min_result.w) { min_result.w = evals.w; min_fade.w = i; }
                max_result.x = max(max_result.x, evals.x);
                max_result.y = max(max_result.y, evals.y);
                max_result.z = max(max_result.z, evals.z);
                max_result.w = max(max_result.w, evals.w);
            }

            ret.x = (decltype(TypeIdx4::x))adjust_mask_analyze(each_fade_count, valid_mask_count, whole_min_result, min_fade.x, min_result.x, max_result.x);
            ret.y = (decltype(TypeIdx4::y))adjust_mask_analyze(each_fade_count, valid_mask_count, whole_min_result, min_fade.y, min_result.y, max_result.y);
            ret.z = (decltype(TypeIdx4::z))adjust_mask_analyze(each_fade_count, valid_mask_count, whole_min_result, min_fade.z, min_result.z, max_result.z);
            ret.w = (decltype(TypeIdx4::w))adjust_mask_analyze(each_fade_count, valid_mask_count, whole_min_result, min_fade.w, min_result.w, max_result.w);

            ret = apply_mask<TypeIdx4, TypeMask4>(type4_set1<TypeIdx4>(-1), ret, *(const TypeMask4*)ptr_mask, mask_threshold);
        }
        if (imgy < logo_h) {
            *(TypeIdx4 *)(ptr_dst_min_fade_idx) = ret;
        }
    }

    __shared__ int shared_tmp[DELOGO_BLOCK_X * DELOGO_BLOCK_Y / WARP_SIZE];
    valid_mask_count = block_sum<decltype(whole_min_result), DELOGO_BLOCK_X, DELOGO_BLOCK_Y>(valid_mask_count, shared_tmp);
    whole_min_result = block_min<decltype(whole_min_result), DELOGO_BLOCK_X, DELOGO_BLOCK_Y>(whole_min_result, shared_tmp);

    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        ptr_temp_whole_min_result_valid_mask_count[gid] = make_int2(whole_min_result, valid_mask_count);
    }
    //グローバルメモリにAtomic演算を使ってブロック単位の結果を足しこむ
    if (lid <= DELOGO_PRE_DIV_COUNT) {
        atomicAdd(&ptr_temp_each_fade_count[lid], each_fade_count[lid]);
    }
}

template<typename TypeMask4, int pos>
__inline__ __device__
int gen_adjust_mask(int& valid_mask_count,
    const uint8_t *__restrict__ ptr_eval_result, //TypeMask4
    const int mask_size,
    const int min_fade_index, const float prewitt_threshold,
    const int mask_threshold) {
    static_assert(0 <= pos && pos <= 3, "0 <= pos && pos <= 3");
    int ret = 0;
    if (min_fade_index >= 0) {
        const TypeMask4 *ptr = (const TypeMask4 *)(ptr_eval_result + mask_size * min_fade_index);
        decltype(TypeMask4::x) eval_result;
        switch (pos) {
        case 0: eval_result = ptr[0].x; break;
        case 1: eval_result = ptr[0].y; break;
        case 2: eval_result = ptr[0].z; break;
        case 3:
        default: eval_result = ptr[0].w; break;
        }
        if (eval_result < prewitt_threshold) {
            valid_mask_count++;
            ret = mask_threshold+1;
        }
    }
    return ret;
}

template<typename Type>
constexpr Type constpow(Type base, int exp) noexcept {
    return (exp == 0 ? (Type)1 : base * constpow<Type>(base, exp-1));
}

//prewitt_threshold_baseに対する倍率を配列で保持
__constant__ float g_threshold_adj_mul[DELOGO_ADJMASK_DIV_COUNT];

//処理単位: 4要素/thread
//ブロック構成: DELOGO_BLOCK_X * DELOGO_BLOCK_Y * DELOGO_ADJMASK_DIV_COUNT
template<typename TypeIdx4, typename TypeMask4>
__global__ void kernel_create_adjust_mask2(
    uint8_t *__restrict__ ptr_dst_adjusted_mask, //TypeMask4
    int *__restrict__ ptr_temp_valid_mask_count,
    int *__restrict__ ptr_target_count,
    const uint8_t *__restrict__ ptr_eval_result, //TypeMask4
    const int mask_pitch, const int mask_size, const int logo_w, const int logo_h,
    const uint8_t *__restrict__ ptr_min_fade_idx, //TypeIdx4
    const int min_fade_idx_pitch,
    const int mask_threshold,
    const float *__restrict__ ptr_temp_eval, const int eval_blocks,
    const int2 *__restrict__ ptr_temp_whole_min_result_valid_mask_count, const int block_count,
    const int *__restrict__ ptr_temp_each_fade_count,
    const int orig_valid_mask_count
) {
    const int imgx = blockIdx.x * DELOGO_BLOCK_X /*blockDim.x*/ + threadIdx.x;
    const int imgy = (blockIdx.y * DELOGO_BLOCK_Y + threadIdx.y);
    const int lid = threadIdx.y * DELOGO_BLOCK_X + threadIdx.x;
    const int warp_lane = lid & (WARP_SIZE - 1);

    //eval_resultの集計
    __shared__ int eval_results[DELOGO_PRE_DIV_COUNT+1];
    if (lid <= DELOGO_PRE_DIV_COUNT) {
        eval_results[lid] = 0;
    }
    for (int j = threadIdx.y; j <= DELOGO_PRE_DIV_COUNT; j += DELOGO_BLOCK_Y) {
        int tmp = 0;
        for (int i = threadIdx.x; i < eval_blocks; i += DELOGO_BLOCK_X) {
            tmp += ptr_temp_eval[j * eval_blocks + i];
        }
        tmp = warp_sum<int, WARP_SIZE>(tmp);
        if (warp_lane == 0) {
            atomicAdd(&eval_results[j], tmp);
        }
    }
#if 0
    if (imgx == 0 && imgy == 0 && blockIdx.z == 0) {
        for (int i = 0; i <= DELOGO_PRE_DIV_COUNT; i++) {
            printf("eval_results[%d]=%d\n", i, eval_results[i]);
        }
    }
#endif

    //shared_whole_min_result / shared_valid_mask_evalの集計
    __shared__ int shared_whole_min_result;
    __shared__ int shared_valid_mask_eval;
    if (lid == 0) {
        shared_whole_min_result = type_min<int>();
        shared_valid_mask_eval = 0;
    }
    {
        int whole_min_result = type_min<int>();
        int valid_mask_eval = 0;
        if (lid < block_count) {
            int2 temp = ptr_temp_whole_min_result_valid_mask_count[lid];
            whole_min_result = temp.x;
            valid_mask_eval = temp.y;
        }
        whole_min_result = warp_min<int, WARP_SIZE>(whole_min_result);
        valid_mask_eval = warp_sum<int, WARP_SIZE>(valid_mask_eval);
        if (warp_lane == 0) {
            atomicAdd(&shared_whole_min_result, whole_min_result);
            atomicAdd(&shared_valid_mask_eval, valid_mask_eval);
        }
    }
#if 0
    if (imgx == 0 && imgy == 0 && blockIdx.z == 0) {
        printf("[gpu] shared_whole_min_result=%d, shared_valid_mask_eval=%d\n", shared_whole_min_result, shared_valid_mask_eval);
    }
#endif
    __shared__ int each_fade_count[DELOGO_PRE_DIV_COUNT+1];
    if (lid <= DELOGO_PRE_DIV_COUNT) {
        each_fade_count[lid] = ptr_temp_each_fade_count[lid];
    }
    __syncthreads();

    __shared__ float shared_prewitt_threshold_base;
    if (lid == 0) {
        int min_fade_index = -1;
        int max_fade_count = -1;
        for (int i = 0; i <= DELOGO_PRE_DIV_COUNT; i++) {
            if (each_fade_count[i] > max_fade_count) {
                max_fade_count = each_fade_count[i];
                min_fade_index = i;
            }
        }

        const float aveResult = (orig_valid_mask_count > 0) ? eval_results[min_fade_index] / orig_valid_mask_count : 0;

        shared_prewitt_threshold_base = shared_whole_min_result * 2.0f + 100.0f;
        const float rate = 0.5f + (float)min((aveResult - 400.0f) * (1.0f / 200.0f), 4.0f) * 0.08f;
        const int target_count = (int)((float)shared_valid_mask_eval * (1.0f - rate));
#if 0
        if (imgx == 0 && imgy == 0 && blockIdx.z == 0) {
            printf("[gpu] shared_prewitt_threshold_base = %f, target_count = %d\n", shared_prewitt_threshold_base, target_count);
        }
#endif
        ptr_target_count[0] = target_count;
    }
    __syncthreads();

    //prewitt_thresholdを並列に計算する
    //prewitt_threshold_baseに対する倍率を配列で保持
    const float prewitt_threshold = shared_prewitt_threshold_base * g_threshold_adj_mul[blockIdx.z];

    ptr_dst_adjusted_mask  += imgy * mask_pitch         + imgx * sizeof(TypeMask4) + mask_size * blockIdx.z;
    ptr_min_fade_idx       += imgy * min_fade_idx_pitch + imgx * sizeof(TypeIdx4);
    ptr_eval_result        += imgy * mask_pitch         + imgx * sizeof(TypeMask4);

    int valid_mask_count = 0;

    TypeMask4 ret = type4_set1<TypeMask4>(0);
    if (imgx < (logo_w>>2)) {
        if (2 <= imgy && imgy < logo_h-2) {
            TypeIdx4 min_fade_idx = *(TypeIdx4 *)ptr_min_fade_idx;

            ret.x = gen_adjust_mask<TypeMask4, 0>(valid_mask_count, ptr_eval_result, mask_size, min_fade_idx.x, prewitt_threshold, mask_threshold);
            ret.y = gen_adjust_mask<TypeMask4, 1>(valid_mask_count, ptr_eval_result, mask_size, min_fade_idx.y, prewitt_threshold, mask_threshold);
            ret.z = gen_adjust_mask<TypeMask4, 2>(valid_mask_count, ptr_eval_result, mask_size, min_fade_idx.z, prewitt_threshold, mask_threshold);
            ret.w = gen_adjust_mask<TypeMask4, 3>(valid_mask_count, ptr_eval_result, mask_size, min_fade_idx.w, prewitt_threshold, mask_threshold);
        }
        if (imgy < logo_h) {
            *(TypeMask4 *)ptr_dst_adjusted_mask = ret;
        }
    }

    __shared__ int shared_tmp[DELOGO_BLOCK_X * DELOGO_BLOCK_Y / WARP_SIZE];
    valid_mask_count = block_sum<decltype(valid_mask_count), DELOGO_BLOCK_X, DELOGO_BLOCK_Y>(valid_mask_count, shared_tmp);

    if (lid == 0) {
        const int gid = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
        ptr_temp_valid_mask_count[gid] = valid_mask_count;
    }
}

//処理単位: 4要素/thread
//ブロック構成: DELOGO_BLOCK_X * DELOGO_BLOCK_Y
template<typename TypeMask4>
__global__ void kernel_create_adjust_mask3(
    uint8_t *__restrict__ ptr_dst_adjusted_mask, //TypeMask4
    const int *__restrict__ ptr_temp_valid_mask_count, const int block_count,
    const uint8_t *__restrict__ ptr_src_adjusted_mask, //TypeMask4
    const int mask_pitch, const int mask_size, const int logo_w, const int logo_h,
    const int *__restrict__ ptr_target_count
) {
    const int imgx = blockIdx.x * DELOGO_BLOCK_X /*blockDim.x*/ + threadIdx.x;
    const int imgy = (blockIdx.y * DELOGO_BLOCK_Y + threadIdx.y);
    const int lid = threadIdx.y * DELOGO_BLOCK_X + threadIdx.x;
    const int warp_lane = lid & (WARP_SIZE - 1);

    __shared__ int mask_count[DELOGO_ADJMASK_DIV_COUNT];
    if (lid < DELOGO_ADJMASK_DIV_COUNT) {
        mask_count[lid] = 0;
    }
    for (int j = threadIdx.y; j < DELOGO_ADJMASK_DIV_COUNT; j += DELOGO_BLOCK_Y) {
        int tmp = 0;
        for (int i = threadIdx.x; i < block_count; i += DELOGO_BLOCK_X) {
            tmp += ptr_temp_valid_mask_count[j * block_count + i];
        }
        tmp = warp_sum<int, WARP_SIZE>(tmp);
        if (warp_lane == 0) {
            atomicAdd(&mask_count[j], tmp);
        }
    }
    __syncthreads();

    __shared__ int shared_tmp;
    if (lid == 0) {
        shared_tmp = 0;
        const int target_count = ptr_target_count[0];
        for (int i = 0; i < DELOGO_ADJMASK_DIV_COUNT; i++) {
            if (mask_count[i] >= target_count) {
                shared_tmp = i;
                break;
            }
        }
#if 0
        for (int i = 0; i < DELOGO_ADJMASK_DIV_COUNT; i++) {
            printf("mask_count[%d]=%d\n", i, mask_count[i]);
        }
#endif
    }
    __syncthreads();
    if (imgx < (logo_w>>2) && imgy < logo_h) {
        ptr_src_adjusted_mask += shared_tmp * mask_size;
        ptr_src_adjusted_mask += imgy * mask_pitch + imgx * sizeof(TypeMask4);
        ptr_dst_adjusted_mask += imgy * mask_pitch + imgx * sizeof(TypeMask4);
        *(TypeMask4 *)ptr_dst_adjusted_mask = *(TypeMask4 *)ptr_src_adjusted_mask;
    }
}

RGY_ERR NVEncFilterDelogo::createLogoMask(int maskThreshold) {
    const auto pLogoData = &m_sProcessData[LOGO__Y];
    if (pLogoData->width % 4 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("frame width must be mod4\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const dim3 blockSize(DELOGO_BLOCK_X, DELOGO_BLOCK_Y);
    const dim3 gridSize(divCeil(pLogoData->width, blockSize.x * 4), divCeil(pLogoData->height, blockSize.y * DELOGO_BLOCK_LOOP_Y), 1);
    const int blockCount = gridSize.x * gridSize.y * gridSize.z;

    if (m_createLogoMaskValidMaskCount.nSize < blockCount * sizeof(int)) {
        auto sts = m_createLogoMaskValidMaskCount.alloc(blockCount * sizeof(int));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error allocating memory for counting valid mask: %s.\n"),
                get_err_mes(sts));
            return sts;
        }
    }
    kernel_proc_prewitt<int16x2x4, short4, 1, true, false, false, true><<<gridSize, blockSize>>>(
        (uint8_t *)m_mask->frame.ptr[0], nullptr, (int *)m_createLogoMaskValidMaskCount.ptrDevice,
        (const uint8_t *)pLogoData->pDevLogo->frame.ptr[0], pLogoData->pDevLogo->frame.pitch[0],
        nullptr,
        m_mask->frame.pitch[0],
        pLogoData->width, pLogoData->height,
        m_mask->frame.pitch[0] * pLogoData->height,
        maskThreshold);

    auto sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createLogoMask(kernel_proc_prewitt): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
#if DELOGO_DEBUG_CUDA
    sts = err_to_rgy(cudaThreadSynchronize());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(cudaThreadSynchronize): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    debug_out_csv<short>(m_mask.get(), _T("m_mask.csv"));
#endif

    //maskValidCountのGPU側の計算結果をCPUに転送
    sts = m_createLogoMaskValidMaskCount.copyDtoH();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createLogoMask(m_createLogoMaskValidMaskCount.copyDtoH): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    //maskValidCountの集計
    //ブロックごとに算出された値をすべて加算する
    m_maskValidCount = 0;
    auto pMaskValidCountTemp = (const int *)m_createLogoMaskValidMaskCount.ptrHost;
    for (int i = 0; i < blockCount; i++) {
        m_maskValidCount += pMaskValidCountTemp[i];
    }
    return RGY_ERR_NONE;
}

template<typename Type4, int range>
RGY_ERR run_erosion(CUFrameBuf *ptr_mask_nr, const CUFrameBuf *ptr_mask, int mask_threshold) {
    dim3 blockSize(DELOGO_BLOCK_X, DELOGO_BLOCK_Y);
    dim3 gridSize(divCeil(ptr_mask->frame.width, blockSize.x * 4), divCeil(ptr_mask->frame.height, blockSize.y * DELOGO_BLOCK_LOOP_Y), 1);

    kernel_erosion<Type4, range><<<gridSize, blockSize>>>(
        (uint8_t *)ptr_mask_nr->frame.ptr[0], ptr_mask->frame.pitch[0],
        ptr_mask->frame.width, ptr_mask->frame.height,
        (const uint8_t *)ptr_mask->frame.ptr[0], mask_threshold);

    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterDelogo::createNRMask(CUFrameBuf *ptr_mask_nr, const CUFrameBuf *ptr_mask, int nr_value) {
    static const std::map<int, decltype(&run_erosion<short4, 1>)> create_nr_func_list = {
        { 1, run_erosion<short4, 1> },
        { 2, run_erosion<short4, 2> },
        { 3, run_erosion<short4, 3> },
        { 4, run_erosion<short4, 4> },
    };

    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (nr_value > 0) {
        if (create_nr_func_list.count(nr_value) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported NRValue for create_nr_func_list: %d\n"), nr_value);
            return RGY_ERR_UNSUPPORTED;
        }
        auto sts = create_nr_func_list.at(nr_value)(ptr_mask_nr, ptr_mask, m_maskThreshold);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at createNRMask(kernel_erosion): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
#if DELOGO_DEBUG_CUDA
        sts = err_to_rgy(cudaThreadSynchronize());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at createNRMask(cudaThreadSynchronize[run_erosion]): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
        debug_out_csv<short>(ptr_mask_nr, _T("m_maskNR.csv"));
#endif
    } else {
        //nrが必要なければ、そのままコピー
        auto sts = err_to_rgy(cudaMemcpyAsync(ptr_mask_nr->frame.ptr[0], ptr_mask->frame.ptr[0],
            ptr_mask->frame.pitch[0] * ptr_mask->frame.height,
            cudaMemcpyDeviceToDevice));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at createNRMask(cudaMemcpyAsync): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
#if DELOGO_DEBUG_CUDA
        sts = err_to_rgy(cudaThreadSynchronize());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at createNRMask(cudaThreadSynchronize[cudaMemcpyAsync]): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
#endif
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDelogo::createAdjustedMask(const RGYFrameInfo *frame_logo) {

    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // NR=0で評価する
    std::vector<float> eval_results(DELOGO_PRE_DIV_COUNT+1);
    cudaEventRecord(*m_adjMaskStream.heEval.get(), cudaStreamDefault);
    cudaStreamWaitEvent(*m_adjMaskStream.stEval.get(), *m_adjMaskStream.heEval.get(), 0);

    if (m_fadeValueAdjust.nSize < sizeof(float) * eval_results.size()) {
        AddMessage(RGY_LOG_ERROR, _T("Not enough buffer: m_fadeValueAdjust.\n"));
        return RGY_ERR_UNKNOWN;
    }
    auto sts = autoFadeCoef2Run(true, frame_logo, 0, pDelogoParam->delogo.NRArea,
        (const float *)m_fadeValueAdjust.ptrDevice, (int)eval_results.size(),
        m_adjMaskStream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const auto stream = *m_adjMaskStream.stEval.get();
    const int logo_w = m_mask->frame.width;
    const int logo_h = m_mask->frame.height;
    const dim3 blockSize(DELOGO_BLOCK_X, DELOGO_BLOCK_Y);
    const dim3 gridSize(divCeil(logo_w, blockSize.x * 4), divCeil(logo_h, blockSize.y));
    const int blockCount = gridSize.x * gridSize.y;

    //fade値をリセット(あとでkernel_create_adjust_mask1で加算していく)
    if (m_adjMaskEachFadeCount.nSize < sizeof(int) * (DELOGO_PRE_DIV_COUNT + 1)) {
        AddMessage(RGY_LOG_ERROR, _T("error: not enough buffer m_adjMaskEachFadeCount.\n"));
        return RGY_ERR_UNKNOWN;
    }
    sts = err_to_rgy(cudaMemsetAsync(m_adjMaskEachFadeCount.ptrDevice, 0, sizeof(int) * (DELOGO_PRE_DIV_COUNT+1), stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(cudaMemset): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    if (m_adjMaskMinResAndValidMaskCount.nSize < sizeof(int2) * blockCount) {
        AddMessage(RGY_LOG_ERROR, _T("error: not enough buffer m_adjMaskMinResAndValidMaskCount.\n"));
        return RGY_ERR_UNKNOWN;
    }

    static_assert(DELOGO_PRE_DIV_COUNT < std::numeric_limits<decltype(char4::x)>::max(), "DELOGO_PRE_DIV_COUNT < std::numeric_limits<decltype(char4::x)>::max()");
    kernel_create_adjust_mask1<char4, short4><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)m_adjMaskMinIndex->frame.ptr[0], m_adjMaskMinIndex->frame.pitch[0],
        (int *)m_adjMaskEachFadeCount.ptrDevice, (int2 *)m_adjMaskMinResAndValidMaskCount.ptrDevice,
        m_mask->frame.pitch[0], logo_w, logo_h, m_mask->frame.pitch[0] * logo_h,
        (const uint8_t *)m_mask->frame.ptr[0], (const uint8_t *)m_bufEval[0]->frame.ptr[0], m_maskThreshold);
    sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(kernel_create_adjust_mask1): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
#if DELOGO_DEBUG_CUDA
    sts = err_to_rgy(cudaThreadSynchronize());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(cudaThreadSynchronize(1)): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    debug_out_csv<char>(m_adjMaskMinIndex.get(), _T("m_adjMaskMinIndex.csv"));
#endif
#if 0
    //計算結果をCPUに転送
    sts = m_adjMaskEachFadeCount.copyDtoHAsync(stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy data from GPU(m_adjMaskEachFadeCount): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    sts = m_adjMaskMinResAndValidMaskCount.copyDtoHAsync(stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy data from GPU(m_adjMaskMinResAndValidMaskCount): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    sts = autoFadeCoef2Collect(eval_results, 0, *m_adjMaskStream.heEvalCopyFin.get());
    if (sts != RGY_ERR_NONE) return sts;
    cudaStreamSynchronize(stream);

    //GPUでの計算結果はブロックごとの値なので、最終的な集計はCPUで行う
    auto temp_whole_min_result_valid_mask_count = (const int2 *)m_adjMaskMinResAndValidMaskCount.ptrHost;
    int whole_min_result = std::numeric_limits<int>::max();
    int valid_mask_eval = 0;
    for (int i = 0; i < blockCount; i++) {
        whole_min_result = std::min(whole_min_result, temp_whole_min_result_valid_mask_count[i].x);
        valid_mask_eval += temp_whole_min_result_valid_mask_count[i].y;
    }
#if DELOGO_DEBUG_CUDA
    AddMessage(RGY_LOG_DEBUG, _T("whole_min_result = %d, valid_mask_eval = %d\n"), whole_min_result, valid_mask_eval);
#endif

    //fade値は集計は必要なく直接利用できる
    auto each_fade_count = (const int *)m_adjMaskEachFadeCount.ptrHost;
    int min_fade_index = -1;
    int max_fade_count = -1;
    for (int i = 0; i <= DELOGO_PRE_DIV_COUNT; i++) {
#if DELOGO_DEBUG_CUDA
        AddMessage(RGY_LOG_DEBUG, _T("each_fade_count[%d] = %d\n"), i, each_fade_count[i]);
#endif
        if (each_fade_count[i] > max_fade_count) {
            max_fade_count = each_fade_count[i];
            min_fade_index = i;
        }
    }

    const float aveResult = (m_maskValidCount > 0) ? eval_results[min_fade_index] / m_maskValidCount : 0;

    const float prewitt_threshold_base = whole_min_result * 2.0f + 100.0f;
    const float rate = 0.5f + (float)std::min((aveResult - 400.0f) * (1.0f / 200.0f), 4.0f) * 0.08f;
    const int target_count = (int)((float)valid_mask_eval * (1.0f - rate));
#if DELOGO_DEBUG_CUDA
    AddMessage(RGY_LOG_DEBUG, _T("prewitt_threshold_base = %f, target_count = %d\n"), prewitt_threshold_base, target_count);
#endif
#endif
    static bool threshold_mul_initialized = false;
    if (!threshold_mul_initialized) {
        //constantメモリを初期化
        std::array<float, _countof(g_threshold_adj_mul)> threshold_adj_mul;
        for (int i = 0; i < (int)threshold_adj_mul.size(); i++) {
            threshold_adj_mul[i] = constpow<float>((float)DELOGO_ADJMASK_POW_BASE, i);
        }
        sts = err_to_rgy(cudaMemcpyToSymbol(g_threshold_adj_mul, threshold_adj_mul.data(), sizeof(g_threshold_adj_mul), 0, cudaMemcpyHostToDevice));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy data to symbol(g_threshold_adj_mul): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
        threshold_mul_initialized = true;
    }

    const dim3 gridSize2(gridSize.x, gridSize.y, DELOGO_ADJMASK_DIV_COUNT);
    if (m_adjMask2ValidMaskCount.nSize < sizeof(int) * gridSize2.x * gridSize2.y * gridSize2.z) {
        AddMessage(RGY_LOG_ERROR, _T("error: not enough buffer m_adjMask2ValidMaskCount.\n"));
        return RGY_ERR_UNKNOWN;
    }

    kernel_create_adjust_mask2<char4, short4><<<gridSize2, blockSize, 0, stream>>>(
        (uint8_t *)m_adjMaskThresholdTest->frame.ptr[0], //TypeMask4
        (int *)m_adjMask2ValidMaskCount.ptrDevice,
        (int *)m_adjMask2TargetCount.get(),
        (const uint8_t *)m_bufEval[0]->frame.ptr[0], //TypeMask4
        m_bufEval[0]->frame.pitch[0], m_bufEval[0]->frame.pitch[0] * logo_h,
        logo_w, logo_h,
        (const uint8_t *)m_adjMaskMinIndex->frame.ptr[0], m_adjMaskMinIndex->frame.pitch[0],
        m_maskThreshold,
        (const float *)m_evalCounter[0].ptrDevice, m_evalStream[0].evalBlocks,
        (const int2 *)m_adjMaskMinResAndValidMaskCount.ptrDevice, blockCount,
        (const int *)m_adjMaskEachFadeCount.ptrDevice, m_maskValidCount);

    sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(kernel_create_adjust_mask2): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
#if DELOGO_DEBUG_CUDA
    sts = err_to_rgy(cudaThreadSynchronize());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(cudaThreadSynchronize(2)): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    debug_out_csv<char>(m_adjMaskThresholdTest.get(), _T("m_adjMaskThresholdTest.csv"));
#endif
#if 1
    kernel_create_adjust_mask3<short4><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)m_maskAdjusted->frame.ptr[0],  //TypeMask4
        (const int *)m_adjMask2ValidMaskCount.ptrDevice, blockCount,
        (const uint8_t *)m_adjMaskThresholdTest->frame.ptr[0], //TypeMask4
        m_maskAdjusted->frame.pitch[0], m_maskAdjusted->frame.pitch[0] * logo_h,
        logo_w, logo_h,
        (const int *)m_adjMask2TargetCount.get());
#if DELOGO_DEBUG_CUDA
    sts = err_to_rgy(cudaThreadSynchronize());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at createAdjustedMask(cudaThreadSynchronize(3)): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
#endif
    cudaEventRecord(*m_adjMaskStream.heEvalCopyFin.get(), stream);
#else
    sts = m_adjMask2ValidMaskCount.copyDtoH();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy data from GPU(m_adjMask2ValidMaskCount): %s.\n"),
            get_err_mes(sts));
        return sts;
    }

    int target_index = 0;
    auto *ptr_temp_valid_mask_count = (const int *)m_adjMask2ValidMaskCount.ptrHost;
    for (int i = 0; i < gridSize2.z; i++) {
        int mask_count = 0;
        for (int ib = 0; ib < blockCount; ib++) {
            mask_count += ptr_temp_valid_mask_count[i * blockCount + ib];
        }
#if DELOGO_DEBUG_CUDA
        AddMessage(RGY_LOG_DEBUG, _T("mask_count[%d] = %d\n"), i, mask_count);
#endif
        if (mask_count >= target_count) {
            target_index = i;
            break;
        }
    }

    sts = err_to_rgy(cudaMemcpy2DAsync(m_maskAdjusted->frame.ptr, m_maskAdjusted->frame.pitch,
        m_adjMaskThresholdTest->frame.ptr + target_index * m_adjMaskThresholdTest->frame.pitch * logo_h, m_adjMaskThresholdTest->frame.pitch,
        logo_w, logo_h,
        cudaMemcpyDeviceToDevice,
        *m_adjMaskStream.stEvalSub.get()));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy data createAdjustedMask(m_adjMask2ValidMaskCount): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    cudaEventRecord(*m_adjMaskStream.heEvalCopyFin.get(), *m_adjMaskStream.stEvalSub.get());
#endif
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
RGY_ERR runDelogoYMultiFadeKernel(
    const CUFrameBuf *pDevBufDst,       //出力先、fade_n分のメモリが必要
    const ProcessDataDelogo *logo_data, //logoの情報
    const RGYFrameInfo *srcFrame,          //delogoを行うフレームの情報
    const bool multi_src,               //入力(frame_logo)も複数枚の入力(fade_n)を持つ
    const float *ptrDevFadeDepth,       //fade * depthの情報 (fade_n分の配列)
    const int fade_n,                   //並列処理するfadeの数
    cudaStream_t stream
) {
    //delogoは他のkernelとは異なり、1ピクセル=1スレッドなことに注意
    const dim3 blockSize(DELOGO_BLOCK_X, DELOGO_BLOCK_Y);
    const dim3 gridSize(divCeil(logo_data->width, blockSize.x), divCeil(logo_data->height, blockSize.y), fade_n);

    kernel_delogo_multi_fade<Type, short, bit_depth, true><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pDevBufDst->frame.ptr[0], pDevBufDst->frame.pitch[0], pDevBufDst->frame.pitch[0] * logo_data->height,
        (const uint8_t *)srcFrame->ptr[0] + logo_data->j_start * srcFrame->pitch[0] + logo_data->i_start * sizeof(Type),
        srcFrame->pitch[0], srcFrame->width,
        (multi_src) ? srcFrame->pitch[0] * srcFrame->height : 0,
        (const uint8_t *)logo_data->pDevLogo->frame.ptr[0], logo_data->pDevLogo->frame.pitch[0],
        logo_data->width, logo_data->height,
        ptrDevFadeDepth);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterDelogo::runDelogoYMultiFade(
    const RGYFrameInfo *frame_logo,    //delogoを行うフレームの情報
    const bool multi_src,           //入力(frame_logo)も複数枚の入力(fade_n)を持つ
    const int nr_value,             //この処理の時のnr_value (delogo処理には関係ないが、出力先のバッファを決めるために指定が必要)
    const float *ptrDevFadeDepth,   //fade * depthの情報 (fade_n分の配列)
    const int fade_n,               //並列処理するfadeの数
    cudaStream_t stream
) {
    static const decltype(&runDelogoYMultiFadeKernel<uint16_t, 16>) delogo_multi_fade_func_list[2] = {
        runDelogoYMultiFadeKernel<uint8_t,   8>,
        runDelogoYMultiFadeKernel<uint16_t, 16>,
    };
    if (nr_value > LOGO_NR_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("nr_value: %d > LOGO_NR_MAX: %d\n"), nr_value);
        return RGY_ERR_INVALID_CALL;
    }

    auto sts = delogo_multi_fade_func_list[RGY_CSP_BIT_DEPTH[frame_logo->csp] > 8 ? 1 : 0](
        m_bufDelogo[nr_value].get(),
        &m_sProcessData[LOGO__Y], frame_logo, multi_src, ptrDevFadeDepth, fade_n,
        stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at runDelogoYMultiFade(kernel_delogo_multi_fade): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
#if DELOGO_DEBUG_CUDA
    cudaerr = err_to_rgy(cudaThreadSynchronize());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at runDelogoYMultiFade(cudaThreadSynchronize): %s.\n"),
            get_err_mes(sts));
        return sts;
    }
    debug_out_csv<short>(m_bufDelogo[nr_value].get(), _T("delogo_result.csv"));
#endif
    return RGY_ERR_NONE;
}

template<typename Type4, typename TypeMask4, int range>
RGY_ERR runSmoothKernel(
    uint8_t *ptr_dst, const int dst_pitch, const int dst_size,       //スムージング結果の出力先
    const uint8_t *ptr_src, const int src_pitch, const int src_size, //スムージングを行う対象
    const int logo_w, const int logo_h, //ロゴの情報
    const TypeMask4 *ptr_mask, const int mask_pitch, int mask_threshold,  //スムージングを行うピクセルを決めるマスク
    const float *smooth_kernel, //スムージングのカーネル(GPU上にあるデータ)
    int smooth_n,               //並列処理する数
    cudaStream_t stream
) {
    const dim3 blockSize(DELOGO_BLOCK_X, DELOGO_BLOCK_Y);
    const dim3 gridSize(divCeil(logo_w, blockSize.x * 4), divCeil(logo_h, blockSize.y * DELOGO_BLOCK_LOOP_Y), smooth_n);

    kernel_proc_symmetry<Type4, TypeMask4, range><<<gridSize, blockSize, 0, stream>>>(
        ptr_dst, dst_pitch, dst_size,
        ptr_src, src_pitch, src_size,
        logo_w, logo_h,
        (const uint8_t *)ptr_mask, mask_pitch,
        mask_threshold,
        smooth_kernel, smooth_kernel);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterDelogo::runSmooth(
    const int smooth_n, //並列処理する数
    const int nr_value, const int nr_area,
    cudaStream_t stream) {
    static const std::map<int, decltype(&runSmoothKernel<short4, short4, 1>)> smooth_func_list = {
        { 1, runSmoothKernel<short4, short4, 1> },
        { 2, runSmoothKernel<short4, short4, 2> },
        { 3, runSmoothKernel<short4, short4, 3> },
        { 4, runSmoothKernel<short4, short4, 4> },
    };
    if (m_bufDelogo[nr_value]->frame.width % 4 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("frame width must be mod4\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (nr_value > 0) {
        if (smooth_func_list.count(nr_value) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported nr_value for smooth_func_list: %d\n"), nr_value);
            return RGY_ERR_UNSUPPORTED;
        }
        //入出力のバッファは、nr_valueに対応するものを使用する
        const int logo_w = m_mask->frame.width;
        const int logo_h = m_mask->frame.height;
        const CUFrameBuf *ptr_mask = (nr_area > 0) ? m_maskNR.get() : m_mask.get();
        auto sts = smooth_func_list.at(nr_value)(
            (uint8_t *)m_bufDelogoNR[nr_value]->frame.ptr[0], m_bufDelogoNR[nr_value]->frame.pitch[0], m_bufDelogoNR[nr_value]->frame.pitch[0] * logo_h,
            (const uint8_t *)m_bufDelogo[nr_value]->frame.ptr[0], m_bufDelogo[nr_value]->frame.pitch[0], m_bufDelogo[nr_value]->frame.pitch[0] * logo_h,
            logo_w, logo_h,
            (const short4 *)ptr_mask->frame.ptr[0], ptr_mask->frame.pitch[0], m_maskThreshold,
            (float *)m_smoothKernel.get(),
            smooth_n,
            stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at runSmooth(kernel_proc_symmetry): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
#if DELOGO_DEBUG_CUDA
        sts = err_to_rgy(cudaThreadSynchronize());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at runSmooth(cudaThreadSynchronize): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
        debug_out_csv<short>(m_bufDelogoNR[nr_value].get(), _T("m_bufDelogoNR[nr_value].csv"));
#endif
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDelogo::prewittEvaluateRun(
    const bool store_pixel_result, //評価結果をピクセルごとにm_bufEvalに格納するかどうか
    const CUFrameBuf *target,      //評価対象のデータ
    const CUFrameBuf *mask,        //処理時に使用するmask
    const int nr_value,            //この処理の時のnr_value (delogo処理には関係ないが、入出力のバッファを決めるために指定が必要)
    const int eval_n,              //同時処理する数
    DelogoEvalStreams& evalst
) {
    const int logo_w = m_sProcessData[LOGO__Y].width;
    const int logo_h = m_sProcessData[LOGO__Y].height;
    if (logo_w % 4 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("logo width must be mod4.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (eval_n == 0) {
        AddMessage(RGY_LOG_ERROR, _T("eval_n == 0.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (eval_n > DELOGO_PARALLEL_FADE) {
        AddMessage(RGY_LOG_ERROR, _T("eval_n > DELOGO_PARALLEL_FADE.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const auto stream = *evalst.stEval.get();
    const dim3 blockSize(DELOGO_BLOCK_X, DELOGO_BLOCK_Y);
    const dim3 gridSize(divCeil(logo_w, blockSize.x * 4), divCeil(logo_h, blockSize.y * DELOGO_BLOCK_LOOP_Y), eval_n);
    m_evalStream[nr_value].evalBlocks = gridSize.x * gridSize.y;

    if (m_evalCounter[nr_value].nSize < gridSize.x * gridSize.y * gridSize.z * sizeof(float)) {
        AddMessage(RGY_LOG_ERROR, _T("error: Not enough buffer for m_evalCounter[nr_value]\n"));
        return RGY_ERR_UNKNOWN;
    }
    if (store_pixel_result) {
        //ピクセルごとの評価結果をバッファに出力
        kernel_proc_prewitt<short4, short4, 2, true, true, true, false><<<gridSize, blockSize, 0, stream>>>(
            (uint8_t *)m_bufEval[nr_value]->frame.ptr[0], (float *)m_evalCounter[nr_value].ptrDevice, nullptr,
            (const uint8_t *)target->frame.ptr[0], target->frame.pitch[0],
            (const uint8_t *)mask->frame.ptr[0], mask->frame.pitch[0],
            logo_w, logo_h, mask->frame.pitch[0] * logo_h,
            m_maskThreshold);
    } else {
        //ピクセルごとの評価結果は出力しない
        kernel_proc_prewitt<short4, short4, 2, false, true, true, false><<<gridSize, blockSize, 0, stream>>>(
            nullptr, (float *)m_evalCounter[nr_value].ptrDevice, nullptr,
            (const uint8_t *)target->frame.ptr[0], target->frame.pitch[0],
            (const uint8_t *)mask->frame.ptr[0], mask->frame.pitch[0],
            logo_w, logo_h, mask->frame.pitch[0] * logo_h,
            m_maskThreshold);
    }
    auto sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at prewittEvaluate(kernel_proc_prewitt<store_pixel_result=%s>): %s.\n"),
            store_pixel_result ? _T("true") : _T("false"),
            get_err_mes(sts));
        return sts;
    }
#if DELOGO_DEBUG_CUDA
    if (store_pixel_result) {
        sts = err_to_rgy(cudaThreadSynchronize());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at prewittEvaluateRun(cudaThreadSynchronize): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
        debug_out_csv<short>(m_bufEval[nr_value].get(), _T("m_bufEval[nr_value].csv"));
    }
#endif //#if DELOGO_DEBUG_CUDA
    //評価結果(集計版)の一時データをGPU→CPUに転送
    if (evalst.stEvalSub) {
        cudaEventRecord(*evalst.heEval.get(), stream);
        cudaStreamWaitEvent(*evalst.stEvalSub.get(), *evalst.heEval.get(), 0);
        sts = m_evalCounter[nr_value].copyDtoHAsync(*evalst.stEvalSub.get());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at prewittEvaluate(m_evalCounter[nr_value].copyDtoHAsync): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
        cudaEventRecord(*evalst.heEvalCopyFin.get(), *evalst.stEvalSub.get());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDelogo::autoFadeCoef2Run(
    const bool store_pixel_result, //評価結果をピクセルごとにm_bufEvalに格納するかどうか
    const RGYFrameInfo *frame_logo,   //delogoを行うフレームの情報
    const int nr_value,            //この処理の時のnr_value
    const int nr_area,             //この処理の時のnr_area
    const float *ptrDevFadeDepth,  //fade * depthの情報 (calc_n分の配列)
    const int calc_n,              //同時処理する数
    DelogoEvalStreams& evalst
) {
    if (m_sProcessData[LOGO__Y].pDevLogo->frame.width % 4 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("frame width must be mod4\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (calc_n == 0) {
        AddMessage(RGY_LOG_ERROR, _T("calc_n == 0.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const auto stream = *evalst.stEval.get();
    auto sts = runDelogoYMultiFade(frame_logo, false, nr_value, ptrDevFadeDepth, calc_n, stream);
    if (sts != RGY_ERR_NONE) return sts;

    auto ptrDevTarget = m_bufDelogo[nr_value].get();
    if (nr_value > 0) {
        sts = runSmooth(calc_n, nr_value, nr_area, stream);
        if (sts != RGY_ERR_NONE) return sts;
        ptrDevTarget = m_bufDelogoNR[nr_value].get();
    }

    sts = prewittEvaluateRun(store_pixel_result, ptrDevTarget, m_mask.get(), nr_value, calc_n, evalst);
    if (sts != RGY_ERR_NONE) return sts;

    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDelogo::autoFadeCoef2Collect(
    std::vector<float>& eval,      //評価結果を出力(格納)する場所、vectorのサイズが同時処理する数
    const int nr_value,            //この処理の時のnr_value
    cudaEvent_t eventCopyFin
) {
    if (eval.size() == 0) {
        AddMessage(RGY_LOG_ERROR, _T("eval.size() == 0.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    cudaEventSynchronize(eventCopyFin);
    //一時データ(ブロックごとの出力)をCPU側で最終集計する
    auto ptr_eval_temp = (const float *)m_evalCounter[nr_value].ptrHost;
    std::fill(eval.begin(), eval.end(), 0.0f);
    for (size_t ieval = 0; ieval < eval.size(); ieval++) {
        for (int ib = 0; ib < m_evalStream[nr_value].evalBlocks; ib++) {
            eval[ieval] += ptr_eval_temp[ieval * m_evalStream[nr_value].evalBlocks + ib];
        }
    }

    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDelogo::logoNR(RGYFrameInfo *pFrame, int nr_value) {
    static const std::map<std::pair<RGY_CSP, int>, decltype(&runSmoothKernel<uchar4, short4, 1>)> smooth_func_list = {
        { std::make_pair(RGY_CSP_Y8,  1), runSmoothKernel<uchar4,  short4, 1> },
        { std::make_pair(RGY_CSP_Y8,  2), runSmoothKernel<uchar4,  short4, 2> },
        { std::make_pair(RGY_CSP_Y8,  3), runSmoothKernel<uchar4,  short4, 3> },
        { std::make_pair(RGY_CSP_Y8,  4), runSmoothKernel<uchar4,  short4, 4> },
        { std::make_pair(RGY_CSP_Y16, 1), runSmoothKernel<ushort4, short4, 1> },
        { std::make_pair(RGY_CSP_Y16, 2), runSmoothKernel<ushort4, short4, 2> },
        { std::make_pair(RGY_CSP_Y16, 3), runSmoothKernel<ushort4, short4, 3> },
        { std::make_pair(RGY_CSP_Y16, 4), runSmoothKernel<ushort4, short4, 4> },
    };

    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (nr_value > 0) {
        auto sts = createNRMask(m_maskNRAdjusted.get(), m_maskAdjusted.get(), pDelogoParam->delogo.NRArea);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        const auto key = std::make_pair(RGY_CSP_BIT_DEPTH[pFrame->csp] > 8 ? RGY_CSP_Y16 : RGY_CSP_Y8, nr_value);
        if (smooth_func_list.count(key) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported nr_value for smooth_func_list: %d\n"), nr_value);
            return RGY_ERR_UNSUPPORTED;
        }
        const int src_pixel_size = RGY_CSP_BIT_DEPTH[pFrame->csp] > 8 ? 2 : 1;
        const auto logodata = &m_sProcessData[LOGO__Y];
        //入出力のバッファは、nr_valueに対応するものを使用する
        sts = smooth_func_list.at(key)(
            (uint8_t *)m_NRProcTemp->frame.ptr[0], m_NRProcTemp->frame.pitch[0], m_NRProcTemp->frame.pitch[0] * m_NRProcTemp->frame.height,
            (const uint8_t *)pFrame->ptr[0] + logodata->j_start * pFrame->pitch[0] + logodata->i_start * src_pixel_size,
            pFrame->pitch[0], pFrame->pitch[0] * pFrame->height,
            logodata->width, logodata->height,
            (const short4 *)m_maskNRAdjusted->frame.ptr[0], m_maskNRAdjusted->frame.pitch[0], m_maskThreshold,
            (float *)m_smoothKernel.get(),
            1, cudaStreamDefault);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at logoNR(kernel_proc_symmetry): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
        sts = err_to_rgy(cudaMemcpy2DAsync(
            (uint8_t *)pFrame->ptr[0] + logodata->j_start * pFrame->pitch[0] + logodata->i_start * src_pixel_size, pFrame->pitch[0],
            m_NRProcTemp->frame.ptr[0], m_NRProcTemp->frame.pitch[0],
            logodata->width, logodata->height,
            cudaMemcpyDeviceToDevice));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at logoNR(cudaMemcpy2DAsync): %s.\n"),
                get_err_mes(sts));
            return sts;
        }
    }
    return RGY_ERR_NONE;
}
