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
#include "convert_csp.h"
#include "NVEncFilterConvolution3d.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int C3D_THRESHOLD_MIN = 0;
static const int C3D_THRESHOLD_MAX = 255;
static const int C3D_BLOCK_X       = 32;
static const int C3D_BLOCK_Y       = 8;

template<typename Type>
__device__ void convolution3d_load(
    float temp[C3D_BLOCK_Y + 2][C3D_BLOCK_X + 2],
    const uint8_t *__restrict__ pFrame, const int srcPitch,
    const int lx, const int ly, const int blockimgx, const int blockimgy,
    const int width, const int height) {
#define SRCPTR(ptr, pitch, ix, iy) (const Type *)((ptr) + clamp((iy), 0, height-1) * (pitch) + clamp((ix), 0, width-1) * sizeof(Type))
 
    if (true)       temp[            ly][            lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx               + lx - 1, blockimgy               + ly - 1));
    if (lx < 2)     temp[            ly][C3D_BLOCK_X+lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx + C3D_BLOCK_X + lx - 1, blockimgy               + ly - 1));
    if (ly < 2) {   temp[C3D_BLOCK_Y+ly][            lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx               + lx - 1, blockimgy + C3D_BLOCK_Y + ly - 1));
        if (lx < 2) temp[C3D_BLOCK_Y+ly][C3D_BLOCK_X+lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx + C3D_BLOCK_X + lx - 1, blockimgy + C3D_BLOCK_Y + ly - 1));
    }
#undef SRCPTR
}

__device__ float convolution3d_check_threshold(
    const float orig, const float pixel, const float thresh
) {
    return (fabsf(orig - pixel) <= thresh) ? orig : pixel;
}

template<int s0, int s1, int s2>
__device__ float convolution3d_spatial(
    const float temp[C3D_BLOCK_Y + 2][C3D_BLOCK_X + 2],
    const int lx, const int ly,
    const float src,
    const float threshold_spatial
) {
    float val0 = 0.0f;
    float val1 = 0.0f;
    float val2 = 0.0f;
    val0 += convolution3d_check_threshold(temp[ly+0][lx+0], src, threshold_spatial) * (float)(s0 * s0);
    val0 += convolution3d_check_threshold(temp[ly+0][lx+1], src, threshold_spatial) * (float)(s0 * s1);
    val0 += convolution3d_check_threshold(temp[ly+0][lx+2], src, threshold_spatial) * (float)(s0 * s2);
    val1 += convolution3d_check_threshold(temp[ly+1][lx+0], src, threshold_spatial) * (float)(s1 * s0);
    val1 += convolution3d_check_threshold(temp[ly+1][lx+1], src, threshold_spatial) * (float)(s1 * s1);
    val1 += convolution3d_check_threshold(temp[ly+1][lx+2], src, threshold_spatial) * (float)(s1 * s2);
    val2 += convolution3d_check_threshold(temp[ly+2][lx+0], src, threshold_spatial) * (float)(s2 * s0);
    val2 += convolution3d_check_threshold(temp[ly+2][lx+1], src, threshold_spatial) * (float)(s2 * s1);
    val2 += convolution3d_check_threshold(temp[ly+2][lx+2], src, threshold_spatial) * (float)(s2 * s2);
    int stotal = s0 + s1 + s2;
    return (val0 + val1 + val2) * (1.0f / (float)(stotal * stotal));
}

template<typename Type, int depth, bool fast, int s0, int s1, int s2, int t0, int t1, int t2>
__global__ void kernel_denoise_convolution3d(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pPrev, const uint8_t *__restrict__ pCur, const uint8_t *__restrict__ pNext,
    const int srcPitch, const int width, const int height,
    const float threshold_spatial, const float threshold_temporal) {
    const int lx = threadIdx.x; //スレッド数=C3D_BLOCK_X
    const int ly = threadIdx.y; //スレッド数=C3D_BLOCK_Y
    const int blockimgx = blockIdx.x * C3D_BLOCK_X;
    const int blockimgy = blockIdx.y * C3D_BLOCK_Y;
    const int imgx = blockimgx + lx;
    const int imgy = blockimgy + ly;
    __shared__ float temp_src[(fast) ? 1 : 3][C3D_BLOCK_Y + 2][C3D_BLOCK_X + 2];
#define GETPTR(ptr, pitch, ix, iy) (Type *)((ptr) + (iy) * (pitch) + (ix) * sizeof(Type))

    static const int SRC_CUR = (fast) ? 0 : 1;

    convolution3d_load<Type>(temp_src[SRC_CUR], pCur, srcPitch, lx, ly, blockimgx, blockimgy, width, height);
    if (!fast) {
        convolution3d_load<Type>(temp_src[0], pPrev, srcPitch, lx, ly, blockimgx, blockimgy, width, height);
        convolution3d_load<Type>(temp_src[2], pNext, srcPitch, lx, ly, blockimgx, blockimgy, width, height);
    }
    __syncthreads();

    if (imgx < width && imgy < height) {
        const float src = temp_src[SRC_CUR][ly + 1][lx + 1];
        const float cur = convolution3d_spatial<s0, s1, s2>(temp_src[SRC_CUR], lx, ly, src, threshold_spatial);

        float prev = 0.0f;
        float next = 0.0f;
        if (fast) {
            prev = convolution3d_check_threshold((float)(*GETPTR(pPrev, srcPitch, imgx, imgy)), src, threshold_temporal);
            next = convolution3d_check_threshold((float)(*GETPTR(pNext, srcPitch, imgx, imgy)), src, threshold_temporal);
        } else {
            prev = convolution3d_spatial<s0, s1, s2>(temp_src[0], lx, ly, src, threshold_temporal);
            next = convolution3d_spatial<s0, s1, s2>(temp_src[2], lx, ly, src, threshold_temporal);
        }
        float result = 0.0f;
        result += prev * (float)t0;
        result += cur  * (float)t1;
        result += next * (float)t2;
        result *= (1.0f / (float)(t0 + t1 + t2));

        *GETPTR(pDst, dstPitch, imgx, imgy) = (Type)clamp(result + 0.5f, 0.0f, (float)((1 << depth) - 1) + 1e-6f);
    }
#undef GETPTR
}

template<typename Type, int depth, bool fast, int s0, int s1, int s2, int t0, int t1, int t2>
static cudaError_t denoise_convolution3d_plane(RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pPrevPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pNextPlane,
    const float threshold_spatial, const float threshold_temporal, cudaStream_t stream) {
    dim3 blockSize(C3D_BLOCK_X, C3D_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));
    kernel_denoise_convolution3d<Type, depth, fast, s0, s1, s2, t0, t1, t2><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (const uint8_t *)pPrevPlane->ptr[0], (const uint8_t *)pInputPlane->ptr[0], (const uint8_t *)pNextPlane->ptr[0],
        pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
        threshold_spatial, threshold_temporal);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int depth, bool fast, int s0, int s1, int s2, int t0, int t1, int t2>
static cudaError_t denoise_convolution3d_frame_weight(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pPrevFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pNextFrame,
    const float threshYspatial, const float threshCspatial, const float threshYtemporal, const float threshCtemporal,
    cudaStream_t stream) {
    cudaError_t cudaerr = cudaSuccess;
    const auto planePrevY  = getPlane(pPrevFrame,   RGY_PLANE_Y);
    const auto planePrevU  = getPlane(pPrevFrame,   RGY_PLANE_U);
    const auto planePrevV  = getPlane(pPrevFrame,   RGY_PLANE_V);
    const auto planeInputY = getPlane(pInputFrame,  RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame,  RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame,  RGY_PLANE_V);
    const auto planeNextY  = getPlane(pNextFrame,   RGY_PLANE_Y);
    const auto planeNextU  = getPlane(pNextFrame,   RGY_PLANE_U);
    const auto planeNextV  = getPlane(pNextFrame,   RGY_PLANE_V);
    auto planeOutputY      = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU      = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV      = getPlane(pOutputFrame, RGY_PLANE_V);

    cudaerr = denoise_convolution3d_plane<Type, depth, fast, s0, s1, s2, t0, t1, t2>(&planeOutputY, &planePrevY, &planeInputY, &planeNextY, threshYspatial, threshYtemporal, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = denoise_convolution3d_plane<Type, depth, fast, s0, s1, s2, t0, t1, t2>(&planeOutputU, &planePrevU, &planeInputU, &planeNextU, threshCspatial, threshCtemporal, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = denoise_convolution3d_plane<Type, depth, fast, s0, s1, s2, t0, t1, t2>(&planeOutputV, &planePrevV, &planeInputV, &planeNextV, threshCspatial, threshCtemporal, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int depth>
static RGY_ERR denoise_convolution3d_frame(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pPrevFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pNextFrame,
    const bool fastMode, const VppConvolution3dMatrix matrix,
    const float threshYspatial, const float threshCspatial, const float threshYtemporal, const float threshCtemporal,
    cudaStream_t stream) {
    if (fastMode) {
        if (matrix == VppConvolution3dMatrix::Standard) {
            auto cudaerr = denoise_convolution3d_frame_weight<Type, depth, true, 1, 2, 1, 1, 2, 1>(
                pOutputFrame, pPrevFrame, pInputFrame, pNextFrame, threshYspatial, threshYtemporal, threshCspatial, threshCtemporal, stream);
            return err_to_rgy(cudaerr);
        } else if (matrix == VppConvolution3dMatrix::Simple) {
            auto cudaerr = denoise_convolution3d_frame_weight<Type, depth, true, 1, 1, 1, 1, 1, 1>(
                pOutputFrame, pPrevFrame, pInputFrame, pNextFrame, threshYspatial, threshYtemporal, threshCspatial, threshCtemporal, stream);
            return err_to_rgy(cudaerr);
        }
    } else {
        if (matrix == VppConvolution3dMatrix::Standard) {
            auto cudaerr = denoise_convolution3d_frame_weight<Type, depth, false, 1, 2, 1, 1, 2, 1>(
                pOutputFrame, pPrevFrame, pInputFrame, pNextFrame, threshYspatial, threshYtemporal, threshCspatial, threshCtemporal, stream);
            return err_to_rgy(cudaerr);
        } else if (matrix == VppConvolution3dMatrix::Simple) {
            auto cudaerr = denoise_convolution3d_frame_weight<Type, depth, false, 1, 1, 1, 1, 1, 1>(
                pOutputFrame, pPrevFrame, pInputFrame, pNextFrame, threshYspatial, threshYtemporal, threshCspatial, threshCtemporal, stream);
            return err_to_rgy(cudaerr);
        }
    }
    return RGY_ERR_UNSUPPORTED;
}

NVEncFilterConvolution3d::NVEncFilterConvolution3d() : m_bInterlacedWarn(false), m_prevFrames(), m_cacheIdx(0) {
    m_name = _T("convolution3d");
}

NVEncFilterConvolution3d::~NVEncFilterConvolution3d() {
    close();
}

RGY_ERR NVEncFilterConvolution3d::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto param = std::dynamic_pointer_cast<NVEncFilterParamConvolution3d>(pParam);
    if (!param) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (param->frameOut.height <= 0 || param->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (param->convolution3d.threshYspatial < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < param->convolution3d.threshYspatial) {
        AddMessage(RGY_LOG_ERROR, _T("ythresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (param->convolution3d.threshCspatial < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < param->convolution3d.threshCspatial) {
        AddMessage(RGY_LOG_ERROR, _T("cthresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (param->convolution3d.threshYtemporal < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < param->convolution3d.threshYtemporal) {
        AddMessage(RGY_LOG_ERROR, _T("t_ythresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (param->convolution3d.threshCtemporal < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < param->convolution3d.threshCtemporal) {
        AddMessage(RGY_LOG_ERROR, _T("t_cthresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_param
        || std::dynamic_pointer_cast<NVEncFilterParamConvolution3d>(m_param)->convolution3d != param->convolution3d) {
        sts = AllocFrameBuf(param->frameOut, 1);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    if (!m_prevFrames.front() ||
        cmpFrameInfoCspResolution(&m_prevFrames.front()->frame, &param->frameOut)) {
        for (auto& f : m_prevFrames) {
            f.reset(new CUFrameBuf(param->frameOut));
            f->releasePtr();
            sts = f->alloc();
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_cacheIdx = 0;
    }

    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        param->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamConvolution3d::print() const {
    return convolution3d.print();
}

RGY_ERR NVEncFilterConvolution3d::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto param = std::dynamic_pointer_cast<NVEncFilterParamConvolution3d>(m_param);
    if (!param) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (pInputFrame->ptr[0] == nullptr && m_nFrameIdx >= m_cacheIdx) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }

    auto frameNext = pInputFrame;

    //十分な数のフレームがたまった、あるいはdrainモードならフレームを出力
    if (m_cacheIdx >= 1) {
        //出力先のフレーム
        CUFrameBuf *pOutFrame = nullptr;
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            pOutFrame = m_frameBuf[0].get();
            ppOutputFrames[0] = &pOutFrame->frame;
        }
        if (pInputFrame->ptr[0]) {
            const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
            if (memcpyKind != cudaMemcpyDeviceToDevice) {
                AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
                return RGY_ERR_INVALID_PARAM;
            }
            if (m_param->frameOut.csp != m_param->frameIn.csp) {
                AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
                return RGY_ERR_INVALID_PARAM;
            }
        }

        auto framePrev = &m_prevFrames[std::max(m_cacheIdx-2, 0) % m_prevFrames.size()]->frame;
        auto frameCur  = &m_prevFrames[        (m_cacheIdx-1)    % m_prevFrames.size()]->frame;
        if (frameNext->ptr[0] == nullptr) {
            frameNext = frameCur;
        }

        pOutFrame->frame.inputFrameId = frameCur->inputFrameId;
        pOutFrame->frame.duration     = frameCur->duration;
        pOutFrame->frame.timestamp    = frameCur->timestamp;

        static const std::map<RGY_CSP, decltype(denoise_convolution3d_frame<uint8_t, 8>)*> denoise_list = {
            { RGY_CSP_YV12,      denoise_convolution3d_frame<uint8_t,   8> },
            { RGY_CSP_YV12_16,   denoise_convolution3d_frame<uint16_t, 16> },
            { RGY_CSP_YUV444,    denoise_convolution3d_frame<uint8_t,   8> },
            { RGY_CSP_YUV444_16, denoise_convolution3d_frame<uint16_t, 16> },
        };
        if (denoise_list.count(frameNext->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[frameNext->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        const float thresholdMul = (float)(1 << (RGY_CSP_BIT_DEPTH[frameNext->csp] - 8));
        denoise_list.at(frameNext->csp)(&pOutFrame->frame, framePrev, frameCur, frameNext,
            param->convolution3d.fast, param->convolution3d.matrix,
            param->convolution3d.threshYspatial  * thresholdMul,
            param->convolution3d.threshCspatial  * thresholdMul,
            param->convolution3d.threshYtemporal * thresholdMul,
            param->convolution3d.threshCtemporal * thresholdMul,
            stream);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at convolution3d(%s): %s.\n"),
                RGY_CSP_NAMES[frameNext->csp],
                get_err_mes(err_to_rgy(cudaerr)));
            return err_to_rgy(cudaerr);
        }
        m_nFrameIdx++;
    } else {
        //出力フレームなし
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    //sourceキャッシュにコピー
    if (pInputFrame->ptr[0]) {
        auto cacheFrame = &m_prevFrames[m_cacheIdx++ % m_prevFrames.size()]->frame;
        sts = copyFrameAsync(cacheFrame, frameNext, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set frame to data cache: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFrameProp(cacheFrame, frameNext);
    }
    return sts;
}

void NVEncFilterConvolution3d::close() {
    m_frameBuf.clear();
    for (auto& f : m_prevFrames) {
        f.reset();
    }
    m_bInterlacedWarn = false;
    m_cacheIdx = 0;
}
