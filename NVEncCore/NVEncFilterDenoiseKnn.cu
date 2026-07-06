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
#include "NVEncFilterDenoiseKnn.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int KNN_RADIUS_MAX = 5;
static const int KNN_TEMPORAL_MAX = 2;

template<typename Type, int knn_radius, int temporal_d, int bit_depth>
__global__ void kernel_denoise_knn(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texPrev2, cudaTextureObject_t texPrev1, cudaTextureObject_t texSrc, cudaTextureObject_t texNext1, cudaTextureObject_t texNext2,
    const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    const float knn_window_area = (float)((2 * knn_radius + 1) * (2 * knn_radius + 1));
    const float inv_knn_window_area = 1.0f / knn_window_area;
    //temporal_d == 0 では inv_temporal_frames = 1, distT = 0 に定数化され、従来の空間のみの処理と完全に一致する
    const float inv_temporal_frames = 1.0f / (float)(2 * temporal_d + 1);
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        float fCount = 0.0f;
        float sumWeights = 0.0f;
        float sum = 0.0f;
        float center = (float)tex2D<Type>(texSrc, x, y) * (1.0f / (1<<bit_depth));

        #pragma unroll
        for (int t = -temporal_d; t <= temporal_d; t++) {
            const cudaTextureObject_t tex = (t == -2) ? texPrev2 : ((t == -1) ? texPrev1 : ((t == 0) ? texSrc : ((t == 1) ? texNext1 : texNext2)));
            //時間方向の距離ペナルティ、空間方向と同様に窓サイズで正規化
            const float distT = (float)(t * t) * (inv_temporal_frames * inv_temporal_frames);
            #pragma unroll
            for (int i = -knn_radius; i <= knn_radius; i++) {
                #pragma unroll
                for (int j = -knn_radius; j <= knn_radius; j++) {
                    float clrIJ = (float)tex2D<Type>(tex, x + (float)j, y + (float)i) * (1.0f / (1<<bit_depth));
                    float distanceIJ = (center - clrIJ) * (center - clrIJ);

                    float weightIJ = __expf(-(distanceIJ * strength + (float)(i * i + j * j) * inv_knn_window_area + distT));

                    sum += clrIJ * weightIJ;

                    sumWeights += weightIJ;

                    fCount += (weightIJ > weight_threshold) ? inv_knn_window_area * inv_temporal_frames : 0;
                }
            }
        }
        float lerpQ = (fCount > lerp_threshold) ? lerpC : 1.0f - lerpC;

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(lerpf(sum * __frcp_rn(sumWeights), center, lerpQ) * (1<<bit_depth));
    }
}

template<typename Type, int bit_depth>
void denoise_knn(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texPrev2, cudaTextureObject_t texPrev1, cudaTextureObject_t texSrc, cudaTextureObject_t texNext1, cudaTextureObject_t texNext2,
    int radius, int temporal_d, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold,
    cudaStream_t stream) {
    //radius=5はよりレジスタを使うので、ブロック当たりのスレッド数を低減
    dim3 blockSize = (radius >= 5) ? dim3(32, 16) : dim3(64, 16);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
#define KNN_KERNEL(KNN_RADIUS, TEMPORAL_D) \
    case ((KNN_RADIUS) * 4 + (TEMPORAL_D)): \
        kernel_denoise_knn<Type, (KNN_RADIUS), (TEMPORAL_D), bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, \
            texPrev2, texPrev1, texSrc, texNext1, texNext2, \
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold); \
        break;
    switch (radius * 4 + temporal_d) {
    KNN_KERNEL(1, 0) KNN_KERNEL(1, 1) KNN_KERNEL(1, 2)
    KNN_KERNEL(2, 0) KNN_KERNEL(2, 1) KNN_KERNEL(2, 2)
    KNN_KERNEL(3, 0) KNN_KERNEL(3, 1) KNN_KERNEL(3, 2)
    KNN_KERNEL(4, 0) KNN_KERNEL(4, 1) KNN_KERNEL(4, 2)
    KNN_KERNEL(5, 0) KNN_KERNEL(5, 1) KNN_KERNEL(5, 2)
    default:
        break;
    }
#undef KNN_KERNEL
}

template<typename Type>
cudaError_t textureCreateDenoiseKnn(cudaTextureObject_t& tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = ptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = filterMode;
    texDesc.readMode         = readMode;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}

template<typename Type, int bit_depth>
static cudaError_t denoise_knn_plane(RGYFrameInfo *pOutputFrame, const std::array<const RGYFrameInfo *, 5> &pSrc,
    int radius, int temporal_d, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold,
    cudaStream_t stream) {
    //中央(pSrc[2])が現在フレーム、その前後temporal_dフレーム分のテクスチャを作成する
    //先頭/末尾でクランプされ同じフレームが渡された場合はテクスチャを共有する
    std::array<cudaTextureObject_t, 5> texSrc = { 0, 0, 0, 0, 0 };
    cudaError_t cudaerr = cudaSuccess;
    auto destroyTextures = [&]() {
        for (int t = 2 - temporal_d; t <= 2 + temporal_d; t++) {
            bool shared = false;
            for (int i = 2 - temporal_d; i < t; i++) {
                if (texSrc[i] == texSrc[t]) {
                    shared = true;
                    break;
                }
            }
            if (!shared && texSrc[t] != 0) {
                const auto err = cudaDestroyTextureObject(texSrc[t]);
                if (err != cudaSuccess) {
                    return err;
                }
            }
        }
        return cudaSuccess;
    };
    for (int t = 2 - temporal_d; t <= 2 + temporal_d; t++) {
        for (int i = 2 - temporal_d; i < t; i++) {
            if (pSrc[i]->ptr[0] == pSrc[t]->ptr[0]) {
                texSrc[t] = texSrc[i];
                break;
            }
        }
        if (texSrc[t] == 0) {
            cudaerr = textureCreateDenoiseKnn<Type>(texSrc[t], cudaFilterModePoint, cudaReadModeElementType, pSrc[t]->ptr[0], pSrc[t]->pitch[0], pSrc[t]->width, pSrc[t]->height);
            if (cudaerr != cudaSuccess) {
                destroyTextures();
                return cudaerr;
            }
        }
    }
    for (int t = 0; t < 5; t++) {
        //カーネルから参照されないスロットには現在フレームのテクスチャを入れておく
        if (texSrc[t] == 0) {
            texSrc[t] = texSrc[2];
        }
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        texSrc[0], texSrc[1], texSrc[2], texSrc[3], texSrc[4], radius, temporal_d, strength, lerpC, weight_threshold, lerp_threshold, stream);
    cudaerr = cudaGetLastError();
    const auto destroyerr = destroyTextures();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    if (destroyerr != cudaSuccess) {
        return destroyerr;
    }
    return cudaSuccess;
}

template<typename Type, int bit_depth>
static cudaError_t denoise_knn_frame(RGYFrameInfo *pOutputFrame, const std::array<const RGYFrameInfo *, 5> &pSrc,
    int radius, int temporal_d, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold,
    cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pSrc[2]->csp]; iplane++) {
        std::array<RGYFrameInfo, 5> planeSrc;
        std::array<const RGYFrameInfo *, 5> planeSrcPtr;
        for (int t = 0; t < 5; t++) {
            planeSrc[t] = getPlane(pSrc[t], (RGY_PLANE)iplane);
            planeSrcPtr[t] = &planeSrc[t];
        }
        auto planeOutput = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto cudaerr = denoise_knn_plane<Type, bit_depth>(&planeOutput, planeSrcPtr, radius, temporal_d, strength, lerpC, weight_threshold, lerp_threshold, stream);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    return cudaSuccess;
}

NVEncFilterDenoiseKnn::NVEncFilterDenoiseKnn() : m_bInterlacedWarn(false), m_prevFrames(), m_cacheIdx(0) {
    m_name = _T("knn");
}

NVEncFilterDenoiseKnn::~NVEncFilterDenoiseKnn() {
    close();
}

RGY_ERR NVEncFilterDenoiseKnn::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(pParam);
    if (!pKnnParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pKnnParam->frameOut.height <= 0 || pKnnParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius > KNN_RADIUS_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.d < 0 || pKnnParam->knn.d > KNN_TEMPORAL_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("d must be 0 - %d.\n"), KNN_TEMPORAL_MAX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.strength <= 0.0 || 1.0 < pKnnParam->knn.strength) {
        // 有効化されたフィルタで strength == 0 は意味がなく、
        // host 側の 1/(strength*strength) 計算で NaN フレームが出力される。
        AddMessage(RGY_LOG_ERROR, _T("strength should be greater than 0.0, up to 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerpC < 0.0 || 1.0 < pKnnParam->knn.lerpC) {
        AddMessage(RGY_LOG_ERROR, _T("lerpC should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerp_threshold < 0.0 || 1.0 < pKnnParam->knn.lerp_threshold) {
        AddMessage(RGY_LOG_ERROR, _T("th_lerp should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.weight_threshold < 0.0 || 1.0 < pKnnParam->knn.weight_threshold) {
        AddMessage(RGY_LOG_ERROR, _T("th_weight should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = AllocFrameBuf(pKnnParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pKnnParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (pKnnParam->knn.d > 0) {
        //convolution3dと同様に前後フレームをキャッシュし、dフレーム遅れで出力する
        const int cacheFrames = 2 * pKnnParam->knn.d + 1;
        if ((int)m_prevFrames.size() != cacheFrames
            || !m_prevFrames.front()
            || cmpFrameInfoCspResolution(&m_prevFrames.front()->frame, &pKnnParam->frameOut)) {
            m_prevFrames.clear();
            m_prevFrames.resize(cacheFrames);
            for (auto& f : m_prevFrames) {
                f.reset(new CUFrameBuf(pKnnParam->frameOut));
                f->releasePtr();
                sts = f->alloc();
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
                    return sts;
                }
            }
        }
        m_cacheIdx = 0;
        m_nFrameIdx = 0;
        //遅延が発生するため、タイムスタンプ等はフィルタ側で設定する
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    } else {
        m_prevFrames.clear();
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseKnn::print() const {
    return knn.print();
}

RGY_ERR NVEncFilterDenoiseKnn::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(m_param);
    if (!pKnnParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int temporal_d = pKnnParam->knn.d;

    if (pInputFrame->ptr[0] == nullptr
        && (temporal_d == 0 || m_nFrameIdx >= m_cacheIdx)) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(denoise_knn_frame<uint8_t, 8>)*> denoise_list = {
        { RGY_CSP_YV12,      denoise_knn_frame<uint8_t,   8> },
        { RGY_CSP_YV12_16,   denoise_knn_frame<uint16_t, 16> },
        { RGY_CSP_YUV444,    denoise_knn_frame<uint8_t,   8> },
        { RGY_CSP_YUV444_16, denoise_knn_frame<uint16_t, 16> },
    };

    if (temporal_d == 0) {
        //空間のみ(従来)のパス、遅延なし
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
        }
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        if (interlaced(*pInputFrame)) {
            return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
        }
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (denoise_list.count(pInputFrame->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        const std::array<const RGYFrameInfo *, 5> pSrc = { pInputFrame, pInputFrame, pInputFrame, pInputFrame, pInputFrame };
        sts = err_to_rgy(denoise_list.at(pInputFrame->csp)(ppOutputFrames[0], pSrc, pKnnParam->knn.radius, 0, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold, stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at knn(%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp],
                get_err_mes(sts));
            return sts;
        }
        return sts;
    }

    //temporal_d > 0: convolution3dと同様に前後フレームをキャッシュし、temporal_dフレーム遅れで出力する
    if (pInputFrame->ptr[0]) {
        if (interlaced(*pInputFrame)) {
            AddMessage(RGY_LOG_ERROR, _T("d > 0 does not support interlaced processing.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (denoise_list.count(pInputFrame->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        //sourceキャッシュにコピー
        auto cacheFrame = &m_prevFrames[m_cacheIdx % m_prevFrames.size()]->frame;
        sts = copyFrameAsync(cacheFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set frame to data cache: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFrameProp(cacheFrame, pInputFrame);
        m_cacheIdx++;
    }

    //出力するフレームの前後temporal_dフレームがそろうまでは出力しない
    if (pInputFrame->ptr[0] != nullptr && m_cacheIdx < m_nFrameIdx + temporal_d + 1) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }

    CUFrameBuf *pOutFrame = m_frameBuf[0].get();
    *pOutputFrameNum = 1;
    ppOutputFrames[0] = &pOutFrame->frame;

    //出力フレームの前後temporal_dフレームを集める(先頭/末尾はクランプ)
    std::array<const RGYFrameInfo *, 5> pSrc = { nullptr, nullptr, nullptr, nullptr, nullptr };
    for (int t = -2; t <= 2; t++) {
        const int idx = std::max(0, std::min(m_nFrameIdx + t, m_cacheIdx - 1));
        pSrc[t + 2] = &m_prevFrames[idx % m_prevFrames.size()]->frame;
    }
    const RGYFrameInfo *frameCur = pSrc[2];
    pOutFrame->frame.picstruct    = frameCur->picstruct;
    pOutFrame->frame.inputFrameId = frameCur->inputFrameId;
    pOutFrame->frame.duration     = frameCur->duration;
    pOutFrame->frame.timestamp    = frameCur->timestamp;
    pOutFrame->frame.flags        = frameCur->flags;
    pOutFrame->frame.dataList     = frameCur->dataList;

    sts = err_to_rgy(denoise_list.at(frameCur->csp)(&pOutFrame->frame, pSrc, pKnnParam->knn.radius, temporal_d, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold, stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at knn(%s): %s.\n"),
            RGY_CSP_NAMES[frameCur->csp],
            get_err_mes(sts));
        return sts;
    }
    m_nFrameIdx++;
    return sts;
}

void NVEncFilterDenoiseKnn::close() {
    m_frameBuf.clear();
    m_prevFrames.clear();
    m_cacheIdx = 0;
    m_bInterlacedWarn = false;
}
