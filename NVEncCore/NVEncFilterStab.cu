// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterStab.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int STAB_FFT_N = 256;
static const int STAB_FFT_LOG2_N = 8;
static const __device__  float STAB_PI_F = 3.14159265358979323846f;

static float stab_parabolic_refine(float a, float b, float c) {
    const float denom = a - 2.0f * b + c;
    if (std::abs(denom) < 1e-9f) return 0.0f;
    const float frac = 0.5f * (a - c) / denom;
    return clamp(frac, -0.5f, 0.5f);
}

__device__ __forceinline__ int stab_bitrev8(int x) {
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4);
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2);
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1);
    return x;
}

template<typename Type, int bit_depth>
__global__ void kernel_stab_luma_downsample(const uint8_t *src, const int srcPitch, const int width, const int height, float2 *dst) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= STAB_FFT_N || oy >= STAB_FFT_N) return;

    const float sx0f = (float)ox * (float)width / (float)STAB_FFT_N;
    const float sx1f = (float)(ox + 1) * (float)width / (float)STAB_FFT_N;
    const float sy0f = (float)oy * (float)height / (float)STAB_FFT_N;
    const float sy1f = (float)(oy + 1) * (float)height / (float)STAB_FFT_N;
    const int sx0 = (int)sx0f;
    const int sy0 = (int)sy0f;
    int sx1 = (int)sx1f; if (sx1 <= sx0) sx1 = sx0 + 1;
    int sy1 = (int)sy1f; if (sy1 <= sy0) sy1 = sy0 + 1;

    float sum = 0.0f;
    int count = 0;
    for (int sy = sy0; sy < sy1 && sy < height; sy++) {
        for (int sx = sx0; sx < sx1 && sx < width; sx++) {
            const auto ptr = (const Type *)(src + sy * srcPitch + sx * sizeof(Type));
            sum += (float)ptr[0];
            count++;
        }
    }
    static const int max_val = (1 << bit_depth) - 1;
    const float v = (count > 0) ? (sum / (float)count / (float)max_val) : 0.0f;
    dst[oy * STAB_FFT_N + ox] = make_float2(v, 0.0f);
}

__global__ void kernel_stab_fft_1d(const float2 *input, float2 *output, const int stride, const float direction) {
    __shared__ float2 sdata[STAB_FFT_N];
    const int tid = threadIdx.x;
    const int wg = blockIdx.x;
    const int base = (stride == 1) ? (wg * STAB_FFT_N) : wg;

    const int i0 = tid;
    const int i1 = tid + STAB_FFT_N / 2;
    sdata[stab_bitrev8(i0)] = input[base + i0 * stride];
    sdata[stab_bitrev8(i1)] = input[base + i1 * stride];
    __syncthreads();

    for (int s = 0; s < STAB_FFT_LOG2_N; s++) {
        const int bflyHalf = 1 << s;
        const int m = bflyHalf << 1;
        const int k = tid & (bflyHalf - 1);
        const int group = tid >> s;
        const int idx0 = (group << (s + 1)) + k;
        const int idx1 = idx0 + bflyHalf;
        const float2 c0 = sdata[idx0];
        const float2 c1 = sdata[idx1];
        const float angle = direction * (-2.0f * STAB_PI_F) * (float)k / (float)m;
        float ws, wc;
        sincosf(angle, &ws, &wc);
        const float2 t = make_float2(wc * c1.x - ws * c1.y, wc * c1.y + ws * c1.x);
        __syncthreads();
        sdata[idx0] = make_float2(c0.x + t.x, c0.y + t.y);
        sdata[idx1] = make_float2(c0.x - t.x, c0.y - t.y);
        __syncthreads();
    }

    const float norm = (direction < 0.0f) ? (1.0f / (float)STAB_FFT_N) : 1.0f;
    output[base + i0 * stride] = make_float2(sdata[i0].x * norm, sdata[i0].y * norm);
    output[base + i1 * stride] = make_float2(sdata[i1].x * norm, sdata[i1].y * norm);
}

__global__ void kernel_stab_cross_spectrum(const float2 *cur, const float2 *prev, float2 *out, const int total) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const float2 a = cur[gid];
    const float2 b = prev[gid];
    const float2 g = make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
    const float mag = sqrtf(g.x * g.x + g.y * g.y);
    out[gid] = (mag > 1e-12f) ? make_float2(g.x / mag, g.y / mag) : make_float2(0.0f, 0.0f);
}

template<typename Type>
__device__ __forceinline__ float stab_sample(const uint8_t *src, const int srcPitch, const int width, const int height,
    int x, int y, const int borderMode, const int fillValue) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        if (borderMode == VPP_STAB_BORDER_BLACK) {
            return (float)fillValue;
        } else if (borderMode == VPP_STAB_BORDER_CLAMP) {
            x = clamp(x, 0, width - 1);
            y = clamp(y, 0, height - 1);
        } else {
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = clamp(x, 0, width - 1);
            y = clamp(y, 0, height - 1);
        }
    }
    const auto ptr = (const Type *)(src + y * srcPitch + x * sizeof(Type));
    return (float)ptr[0];
}

template<typename Type, int bit_depth>
__global__ void kernel_stab_warp(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const float shiftX, const float shiftY, const int borderMode, const int fillValue) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= width || oy >= height) return;

    const float sxF = (float)ox + 0.5f - shiftX;
    const float syF = (float)oy + 0.5f - shiftY;
    const float sxc = sxF - 0.5f;
    const float syc = syF - 0.5f;
    const int sx0 = (int)floorf(sxc);
    const int sy0 = (int)floorf(syc);
    const float fx = sxc - (float)sx0;
    const float fy = syc - (float)sy0;

    const float v00 = stab_sample<Type>(src, srcPitch, width, height, sx0,     sy0,     borderMode, fillValue);
    const float v10 = stab_sample<Type>(src, srcPitch, width, height, sx0 + 1, sy0,     borderMode, fillValue);
    const float v01 = stab_sample<Type>(src, srcPitch, width, height, sx0,     sy0 + 1, borderMode, fillValue);
    const float v11 = stab_sample<Type>(src, srcPitch, width, height, sx0 + 1, sy0 + 1, borderMode, fillValue);
    const float v = v00 * (1.0f - fx) * (1.0f - fy)
        + v10 * fx * (1.0f - fy)
        + v01 * (1.0f - fx) * fy
        + v11 * fx * fy;
    static const int max_val = (1 << bit_depth) - 1;
    auto dstPix = (Type *)(dst + oy * dstPitch + ox * sizeof(Type));
    dstPix[0] = (Type)clamp((int)(v + 0.5f), 0, max_val);
}

template<typename Type, int bit_depth>
static RGY_ERR stab_downsample_y(const RGYFrameInfo *pInputFrame, CUMemBuf *srcReal, cudaStream_t stream) {
    const auto planeY = getPlane(pInputFrame, RGY_PLANE_Y);
    dim3 block(32, 8);
    dim3 grid(divCeil(STAB_FFT_N, block.x), divCeil(STAB_FFT_N, block.y));
    kernel_stab_luma_downsample<Type, bit_depth><<<grid, block, 0, stream>>>(planeY.ptr[0], planeY.pitch[0],
        planeY.width, planeY.height, (float2 *)srcReal->ptr);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR stab_downsample(const RGYFrameInfo *pInputFrame, CUMemBuf *srcReal, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return stab_downsample_y<uint16_t, 16>(pInputFrame, srcReal, stream);
    }
    return stab_downsample_y<uint8_t, 8>(pInputFrame, srcReal, stream);
}

static RGY_ERR stab_fft_1d(CUMemBuf *input, CUMemBuf *output, const int stride, const float direction, cudaStream_t stream) {
    kernel_stab_fft_1d<<<STAB_FFT_N, STAB_FFT_N / 2, 0, stream>>>((const float2 *)input->ptr, (float2 *)output->ptr, stride, direction);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR stab_cross(CUMemBuf *cur, CUMemBuf *prev, CUMemBuf *out, cudaStream_t stream) {
    const int total = STAB_FFT_N * STAB_FFT_N;
    const int block = 256;
    const int grid = divCeil(total, block);
    kernel_stab_cross_spectrum<<<grid, block, 0, stream>>>((const float2 *)cur->ptr, (const float2 *)prev->ptr, (float2 *)out->ptr, total);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

template<typename Type, int bit_depth>
static RGY_ERR stab_warp_plane(RGYFrameInfo *pOut, const RGYFrameInfo *pInputFrame, RGY_PLANE plane,
    float shiftX, float shiftY, int border, int fillValue, cudaStream_t stream) {
    const auto src = getPlane(pInputFrame, plane);
    auto dst = getPlane(pOut, plane);
    dim3 block(32, 8);
    dim3 grid(divCeil(src.width, block.x), divCeil(src.height, block.y));
    kernel_stab_warp<Type, bit_depth><<<grid, block, 0, stream>>>(src.ptr[0], src.pitch[0],
        dst.ptr[0], dst.pitch[0], src.width, src.height, shiftX, shiftY, border, fillValue);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR stab_warp_frame(RGYFrameInfo *pOut, const RGYFrameInfo *pInputFrame, const VppStab& prm,
    float shiftX, float shiftY, cudaStream_t stream) {
    const int planes = std::min<int>(RGY_CSP_PLANES[pInputFrame->csp], RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]);
    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const auto pY = getPlane(pInputFrame, RGY_PLANE_Y);
    int subX = 1, subY = 1;
    if (planes >= 2) {
        const auto pU = getPlane(pInputFrame, RGY_PLANE_U);
        if (pU.width > 0) subX = std::max(1, pY.width / pU.width);
        if (pU.height > 0) subY = std::max(1, pY.height / pU.height);
    }
    const int chromaFill = 1 << (bitDepth - 1);
    for (int i = 0; i < planes; i++) {
        const float planeShiftX = (i == 0) ? shiftX : (shiftX / (float)subX);
        const float planeShiftY = (i == 0) ? shiftY : (shiftY / (float)subY);
        const int fillValue = (i == 0) ? 0 : chromaFill;
        RGY_ERR sts = RGY_ERR_NONE;
        if (bitDepth > 8) {
            sts = stab_warp_plane<uint16_t, 16>(pOut, pInputFrame, (RGY_PLANE)i, planeShiftX, planeShiftY, prm.border, fillValue, stream);
        } else {
            sts = stab_warp_plane<uint8_t, 8>(pOut, pInputFrame, (RGY_PLANE)i, planeShiftX, planeShiftY, prm.border, fillValue, stream);
        }
        if (sts != RGY_ERR_NONE) return sts;
    }
    return copyPlaneAlphaAsync(pOut, pInputFrame, stream);
}

NVEncFilterStab::NVEncFilterStab() :
    m_srcReal(),
    m_curFreq(),
    m_prevFreq(),
    m_corrFreq(),
    m_corrReal(),
    m_corrHost(),
    m_havePrev(false),
    m_smoothShiftX(0.0f),
    m_smoothShiftY(0.0f),
    m_haveSmoothing(false),
    m_lowTrustFrames(0) {
    m_name = _T("stab");
}

NVEncFilterStab::~NVEncFilterStab() {
    close();
}

RGY_ERR NVEncFilterStab::checkParam(const std::shared_ptr<NVEncFilterParamStab> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->stab.strength = clamp(prm->stab.strength, 0.0f, 1.0f);
    prm->stab.damping = clamp(prm->stab.damping, 0.0f, 1.0f);
    prm->stab.trust_threshold = clamp(prm->stab.trust_threshold, 0.0f, 1.0f);
    prm->stab.max_shift = clamp(prm->stab.max_shift, 1.0f, 256.0f);
    if (prm->stab.border < VPP_STAB_BORDER_BLACK || prm->stab.border > VPP_STAB_BORDER_MIRROR) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid border mode: %d.\n"), prm->stab.border);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto csp = prm->frameIn.csp;
    const auto dataType = RGY_CSP_DATA_TYPE[csp];
    if (dataType != RGY_DATA_TYPE_U8 && dataType != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("stab requires 8-16bit integer input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterStab::allocWorkBuf(std::unique_ptr<CUMemBuf>& buf, size_t bytes, const TCHAR *label) {
    if (!buf || buf->nSize != bytes) {
        buf = std::make_unique<CUMemBuf>(bytes);
        const auto sts = buf->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("stab: failed to allocate %s buffer (%zu bytes): %s.\n"), label, bytes, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterStab::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamStab>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) return sts;

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) return RGY_ERR_MEMORY_ALLOC;
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const size_t fftBytes = (size_t)STAB_FFT_N * (size_t)STAB_FFT_N * sizeof(float2);
    if ((sts = allocWorkBuf(m_srcReal,  fftBytes, _T("srcReal")))  != RGY_ERR_NONE) return sts;
    if ((sts = allocWorkBuf(m_curFreq,  fftBytes, _T("curFreq")))  != RGY_ERR_NONE) return sts;
    if ((sts = allocWorkBuf(m_prevFreq, fftBytes, _T("prevFreq"))) != RGY_ERR_NONE) return sts;
    if ((sts = allocWorkBuf(m_corrFreq, fftBytes, _T("corrFreq"))) != RGY_ERR_NONE) return sts;
    if ((sts = allocWorkBuf(m_corrReal, fftBytes, _T("corrReal"))) != RGY_ERR_NONE) return sts;
    m_corrHost.assign((size_t)STAB_FFT_N * (size_t)STAB_FFT_N * 2, 0.0f);

    m_havePrev = false;
    m_smoothShiftX = 0.0f;
    m_smoothShiftY = 0.0f;
    m_haveSmoothing = false;
    m_lowTrustFrames = 0;

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamStab::print() const {
    return stab.print();
}

RGY_ERR NVEncFilterStab::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) return sts;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) return RGY_ERR_INVALID_PARAM;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamStab>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;

    if ((sts = stab_downsample(pInputFrame, m_srcReal.get(), stream)) != RGY_ERR_NONE) return sts;
    if ((sts = stab_fft_1d(m_srcReal.get(), m_curFreq.get(), 1, +1.0f, stream)) != RGY_ERR_NONE) return sts;
    if ((sts = stab_fft_1d(m_curFreq.get(), m_curFreq.get(), STAB_FFT_N, +1.0f, stream)) != RGY_ERR_NONE) return sts;

    bool haveCorrelation = false;
    int peakX = 0, peakY = 0;
    float peakValue = 0.0f, meanValue = 1.0f, refineX = 0.0f, refineY = 0.0f;
    if (m_havePrev) {
        if ((sts = stab_cross(m_curFreq.get(), m_prevFreq.get(), m_corrFreq.get(), stream)) != RGY_ERR_NONE) return sts;
        if ((sts = stab_fft_1d(m_corrFreq.get(), m_corrReal.get(), 1, -1.0f, stream)) != RGY_ERR_NONE) return sts;
        if ((sts = stab_fft_1d(m_corrReal.get(), m_corrReal.get(), STAB_FFT_N, -1.0f, stream)) != RGY_ERR_NONE) return sts;
        auto cudaerr = cudaMemcpyAsync(m_corrHost.data(), m_corrReal->ptr, m_corrReal->nSize, cudaMemcpyDeviceToHost, stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        cudaerr = cudaStreamSynchronize(stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

        const int N = STAB_FFT_N;
        float pk = m_corrHost[0];
        int pi = 0;
        double sum = 0.0;
        for (int i = 0; i < N * N; i++) {
            const float v = m_corrHost[(size_t)i * 2];
            sum += (double)v;
            if (v > pk) {
                pk = v;
                pi = i;
            }
        }
        peakX = pi % N;
        peakY = pi / N;
        peakValue = pk;
        meanValue = (float)(sum / (double)(N * N));
        auto getPx = [&](int x, int y) -> float {
            if (x < 0) x += N; else if (x >= N) x -= N;
            if (y < 0) y += N; else if (y >= N) y -= N;
            return m_corrHost[((size_t)y * (size_t)N + (size_t)x) * 2];
        };
        refineX = stab_parabolic_refine(getPx(peakX - 1, peakY), getPx(peakX, peakY), getPx(peakX + 1, peakY));
        refineY = stab_parabolic_refine(getPx(peakX, peakY - 1), getPx(peakX, peakY), getPx(peakX, peakY + 1));
        haveCorrelation = true;
    }

    float rawShiftX = 0.0f, rawShiftY = 0.0f, trust = 0.0f;
    if (haveCorrelation) {
        int sx = peakX;
        int sy = peakY;
        if (sx > STAB_FFT_N / 2) sx -= STAB_FFT_N;
        if (sy > STAB_FFT_N / 2) sy -= STAB_FFT_N;
        rawShiftX = ((float)sx + refineX) * (float)prm->frameOut.width / (float)STAB_FFT_N;
        rawShiftY = ((float)sy + refineY) * (float)prm->frameOut.height / (float)STAB_FFT_N;
        trust = (meanValue > 1e-12f) ? (peakValue / meanValue) : 0.0f;
        const float trustNorm = trust / (trust + 100.0f);
        rawShiftX = clamp(rawShiftX, -prm->stab.max_shift, prm->stab.max_shift);
        rawShiftY = clamp(rawShiftY, -prm->stab.max_shift, prm->stab.max_shift);
        if (trustNorm < prm->stab.trust_threshold) {
            m_lowTrustFrames++;
        } else if (!m_haveSmoothing) {
            m_smoothShiftX = rawShiftX;
            m_smoothShiftY = rawShiftY;
            m_haveSmoothing = true;
        } else {
            const float d = prm->stab.damping;
            m_smoothShiftX = d * m_smoothShiftX + (1.0f - d) * rawShiftX;
            m_smoothShiftY = d * m_smoothShiftY + (1.0f - d) * rawShiftY;
        }
    }

    if (m_haveSmoothing) {
        sts = stab_warp_frame(ppOutputFrames[0], pInputFrame, prm->stab,
            prm->stab.strength * m_smoothShiftX, prm->stab.strength * m_smoothShiftY, stream);
    } else {
        sts = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    }
    if (sts != RGY_ERR_NONE) return sts;

    std::swap(m_curFreq, m_prevFreq);
    m_havePrev = true;

    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = pInputFrame->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    ppOutputFrames[0]->flags = pInputFrame->flags;
    return RGY_ERR_NONE;
}

void NVEncFilterStab::close() {
    if (m_lowTrustFrames > 0) {
        AddMessage(RGY_LOG_INFO, _T("stab: rejected %d low-trust frames during analysis.\n"), m_lowTrustFrames);
    }
    m_srcReal.reset();
    m_curFreq.reset();
    m_prevFreq.reset();
    m_corrFreq.reset();
    m_corrReal.reset();
    m_corrHost.clear();
    m_havePrev = false;
    m_smoothShiftX = 0.0f;
    m_smoothShiftY = 0.0f;
    m_haveSmoothing = false;
    m_lowTrustFrames = 0;
    m_frameBuf.clear();
}
