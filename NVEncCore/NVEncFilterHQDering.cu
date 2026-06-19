// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterHQDering.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int HQDERING_BLOCK_X = 32;
static const int HQDERING_BLOCK_Y = 8;
static const int HQDERING_KERNEL_RADIUS_MAX = 10;

template<typename Type>
__device__ __forceinline__ int hqd_read_pix_clamp(const uint8_t *frame, const int pitch, const int width, const int height, int x, int y) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const auto ptr = (const Type *)(frame + y * pitch + x * sizeof(Type));
    return (int)ptr[0];
}

template<int bit_depth>
__device__ __forceinline__ int hqd_levels_ramp(const int g, const int mthrHbd) {
    static const int max_val = (1 << bit_depth) - 1;
    if (max_val > mthrHbd) {
        long long num = (long long)(g - mthrHbd) * (long long)max_val;
        return clamp((int)(num / (long long)(max_val - mthrHbd)), 0, max_val);
    }
    return (g >= mthrHbd) ? max_val : 0;
}

template<typename Type, int bit_depth>
__global__ void kernel_hqdering_edge(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int mthrHbd, const int edgeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int tl = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y - 1);
    const int tc = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y - 1);
    const int tr = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y - 1);
    const int cl = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y);
    const int cc = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y);
    const int cr = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y);
    const int bl = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y + 1);
    const int bc = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y + 1);
    const int br = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y + 1);

    int g = 0;
    if (edgeMode == 1) {
        const int gx = -tl - 2 * cl - bl + tr + 2 * cr + br;
        const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        g = abs(gx) + abs(gy);
    } else if (edgeMode == 2) {
        const int gx = (-tl + tr) + (-cl + cr) + (-bl + br);
        const int gy = (-tl - tc - tr) + (bl + bc + br);
        g = ((abs(gx) + abs(gy)) * 4) / 3;
    } else if (edgeMode == 3) {
        const int gx = -3 * tl + 3 * tr - 10 * cl + 10 * cr - 3 * bl + 3 * br;
        const int gy = -3 * tl - 10 * tc - 3 * tr + 3 * bl + 10 * bc + 3 * br;
        g = (abs(gx) + abs(gy)) / 4;
    } else if (edgeMode == 4) {
        const int n  =  5 * (tl + tc + tr) - 3 * (cl + cr + bl + bc + br);
        const int ne =  5 * (tc + tr + cr) - 3 * (tl + cl + bl + bc + br);
        const int e  =  5 * (tr + cr + br) - 3 * (tl + tc + cl + bl + bc);
        const int se =  5 * (cr + br + bc) - 3 * (tl + tc + tr + cl + bl);
        const int s  =  5 * (bl + bc + br) - 3 * (tl + tc + tr + cl + cr);
        const int sw =  5 * (cl + bl + bc) - 3 * (tl + tc + tr + cr + br);
        const int w  =  5 * (tl + cl + bl) - 3 * (tc + tr + cr + bc + br);
        const int nw =  5 * (tl + tc + cl) - 3 * (tr + cr + bl + bc + br);
        int m = max(max(max(n, ne), max(e, se)), max(max(s, sw), max(w, nw)));
        g = (max(m, 0) * 8) / 15;
    } else if (edgeMode == 5) {
        g = abs(4 * cc - tc - cl - cr - bc) * 2;
    } else {
        const int n2c = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y - 2);
        const int n1l = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y - 1);
        const int n1c = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y - 1);
        const int n1r = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y - 1);
        const int c2l = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 2, y);
        const int c1l = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y);
        const int ccc = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y);
        const int c1r = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y);
        const int c2r = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 2, y);
        const int s1l = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y + 1);
        const int s1c = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y + 1);
        const int s1r = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y + 1);
        const int s2c = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y + 2);
        const int logResp = 16 * ccc
            - (n2c + c2l + c2r + s2c)
            - 2 * (n1c + c1l + c1r + s1c)
            - (n1l + n1r + s1l + s1r);
        g = abs(logResp) / 2;
    }

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)hqd_levels_ramp<bit_depth>(g, mthrHbd);
}

template<typename Type>
__global__ void kernel_hqdering_expand3x3(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int m = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            m = max(m, hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy));
        }
    }
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_hqdering_blur_h(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int radius, const float sigma) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);
    float vSum = 0.0f;
    float wSum = 0.0f;
    for (int i = -HQDERING_KERNEL_RADIUS_MAX; i <= HQDERING_KERNEL_RADIUS_MAX; i++) {
        if (i < -radius || i > radius) continue;
        const float wi = expf(-(float)(i * i) * invTwoSigmaSq);
        vSum += (float)hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x + i, y) * wi;
        wSum += wi;
    }
    const int v = clamp((int)(vSum / wSum + 0.5f), 0, max_val);
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)v;
}

template<typename Type, int bit_depth>
__global__ void kernel_hqdering_blur_v(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int radius, const float sigma) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);
    float vSum = 0.0f;
    float wSum = 0.0f;
    for (int i = -HQDERING_KERNEL_RADIUS_MAX; i <= HQDERING_KERNEL_RADIUS_MAX; i++) {
        if (i < -radius || i > radius) continue;
        const float wi = expf(-(float)(i * i) * invTwoSigmaSq);
        vSum += (float)hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x, y + i) * wi;
        wSum += wi;
    }
    const int v = clamp((int)(vSum / wSum + 0.5f), 0, max_val);
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)v;
}

template<typename Type, int bit_depth>
__global__ void kernel_hqdering_combine(const uint8_t *src, const int srcPitch, const uint8_t *blurred, const int blurredPitch,
    const uint8_t *mask, const int maskPitch, const uint8_t *edgeMask, const int edgeMaskPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height, const int showmask, const int protect) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const int s = hqd_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const int b = hqd_read_pix_clamp<Type>(blurred, blurredPitch, width, height, x, y);
    const int mk = hqd_read_pix_clamp<Type>(mask, maskPitch, width, height, x, y);
    int effectiveMask = mk;
    if (protect) {
        effectiveMask = max(mk - hqd_read_pix_clamp<Type>(edgeMask, edgeMaskPitch, width, height, x, y), 0);
    }
    int out = effectiveMask;
    if (!showmask) {
        const float m = (float)effectiveMask / (float)max_val;
        out = clamp((int)((float)s + ((float)b - (float)s) * m + 0.5f), 0, max_val);
    }
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

static int hqdering_edge_mode(const tstring& edge) {
    if (edge == _T("sobel")) return 1;
    if (edge == _T("prewitt")) return 2;
    if (edge == _T("scharr")) return 3;
    if (edge == _T("kirsch")) return 4;
    if (edge == _T("laplacian")) return 5;
    return 0;
}

template<typename Type, int bit_depth>
static RGY_ERR hqdering_process_y_typed(RGYFrameInfo *pOut, const RGYFrameInfo *pInput,
    RGYFrameInfo *pEdgeMask, RGYFrameInfo *pRingMask, RGYFrameInfo *pMorphTmp,
    RGYFrameInfo *pHBlurred, RGYFrameInfo *pBlurred,
    const VppDering& prm, const int mthrHbd, const int kernelRadius, cudaStream_t stream) {
    dim3 blockSize(HQDERING_BLOCK_X, HQDERING_BLOCK_Y);
    dim3 gridSize(divCeil(pInput->width, blockSize.x), divCeil(pInput->height, blockSize.y));
    const int edgeMode = hqdering_edge_mode(prm.edge);
    const auto width = pInput->width;
    const auto height = pInput->height;

    kernel_hqdering_edge<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInput->ptr[0], pInput->pitch[0],
        pEdgeMask->ptr[0], pEdgeMask->pitch[0], width, height, mthrHbd, edgeMode);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    const RGYFrameInfo *pIn = pEdgeMask;
    RGYFrameInfo *pRing = pEdgeMask;
    for (int i = 0; i < prm.mrad; i++) {
        RGYFrameInfo *pMorphOut = ((i & 1) == 0) ? pRingMask : pMorphTmp;
        kernel_hqdering_expand3x3<Type><<<gridSize, blockSize, 0, stream>>>(pIn->ptr[0], pIn->pitch[0],
            pMorphOut->ptr[0], pMorphOut->pitch[0], width, height);
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        pIn = pMorphOut;
        pRing = pMorphOut;
    }

    kernel_hqdering_blur_h<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInput->ptr[0], pInput->pitch[0],
        pHBlurred->ptr[0], pHBlurred->pitch[0], width, height, kernelRadius, prm.sigma);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_hqdering_blur_v<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pHBlurred->ptr[0], pHBlurred->pitch[0],
        pBlurred->ptr[0], pBlurred->pitch[0], width, height, kernelRadius, prm.sigma);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_hqdering_combine<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInput->ptr[0], pInput->pitch[0],
        pBlurred->ptr[0], pBlurred->pitch[0], pRing->ptr[0], pRing->pitch[0],
        pEdgeMask->ptr[0], pEdgeMask->pitch[0], pOut->ptr[0], pOut->pitch[0],
        width, height, prm.showmask ? 1 : 0, prm.protect ? 1 : 0);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    return RGY_ERR_NONE;
}

static RGY_ERR hqdering_process_y(RGYFrameInfo *pOut, const RGYFrameInfo *pInput,
    RGYFrameInfo *pEdgeMask, RGYFrameInfo *pRingMask, RGYFrameInfo *pMorphTmp,
    RGYFrameInfo *pHBlurred, RGYFrameInfo *pBlurred,
    const VppDering& prm, const int mthrHbd, const int kernelRadius, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInput->csp] > 8) {
        return hqdering_process_y_typed<uint16_t, 16>(pOut, pInput, pEdgeMask, pRingMask, pMorphTmp, pHBlurred, pBlurred, prm, mthrHbd, kernelRadius, stream);
    }
    return hqdering_process_y_typed<uint8_t, 8>(pOut, pInput, pEdgeMask, pRingMask, pMorphTmp, pHBlurred, pBlurred, prm, mthrHbd, kernelRadius, stream);
}

NVEncFilterHQDering::NVEncFilterHQDering() :
    m_edgeMask(),
    m_ringMask(),
    m_morphTmp(),
    m_hBlurred(),
    m_blurred() {
    m_name = _T("hqdering");
}

NVEncFilterHQDering::~NVEncFilterHQDering() {
    close();
}

RGY_ERR NVEncFilterHQDering::checkParam(const std::shared_ptr<NVEncFilterParamHQDering> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("hqdering requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto csp = prm->frameIn.csp;
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[csp];
    const auto dataType = RGY_CSP_DATA_TYPE[csp];
    if (dataType != RGY_DATA_TYPE_U8 && dataType != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("hqdering requires 8-16bit integer input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (chromaFormat != RGY_CHROMAFMT_YUV420
        && chromaFormat != RGY_CHROMAFMT_YUV422
        && chromaFormat != RGY_CHROMAFMT_YUV444
        && chromaFormat != RGY_CHROMAFMT_MONOCHROME) {
        AddMessage(RGY_LOG_ERROR, _T("hqdering requires YUV or monochrome input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_PLANES[csp] <= 1 && chromaFormat != RGY_CHROMAFMT_MONOCHROME) {
        AddMessage(RGY_LOG_ERROR, _T("hqdering does not support packed YUV input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.mrad < 1 || prm->dering.mrad > 3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mrad=%d: must be in [1, 3].\n"), prm->dering.mrad);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.mthr < 0 || prm->dering.mthr > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mthr=%d: must be in [0, 255].\n"), prm->dering.mthr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dering.sigma >= 0.5f && prm->dering.sigma <= 5.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid sigma=%.2f: must be in [0.5, 5.0].\n"), prm->dering.sigma);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.edge != _T("log")
        && prm->dering.edge != _T("sobel")
        && prm->dering.edge != _T("prewitt")
        && prm->dering.edge != _T("scharr")
        && prm->dering.edge != _T("kirsch")
        && prm->dering.edge != _T("laplacian")) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid edge=%s: must be log|sobel|prewitt|scharr|kirsch|laplacian.\n"),
            prm->dering.edge.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterHQDering::allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label) {
    if (!frame || frame->frame.width != frameInfo.width || frame->frame.height != frameInfo.height || frame->frame.csp != frameInfo.csp) {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        frame->releasePtr();
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate hqdering %s buffer: %s.\n"), label, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterHQDering::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamHQDering>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) return sts;

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) return RGY_ERR_MEMORY_ALLOC;
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    sts = allocWorkFrame(m_edgeMask, prm->frameIn, _T("edgeMask"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_ringMask, prm->frameIn, _T("ringMask"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_morphTmp, prm->frameIn, _T("morphTmp"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_hBlurred, prm->frameIn, _T("hBlurred"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_blurred, prm->frameIn, _T("blurred"));
    if (sts != RGY_ERR_NONE) return sts;

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

tstring NVEncFilterParamHQDering::print() const {
    return dering.print();
}

RGY_ERR NVEncFilterHQDering::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) return sts;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) return RGY_ERR_INVALID_PARAM;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamHQDering>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int mthrHbd = (int)((long long)prm->dering.mthr * maxVal / 255);
    int kernelRadius = (int)std::ceil(2.0f * prm->dering.sigma);
    kernelRadius = clamp(kernelRadius, 1, HQDERING_KERNEL_RADIUS_MAX);

    auto planeOutY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    sts = hqdering_process_y(&planeOutY, pInputFrame,
        &m_edgeMask->frame, &m_ringMask->frame, &m_morphTmp->frame, &m_hBlurred->frame, &m_blurred->frame,
        prm->dering, mthrHbd, kernelRadius, stream);
    if (sts != RGY_ERR_NONE) return sts;

    const int copyPlanes = std::min<int>(RGY_CSP_PLANES[pInputFrame->csp], RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]);
    for (int iplane = 1; iplane < copyPlanes; iplane++) {
        const auto planeInput = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(ppOutputFrames[0], (RGY_PLANE)iplane);
        sts = copyPlaneAsync(&planeOutput, &planeInput, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return copyPlaneAlphaAsync(ppOutputFrames[0], pInputFrame, stream);
}

void NVEncFilterHQDering::close() {
    m_edgeMask.reset();
    m_ringMask.reset();
    m_morphTmp.reset();
    m_hBlurred.reset();
    m_blurred.reset();
    m_frameBuf.clear();
}
