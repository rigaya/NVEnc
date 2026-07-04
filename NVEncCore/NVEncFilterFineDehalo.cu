// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "convert_csp.h"
#include "NVEncFilterFineDehalo.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int FINEDEHALO_BLOCK_X = 32;
static const int FINEDEHALO_BLOCK_Y = 8;
enum {
    FINEDEHALO_MORPH_SQUARE = 0,
    FINEDEHALO_MORPH_BOTH = 1,
    FINEDEHALO_MORPH_HORIZONTAL = 2,
    FINEDEHALO_MORPH_VERTICAL = 3,
};

template<typename Type>
__device__ __forceinline__ int fdh_read_pix_clamp(const uint8_t *frame, const int pitch, const int width, const int height, int x, int y) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const auto ptr = (const Type *)(frame + y * pitch + x * sizeof(Type));
    return (int)ptr[0];
}

template<typename Type, int bit_depth>
__device__ __forceinline__ int fdh_ramp(const int v, const int lo, const int hi) {
    static const int max_val = (1 << bit_depth) - 1;
    if (hi > lo) {
        if (v <= lo) return 0;
        if (v >= hi) return max_val;
        long long num = (long long)(v - lo) * (long long)max_val;
        const long long den = (long long)(hi - lo);
        return clamp((int)((num + den / 2) / den), 0, max_val);
    }
    return (v >= lo) ? max_val : 0;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_edge_raw(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int edgeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;

    const int tl = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y - 1);
    const int tc = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y - 1);
    const int tr = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y - 1);
    const int cl = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y);
    const int cr = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y);
    const int bl = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x - 1, y + 1);
    const int bc = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x,     y + 1);
    const int br = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x + 1, y + 1);

    int g = 0;
    if (edgeMode == 1) {
        const int gx = -3 * tl + 3 * tr - 10 * cl + 10 * cr - 3 * bl + 3 * br;
        const int gy = -3 * tl - 10 * tc - 3 * tr + 3 * bl + 10 * bc + 3 * br;
        g = (abs(gx) + abs(gy)) / 4;
    } else if (edgeMode == 2) {
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
    } else if (edgeMode == 3) {
        const int cc = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
        g = abs(4 * cc - tc - cl - cr - bc) * 2;
    } else if (edgeMode == 4) {
        const int gx = -tl - 2 * cl - bl + tr + 2 * cr + br;
        const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        g = abs(gx) + abs(gy);
    } else {
        const int p90 = tl + tc + tr - bl - bc - br;
        const int p180 = tl + cl + bl - tr - cr - br;
        const int p45 = cl + tl + tc - br - cr - bc;
        const int p135 = bl + cl + bc - tr - cr - tc;
        g = max(max(abs(p90), abs(p180)), max(abs(p45), abs(p135)));
    }

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(g, 0, max_val);
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_ramp_mask(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int lo, const int hi) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int v = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const int out = fdh_ramp<Type, bit_depth>(v, lo, hi);
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type>
__global__ void kernel_fdh_morph_3x3(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int mode, const bool expand) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int m = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const bool use = (mode == FINEDEHALO_MORPH_SQUARE)
                || (mode == FINEDEHALO_MORPH_BOTH && (dx == 0 || dy == 0))
                || (mode == FINEDEHALO_MORPH_HORIZONTAL && dy == 0)
                || (mode == FINEDEHALO_MORPH_VERTICAL && dx == 0);
            if (!use) continue;
            const int v = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
            m = expand ? max(m, v) : min(m, v);
        }
    }
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_mul_clamp(const uint8_t *src, const int srcPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height, const float mul) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const int v = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const int out = clamp((int)((float)v * mul + 0.5f), 0, max_val);
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_removegrain20_approx(const uint8_t *src, const int srcPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
        dstPix[0] = (Type)fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
        return;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
        }
    }
    const int out = (sum + 4) / 9;
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_shr_med(const uint8_t *strong, const int strongPitch, const uint8_t *shrink, const int shrinkPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height, const bool excl) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int s = fdh_read_pix_clamp<Type>(strong, strongPitch, width, height, x, y);
    const int out = excl ? max(s, fdh_read_pix_clamp<Type>(shrink, shrinkPitch, width, height, x, y)) : s;
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_outside(const uint8_t *large, const int largePitch, const uint8_t *shrMed, const int shrMedPitch,
    const uint8_t *strong, const int strongPitch, uint8_t *dst, const int dstPitch, const int width, const int height,
    const float edgeproc) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const int largePix = fdh_read_pix_clamp<Type>(large, largePitch, width, height, x, y);
    const int shrMedPix = fdh_read_pix_clamp<Type>(shrMed, shrMedPitch, width, height, x, y);
    const int strongPix = fdh_read_pix_clamp<Type>(strong, strongPitch, width, height, x, y);
    const float edgeAdd = (edgeproc > 0.0f) ? (float)strongPix * edgeproc * 0.66f : 0.0f;
    const int out = clamp((int)((float)max(largePix - shrMedPix, 0) * 2.0f + edgeAdd + 0.5f), 0, max_val);
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_combine(const uint8_t *src, const int srcPitch, const uint8_t *dehaloed, const int dehaloedPitch,
    const uint8_t *outside, const int outsidePitch, uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const int s = fdh_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const int d = fdh_read_pix_clamp<Type>(dehaloed, dehaloedPitch, width, height, x, y);
    const int mask = fdh_read_pix_clamp<Type>(outside, outsidePitch, width, height, x, y);
    const float m = (float)mask / (float)max_val;
    const int out = clamp((int)((float)s + ((float)d - (float)s) * m + 0.5f), 0, max_val);
    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

static int fdh_edge_mode(const tstring& edge) {
    if (edge == _T("scharr")) return 1;
    if (edge == _T("kirsch")) return 2;
    if (edge == _T("laplacian")) return 3;
    if (edge == _T("sobel")) return 4;
    return 0;
}

static int fdh_morph_multi_mode(const int sw, const int sh, const bool ellipse) {
    if (sw > 0 && sh > 0) {
        return (ellipse && (sw % 3) != 1) ? FINEDEHALO_MORPH_BOTH : FINEDEHALO_MORPH_SQUARE;
    }
    return (sw > 0) ? FINEDEHALO_MORPH_HORIZONTAL : FINEDEHALO_MORPH_VERTICAL;
}

template<typename Type>
static RGY_ERR fdh_morph_multi(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGYFrameInfo *pTmp,
    const int width, const int height, const int rx, const int ry, const bool expand, const bool ellipse,
    const dim3 gridSize, const dim3 blockSize, cudaStream_t stream) {
    const int iter = std::max(rx, ry);
    const RGYFrameInfo *pCur = pSrc;
    for (int i = 0; i < iter; i++) {
        const int sw = std::max(rx - i, 0);
        const int sh = std::max(ry - i, 0);
        const int mode = fdh_morph_multi_mode(sw, sh, ellipse);
        RGYFrameInfo *pNext = nullptr;
        if (i == iter - 1) {
            pNext = pDst;
        } else {
            const bool firstToDst = (iter & 1) != 0;
            const bool useDst = firstToDst ? ((i & 1) == 0) : ((i & 1) != 0);
            pNext = useDst ? pDst : pTmp;
        }
        kernel_fdh_morph_3x3<Type><<<gridSize, blockSize, 0, stream>>>(pCur->ptr[0], pCur->pitch[0],
            pNext->ptr[0], pNext->pitch[0], width, height, mode, expand);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        pCur = pNext;
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR finedehalo_process_y_typed(RGYFrameInfo *pOut, const RGYFrameInfo *pInput, const RGYFrameInfo *pDehaloed,
    RGYFrameInfo *pEdges, RGYFrameInfo *pStrong, RGYFrameInfo *pLarge, RGYFrameInfo *pLight, RGYFrameInfo *pShrink,
    RGYFrameInfo *pOutside, RGYFrameInfo *pMorphTmp, RGYFrameInfo *pShrMed,
    const VppFineDehalo& prm, const int thmi, const int thma, const int thlimi, const int thlima, cudaStream_t stream) {
    dim3 blockSize(FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInput->width, blockSize.x), divCeil(pInput->height, blockSize.y));
    const int edgeMode = fdh_edge_mode(prm.edge);
    const auto width = pInput->width;
    const auto height = pInput->height;
    const float absRx = (prm.rx >= 0.0f) ? prm.rx : -prm.rx;
    const float absRy = (prm.ry >= 0.0f) ? prm.ry : -prm.ry;
    const int rx = std::max(1, (int)(absRx + 0.5f));
    const int ry = std::max(1, (int)(absRy + 0.5f));

    kernel_fdh_edge_raw<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInput->ptr[0], pInput->pitch[0],
        pEdges->ptr[0], pEdges->pitch[0], width, height, edgeMode);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_ramp_mask<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pEdges->ptr[0], pEdges->pitch[0],
        pStrong->ptr[0], pStrong->pitch[0], width, height, thmi, thma);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_ramp_mask<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pEdges->ptr[0], pEdges->pitch[0],
        pLight->ptr[0], pLight->pitch[0], width, height, thlimi, thlima);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    auto sts = fdh_morph_multi<Type>(pLarge, pStrong, pMorphTmp, width, height, rx, ry, true, false, gridSize, blockSize, stream);
    if (sts != RGY_ERR_NONE) return sts;
    sts = fdh_morph_multi<Type>(pShrink, pLight, pMorphTmp, width, height, rx, ry, true, true, gridSize, blockSize, stream);
    if (sts != RGY_ERR_NONE) return sts;
    kernel_fdh_mul_clamp<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pShrink->ptr[0], pShrink->pitch[0],
        pMorphTmp->ptr[0], pMorphTmp->pitch[0], width, height, 4.0f);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    sts = fdh_morph_multi<Type>(pShrink, pMorphTmp, pLight, width, height, rx, ry, false, true, gridSize, blockSize, stream);
    if (sts != RGY_ERR_NONE) return sts;
    kernel_fdh_removegrain20_approx<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pShrink->ptr[0], pShrink->pitch[0],
        pMorphTmp->ptr[0], pMorphTmp->pitch[0], width, height);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_removegrain20_approx<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pMorphTmp->ptr[0], pMorphTmp->pitch[0],
        pShrink->ptr[0], pShrink->pitch[0], width, height);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_shr_med<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pStrong->ptr[0], pStrong->pitch[0],
        pShrink->ptr[0], pShrink->pitch[0], pShrMed->ptr[0], pShrMed->pitch[0], width, height, prm.excl);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_outside<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pLarge->ptr[0], pLarge->pitch[0],
        pShrMed->ptr[0], pShrMed->pitch[0], pStrong->ptr[0], pStrong->pitch[0],
        pOutside->ptr[0], pOutside->pitch[0], width, height, prm.edgeproc);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    if (prm.showmask == 1) {
        return copyPlaneAsync(pOut, pOutside, stream);
    }
    kernel_fdh_removegrain20_approx<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pOutside->ptr[0], pOutside->pitch[0],
        pShrMed->ptr[0], pShrMed->pitch[0], width, height);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_mul_clamp<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pShrMed->ptr[0], pShrMed->pitch[0],
        pOutside->ptr[0], pOutside->pitch[0], width, height, 2.0f);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    if (prm.showmask == 2) {
        return copyPlaneAsync(pOut, pShrink, stream);
    } else if (prm.showmask == 3) {
        return copyPlaneAsync(pOut, pEdges, stream);
    } else if (prm.showmask == 4) {
        return copyPlaneAsync(pOut, pStrong, stream);
    }
    kernel_fdh_combine<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInput->ptr[0], pInput->pitch[0],
        pDehaloed->ptr[0], pDehaloed->pitch[0], pOutside->ptr[0], pOutside->pitch[0],
        pOut->ptr[0], pOut->pitch[0], width, height);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    return RGY_ERR_NONE;
}

static RGY_ERR finedehalo_process_y(RGYFrameInfo *pOut, const RGYFrameInfo *pInput, const RGYFrameInfo *pDehaloed,
    RGYFrameInfo *pEdges, RGYFrameInfo *pStrong, RGYFrameInfo *pLarge, RGYFrameInfo *pLight, RGYFrameInfo *pShrink,
    RGYFrameInfo *pOutside, RGYFrameInfo *pMorphTmp, RGYFrameInfo *pShrMed,
    const VppFineDehalo& prm, const int thmi, const int thma, const int thlimi, const int thlima, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInput->csp] > 8) {
        return finedehalo_process_y_typed<uint16_t, 16>(pOut, pInput, pDehaloed, pEdges, pStrong, pLarge, pLight, pShrink, pOutside, pMorphTmp, pShrMed, prm, thmi, thma, thlimi, thlima, stream);
    }
    return finedehalo_process_y_typed<uint8_t, 8>(pOut, pInput, pDehaloed, pEdges, pStrong, pLarge, pLight, pShrink, pOutside, pMorphTmp, pShrMed, prm, thmi, thma, thlimi, thlima, stream);
}

NVEncFilterFineDehalo::NVEncFilterFineDehalo() :
    m_dehalo(),
    m_edges(),
    m_strong(),
    m_large(),
    m_light(),
    m_shrink(),
    m_outside(),
    m_morphTmp(),
    m_shrMed() {
    m_name = _T("finedehalo");
}

NVEncFilterFineDehalo::~NVEncFilterFineDehalo() {
    close();
}

RGY_ERR NVEncFilterFineDehalo::checkParam(const std::shared_ptr<NVEncFilterParamFineDehalo> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo requires input width/height >= 4 (got %dx%d).\n"), prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (interlaced(prm->frameIn)) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo does not support interlaced input. Please deinterlace before finedehalo.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto &p = prm->finedehalo;
    if (!(p.rx >= 0.5f && p.rx <= 10.0f) || !(p.ry >= 0.5f && p.ry <= 10.0f)) return RGY_ERR_INVALID_PARAM;
    if (!(p.darkstr >= 0.0f && p.darkstr <= 1.0f) || !(p.brightstr >= 0.0f && p.brightstr <= 1.0f)) return RGY_ERR_INVALID_PARAM;
    if (p.lowsens < 0 || p.lowsens > 100 || p.highsens < 0 || p.highsens > 100) return RGY_ERR_INVALID_PARAM;
    if (!(p.ss >= 1.0f && p.ss <= 4.0f)) return RGY_ERR_INVALID_PARAM;
    if (!((p.searchRade == FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO) || (p.searchRade >= 1 && p.searchRade <= 10))) return RGY_ERR_INVALID_PARAM;
    if (!((p.searchRadi == FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO) || (p.searchRadi >= 1 && p.searchRadi <= 10))) return RGY_ERR_INVALID_PARAM;
    if (p.thmi < 0 || p.thmi > 255 || p.thma < 0 || p.thma > 255) return RGY_ERR_INVALID_PARAM;
    if (p.thlimi < 0 || p.thlimi > 255 || p.thlima < 0 || p.thlima > 255) return RGY_ERR_INVALID_PARAM;
    if (p.showmask < 0 || p.showmask > 4) return RGY_ERR_INVALID_PARAM;
    if (!(p.edgeproc >= 0.0f && p.edgeproc <= 1.0f)) return RGY_ERR_INVALID_PARAM;
    if (p.edge != _T("prewitt") && p.edge != _T("sobel") && p.edge != _T("scharr") && p.edge != _T("kirsch") && p.edge != _T("laplacian")) return RGY_ERR_INVALID_PARAM;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterFineDehalo::allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label) {
    if (!frame || frame->frame.width != frameInfo.width || frame->frame.height != frameInfo.height || frame->frame.csp != frameInfo.csp) {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        frame->releasePtr();
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate finedehalo %s buffer: %s.\n"), label, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterFineDehalo::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamFineDehalo>(pParam);
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

    auto prmDh = std::make_shared<NVEncFilterParamDehalo>();
    prmDh->dehalo.enable = true;
    prmDh->dehalo.mode = prm->finedehalo.mode;
    prmDh->dehalo.rx = prm->finedehalo.rx;
    prmDh->dehalo.ry = prm->finedehalo.ry;
    prmDh->dehalo.darkstr = prm->finedehalo.darkstr;
    prmDh->dehalo.brightstr = prm->finedehalo.brightstr;
    prmDh->dehalo.lowsens = prm->finedehalo.lowsens;
    prmDh->dehalo.highsens = prm->finedehalo.highsens;
    prmDh->dehalo.ss = prm->finedehalo.ss;
    prmDh->dehalo.searchRade = prm->finedehalo.searchRade;
    prmDh->dehalo.searchRadi = prm->finedehalo.searchRadi;
    prmDh->frameIn = prm->frameIn;
    prmDh->frameOut = prm->frameIn;
    prmDh->baseFps = prm->baseFps;
    prmDh->bOutOverwrite = false;
    m_dehalo = std::make_unique<NVEncFilterDehalo>();
    sts = m_dehalo->init(prmDh, m_pLog);
    if (sts != RGY_ERR_NONE) return sts;

    sts = allocWorkFrame(m_edges, prm->frameIn, _T("edges"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_strong, prm->frameIn, _T("strong"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_large, prm->frameIn, _T("large"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_light, prm->frameIn, _T("light"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_shrink, prm->frameIn, _T("shrink"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_outside, prm->frameIn, _T("outside"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_morphTmp, prm->frameIn, _T("morphTmp"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_shrMed, prm->frameIn, _T("shrMed"));
    if (sts != RGY_ERR_NONE) return sts;

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

tstring NVEncFilterParamFineDehalo::print() const {
    return finedehalo.print();
}

RGY_ERR NVEncFilterFineDehalo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) return sts;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo does not support interlaced input. Please deinterlace before finedehalo.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) return RGY_ERR_INVALID_PARAM;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamFineDehalo>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;

    RGYFrameInfo *dehaloOut[1] = { nullptr };
    int dehaloOutNum = 0;
    sts = m_dehalo->filter(const_cast<RGYFrameInfo *>(pInputFrame), dehaloOut, &dehaloOutNum, stream);
    if (sts != RGY_ERR_NONE || dehaloOutNum != 1 || dehaloOut[0] == nullptr) return (sts != RGY_ERR_NONE) ? sts : RGY_ERR_UNKNOWN;

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int thmi = (int)((long long)prm->finedehalo.thmi * maxVal / 255);
    const int thma = (int)((long long)prm->finedehalo.thma * maxVal / 255);
    const int thlimi = (int)((long long)prm->finedehalo.thlimi * maxVal / 255);
    const int thlima = (int)((long long)prm->finedehalo.thlima * maxVal / 255);

    auto planeOutY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    sts = finedehalo_process_y(&planeOutY, pInputFrame, dehaloOut[0],
        &m_edges->frame, &m_strong->frame, &m_large->frame, &m_light->frame, &m_shrink->frame,
        &m_outside->frame, &m_morphTmp->frame, &m_shrMed->frame,
        prm->finedehalo, thmi, thma, thlimi, thlima, stream);
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

void NVEncFilterFineDehalo::close() {
    m_dehalo.reset();
    m_edges.reset();
    m_strong.reset();
    m_large.reset();
    m_light.reset();
    m_shrink.reset();
    m_outside.reset();
    m_morphTmp.reset();
    m_shrMed.reset();
    m_frameBuf.clear();
}
