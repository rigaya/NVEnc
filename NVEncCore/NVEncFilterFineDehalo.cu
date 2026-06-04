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

template<typename Type>
__device__ __forceinline__ int fdh_read_pix_clamp(const RGYFrameInfo frame, int x, int y) {
    x = clamp(x, 0, frame.width - 1);
    y = clamp(y, 0, frame.height - 1);
    const auto ptr = (const Type *)((const uint8_t *)frame.ptr[0] + y * frame.pitch[0] + x * sizeof(Type));
    return (int)ptr[0];
}

template<typename Type, int bit_depth>
__device__ __forceinline__ int fdh_ramp(const int v, const int lo, const int hi) {
    static const int max_val = (1 << bit_depth) - 1;
    if (hi > lo) {
        long long num = (long long)(v - lo) * (long long)max_val;
        return clamp((int)(num / (long long)(hi - lo)), 0, max_val);
    }
    return (v >= lo) ? max_val : 0;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_edge(const RGYFrameInfo src, RGYFrameInfo dst, const int lo, const int hi, const int edgeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;

    const int tl = fdh_read_pix_clamp<Type>(src, x - 1, y - 1);
    const int tc = fdh_read_pix_clamp<Type>(src, x,     y - 1);
    const int tr = fdh_read_pix_clamp<Type>(src, x + 1, y - 1);
    const int cl = fdh_read_pix_clamp<Type>(src, x - 1, y);
    const int cc = fdh_read_pix_clamp<Type>(src, x,     y);
    const int cr = fdh_read_pix_clamp<Type>(src, x + 1, y);
    const int bl = fdh_read_pix_clamp<Type>(src, x - 1, y + 1);
    const int bc = fdh_read_pix_clamp<Type>(src, x,     y + 1);
    const int br = fdh_read_pix_clamp<Type>(src, x + 1, y + 1);

    int g = 0;
    if (edgeMode == 1) { // scharr
        const int gx = -3 * tl + 3 * tr - 10 * cl + 10 * cr - 3 * bl + 3 * br;
        const int gy = -3 * tl - 10 * tc - 3 * tr + 3 * bl + 10 * bc + 3 * br;
        g = (abs(gx) + abs(gy)) / 4;
    } else if (edgeMode == 2) { // kirsch
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
    } else if (edgeMode == 3) { // laplacian
        g = abs(4 * cc - tc - cl - cr - bc) * 2;
    } else { // prewitt/sobel-compatible
        const int gx = -tl - 2 * cl - bl + tr + 2 * cr + br;
        const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        g = abs(gx) + abs(gy);
    }

    const int out = fdh_ramp<Type, bit_depth>(g, lo, hi);
    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type>
__global__ void kernel_fdh_expand3x3(const RGYFrameInfo src, RGYFrameInfo dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;
    int m = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            m = max(m, fdh_read_pix_clamp<Type>(src, x + dx, y + dy));
        }
    }
    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type>
__global__ void kernel_fdh_inpand3x3(const RGYFrameInfo src, RGYFrameInfo dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;
    int m = fdh_read_pix_clamp<Type>(src, x, y);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            m = min(m, fdh_read_pix_clamp<Type>(src, x + dx, y + dy));
        }
    }
    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_limitmask(const RGYFrameInfo src, const RGYFrameInfo dehaloed, RGYFrameInfo dst, const int lo, const int hi) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;
    const int s = fdh_read_pix_clamp<Type>(src, x, y);
    const int d = fdh_read_pix_clamp<Type>(dehaloed, x, y);
    const int out = fdh_ramp<Type, bit_depth>(max(s - d, 0), lo, hi);
    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_fdh_combine(const RGYFrameInfo src, const RGYFrameInfo dehaloed, const RGYFrameInfo em,
    const RGYFrameInfo linemask, RGYFrameInfo dst, const int showmask) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;
    static const int max_val = (1 << bit_depth) - 1;
    const int s = fdh_read_pix_clamp<Type>(src, x, y);
    const int d = fdh_read_pix_clamp<Type>(dehaloed, x, y);
    const int finalMask = min(max_val - fdh_read_pix_clamp<Type>(em, x, y), fdh_read_pix_clamp<Type>(linemask, x, y));
    int out = finalMask;
    if (showmask != 4) {
        const float m = (float)finalMask / (float)max_val;
        out = clamp((int)((float)s + ((float)d - (float)s) * m + 0.5f), 0, max_val);
    }
    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)out;
}

static int fdh_edge_mode(const tstring& edge) {
    if (edge == _T("scharr")) return 1;
    if (edge == _T("kirsch")) return 2;
    if (edge == _T("laplacian")) return 3;
    return 0;
}

template<typename Type, int bit_depth>
static RGY_ERR finedehalo_process_y_typed(RGYFrameInfo *pOut, const RGYFrameInfo *pInput, const RGYFrameInfo *pDehaloed,
    RGYFrameInfo *pEdges, RGYFrameInfo *pMorphTmp, RGYFrameInfo *pEy, RGYFrameInfo *pEm, RGYFrameInfo *pLineMask,
    const VppFineDehalo& prm, const int thmi, const int thma, const int thlimi, const int thlima, cudaStream_t stream) {
    dim3 blockSize(FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInput->width, blockSize.x), divCeil(pInput->height, blockSize.y));
    const int edgeMode = fdh_edge_mode(prm.edge);

    kernel_fdh_edge<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(*pInput, *pEdges, thmi, thma, edgeMode);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_expand3x3<Type><<<gridSize, blockSize, 0, stream>>>(*pEdges, *pMorphTmp);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_inpand3x3<Type><<<gridSize, blockSize, 0, stream>>>(*pMorphTmp, *pEy);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_expand3x3<Type><<<gridSize, blockSize, 0, stream>>>(*pEy, *pMorphTmp);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_inpand3x3<Type><<<gridSize, blockSize, 0, stream>>>(*pMorphTmp, *pEm);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_fdh_limitmask<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(*pInput, *pDehaloed, *pLineMask, thlimi, thlima);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    if (prm.showmask == 1) {
        return copyPlaneAsync(pOut, pEdges, stream);
    } else if (prm.showmask == 2) {
        return copyPlaneAsync(pOut, pEm, stream);
    } else if (prm.showmask == 3) {
        return copyPlaneAsync(pOut, pLineMask, stream);
    }
    kernel_fdh_combine<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(*pInput, *pDehaloed, *pEm, *pLineMask, *pOut, prm.showmask);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    return RGY_ERR_NONE;
}

static RGY_ERR finedehalo_process_y(RGYFrameInfo *pOut, const RGYFrameInfo *pInput, const RGYFrameInfo *pDehaloed,
    RGYFrameInfo *pEdges, RGYFrameInfo *pMorphTmp, RGYFrameInfo *pEy, RGYFrameInfo *pEm, RGYFrameInfo *pLineMask,
    const VppFineDehalo& prm, const int thmi, const int thma, const int thlimi, const int thlima, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInput->csp] > 8) {
        return finedehalo_process_y_typed<uint16_t, 16>(pOut, pInput, pDehaloed, pEdges, pMorphTmp, pEy, pEm, pLineMask, prm, thmi, thma, thlimi, thlima, stream);
    }
    return finedehalo_process_y_typed<uint8_t, 8>(pOut, pInput, pDehaloed, pEdges, pMorphTmp, pEy, pEm, pLineMask, prm, thmi, thma, thlimi, thlima, stream);
}

NVEncFilterFineDehalo::NVEncFilterFineDehalo() :
    m_dehalo(),
    m_edges(),
    m_morphTmp(),
    m_ey(),
    m_em(),
    m_linemask() {
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
    auto &p = prm->finedehalo;
    if (!(p.rx >= 0.5f && p.rx <= 10.0f) || !(p.ry >= 0.5f && p.ry <= 10.0f)) return RGY_ERR_INVALID_PARAM;
    if (!(p.darkstr >= 0.0f && p.darkstr <= 1.0f) || !(p.brightstr >= 0.0f && p.brightstr <= 1.0f)) return RGY_ERR_INVALID_PARAM;
    if (p.lowsens < 0 || p.lowsens > 100 || p.highsens < 0 || p.highsens > 100) return RGY_ERR_INVALID_PARAM;
    if (!(p.ss >= 1.0f && p.ss <= 4.0f)) return RGY_ERR_INVALID_PARAM;
    if (p.thmi < 0 || p.thmi > 255 || p.thma < 0 || p.thma > 255) return RGY_ERR_INVALID_PARAM;
    if (p.thlimi < 0 || p.thlimi > 255 || p.thlima < 0 || p.thlima > 255) return RGY_ERR_INVALID_PARAM;
    if (p.showmask < 0 || p.showmask > 4) return RGY_ERR_INVALID_PARAM;
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
    prmDh->dehalo.rx = prm->finedehalo.rx;
    prmDh->dehalo.ry = prm->finedehalo.ry;
    prmDh->dehalo.darkstr = prm->finedehalo.darkstr;
    prmDh->dehalo.brightstr = prm->finedehalo.brightstr;
    prmDh->dehalo.lowsens = prm->finedehalo.lowsens;
    prmDh->dehalo.highsens = prm->finedehalo.highsens;
    prmDh->dehalo.ss = prm->finedehalo.ss;
    prmDh->frameIn = prm->frameIn;
    prmDh->frameOut = prm->frameIn;
    prmDh->baseFps = prm->baseFps;
    prmDh->bOutOverwrite = false;
    m_dehalo = std::make_unique<NVEncFilterDehalo>();
    sts = m_dehalo->init(prmDh, m_pLog);
    if (sts != RGY_ERR_NONE) return sts;

    sts = allocWorkFrame(m_edges, prm->frameIn, _T("edges"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_morphTmp, prm->frameIn, _T("morphTmp"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_ey, prm->frameIn, _T("ey"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_em, prm->frameIn, _T("em"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_linemask, prm->frameIn, _T("linemask"));
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
    if (interlaced(*pInputFrame)) return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
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
        &m_edges->frame, &m_morphTmp->frame, &m_ey->frame, &m_em->frame, &m_linemask->frame,
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
    m_morphTmp.reset();
    m_ey.reset();
    m_em.reset();
    m_linemask.reset();
    m_frameBuf.clear();
}
