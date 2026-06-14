#include "NVEncFilterRtgmcEdi.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "convert_csp.h"
#include "rgy_cuda_util_kernel.h"

#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

namespace {
static constexpr int RTGMC_EDI_BLOCK_X = 32;
static constexpr int RTGMC_EDI_BLOCK_Y = 8;

static bool rtgmcEdiModeIsBob(const VppRtgmcEdiMode mode) {
    return mode == VppRtgmcEdiMode::Bob || mode == VppRtgmcEdiMode::BobChromaMerge;
}

static bool rtgmcEdiModeIsLightweight(const VppRtgmcEdiMode mode) {
    return rtgmcEdiModeIsBob(mode)
        || mode == VppRtgmcEdiMode::Yadif
        || mode == VppRtgmcEdiMode::cYadif
        || mode == VppRtgmcEdiMode::RepYadif
        || mode == VppRtgmcEdiMode::RepcYadif;
}

static bool rtgmcEdiModeUsesTemporalYadif(const VppRtgmcEdiMode mode) {
    return mode == VppRtgmcEdiMode::Yadif
        || mode == VppRtgmcEdiMode::cYadif
        || mode == VppRtgmcEdiMode::RepYadif
        || mode == VppRtgmcEdiMode::RepcYadif;
}

static const TCHAR *rtgmcEdiModeDetail(const VppRtgmcEdiMode mode) {
    switch (mode) {
    case VppRtgmcEdiMode::Bob:
        return _T(" (bob chroma + selected luma)");
    case VppRtgmcEdiMode::Yadif:
        return _T(" (Yadifmod2 temporal)");
    case VppRtgmcEdiMode::cYadif:
        return _T(" (Yadif temporal)");
    case VppRtgmcEdiMode::TDeint:
        return _T(" (TDeint cubic)");
    case VppRtgmcEdiMode::RepYadif:
        return _T(" (Yadifmod2 planar)");
    case VppRtgmcEdiMode::RepcYadif:
        return _T(" (Yadif temporal + Repair mode2)");
    case VppRtgmcEdiMode::NNEDI3:
        return _T(" (nnedi3 adapter)");
    default:
        return _T("");
    }
}

static void rtgmcEdiCopyFrameProps(RGYFrameInfo *dst, const RGYFrameInfo *src, const VppRtgmcEdiMode mode) {
    copyFramePropWithoutRes(dst, src);
    dst->picstruct = rtgmcEdiModeIsLightweight(mode) ? RGY_PICSTRUCT_FRAME : src->picstruct;
}

static RGY_ERR recordCudaEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto sts = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}

template<typename Type>
__device__ int readPix(
    const uint8_t *__restrict__ plane, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const Type *)(plane + y * pitch + x * sizeof(Type)));
}

template<typename T>
__device__ __forceinline__ T max3(T a, T b, T c) {
    return max(max(a, b), c);
}

template<typename T>
__device__ __forceinline__ T min3(T a, T b, T c) {
    return min(min(a, b), c);
}

template<typename Type>
__device__ int rtgmcEdiYadifSpatial(
    const uint8_t *__restrict__ src, const int ix, const int iy,
    const int pitch, const int width, const int height
) {
    int ym1[7], yp1[7];
#pragma unroll
    for (int dx = -3; dx <= 3; dx++) {
        ym1[dx + 3] = readPix<Type>(src, ix + dx, iy - 1, pitch, width, height);
        yp1[dx + 3] = readPix<Type>(src, ix + dx, iy + 1, pitch, width, height);
    }

    const int score[5] = {
        abs(ym1[2] - yp1[2]) + abs(ym1[3] - yp1[3]) + abs(ym1[4] - yp1[4]) - 1,
        abs(ym1[1] - yp1[3]) + abs(ym1[2] - yp1[4]) + abs(ym1[3] - yp1[5]),
        abs(ym1[0] - yp1[4]) + abs(ym1[1] - yp1[5]) + abs(ym1[2] - yp1[6]),
        abs(ym1[3] - yp1[1]) + abs(ym1[4] - yp1[2]) + abs(ym1[5] - yp1[3]),
        abs(ym1[4] - yp1[0]) + abs(ym1[5] - yp1[1]) + abs(ym1[6] - yp1[2])
    };
    int minscore = score[0];
    int minidx = 0;
    if (score[1] < minscore) {
        minscore = score[1];
        minidx = 1;
    }
    if (score[2] < minscore) {
        minscore = score[2];
        minidx = 2;
    }
    if (score[3] < minscore) {
        minscore = score[3];
        minidx = 3;
    }
    if (score[4] < minscore) {
        minscore = score[4];
        minidx = 4;
    }

    switch (minidx) {
    case 0: return (ym1[3] + yp1[3]) >> 1;
    case 1: return (ym1[2] + yp1[4]) >> 1;
    case 2: return (ym1[1] + yp1[5]) >> 1;
    case 3: return (ym1[4] + yp1[2]) >> 1;
    case 4:
    default:return (ym1[5] + yp1[1]) >> 1;
    }
}

template<typename Type>
__device__ int rtgmcEdiTDeintSpatial(
    const uint8_t *__restrict__ src, const int ix, const int iy,
    const int pitch, const int width, const int height
) {
    const int up2 = readPix<Type>(src, ix, iy - 3, pitch, width, height);
    const int up1 = readPix<Type>(src, ix, iy - 1, pitch, width, height);
    const int dn1 = readPix<Type>(src, ix, iy + 1, pitch, width, height);
    const int dn2 = readPix<Type>(src, ix, iy + 3, pitch, width, height);
    const int cubic = (-up2 + 9 * up1 + 9 * dn1 - dn2 + 8) >> 4;
    return clamp(cubic, min(up1, dn1), max(up1, dn1));
}

template<typename Type>
__device__ int rtgmcEdiYadifTemporal(
    const uint8_t *__restrict__ prevSrc,
    const uint8_t *__restrict__ curSrc,
    const uint8_t *__restrict__ nextSrc,
    const int ix, const int iy,
    const int prevPitch, const int curPitch, const int nextPitch,
    const int width, const int height,
    const int valSpatial,
    const int fieldSecond
) {
    const int t00m1 = readPix<Type>(prevSrc, ix, iy - 1, prevPitch, width, height);
    const int t00p1 = readPix<Type>(prevSrc, ix, iy + 1, prevPitch, width, height);
    const int t10m1 = readPix<Type>(curSrc,  ix, iy - 1, curPitch,  width, height);
    const int t10p1 = readPix<Type>(curSrc,  ix, iy + 1, curPitch,  width, height);
    const int t20m1 = readPix<Type>(nextSrc, ix, iy - 1, nextPitch, width, height);
    const int t20p1 = readPix<Type>(nextSrc, ix, iy + 1, nextPitch, width, height);

    const uint8_t *src01 = fieldSecond ? curSrc : prevSrc;
    const uint8_t *src12 = fieldSecond ? nextSrc : curSrc;
    const int src01Pitch = fieldSecond ? curPitch : prevPitch;
    const int src12Pitch = fieldSecond ? nextPitch : curPitch;

    const int t01m2 = readPix<Type>(src01, ix, iy - 2, src01Pitch, width, height);
    const int t01_0  = readPix<Type>(src01, ix, iy + 0, src01Pitch, width, height);
    const int t01p2  = readPix<Type>(src01, ix, iy + 2, src01Pitch, width, height);
    const int t12m2 = readPix<Type>(src12, ix, iy - 2, src12Pitch, width, height);
    const int t12_0  = readPix<Type>(src12, ix, iy + 0, src12Pitch, width, height);
    const int t12p2  = readPix<Type>(src12, ix, iy + 2, src12Pitch, width, height);

    const int tm2 = (t01m2 + t12m2) >> 1;
    const int t_0 = (t01_0 + t12_0) >> 1;
    const int tp2 = (t01p2 + t12p2) >> 1;

    const int diff = max3(
        abs(t01_0 - t12_0) >> 1,
        (abs(t00m1 - t10m1) + abs(t00p1 - t10p1)) >> 1,
        (abs(t20m1 - t10m1) + abs(t10p1 - t20p1)) >> 1);
    return max(min(valSpatial, t_0 + diff), t_0 - diff);
}

template<typename Type>
__device__ int rtgmcEdiYadifEdge(
    const uint8_t *__restrict__ src, const int ix, const int iy,
    const int pitch, const int width, const int height,
    const int targetField
) {
    if (targetField == 0) {
        if (iy == 0) {
            return readPix<Type>(src, ix, 1, pitch, width, height);
        }
        if (iy == height - 2) {
            return (readPix<Type>(src, ix, height - 3, pitch, width, height)
                + readPix<Type>(src, ix, height - 1, pitch, width, height) + 1) >> 1;
        }
    } else {
        if (iy == 1) {
            return (readPix<Type>(src, ix, 0, pitch, width, height)
                + readPix<Type>(src, ix, 2, pitch, width, height) + 1) >> 1;
        }
        if (iy == height - 1) {
            return readPix<Type>(src, ix, height - 2, pitch, width, height);
        }
    }
    return -1;
}

template<typename Type>
__device__ int rtgmcEdiRepairMode2(
    const int edi,
    const uint8_t *__restrict__ src, const int ix, const int iy,
    const int pitch, const int width, const int height
) {
    if (ix <= 0 || iy <= 0 || ix >= width - 1 || iy >= height - 1) {
        return edi;
    }

    int a[9] = {
        readPix<Type>(src, ix - 1, iy - 1, pitch, width, height),
        readPix<Type>(src, ix + 0, iy - 1, pitch, width, height),
        readPix<Type>(src, ix + 1, iy - 1, pitch, width, height),
        readPix<Type>(src, ix - 1, iy + 0, pitch, width, height),
        readPix<Type>(src, ix + 0, iy + 0, pitch, width, height),
        readPix<Type>(src, ix + 1, iy + 0, pitch, width, height),
        readPix<Type>(src, ix - 1, iy + 1, pitch, width, height),
        readPix<Type>(src, ix + 0, iy + 1, pitch, width, height),
        readPix<Type>(src, ix + 1, iy + 1, pitch, width, height)
    };
    for (int i = 1; i < 9; i++) {
        int value = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > value) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = value;
    }

    return clamp(edi, a[1], a[7]);
}

template<typename Type>
__device__ int rtgmcEdiLuma(
    const uint8_t *__restrict__ bobSrc,
    const uint8_t *__restrict__ ediPrevSrc,
    const uint8_t *__restrict__ ediSrc,
    const uint8_t *__restrict__ ediNextSrc,
    const int ix, const int iy,
    const int bobPitch, const int ediPrevPitch, const int ediPitch, const int ediNextPitch,
    const int width, const int height,
    const int targetField,
    const int fieldSecond,
    const int mode
) {
    const int bobValue = readPix<Type>(bobSrc, ix, iy, bobPitch, width, height);
    const int ediValue = readPix<Type>(ediSrc, ix, iy, ediPitch, width, height);
    if (mode == 6) {
        return ediValue;
    }
    if ((iy & 1) != targetField) {
        return ediValue;
    }
    if (mode == 0) {
        return bobValue;
    }
    if (mode == 1 || mode == 2 || mode == 4) {
        const int edge = rtgmcEdiYadifEdge<Type>(ediSrc, ix, iy, ediPitch, width, height, targetField);
        if (edge >= 0) {
            return edge;
        }
        const int spatial = rtgmcEdiYadifSpatial<Type>(ediSrc, ix, iy, ediPitch, width, height);
        return rtgmcEdiYadifTemporal<Type>(ediPrevSrc, ediSrc, ediNextSrc, ix, iy,
            ediPrevPitch, ediPitch, ediNextPitch, width, height, spatial, fieldSecond);
    }
    if (mode == 3) {
        return rtgmcEdiTDeintSpatial<Type>(ediSrc, ix, iy, ediPitch, width, height);
    }
    if (mode == 5) {
        const int edge = rtgmcEdiYadifEdge<Type>(ediSrc, ix, iy, ediPitch, width, height, targetField);
        const int yadifValue = (edge >= 0)
            ? edge
            : rtgmcEdiYadifTemporal<Type>(ediPrevSrc, ediSrc, ediNextSrc, ix, iy,
                ediPrevPitch, ediPitch, ediNextPitch, width, height,
                rtgmcEdiYadifSpatial<Type>(ediSrc, ix, iy, ediPitch, width, height), fieldSecond);
        return rtgmcEdiRepairMode2<Type>(yadifValue, bobSrc, ix, iy, bobPitch, width, height);
    }
    return bobValue;
}

template<typename Type>
__global__ void kernel_rtgmc_edi(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pBobSrc, const int bobPitch,
    const uint8_t *__restrict__ pEdiPrevSrc, const int ediPrevPitch,
    const uint8_t *__restrict__ pEdiSrc, const int ediPitch,
    const uint8_t *__restrict__ pEdiNextSrc, const int ediNextPitch,
    const int width,
    const int height,
    const int planeIndex,
    const int targetField,
    const int fieldSecond,
    const int mode,
    const int maxVal
) {
    (void)planeIndex;
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) {
        return;
    }

    const int value = rtgmcEdiLuma<Type>(
        pBobSrc, pEdiPrevSrc, pEdiSrc, pEdiNextSrc,
        ix, iy, bobPitch, ediPrevPitch, ediPitch, ediNextPitch,
        width, height, targetField, fieldSecond, mode);
    Type *dstPix = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, maxVal);
}

template<typename Type>
RGY_ERR launchRtgmcEdiPlane(RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pBobPlane,
    const RGYFrameInfo *pEdiPrevPlane,
    const RGYFrameInfo *pEdiPlane,
    const RGYFrameInfo *pEdiNextPlane,
    const int planeIndex,
    const int targetField,
    const int fieldSecond,
    const int mode,
    cudaStream_t stream) {
    const dim3 blockSize(RTGMC_EDI_BLOCK_X, RTGMC_EDI_BLOCK_Y);
    const dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));
    const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputPlane->csp];
    const int maxVal = (bitDepth >= 16) ? ((1 << 16) - 1) : ((1 << bitDepth) - 1);
    kernel_rtgmc_edi<Type><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (const uint8_t *)pBobPlane->ptr[0], pBobPlane->pitch[0],
        (const uint8_t *)pEdiPrevPlane->ptr[0], pEdiPrevPlane->pitch[0],
        (const uint8_t *)pEdiPlane->ptr[0], pEdiPlane->pitch[0],
        (const uint8_t *)pEdiNextPlane->ptr[0], pEdiNextPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height,
        planeIndex, targetField, fieldSecond, mode, maxVal);
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type>
RGY_ERR processRtgmcEdiFrame(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pBobInputFrame,
    const RGYFrameInfo *pEdiPrevFrame,
    const RGYFrameInfo *pEdiInputFrame,
    const RGYFrameInfo *pEdiNextFrame,
    const int targetField,
    const int fieldSecond,
    const int mode,
    cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pOutputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        auto planeOutput = getPlane(pOutputFrame, plane);
        const auto planeBob = getPlane(pBobInputFrame, plane);
        const auto planePrev = getPlane(pEdiPrevFrame, plane);
        const auto planeEdi = getPlane(pEdiInputFrame, plane);
        const auto planeNext = getPlane(pEdiNextFrame, plane);
        auto sts = launchRtgmcEdiPlane<Type>(&planeOutput, &planeBob, &planePrev, &planeEdi, &planeNext,
            iplane, targetField, fieldSecond, mode, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}
} // namespace

void NVEncFilterRtgmcEdi::FrameSource::clear() {
    for (auto& buf : m_buf) {
        buf.clear();
    }
    m_nFramesInput = 0;
}

void NVEncFilterRtgmcEdi::FrameSource::resetFrames() {
    // Keep the allocated CUFrameBuf GPU memory; only rewind the logical ring position so the
    // next cold-start fill overwrites the existing buffers in order.
    m_nFramesInput = 0;
}

NVEncFilterRtgmcEdi::FrameSource::FrameSource() :
    m_nFramesInput(0),
    m_buf() {
}

RGY_ERR NVEncFilterRtgmcEdi::FrameSource::alloc(const RGYFrameInfo& frameInfo) {
    if (m_buf[0].frame.ptr[0] != nullptr
        && !cmpFrameInfoCspResolution(&m_buf[0].frame, &frameInfo)) {
        bool allocated = true;
        for (auto& buf : m_buf) {
            if (buf.frame.ptr[0] == nullptr) {
                allocated = false;
                break;
            }
        }
        if (allocated) {
            return RGY_ERR_NONE;
        }
    }
    for (auto& buf : m_buf) {
        auto ret = buf.alloc(frameInfo.width, frameInfo.height, frameInfo.csp);
        if (ret != RGY_ERR_NONE) {
            buf.clear();
            return ret;
        }
    }
    m_nFramesInput = 0;
    return RGY_ERR_NONE;
}

CUFrameBuf *NVEncFilterRtgmcEdi::FrameSource::get(int iframe) {
    iframe = clamp(iframe, 0, m_nFramesInput - 1);
    return &m_buf[iframe % m_buf.size()];
}

int NVEncFilterRtgmcEdi::FrameSource::findIndexByInputFrameId(int inputFrameId) const {
    const int start = std::max(0, m_nFramesInput - (int)m_buf.size());
    for (int iframe = start; iframe < m_nFramesInput; iframe++) {
        const auto& buf = m_buf[iframe % m_buf.size()];
        if (buf.frame.ptr[0] && buf.frame.inputFrameId == inputFrameId) {
            return iframe;
        }
    }
    return -1;
}

RGY_ERR NVEncFilterRtgmcEdi::FrameSource::add(const RGYFrameInfo *pInputFrame, cudaStream_t stream, bool copyChroma) {
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int iframe = m_nFramesInput++;
    auto pDstFrame = get(iframe);
    auto err = RGY_ERR_NONE;
    const int planes = copyChroma ? RGY_CSP_PLANES[pDstFrame->frame.csp] : 1;
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dstPlane = getPlane(&pDstFrame->frame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        err = copyPlaneAsync(&dstPlane, &srcPlane, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    copyFramePropWithoutRes(&pDstFrame->frame, pInputFrame);
    return RGY_ERR_NONE;
}

NVEncFilterRtgmcEdi::NnediAdapterState::NnediAdapterState() :
    filter(),
    outputCsp(),
    cachedFrames({ nullptr, nullptr }),
    cachedKey(),
    cachedEvent(),
    cacheValid(false) {
}

void NVEncFilterRtgmcEdi::NnediAdapterState::clear() {
    filter.reset();
    outputCsp.reset();
    cachedFrames = { nullptr, nullptr };
    cachedKey = FrameKey();
    cachedEvent.reset();
    cacheValid = false;
}

tstring NVEncFilterParamRtgmcEdi::print() const {
    return strsprintf(_T("rtgmc-edi: mode=%s%s, nnsize=%d, nneurons=%d, ediqual=%d, chroma_edi=%s"),
        get_cx_desc(list_vpp_rtgmc_edi_mode, (int)mode),
        rtgmcEdiModeIsLightweight(mode) ? rtgmcEdiModeDetail(mode) : _T(""),
        nnsize, nneurons, ediqual,
        get_cx_desc(list_vpp_rtgmc_chroma_edi_mode, (int)chromaEdi));
}

NVEncFilterRtgmcEdi::NVEncFilterRtgmcEdi() :
    NVEncFilter(),
    m_buildOptions(),
    m_bobSource(),
    m_ediSource(),
    m_inputSource(),
    m_nnediStates(),
    m_nnediAdapterCopyEvent(),
    m_nFrame(0),
    m_lastInputFrameId(-1),
    m_pairFrameIndex(0),
    m_fallbackFrameIndex(0),
    m_useKernel(false) {
    m_name = _T("rtgmc-edi");
    m_pathThrough &= ~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA);
}

NVEncFilterRtgmcEdi::~NVEncFilterRtgmcEdi() {
    close();
}

RGY_ERR NVEncFilterRtgmcEdi::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcEdi> &prm) {
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0
        || prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.csp != prm->frameOut.csp
        || prm->frameIn.width != prm->frameOut.width
        || prm->frameIn.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto dataType = RGY_CSP_DATA_TYPE[prm->frameOut.csp];
    if (dataType != RGY_DATA_TYPE_U8 && dataType != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode < VppRtgmcEdiMode::Passthrough || prm->mode > VppRtgmcEdiMode::NNEDI3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rtgmc-edi mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->chromaEdi == VppRtgmcChromaEdiMode::NNEDI3) {
        if (prm->mode != VppRtgmcEdiMode::NNEDI3) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi chroma_edi=nnedi3 requires mode=nnedi3.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (RGY_CSP_PLANES[prm->frameOut.csp] < 3) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi chroma_edi=nnedi3 requires a planar YUV format with chroma planes.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (prm->mode == VppRtgmcEdiMode::NNEDI3
        && (prm->sourceFrameIn.csp == RGY_CSP_NA || prm->sourceFrameIn.width <= 0 || prm->sourceFrameIn.height <= 0)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi mode nnedi3 requires original source frame info; use it through --vpp-rtgmc, not standalone --vpp-rtgmc-edi.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode != VppRtgmcEdiMode::Passthrough && !rtgmcEdiModeIsLightweight(prm->mode)) {
        if (prm->mode != VppRtgmcEdiMode::NNEDI3) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid rtgmc-edi mode.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (prm->nnsize < 0 || prm->nnsize > 6 || prm->nneurons < 0 || prm->nneurons > 4 || prm->ediqual < 1 || prm->ediqual > 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rtgmc-edi NNEDI3 parameter: nnsize %d, nneurons %d, ediqual %d.\n"),
            prm->nnsize, prm->nneurons, prm->ediqual);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::checkInputs(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame) {
    if (pBobInputFrame->csp != pEdiInputFrame->csp
        || pBobInputFrame->width != pEdiInputFrame->width
        || pBobInputFrame->height != pEdiInputFrame->height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi requires bob/edi inputs to match in csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcEdi> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D rtgmc_edi_block_x=%d -D rtgmc_edi_block_y=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        RTGMC_EDI_BLOCK_X,
        RTGMC_EDI_BLOCK_Y);
    AddMessage(RGY_LOG_DEBUG, _T("Using CUDA kernel for rtgmc-edi: %s\n"),
        char_to_tstring(m_buildOptions).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcEdi>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_useKernel = (RGY_CSP_DATA_TYPE[prm->frameOut.csp] == RGY_DATA_TYPE_U8
        || RGY_CSP_DATA_TYPE[prm->frameOut.csp] == RGY_DATA_TYPE_U16);
    m_lastInputFrameId = -1;
    m_pairFrameIndex = 0;
    m_fallbackFrameIndex = 0;
    m_nFrame = 0;
    for (auto& state : m_nnediStates) {
        state.clear();
    }
    m_nnediAdapterCopyEvent.reset();

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcEdi>(m_param);
    if (prm->mode == VppRtgmcEdiMode::NNEDI3) {
        sts = initNnediAdapterState(m_nnediStates[0], prm, false);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->chromaEdi == VppRtgmcChromaEdiMode::NNEDI3) {
            sts = initNnediAdapterState(m_nnediStates[1], prm, true);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_buildOptions.clear();
        m_useKernel = false;
    } else if (m_useKernel
        && (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp])) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-edi kernel.\n"));
            return sts;
        }
    }

    if (rtgmcEdiModeUsesTemporalYadif(prm->mode)) {
        sts = m_bobSource.alloc(prm->frameOut);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi bob source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = m_ediSource.alloc(prm->frameOut);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = m_inputSource.alloc(prm->frameIn);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi input source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else if (prm->mode == VppRtgmcEdiMode::NNEDI3) {
        m_bobSource.clear();
        m_ediSource.clear();
        sts = m_inputSource.alloc(prm->sourceFrameIn);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi NNEDI source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        m_bobSource.clear();
        m_ediSource.clear();
        m_inputSource.clear();
    }

    sts = AllocFrameBuf(prm->frameOut, rtgmcEdiModeUsesTemporalYadif(prm->mode) ? 2 : 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::initNnediAdapterState(NnediAdapterState &state, const std::shared_ptr<NVEncFilterParamRtgmcEdi> &prm, const bool chroma) {
    auto filter = std::make_unique<NVEncFilterNnedi>();
    auto nnedi = std::make_shared<NVEncFilterParamNnedi>();
    nnedi->nnedi.enable = true;
    nnedi->nnedi.field = VPP_NNEDI_FIELD_BOB;
    nnedi->nnedi.nsize = (VppNnediNSize)(chroma ? VPP_NNEDI_NSIZE_8x4 : prm->nnsize);
    nnedi->nnedi.nns = rgy_nnedi_nns_value(chroma ? 0 : prm->nneurons);
    nnedi->nnedi.quality = (VppNnediQuality)(chroma ? VPP_NNEDI_QUALITY_FAST : prm->ediqual);
    nnedi->nnedi.processPlane = chroma
        ? std::array<bool, 4>{ false, true, true, false }
        : std::array<bool, 4>{ true, false, false, false };
    nnedi->nnedi.prescreen = 2;
    nnedi->nnedi.errortype = VPP_NNEDI_ETYPE_ABS;
    nnedi->nnedi.clamp = 1;
    nnedi->nnedi.doubleHeight = false;
    nnedi->hModule = prm->hModule;
    nnedi->frameIn = prm->sourceFrameIn;
    nnedi->frameOut = prm->sourceFrameIn;
    nnedi->baseFps = prm->sourceBaseFps;
    nnedi->timebase = prm->sourceTimebase;
    nnedi->bOutOverwrite = false;
    auto sts = filter->init(nnedi, m_pLog);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize rtgmc-edi %s NNEDI adapter: %s.\n"),
            chroma ? _T("chroma") : _T("main"), get_err_mes(sts));
        return sts;
    }
    if (nnedi->frameOut.width != prm->frameOut.width || nnedi->frameOut.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR,
            _T("rtgmc-edi %s NNEDI adapter output size does not match Bob frame size: NNEDI %dx%d, Bob %dx%d.\n"),
            chroma ? _T("chroma") : _T("main"),
            nnedi->frameOut.width, nnedi->frameOut.height, prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (nnedi->frameOut.csp != prm->frameOut.csp) {
        auto cspconv = std::make_unique<NVEncFilterCspCrop>();
        auto cspPrm = std::make_shared<NVEncFilterParamCrop>();
        cspPrm->frameIn = nnedi->frameOut;
        cspPrm->frameOut = nnedi->frameOut;
        cspPrm->frameOut.csp = prm->frameOut.csp;
        cspPrm->baseFps = nnedi->baseFps;
        cspPrm->bOutOverwrite = false;
        sts = cspconv->init(cspPrm, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to initialize rtgmc-edi %s NNEDI output csp conversion (%s -> %s): %s.\n"),
                chroma ? _T("chroma") : _T("main"),
                RGY_CSP_NAMES[nnedi->frameOut.csp], RGY_CSP_NAMES[prm->frameOut.csp], get_err_mes(sts));
            return sts;
        }
        state.outputCsp = std::move(cspconv);
    } else {
        state.outputCsp.reset();
    }
    state.filter = std::move(filter);
    state.cachedFrames = { nullptr, nullptr };
    state.cachedKey = FrameKey();
    state.cachedEvent.reset();
    state.cacheValid = false;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::runNnediAdapterState(NnediAdapterState &state, const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrame, const RGYFrameInfo **ppSelectedFrame,
    cudaStream_t stream,
    const NVEncFilterParamRtgmcEdi &prm, const bool chroma) {
    UNREFERENCED_PARAMETER(prm);
    if (ppOutputFrame) {
        *ppOutputFrame = nullptr;
    }
    if (ppSelectedFrame) {
        *ppSelectedFrame = nullptr;
    }
    if (!pBobInputFrame || !pBobInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!pSourceInputFrame || !pSourceInputFrame->ptr[0]) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s nnedi3 requires original source input frame.\n"),
            chroma ? _T("chroma") : _T("main"));
        return RGY_ERR_INVALID_CALL;
    }
    if (!state.filter) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter is not initialized.\n"),
            chroma ? _T("chroma") : _T("main"));
        return RGY_ERR_INVALID_CALL;
    }
    const int sourceIndex = m_inputSource.findIndexByInputFrameId(pBobInputFrame->inputFrameId);
    if (sourceIndex < 0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter source frame is missing for Bob inputFrameId=%d.\n"),
            chroma ? _T("chroma") : _T("main"), pBobInputFrame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }
    const auto *pNnediSourceFrame = &m_inputSource.get(sourceIndex)->frame;
    const FrameKey sourceKey(pNnediSourceFrame);
    if (!state.cacheValid || !state.cachedKey.matches(pNnediSourceFrame)) {
        RGYFrameInfo *nnediOut[2] = { nullptr, nullptr };
        int nnediOutNum = 0;
        auto sts = state.filter->filter(const_cast<RGYFrameInfo *>(pNnediSourceFrame), nnediOut, &nnediOutNum, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (nnediOutNum != 2 || !nnediOut[0] || !nnediOut[1] || !nnediOut[0]->ptr[0] || !nnediOut[1]->ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter expected 2 output frames, got %d.\n"),
                chroma ? _T("chroma") : _T("main"), nnediOutNum);
            return RGY_ERR_INVALID_CALL;
        }
        state.cachedFrames = { nnediOut[0], nnediOut[1] };
        state.cachedKey = sourceKey;
        state.cacheValid = true;
        state.cachedEvent.reset();
    }

    RGYFrameInfo *selected = nullptr;
    for (auto *candidate : state.cachedFrames) {
        if (candidate && candidate->inputFrameId == pBobInputFrame->inputFrameId
            && candidate->timestamp == pBobInputFrame->timestamp
            && candidate->duration == pBobInputFrame->duration) {
            selected = candidate;
            break;
        }
    }
    if (selected == nullptr) {
        for (auto *candidate : state.cachedFrames) {
            if (candidate && candidate->inputFrameId == pBobInputFrame->inputFrameId
                && candidate->timestamp == pBobInputFrame->timestamp) {
                selected = candidate;
                break;
            }
        }
    }
    if (selected == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter could not match Bob frame timestamp to cached NNEDI output.\n"),
            chroma ? _T("chroma") : _T("main"));
        AddMessage(RGY_LOG_ERROR, _T("  Bob frame: inputFrameId=%d, timestamp=%lld, duration=%lld.\n"),
            pBobInputFrame->inputFrameId, (long long)pBobInputFrame->timestamp, (long long)pBobInputFrame->duration);
        for (int i = 0; i < (int)state.cachedFrames.size(); i++) {
            const auto *candidate = state.cachedFrames[i];
            if (candidate) {
                AddMessage(RGY_LOG_ERROR, _T("  NNEDI candidate[%d]: inputFrameId=%d, timestamp=%lld, duration=%lld.\n"),
                    i, candidate->inputFrameId, (long long)candidate->timestamp, (long long)candidate->duration);
            }
        }
        return RGY_ERR_INVALID_CALL;
    }
    if (ppSelectedFrame) {
        *ppSelectedFrame = selected;
    }

    std::vector<RGYCudaEvent> copyWaitEvents;
    RGYFrameInfo *copySrc = selected;
    if (state.outputCsp) {
        RGYFrameInfo *converted[1] = { nullptr };
        int convertedNum = 0;
        auto sts = state.outputCsp->filter(selected, converted, &convertedNum, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (convertedNum != 1 || !converted[0] || !converted[0]->ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter csp conversion expected 1 output frame, got %d.\n"),
                chroma ? _T("chroma") : _T("main"), convertedNum);
            return RGY_ERR_INVALID_CALL;
        }
        copySrc = converted[0];
    }
    if (ppOutputFrame) {
        *ppOutputFrame = copySrc;
    }
    state.cachedEvent.reset();
    auto sts = recordCudaEvent(stream, &state.cachedEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

int NVEncFilterRtgmcEdi::targetField(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pParityFrame) {
    int pairIndex = 0;
    if (pBobInputFrame != nullptr && pBobInputFrame->inputFrameId >= 0) {
        if (pBobInputFrame->inputFrameId != m_lastInputFrameId) {
            m_lastInputFrameId = pBobInputFrame->inputFrameId;
            m_pairFrameIndex = 0;
        }
        pairIndex = m_pairFrameIndex++;
    } else {
        pairIndex = m_fallbackFrameIndex++;
    }
    const auto *fieldOrderFrame = (pParityFrame != nullptr) ? pParityFrame : pBobInputFrame;
    const bool bff = fieldOrderFrame != nullptr && (fieldOrderFrame->picstruct & RGY_PICSTRUCT_BFF);
    const int firstTargetField = bff ? 0 : 1;
    return (pairIndex & 1) ? (1 - firstTargetField) : firstTargetField;
}

RGY_ERR NVEncFilterRtgmcEdi::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBobInputFrame,
    const RGYFrameInfo *pEdiPrevFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pEdiNextFrame,
    const NVEncFilterParamRtgmcEdi &prm,
    const int targetField,
    cudaStream_t stream) {
    const bool tff = pEdiInputFrame == nullptr || (pEdiInputFrame->picstruct & RGY_PICSTRUCT_BFF) == 0;
    const int fieldSecond = ((targetField == 0) == tff) ? 1 : 0;
    const int mode = (int)prm.mode;
    const int dataType = RGY_CSP_DATA_TYPE[pOutputFrame->csp];
    if (dataType == RGY_DATA_TYPE_U8) {
        return processRtgmcEdiFrame<uint8_t>(pOutputFrame, pBobInputFrame, pEdiPrevFrame, pEdiInputFrame, pEdiNextFrame, targetField, fieldSecond, mode, stream);
    }
    if (dataType == RGY_DATA_TYPE_U16) {
        return processRtgmcEdiFrame<uint16_t>(pOutputFrame, pBobInputFrame, pEdiPrevFrame, pEdiInputFrame, pEdiNextFrame, targetField, fieldSecond, mode, stream);
    }
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterRtgmcEdi::runTemporalYadif(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream,
    const NVEncFilterParamRtgmcEdi &prm) {
    const bool draining = !pBobInputFrame || !pBobInputFrame->ptr[0];
    if (pBobInputFrame && pBobInputFrame->ptr[0]) {
        auto err = m_bobSource.add(pBobInputFrame, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi bob source frame: %s.\n"), get_err_mes(err));
            return err;
        }
        err = m_ediSource.add(pEdiInputFrame, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi source frame: %s.\n"), get_err_mes(err));
            return err;
        }
        if (pSourceInputFrame && pSourceInputFrame->ptr[0]
            && m_inputSource.findIndexByInputFrameId(pSourceInputFrame->inputFrameId) < 0) {
            err = m_inputSource.add(pSourceInputFrame, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi input source frame: %s.\n"), get_err_mes(err));
                return err;
            }
        }
    }

    while (*pOutputFrameNum < (int)m_frameBuf.size() && m_nFrame < m_ediSource.inframe()) {
        const auto *pBobCur = &m_bobSource.get(m_nFrame)->frame;
        const int srcIndex = m_inputSource.findIndexByInputFrameId(pBobCur->inputFrameId);
        if (srcIndex < 0 || (!draining && srcIndex + 1 >= m_inputSource.inframe())) {
            break;
        }

        auto pOutFrame = m_frameBuf[*pOutputFrameNum].get();
        ppOutputFrames[*pOutputFrameNum] = &pOutFrame->frame;

        const int prevIndex = (srcIndex == 0 && m_inputSource.inframe() > 1) ? srcIndex + 1 : srcIndex - 1;
        const auto *pSrcPrev = &m_inputSource.get(prevIndex)->frame;
        const auto *pSrcCur = &m_inputSource.get(srcIndex + 0)->frame;
        const auto *pSrcNext = &m_inputSource.get(srcIndex + 1)->frame;
        const int target = targetField(pBobCur, pSrcCur);

        auto err = processFrame(&pOutFrame->frame, pBobCur, pSrcPrev, pSrcCur, pSrcNext, prm, target, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        rtgmcEdiCopyFrameProps(&pOutFrame->frame, pBobCur, prm.mode);
        (*pOutputFrameNum)++;
        m_nFrame++;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::runNnediAdapter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream,
    const NVEncFilterParamRtgmcEdi &prm) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    if (!pBobInputFrame || !pBobInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!pSourceInputFrame || !pSourceInputFrame->ptr[0]) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi nnedi3 requires original source input frame.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (pSourceInputFrame && pSourceInputFrame->ptr[0]
        && m_inputSource.findIndexByInputFrameId(pSourceInputFrame->inputFrameId) < 0) {
        auto err = m_inputSource.add(pSourceInputFrame, stream, prm.chromaEdi == VppRtgmcChromaEdiMode::NNEDI3);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi NNEDI source frame: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    const int sourceIndex = m_inputSource.findIndexByInputFrameId(pBobInputFrame->inputFrameId);
    if (sourceIndex < 0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi NNEDI adapter source frame is missing for Bob inputFrameId=%d.\n"),
            pBobInputFrame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }

    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *mainFrame = nullptr;
    const RGYFrameInfo *mainSelected = nullptr;
    auto sts = runNnediAdapterState(m_nnediStates[0], pBobInputFrame, pSourceInputFrame, &mainFrame, &mainSelected, stream, prm, false);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYFrameInfo *chromaFrame = nullptr;
    const bool useChroma = prm.chromaEdi == VppRtgmcChromaEdiMode::NNEDI3;
    if (useChroma) {
        sts = runNnediAdapterState(m_nnediStates[1], pBobInputFrame, pSourceInputFrame, &chromaFrame, nullptr, stream, prm, true);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    auto dstY = getPlane(&pOutFrame->frame, RGY_PLANE_Y);
    const auto srcY = getPlane(mainFrame, RGY_PLANE_Y);
    auto err = copyPlaneAsync(&dstY, &srcY, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-edi NNEDI adapter luma output: %s.\n"), get_err_mes(err));
        return err;
    }

    if (useChroma) {
        const int chromaPlanes = std::min(3, (int)RGY_CSP_PLANES[pOutFrame->frame.csp]);
        for (int iplane = 1; iplane < chromaPlanes; iplane++) {
            auto dstPlane = getPlane(&pOutFrame->frame, (RGY_PLANE)iplane);
            const auto srcPlane = getPlane(chromaFrame, (RGY_PLANE)iplane);
            if (dstPlane.ptr[0] == nullptr || srcPlane.ptr[0] == nullptr) {
                continue;
            }
            err = copyPlaneAsync(&dstPlane, &srcPlane, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-edi NNEDI chroma plane %d: %s.\n"), iplane, get_err_mes(err));
                return err;
            }
        }
    } else {
        const int chromaPlanes = std::min(3, (int)RGY_CSP_PLANES[pOutFrame->frame.csp]);
        for (int iplane = 1; iplane < chromaPlanes; iplane++) {
            auto dstPlane = getPlane(&pOutFrame->frame, (RGY_PLANE)iplane);
            const auto srcPlane = getPlane(pBobInputFrame, (RGY_PLANE)iplane);
            if (dstPlane.ptr[0] == nullptr || srcPlane.ptr[0] == nullptr) {
                continue;
            }
            err = copyPlaneAsync(&dstPlane, &srcPlane, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-edi bob chroma plane %d: %s.\n"), iplane, get_err_mes(err));
                return err;
            }
        }
    }

    m_nnediAdapterCopyEvent.reset();
    sts = recordCudaEvent(stream, &m_nnediAdapterCopyEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    copyFramePropWithoutRes(&pOutFrame->frame, pBobInputFrame);
    pOutFrame->frame.picstruct = RGY_PICSTRUCT_FRAME;
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;
    m_nFrame++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::run_filter_impl(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream,
    const NVEncFilterParamRtgmcEdi &prm) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (prm.mode == VppRtgmcEdiMode::NNEDI3) {
        return runNnediAdapter(pBobInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, stream, prm);
    }

    if (!pBobInputFrame || !pBobInputFrame->ptr[0] || !pEdiInputFrame || !pEdiInputFrame->ptr[0]) {
        if (rtgmcEdiModeUsesTemporalYadif(prm.mode)) {
            return runTemporalYadif(nullptr, nullptr, nullptr, ppOutputFrames, pOutputFrameNum, stream, prm);
        }
        return RGY_ERR_NONE;
    }
    auto inputErr = checkInputs(pBobInputFrame, pEdiInputFrame);
    if (inputErr != RGY_ERR_NONE) {
        return inputErr;
    }

    auto pOutFrame = m_frameBuf[0].get();

    if (m_useKernel) {
        if (rtgmcEdiModeUsesTemporalYadif(prm.mode)) {
            return runTemporalYadif(pBobInputFrame, pEdiInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, stream, prm);
        }
        ppOutputFrames[0] = &pOutFrame->frame;
        *pOutputFrameNum = 1;
        const int target = targetField(pBobInputFrame);
        auto err = processFrame(&pOutFrame->frame, pBobInputFrame, pEdiInputFrame, pEdiInputFrame, pEdiInputFrame, prm, target, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        rtgmcEdiCopyFrameProps(&pOutFrame->frame, pEdiInputFrame, prm.mode);
        return RGY_ERR_NONE;
    }

    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;
    auto copyErr = copyFrameAsync(ppOutputFrames[0], pEdiInputFrame, stream);
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    rtgmcEdiCopyFrameProps(ppOutputFrames[0], pEdiInputFrame, prm.mode);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcEdi::run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcEdi>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return run_filter_impl(pBobInputFrame, pEdiInputFrame, pEdiInputFrame, ppOutputFrames, pOutputFrameNum, stream, *prm);
}

RGY_ERR NVEncFilterRtgmcEdi::run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcEdi>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return run_filter_impl(pBobInputFrame, pEdiInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, stream, *prm);
}

RGY_ERR NVEncFilterRtgmcEdi::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcEdi>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return run_filter_impl(pInputFrame, pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, stream, *prm);
}

void NVEncFilterRtgmcEdi::resetTemporalState() {
    // Reset all time-dependent state without releasing GPU buffers or rebuilt kernel/filter objects.
    // Rewind the source ring buffers but keep their GPU allocations (clear() would memfree them
    // and the next add() would fail with cudaErrorInvalidPitchValue on a pitch=0 destination).
    m_bobSource.resetFrames();
    m_ediSource.resetFrames();
    m_inputSource.resetFrames();
    // NNEDI adapter: clear cached frame references and validity flags, but keep the NNEDI filter
    // object itself (weights loaded) and outputCsp converter to avoid re-building.
    for (auto &state : m_nnediStates) {
        state.cachedFrames = { nullptr, nullptr };
        state.cachedKey = FrameKey();
        state.cachedEvent.reset();
        state.cacheValid = false;
    }
    m_nnediAdapterCopyEvent.reset();
    // Frame counters
    m_nFrame = 0;
    m_lastInputFrameId = -1;
    m_pairFrameIndex = 0;
    m_fallbackFrameIndex = 0;
}

void NVEncFilterRtgmcEdi::close() {
    m_buildOptions.clear();
    m_bobSource.clear();
    m_ediSource.clear();
    m_inputSource.clear();
    for (auto& state : m_nnediStates) {
        state.clear();
    }
    m_nnediAdapterCopyEvent.reset();
    m_frameBuf.clear();
    m_nFrame = 0;
    m_lastInputFrameId = -1;
    m_pairFrameIndex = 0;
    m_fallbackFrameIndex = 0;
    m_useKernel = false;
}
