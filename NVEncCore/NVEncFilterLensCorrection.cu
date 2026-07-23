// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The radial polynomial lens-distortion model implemented here is the standard
// Brown-Conrady model.  This is an independent implementation from that published maths.

#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterLensCorrection.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int LENSC_BLOCK_X = 32;
static const int LENSC_BLOCK_Y = 8;

template<typename Type>
__device__ __forceinline__ float lens_sample(const uint8_t *src, const int srcPitch, const int srcWidth, const int srcHeight,
    const int x, const int y, const float fillValue) {
    if (x < 0 || x >= srcWidth || y < 0 || y >= srcHeight) {
        return fillValue;
    }
    const auto ptr = (const Type *)(src + y * srcPitch + x * (int)sizeof(Type));
    return (float)ptr[0];
}

template<typename Type, int bit_depth>
__global__ void kernel_lenscorrection(
    uint8_t *dst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *src, const int srcPitch, const int srcWidth, const int srcHeight,
    const float k1, const float k2, const float cx, const float cy, const float fillValue) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }
    const float dx = (float)ix - cx * (float)dstWidth;
    const float dy = (float)iy - cy * (float)dstHeight;
    const float r0 = 0.5f * sqrtf((float)dstWidth * (float)dstWidth + (float)dstHeight * (float)dstHeight);
    const float rn = sqrtf(dx * dx + dy * dy) / r0;
    const float rn2 = rn * rn;
    const float scale = 1.0f + k1 * rn2 + k2 * rn2 * rn2;
    const float sx = cx * (float)srcWidth + dx * scale;
    const float sy = cy * (float)srcHeight + dy * scale;

    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;
    const float v00 = lens_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0, y0, fillValue);
    const float v10 = lens_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0 + 1, y0, fillValue);
    const float v01 = lens_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0, y0 + 1, fillValue);
    const float v11 = lens_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0 + 1, y0 + 1, fillValue);
    const float v = v00 * (1.0f - fx) * (1.0f - fy) + v10 * fx * (1.0f - fy)
                  + v01 * (1.0f - fx) * fy          + v11 * fx * fy;
    auto dstPix = (Type *)(dst + iy * dstPitch + ix * (int)sizeof(Type));
    const int maxValue = (1 << bit_depth) - 1;
    dstPix[0] = (Type)clamp((int)(v + 0.5f), 0, maxValue);
}

template<typename Type, int bit_depth>
static RGY_ERR lenscorrection_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    const std::shared_ptr<NVEncFilterParamLensCorrection> prm, const float fillValue, cudaStream_t stream) {
    dim3 block(LENSC_BLOCK_X, LENSC_BLOCK_Y);
    dim3 grid(divCeil(pOutputPlane->width, block.x), divCeil(pOutputPlane->height, block.y));
    kernel_lenscorrection<Type, bit_depth><<<grid, block, 0, stream>>>(
        pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
        prm->lenscorrection.k1, prm->lenscorrection.k2,
        prm->lenscorrection.cx, prm->lenscorrection.cy, fillValue);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}

NVEncFilterLensCorrection::NVEncFilterLensCorrection() {
    m_name = _T("lenscorrection");
}

NVEncFilterLensCorrection::~NVEncFilterLensCorrection() {
    close();
}

RGY_ERR NVEncFilterLensCorrection::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamLensCorrection>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterLensCorrection::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    const float fillValue, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamLensCorrection>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    if (RGY_CSP_BIT_DEPTH[pOutputPlane->csp] > 8) {
        return lenscorrection_plane<uint16_t, 16>(pOutputPlane, pInputPlane, prm, fillValue, stream);
    }
    return lenscorrection_plane<uint8_t, 8>(pOutputPlane, pInputPlane, prm, fillValue, stream);
}

RGY_ERR NVEncFilterLensCorrection::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const float fillValue = (i == 0) ? 0.0f : (float)(1 << (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] - 1));
        auto sts = procPlane(&planeDst, &planeSrc, fillValue, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterLensCorrection::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) return RGY_ERR_NONE;
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    return procFrame(ppOutputFrames[0], pInputFrame, stream);
}

void NVEncFilterLensCorrection::close() {
    m_frameBuf.clear();
}

tstring NVEncFilterParamLensCorrection::print() const {
    return lenscorrection.print();
}
