// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// Projection conversion (equirectangular / rectilinear / cubemap 3x2).

#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterV360.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int V360_BLOCK_X = 32;
static const int V360_BLOCK_Y = 8;
static constexpr float V360_PI_F = 3.14159265358979323846f;

enum { V360_PROJ_EQUIRECT = 0, V360_PROJ_FLAT = 1, V360_PROJ_CUBE = 2 };

template<int OUT_PROJ>
__device__ float3 v360_out_to_ray(const float fx, const float fy, const int W, const int H, const float hfov, int *valid) {
    *valid = 1;
    if constexpr (OUT_PROJ == V360_PROJ_FLAT) {
        const float f = (W * 0.5f) / tanf(hfov * 0.5f);
        return make_float3(fx - W * 0.5f + 0.5f, fy - H * 0.5f + 0.5f, f);
    } else if constexpr (OUT_PROJ == V360_PROJ_EQUIRECT) {
        const float lon = ((fx + 0.5f) / W - 0.5f) * 2.0f * V360_PI_F;
        const float lat = ((fy + 0.5f) / H - 0.5f) * V360_PI_F;
        return make_float3(cosf(lat) * sinf(lon), sinf(lat), cosf(lat) * cosf(lon));
    } else {
        const float cw = W / 3.0f, ch = H / 2.0f;
        const int col = (int)(fx / cw), row = (int)(fy / ch);
        const float a = (fx - col * cw) / cw * 2.0f - 1.0f;
        const float b = (fy - row * ch) / ch * 2.0f - 1.0f;
        const int face = row * 3 + col;
        if (face == 0) return make_float3(1.0f, b, -a);
        if (face == 1) return make_float3(-1.0f, b, a);
        if (face == 2) return make_float3(a, 1.0f, -b);
        if (face == 3) return make_float3(a, -1.0f, b);
        if (face == 4) return make_float3(a, b, 1.0f);
        return make_float3(-a, b, -1.0f);
    }
}

template<int IN_PROJ>
__device__ float2 v360_ray_to_in(const float3 d, const int W, const int H, const float hfov, int *valid) {
    *valid = 1;
    if constexpr (IN_PROJ == V360_PROJ_EQUIRECT) {
        const float lon = atan2f(d.x, d.z);
        const float lat = atan2f(d.y, sqrtf(d.x * d.x + d.z * d.z));
        float u = (lon / (2.0f * V360_PI_F) + 0.5f) * W;
        float v = (lat / V360_PI_F + 0.5f) * H;
        u = fmodf(u + W, (float)W);
        v = clamp(v, 0.0f, H - 1.0f);
        return make_float2(u, v);
    } else if constexpr (IN_PROJ == V360_PROJ_FLAT) {
        if (d.z <= 0.0f) { *valid = 0; return make_float2(0.0f, 0.0f); }
        const float f = (W * 0.5f) / tanf(hfov * 0.5f);
        const float u = d.x / d.z * f + W * 0.5f;
        const float v = d.y / d.z * f + H * 0.5f;
        if (u < 0.0f || u >= W || v < 0.0f || v >= H) *valid = 0;
        return make_float2(u, v);
    } else {
        const float ax = fabsf(d.x), ay = fabsf(d.y), az = fabsf(d.z);
        float a, b, dom; int row, col;
        if (ax >= ay && ax >= az) {
            if (d.x > 0.0f) { a = -d.z; b = d.y; dom = ax; row = 0; col = 0; }
            else            { a =  d.z; b = d.y; dom = ax; row = 0; col = 1; }
        } else if (ay >= ax && ay >= az) {
            if (d.y > 0.0f) { a = d.x; b = -d.z; dom = ay; row = 0; col = 2; }
            else            { a = d.x; b =  d.z; dom = ay; row = 1; col = 0; }
        } else {
            if (d.z > 0.0f) { a =  d.x; b = d.y; dom = az; row = 1; col = 1; }
            else            { a = -d.x; b = d.y; dom = az; row = 1; col = 2; }
        }
        const float uc = a / dom, vc = b / dom;
        const float cw = W / 3.0f, ch = H / 2.0f;
        return make_float2((col + (uc + 1.0f) * 0.5f) * cw, (row + (vc + 1.0f) * 0.5f) * ch);
    }
}

template<typename Type>
__device__ __forceinline__ float v360_sample(const uint8_t *src, const int srcPitch, const int srcWidth, const int srcHeight,
    const int x, const int y, const float fillValue) {
    if (x < 0 || x >= srcWidth || y < 0 || y >= srcHeight) return fillValue;
    return (float)((const Type *)(src + y * srcPitch + x * (int)sizeof(Type)))[0];
}

template<typename Type, int bit_depth, int IN_PROJ, int OUT_PROJ>
__global__ void kernel_v360(
    uint8_t *dst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *src, const int srcPitch, const int srcWidth, const int srcHeight,
    const float m00, const float m01, const float m02,
    const float m10, const float m11, const float m12,
    const float m20, const float m21, const float m22,
    const float out_hfov, const float in_hfov, const float fillValue) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) return;
    int valid;
    const float3 d = v360_out_to_ray<OUT_PROJ>((float)ix, (float)iy, dstWidth, dstHeight, out_hfov, &valid);
    const float3 w = make_float3(m00 * d.x + m01 * d.y + m02 * d.z,
        m10 * d.x + m11 * d.y + m12 * d.z,
        m20 * d.x + m21 * d.y + m22 * d.z);
    int valid2;
    const float2 s = v360_ray_to_in<IN_PROJ>(w, srcWidth, srcHeight, in_hfov, &valid2);
    auto dstPix = (Type *)(dst + iy * dstPitch + ix * (int)sizeof(Type));
    if (!valid || !valid2) {
        dstPix[0] = (Type)fillValue;
        return;
    }
    const int x0 = (int)floorf(s.x), y0 = (int)floorf(s.y);
    const float fx = s.x - x0, fy = s.y - y0;
    const float v00 = v360_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0, y0, fillValue);
    const float v10 = v360_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0 + 1, y0, fillValue);
    const float v01 = v360_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0, y0 + 1, fillValue);
    const float v11 = v360_sample<Type>(src, srcPitch, srcWidth, srcHeight, x0 + 1, y0 + 1, fillValue);
    const float val = v00 * (1.0f - fx) * (1.0f - fy) + v10 * fx * (1.0f - fy)
                    + v01 * (1.0f - fx) * fy          + v11 * fx * fy;
    const int maxValue = (1 << bit_depth) - 1;
    dstPix[0] = (Type)clamp((int)(val + 0.5f), 0, maxValue);
}

static void matmul3(const float A[9], const float B[9], float C[9]) {
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
        C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
    }
}

static void computeRot(const float yawDeg, const float pitchDeg, const float rollDeg, float R[9]) {
    const float a = yawDeg * (float)M_PI / 180.0f, b = pitchDeg * (float)M_PI / 180.0f, c = rollDeg * (float)M_PI / 180.0f;
    const float ca = cosf(a), sa = sinf(a), cb = cosf(b), sb = sinf(b), cc = cosf(c), sc = sinf(c);
    const float Ry[9] = { ca, 0.0f, sa, 0.0f, 1.0f, 0.0f, -sa, 0.0f, ca };
    const float Rp[9] = { 1.0f, 0.0f, 0.0f, 0.0f, cb, -sb, 0.0f, sb, cb };
    const float Rr[9] = { cc, -sc, 0.0f, sc, cc, 0.0f, 0.0f, 0.0f, 1.0f };
    float RpRr[9];
    matmul3(Rp, Rr, RpRr);
    matmul3(Ry, RpRr, R);
}

template<typename Type, int bit_depth, int IN_PROJ, int OUT_PROJ>
static RGY_ERR v360_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    const std::shared_ptr<NVEncFilterParamV360> prm, const float fillValue, cudaStream_t stream) {
    float R[9];
    computeRot(prm->v360.yaw, prm->v360.pitch, prm->v360.roll, R);
    dim3 block(V360_BLOCK_X, V360_BLOCK_Y);
    dim3 grid(divCeil(pOutputPlane->width, block.x), divCeil(pOutputPlane->height, block.y));
    kernel_v360<Type, bit_depth, IN_PROJ, OUT_PROJ><<<grid, block, 0, stream>>>(
        pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
        prm->v360.out_hfov * (float)M_PI / 180.0f, prm->v360.in_hfov * (float)M_PI / 180.0f, fillValue);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR v360_dispatch(RGYFrameInfo *out, const RGYFrameInfo *in,
    const std::shared_ptr<NVEncFilterParamV360> prm, const float fillValue, cudaStream_t stream) {
    const int inProj = prm->v360.in_proj, outProj = prm->v360.out_proj;
    if (inProj == V360_PROJ_EQUIRECT && outProj == V360_PROJ_EQUIRECT) return v360_plane<Type, bit_depth, 0, 0>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_EQUIRECT && outProj == V360_PROJ_FLAT)     return v360_plane<Type, bit_depth, 0, 1>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_EQUIRECT && outProj == V360_PROJ_CUBE)     return v360_plane<Type, bit_depth, 0, 2>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_FLAT && outProj == V360_PROJ_EQUIRECT)     return v360_plane<Type, bit_depth, 1, 0>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_FLAT && outProj == V360_PROJ_FLAT)         return v360_plane<Type, bit_depth, 1, 1>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_FLAT && outProj == V360_PROJ_CUBE)         return v360_plane<Type, bit_depth, 1, 2>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_CUBE && outProj == V360_PROJ_EQUIRECT)     return v360_plane<Type, bit_depth, 2, 0>(out, in, prm, fillValue, stream);
    if (inProj == V360_PROJ_CUBE && outProj == V360_PROJ_FLAT)         return v360_plane<Type, bit_depth, 2, 1>(out, in, prm, fillValue, stream);
    return v360_plane<Type, bit_depth, 2, 2>(out, in, prm, fillValue, stream);
}

NVEncFilterV360::NVEncFilterV360() {
    m_name = _T("v360");
}

NVEncFilterV360::~NVEncFilterV360() {
    close();
}

RGY_ERR NVEncFilterV360::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamV360>(pParam);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    if (prm->frameIn.height <= 0 || prm->frameIn.width <= 0) return RGY_ERR_INVALID_PARAM;
    int outW = (prm->v360.w > 0) ? prm->v360.w : prm->frameIn.width;
    int outH = (prm->v360.h > 0) ? prm->v360.h : prm->frameIn.height;
    outW &= ~1; outH &= ~1;
    prm->frameOut.width = outW;
    prm->frameOut.height = outH;
    auto sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterV360::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    const float fillValue, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamV360>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    if (RGY_CSP_BIT_DEPTH[pOutputPlane->csp] > 8) return v360_dispatch<uint16_t, 16>(pOutputPlane, pInputPlane, prm, fillValue, stream);
    return v360_dispatch<uint8_t, 8>(pOutputPlane, pInputPlane, prm, fillValue, stream);
}

RGY_ERR NVEncFilterV360::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const float fillValue = (i == 0) ? 0.0f : (float)(1 << (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] - 1));
        auto sts = procPlane(&planeDst, &planeSrc, fillValue, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterV360::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) return RGY_ERR_NONE;
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) return RGY_ERR_UNSUPPORTED;
    if (m_param->frameOut.csp != m_param->frameIn.csp) return RGY_ERR_UNSUPPORTED;
    return procFrame(ppOutputFrames[0], pInputFrame, stream);
}

void NVEncFilterV360::close() {
    m_frameBuf.clear();
}

tstring NVEncFilterParamV360::print() const {
    return v360.print();
}
