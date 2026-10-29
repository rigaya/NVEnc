// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterOnnx.h"
#include "NVEncFilterOnnxKernels.h"
#include "rgy_cuda_util_kernel.h"

template<typename Type, int bit_depth>
__global__ void kernel_onnx_pack_luma(float *__restrict__ dst,
    const uint8_t *__restrict__ src, const int srcPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const auto srcRow = (const Type *)(src + (size_t)y * srcPitch);
        dst[(size_t)y * width + x] = srcRow[x] * (1.0f / ((1 << bit_depth) - 1));
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_onnx_unpack_luma(uint8_t *__restrict__ dst, const int dstPitch,
    const float *__restrict__ src, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        constexpr int maxValue = (1 << bit_depth) - 1;
        const int value = (int)(src[(size_t)y * width + x] * maxValue + 0.5f);
        auto dstRow = (Type *)(dst + (size_t)y * dstPitch);
        dstRow[x] = (Type)max(0, min(value, maxValue));
    }
}

RGY_ERR run_onnx_pack_luma(float *dst, const RGYFrameInfo *src, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(src->width, (int)block.x), divCeil(src->height, (int)block.y));
    switch (src->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:    kernel_onnx_pack_luma<uint8_t,  8><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_09: kernel_onnx_pack_luma<uint16_t, 9><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_10: kernel_onnx_pack_luma<uint16_t,10><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_12: kernel_onnx_pack_luma<uint16_t,12><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_14: kernel_onnx_pack_luma<uint16_t,14><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: kernel_onnx_pack_luma<uint16_t,16><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_onnx_unpack_luma(RGYFrameInfo *dst, const float *src, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    switch (dst->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:    kernel_onnx_unpack_luma<uint8_t,  8><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_09: kernel_onnx_unpack_luma<uint16_t, 9><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_10: kernel_onnx_unpack_luma<uint16_t,10><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_12: kernel_onnx_unpack_luma<uint16_t,12><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_14: kernel_onnx_unpack_luma<uint16_t,14><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: kernel_onnx_unpack_luma<uint16_t,16><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
__device__ float onnx_sample_chroma(const uint8_t *src, int pitch, int stride, int offset,
    int cw, int ch, int x, int y) {
    const float fxpos = (x + 0.5f) * 0.5f - 0.5f;
    const float fypos = (y + 0.5f) * 0.5f - 0.5f;
    const int x0 = (int)floorf(fxpos), y0 = (int)floorf(fypos);
    const float fx = fxpos - x0, fy = fypos - y0;
    const int xa = max(0, min(x0, cw - 1)) * stride + offset;
    const int xb = max(0, min(x0 + 1, cw - 1)) * stride + offset;
    const auto row0 = (const Type *)(src + (size_t)max(0, min(y0, ch - 1)) * pitch);
    const auto row1 = (const Type *)(src + (size_t)max(0, min(y0 + 1, ch - 1)) * pitch);
    const float a = row0[xa], b = row0[xb], c = row1[xa], d = row1[xb];
    const float top = a + (b - a) * fx, bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

template<typename Type, int bit_depth, bool ycbcr>
__global__ void kernel_onnx_pack(float *__restrict__ dst, const uint8_t *yp, int ypitch,
    const uint8_t *up, int upitch, const uint8_t *vp, int vpitch, int stride, int voffset,
    int sw, int sh, int w, int h, int padL, int padT, OnnxColorCoeffs coeffs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int sx = max(0, min(x - padL, sw - 1)), sy = max(0, min(y - padT, sh - 1));
    const float yv = ((const Type *)(yp + (size_t)sy * ypitch))[sx];
    const float uv = onnx_sample_chroma<Type>(up, upitch, stride, 0, sw / 2, sh / 2, sx, sy);
    const float vv = onnx_sample_chroma<Type>(vp, vpitch, stride, voffset, sw / 2, sh / 2, sx, sy);
    const size_t i = (size_t)y * w + x, plane = (size_t)w * h;
    if (ycbcr) {
        constexpr float maxval = (1 << bit_depth) - 1;
        dst[i] = yv / maxval;
        dst[plane + i] = uv / maxval;
        dst[2 * plane + i] = vv / maxval;
    } else {
        const float yn = (yv - coeffs.yoff) * coeffs.yscale;
        const float un = (uv - coeffs.coff) * coeffs.cscale;
        const float vn = (vv - coeffs.coff) * coeffs.cscale;
        dst[i] = fminf(fmaxf(yn + coeffs.vr * vn, 0.0f), 1.0f);
        dst[plane + i] = fminf(fmaxf(yn + coeffs.ug * un + coeffs.vg * vn, 0.0f), 1.0f);
        dst[2 * plane + i] = fminf(fmaxf(yn + coeffs.ub * un, 0.0f), 1.0f);
    }
}

template<typename Type, int bit_depth, bool ycbcr>
__global__ void kernel_onnx_unpack_y(uint8_t *dst, int pitch, const float *src,
    int w, int h, OnnxColorCoeffs coeffs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * w + x, plane = (size_t)w * h;
    constexpr int maxval = (1 << bit_depth) - 1;
    const float value = ycbcr ? src[i] * maxval
        : (coeffs.ry * src[i] + coeffs.gy * src[plane + i] + coeffs.by * src[2 * plane + i]) * coeffs.yrange + coeffs.yoff;
    ((Type *)(dst + (size_t)y * pitch))[x] = (Type)max(0, min((int)(value + 0.5f), maxval));
}

template<typename Type, int bit_depth, bool ycbcr>
__global__ void kernel_onnx_unpack_uv(uint8_t *up, int upitch, uint8_t *vp, int vpitch,
    int stride, int voffset, const float *src, int w, int h, OnnxColorCoeffs coeffs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w / 2 || y >= h / 2) return;
    const size_t i = (size_t)(2 * y) * w + 2 * x, plane = (size_t)w * h;
    constexpr int maxval = (1 << bit_depth) - 1;
    float ua, va;
    if (ycbcr) {
        ua = (src[plane + i] + src[plane + i + 1] + src[plane + i + w] + src[plane + i + w + 1]) * 0.25f * maxval;
        va = (src[2 * plane + i] + src[2 * plane + i + 1] + src[2 * plane + i + w] + src[2 * plane + i + w + 1]) * 0.25f * maxval;
    } else {
        float u[4], v[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const size_t j = i + (k / 2) * w + (k % 2);
            const float r = src[j], g = src[plane + j], b = src[2 * plane + j];
            u[k] = coeffs.ru * r + coeffs.gu * g + coeffs.bu * b;
            v[k] = coeffs.rv * r + coeffs.gv * g + coeffs.bv * b;
        }
        ua = ((u[0] + u[1] + u[2] + u[3]) * 0.25f) * coeffs.crange + coeffs.coff;
        va = ((v[0] + v[1] + v[2] + v[3]) * 0.25f) * coeffs.crange + coeffs.coff;
    }
    ((Type *)(up + (size_t)y * upitch))[x * stride] = (Type)max(0, min((int)(ua + 0.5f), maxval));
    ((Type *)(vp + (size_t)y * vpitch))[x * stride + voffset] = (Type)max(0, min((int)(va + 0.5f), maxval));
}

template<typename Type, int bit_depth>
__global__ void kernel_onnx_pack_luma_padded(float *__restrict__ dst,
    const uint8_t *src, int pitch, int sw, int sh, int w, int h, int padL, int padT) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int sx = max(0, min(x - padL, sw - 1)), sy = max(0, min(y - padT, sh - 1));
    dst[(size_t)y * w + x] = ((const Type *)(src + (size_t)sy * pitch))[sx] / (float)((1 << bit_depth) - 1);
}

template<typename Type, int bit_depth, bool ycbcr>
static RGY_ERR onnx_pack(float *dst, const RGYFrameInfo *src, int w, int h,
    int padL, int padT, const OnnxColorCoeffs& coeffs, cudaStream_t stream) {
    const bool interleaved = src->csp == RGY_CSP_NV12 || src->csp == RGY_CSP_P010;
    const dim3 block(32, 8), grid(divCeil(w, (int)block.x), divCeil(h, (int)block.y));
    kernel_onnx_pack<Type, bit_depth, ycbcr><<<grid, block, 0, stream>>>(dst,
        src->ptr[0], src->pitch[0], src->ptr[1], src->pitch[1],
        interleaved ? src->ptr[1] : src->ptr[2], interleaved ? src->pitch[1] : src->pitch[2],
        interleaved ? 2 : 1, interleaved ? 1 : 0,
        src->width, src->height, w, h, padL, padT, coeffs);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth, bool ycbcr>
static RGY_ERR onnx_unpack(RGYFrameInfo *dst, const float *src, const OnnxColorCoeffs& coeffs, cudaStream_t stream) {
    const dim3 block(32, 8), grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_onnx_unpack_y<Type, bit_depth, ycbcr><<<grid, block, 0, stream>>>(
        dst->ptr[0], dst->pitch[0], src, dst->width, dst->height, coeffs);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err_to_rgy(err);
    const bool interleaved = dst->csp == RGY_CSP_NV12 || dst->csp == RGY_CSP_P010;
    const dim3 gridUV(divCeil(dst->width / 2, (int)block.x), divCeil(dst->height / 2, (int)block.y));
    kernel_onnx_unpack_uv<Type, bit_depth, ycbcr><<<gridUV, block, 0, stream>>>(
        dst->ptr[1], dst->pitch[1], interleaved ? dst->ptr[1] : dst->ptr[2],
        interleaved ? dst->pitch[1] : dst->pitch[2], interleaved ? 2 : 1,
        interleaved ? 1 : 0, src, dst->width, dst->height, coeffs);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR onnx_pack_luma_padded(float *dst, const RGYFrameInfo *src,
    int w, int h, int padL, int padT, cudaStream_t stream) {
    const dim3 block(32, 8), grid(divCeil(w, (int)block.x), divCeil(h, (int)block.y));
    kernel_onnx_pack_luma_padded<Type, bit_depth><<<grid, block, 0, stream>>>(
        dst, src->ptr[0], src->pitch[0], src->width, src->height, w, h, padL, padT);
    return err_to_rgy(cudaGetLastError());
}

template<bool ycbcr>
static RGY_ERR onnx_pack_dispatch(float *dst, const RGYFrameInfo *src, int w, int h,
    int padL, int padT, const OnnxColorCoeffs& coeffs, cudaStream_t stream) {
    switch (src->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12: return onnx_pack<uint8_t, 8, ycbcr>(dst, src, w, h, padL, padT, coeffs, stream);
    case RGY_CSP_YV12_09: return onnx_pack<uint16_t, 9, ycbcr>(dst, src, w, h, padL, padT, coeffs, stream);
    case RGY_CSP_YV12_10: return onnx_pack<uint16_t, 10, ycbcr>(dst, src, w, h, padL, padT, coeffs, stream);
    case RGY_CSP_YV12_12: return onnx_pack<uint16_t, 12, ycbcr>(dst, src, w, h, padL, padT, coeffs, stream);
    case RGY_CSP_YV12_14: return onnx_pack<uint16_t, 14, ycbcr>(dst, src, w, h, padL, padT, coeffs, stream);
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: return onnx_pack<uint16_t, 16, ycbcr>(dst, src, w, h, padL, padT, coeffs, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

template<bool ycbcr>
static RGY_ERR onnx_unpack_dispatch(RGYFrameInfo *dst, const float *src, const OnnxColorCoeffs& coeffs, cudaStream_t stream) {
    switch (dst->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12: return onnx_unpack<uint8_t, 8, ycbcr>(dst, src, coeffs, stream);
    case RGY_CSP_YV12_09: return onnx_unpack<uint16_t, 9, ycbcr>(dst, src, coeffs, stream);
    case RGY_CSP_YV12_10: return onnx_unpack<uint16_t, 10, ycbcr>(dst, src, coeffs, stream);
    case RGY_CSP_YV12_12: return onnx_unpack<uint16_t, 12, ycbcr>(dst, src, coeffs, stream);
    case RGY_CSP_YV12_14: return onnx_unpack<uint16_t, 14, ycbcr>(dst, src, coeffs, stream);
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: return onnx_unpack<uint16_t, 16, ycbcr>(dst, src, coeffs, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR run_onnx_pack(float *dst, const RGYFrameInfo *src, int w, int h,
    int padL, int padT, bool ycbcr, const OnnxColorCoeffs& coeffs, cudaStream_t stream) {
    return ycbcr ? onnx_pack_dispatch<true>(dst, src, w, h, padL, padT, coeffs, stream)
        : onnx_pack_dispatch<false>(dst, src, w, h, padL, padT, coeffs, stream);
}

RGY_ERR run_onnx_unpack(RGYFrameInfo *dst, const float *src, bool ycbcr,
    const OnnxColorCoeffs& coeffs, cudaStream_t stream) {
    return ycbcr ? onnx_unpack_dispatch<true>(dst, src, coeffs, stream)
        : onnx_unpack_dispatch<false>(dst, src, coeffs, stream);
}

RGY_ERR run_onnx_pack_luma_padded(float *dst, const RGYFrameInfo *src, int w, int h,
    int padL, int padT, cudaStream_t stream) {
    switch (src->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12: return onnx_pack_luma_padded<uint8_t, 8>(dst, src, w, h, padL, padT, stream);
    case RGY_CSP_YV12_09: return onnx_pack_luma_padded<uint16_t, 9>(dst, src, w, h, padL, padT, stream);
    case RGY_CSP_YV12_10: return onnx_pack_luma_padded<uint16_t, 10>(dst, src, w, h, padL, padT, stream);
    case RGY_CSP_YV12_12: return onnx_pack_luma_padded<uint16_t, 12>(dst, src, w, h, padL, padT, stream);
    case RGY_CSP_YV12_14: return onnx_pack_luma_padded<uint16_t, 14>(dst, src, w, h, padL, padT, stream);
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: return onnx_pack_luma_padded<uint16_t, 16>(dst, src, w, h, padL, padT, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

__global__ void kernel_onnx_fill(float *dst, size_t count, float value) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = value;
}

RGY_ERR run_onnx_fill(float *dst, size_t count, float value, cudaStream_t stream) {
    kernel_onnx_fill<<<divCeil(count, (size_t)256), 256, 0, stream>>>(dst, count, value);
    return err_to_rgy(cudaGetLastError());
}

__global__ void kernel_onnx_pad_planes(float *dst, const float *src, int sw, int sh,
    int dw, int dh, int padL, int padT) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const int sx = max(0, min(x - padL, sw - 1)), sy = max(0, min(y - padT, sh - 1));
    #pragma unroll
    for (int c = 0; c < 3; c++)
        dst[(size_t)c * dw * dh + (size_t)y * dw + x] = src[(size_t)c * sw * sh + (size_t)sy * sw + sx];
}

RGY_ERR run_onnx_pad_planes(float *dst, const float *src, int sw, int sh, int dw, int dh,
    int padL, int padT, cudaStream_t stream) {
    const dim3 block(32, 8), grid(divCeil(dw, (int)block.x), divCeil(dh, (int)block.y));
    kernel_onnx_pad_planes<<<grid, block, 0, stream>>>(dst, src, sw, sh, dw, dh, padL, padT);
    return err_to_rgy(cudaGetLastError());
}

__device__ float onnx_sample_float(const float *src, int w, int h, float x, float y) {
    const int x0 = (int)floorf(x), y0 = (int)floorf(y);
    const float wx = x - x0, wy = y - y0;
    const int xa = max(0, min(x0, w - 1)), xb = max(0, min(x0 + 1, w - 1));
    const int ya = max(0, min(y0, h - 1)), yb = max(0, min(y0 + 1, h - 1));
    const float a = src[ya * w + xa] * (1.0f - wx) + src[ya * w + xb] * wx;
    const float b = src[yb * w + xa] * (1.0f - wx) + src[yb * w + xb] * wx;
    return a * (1.0f - wy) + b * wy;
}

__global__ void kernel_onnx_mask_resize(const float *src, float *dst, int sw, int sh, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const float sx = dw > 1 ? ((x + 0.5f) * sw) / dw - 0.5f : 0.0f;
    const float sy = dh > 1 ? ((y + 0.5f) * sh) / dh - 0.5f : 0.0f;
    #pragma unroll
    for (int c = 0; c < 3; c++)
        dst[(size_t)c * dw * dh + (size_t)y * dw + x] = onnx_sample_float(src + (size_t)c * sw * sh, sw, sh, sx, sy);
}

RGY_ERR run_onnx_mask_resize(const float *src, float *dst, int sw, int sh, int dw, int dh, cudaStream_t stream) {
    const dim3 block(32, 8), grid(divCeil(dw, (int)block.x), divCeil(dh, (int)block.y));
    kernel_onnx_mask_resize<<<grid, block, 0, stream>>>(src, dst, sw, sh, dw, dh);
    return err_to_rgy(cudaGetLastError());
}

// OpenCL版と同じ256要素単位で判定し、全体のフラグへ反映する。
__global__ void kernel_onnx_mask_detect(const float *src, int count, int *large) {
    __shared__ int found[256];
    const int lane = threadIdx.x, i = blockIdx.x * blockDim.x + lane;
    found[lane] = i < count && src[i] > 2.0f;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) found[lane] |= found[lane + step];
        __syncthreads();
    }
    if (lane == 0 && found[0]) atomicOr(large, 1);
}

RGY_ERR run_onnx_mask_detect(const float *src, int count, int *large, cudaStream_t stream) {
    auto err = cudaMemsetAsync(large, 0, sizeof(int), stream);
    if (err != cudaSuccess) return err_to_rgy(err);
    kernel_onnx_mask_detect<<<divCeil(count, 256), 256, 0, stream>>>(src, count, large);
    return err_to_rgy(cudaGetLastError());
}

__global__ void kernel_onnx_mask_composite(const float *frame, const float *model, const float *mask,
    const int *large, float *dst, int fw, int fh, int mw, int mh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fw || y >= fh) return;
    const size_t i = (size_t)y * fw + x;
    const float mx = ((x + 0.5f) * mw) / fw - 0.5f;
    const float my = ((y + 0.5f) * mh) / fh - 0.5f;
    const float scale = *large ? (1.0f / 255.0f) : 1.0f;
    #pragma unroll
    for (int c = 0; c < 3; c++)
        dst[(size_t)c * fw * fh + i] = mask[i] > 0.5f
            ? fminf(fmaxf(onnx_sample_float(model + (size_t)c * mw * mh, mw, mh, mx, my) * scale, 0.0f), 1.0f)
            : frame[(size_t)c * fw * fh + i];
}

RGY_ERR run_onnx_mask_composite(const float *frame, const float *model, const float *mask,
    const int *large, float *dst, int fw, int fh, int mw, int mh, cudaStream_t stream) {
    const dim3 block(32, 8), grid(divCeil(fw, (int)block.x), divCeil(fh, (int)block.y));
    kernel_onnx_mask_composite<<<grid, block, 0, stream>>>(frame, model, mask, large, dst, fw, fh, mw, mh);
    return err_to_rgy(cudaGetLastError());
}
