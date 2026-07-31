// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterOnnx.h"
#include "rgy_cuda_util_kernel.h"

__global__ void kernel_onnx_pack_luma(float *__restrict__ dst,
    const uint8_t *__restrict__ src, const int srcPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        dst[(size_t)y * width + x] = src[(size_t)y * srcPitch + x] * (1.0f / 255.0f);
    }
}

__global__ void kernel_onnx_unpack_luma(uint8_t *__restrict__ dst, const int dstPitch,
    const float *__restrict__ src, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int value = (int)(src[(size_t)y * width + x] * 255.0f + 0.5f);
        dst[(size_t)y * dstPitch + x] = (uint8_t)max(0, min(value, 255));
    }
}

RGY_ERR run_onnx_pack_luma(float *dst, const RGYFrameInfo *src, cudaStream_t stream) {
    if (src->csp != RGY_CSP_NV12 && src->csp != RGY_CSP_YV12) return RGY_ERR_UNSUPPORTED;
    const dim3 block(32, 8);
    const dim3 grid(divCeil(src->width, (int)block.x), divCeil(src->height, (int)block.y));
    kernel_onnx_pack_luma<<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_onnx_unpack_luma(RGYFrameInfo *dst, const float *src, cudaStream_t stream) {
    if (dst->csp != RGY_CSP_NV12 && dst->csp != RGY_CSP_YV12) return RGY_ERR_UNSUPPORTED;
    const dim3 block(32, 8);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_onnx_unpack_luma<<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height);
    return err_to_rgy(cudaGetLastError());
}
