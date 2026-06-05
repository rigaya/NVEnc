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

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "convert_csp.h"
#include "NVEncFilterDescale.h"
#include "rgy_avutil.h"
#include "rgy_filter_input_probe.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int DESCALE_BLOCK = 32;

static inline double dsq(double x) { return x * x; }
static inline double dcb(double x) { return x * x * x; }

static inline double sinc_descale(double x) {
    const double pi = 3.14159265358979323846;
    return (x == 0.0) ? 1.0 : std::sin(x * pi) / (x * pi);
}

static double calculate_weight(VppDescaleKernel kernel, int support, double distance, double b, double c) {
    distance = std::fabs(distance);
    switch (kernel) {
    case VppDescaleKernel::Bilinear:
        return std::max(1.0 - distance, 0.0);
    case VppDescaleKernel::Bicubic:
        if (distance < 1.0) {
            return ((12.0 - 9.0 * b - 6.0 * c) * dcb(distance)
                + (-18.0 + 12.0 * b + 6.0 * c) * dsq(distance)
                + (6.0 - 2.0 * b)) / 6.0;
        } else if (distance < 2.0) {
            return ((-b - 6.0 * c) * dcb(distance)
                + (6.0 * b + 30.0 * c) * dsq(distance)
                + (-12.0 * b - 48.0 * c) * distance
                + (8.0 * b + 24.0 * c)) / 6.0;
        }
        return 0.0;
    case VppDescaleKernel::Lanczos2:
    case VppDescaleKernel::Lanczos3:
    case VppDescaleKernel::Lanczos4:
        return distance < support ? sinc_descale(distance) * sinc_descale(distance / support) : 0.0;
    case VppDescaleKernel::Spline16:
        if (distance < 1.0) {
            return 1.0 - (1.0 / 5.0 * distance) - (9.0 / 5.0 * dsq(distance)) + dcb(distance);
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-7.0 / 15.0 * distance) + (4.0 / 5.0 * dsq(distance)) - (1.0 / 3.0 * dcb(distance));
        }
        return 0.0;
    case VppDescaleKernel::Spline36:
        if (distance < 1.0) {
            return 1.0 - (3.0 / 209.0 * distance) - (453.0 / 209.0 * dsq(distance)) + (13.0 / 11.0 * dcb(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-156.0 / 209.0 * distance) + (270.0 / 209.0 * dsq(distance)) - (6.0 / 11.0 * dcb(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (26.0 / 209.0 * distance) - (45.0 / 209.0 * dsq(distance)) + (1.0 / 11.0 * dcb(distance));
        }
        return 0.0;
    case VppDescaleKernel::Spline64:
        if (distance < 1.0) {
            return 1.0 - (3.0 / 2911.0 * distance) - (6387.0 / 2911.0 * dsq(distance)) + (49.0 / 41.0 * dcb(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-2328.0 / 2911.0 * distance) + (4032.0 / 2911.0 * dsq(distance)) - (24.0 / 41.0 * dcb(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (582.0 / 2911.0 * distance) - (1008.0 / 2911.0 * dsq(distance)) + (6.0 / 41.0 * dcb(distance));
        } else if (distance < 4.0) {
            distance -= 3.0;
            return (-97.0 / 2911.0 * distance) + (168.0 / 2911.0 * dsq(distance)) - (1.0 / 41.0 * dcb(distance));
        }
        return 0.0;
    case VppDescaleKernel::Auto:
        return 0.0;
    }
    return 0.0;
}

static inline double round_halfup(double x) {
    return (x < 0) ? std::floor(x + 0.5) : std::floor(x + 0.49999999999999994);
}

static void build_scaling_weights(VppDescaleKernel kernel, int support,
    int src_dim, int dst_dim, double b, double c, double shift, double active_dim,
    VppDescaleBorder border, std::vector<double>& weights) {
    weights.assign((size_t)src_dim * dst_dim, 0.0);
    const double ratio = (double)dst_dim / active_dim;
    for (int i = 0; i < dst_dim; ++i) {
        double total = 0.0;
        const double pos = (i + 0.5) / ratio + shift;
        const double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; ++j) {
            const double xpos = begin_pos + j;
            total += calculate_weight(kernel, support, xpos - pos, b, c);
        }
        for (int j = 0; j < 2 * support; ++j) {
            const double xpos = begin_pos + j;
            double real_pos = xpos;
            if (xpos < 0.0 || xpos > src_dim) {
                if (border == VppDescaleBorder::Zero) {
                    continue;
                } else if (border == VppDescaleBorder::Repeat) {
                    real_pos = (xpos < 0.0) ? 0.0 : src_dim - 0.5;
                } else {
                    real_pos = (xpos < 0.0) ? -xpos : std::min(2.0 * src_dim - xpos, (double)src_dim - 0.5);
                }
            }
            const int idx = (int)std::floor(real_pos);
            const double w = calculate_weight(kernel, support, xpos - pos, b, c) / total;
            weights[(size_t)i * src_dim + idx] += w;
        }
    }
}

static void transpose_matrix(int rows, int cols, const std::vector<double>& src, std::vector<double>& dst) {
    dst.assign((size_t)cols * rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
        }
    }
}

static void multiply_sparse_matrices(int dst_dim, int src_dim,
    const std::vector<int>& lidx, const std::vector<int>& ridx,
    const std::vector<double>& lm, const std::vector<double>& rm,
    std::vector<double>& out) {
    out.assign((size_t)dst_dim * dst_dim, 0.0);
    for (int i = 0; i < dst_dim; ++i) {
        for (int j = 0; j < dst_dim; ++j) {
            double sum = 0.0;
            for (int k = lidx[i]; k < ridx[i]; ++k) {
                sum += lm[(size_t)i * src_dim + k] * rm[(size_t)k * dst_dim + j];
            }
            out[(size_t)i * dst_dim + j] = sum;
        }
    }
}

static void banded_ldlt(int n, int bandwidth, std::vector<double>& mat) {
    const int half = bandwidth / 2;
    const double eps = DBL_EPSILON;
    for (int i = 0; i < n; ++i) {
        const int end = std::min(half + 1, n - i);
        for (int j = 1; j < end; ++j) {
            const double d = mat[(size_t)i * n + i + j] / (mat[(size_t)i * n + i] + eps);
            for (int k = 0; k < end - j; ++k) {
                mat[(size_t)(i + j) * n + i + j + k] -= d * mat[(size_t)i * n + i + j + k];
            }
        }
        const double e = 1.0 / (mat[(size_t)i * n + i] + eps);
        for (int j = 1; j < end; ++j) {
            mat[(size_t)i * n + i + j] *= e;
        }
    }
}

static void multiply_banded_with_diagonal(int n, int bandwidth, std::vector<double>& mat) {
    const int half = bandwidth / 2;
    for (int i = 1; i < n; ++i) {
        const int start = std::max(i - half, 0);
        for (int j = start; j < i; ++j) {
            mat[(size_t)i * n + j] *= mat[(size_t)j * n + j];
        }
    }
}

static void pack_lower_upper_diag(int n, int bandwidth,
    const std::vector<double>& lower_full, const std::vector<double>& upper_full,
    std::vector<float>& lower_packed, std::vector<float>& upper_packed, std::vector<float>& diagonal) {
    const int half = bandwidth / 2;
    const double eps = DBL_EPSILON;
    lower_packed.assign((size_t)half * n, 0.0f);
    upper_packed.assign((size_t)half * n, 0.0f);
    diagonal.assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        const int start = std::max(i - half, 0);
        for (int j = start; j < i; ++j) {
            lower_packed[(size_t)(j - i + half) * n + i] = (float)lower_full[(size_t)i * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        const int start = std::min(i + half, n - 1);
        for (int j = start; j > i; --j) {
            upper_packed[(size_t)(j - i - 1) * n + i] = (float)upper_full[(size_t)i * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        diagonal[i] = (float)(1.0 / (lower_full[(size_t)i * n + i] + eps));
    }
}

static int kernel_support(VppDescaleKernel kernel) {
    switch (kernel) {
    case VppDescaleKernel::Bilinear: return 1;
    case VppDescaleKernel::Bicubic:  return 2;
    case VppDescaleKernel::Spline16: return 2;
    case VppDescaleKernel::Spline36: return 3;
    case VppDescaleKernel::Spline64: return 4;
    case VppDescaleKernel::Lanczos2: return 2;
    case VppDescaleKernel::Lanczos3: return 3;
    case VppDescaleKernel::Lanczos4: return 4;
    case VppDescaleKernel::Auto:     return 2;
    }
    return 2;
}

static RGY_ERR upload_buf(std::unique_ptr<CUMemBuf>& buf, const void *src, size_t bytes) {
    buf = std::make_unique<CUMemBuf>(bytes);
    auto sts = buf->alloc();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return err_to_rgy(cudaMemcpy(buf->ptr, src, bytes, cudaMemcpyHostToDevice));
}

template<typename Type, int bit_depth>
__global__ void kernel_descale_h(float *__restrict__ pDst, const int dstPitchFloats,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int src_h, const int dst_w,
    const int c_band, const int weights_columns,
    const float *__restrict__ weights,
    const int *__restrict__ left_idx, const int *__restrict__ right_idx,
    const float *__restrict__ lower, const float *__restrict__ upper,
    const float *__restrict__ diagonal) {
    const int iy = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy >= src_h) return;
    float *dstRow = pDst + iy * dstPitchFloats;
    const Type *srcRow = (const Type *)(pSrc + iy * srcPitch);
    for (int j = 0; j < dst_w; ++j) {
        const int lj = left_idx[j];
        const int rj = right_idx[j];
        float sum = 0.0f;
        for (int k = lj; k < rj; ++k) {
            const float src_f = (float)srcRow[k] * (1.0f / (float)((1 << bit_depth) - 1));
            sum += weights[j * weights_columns + (k - lj)] * src_f;
        }
        int start = j - c_band;
        if (start < 0) start = 0;
        for (int k = start; k < j; ++k) {
            sum -= lower[(k - j + c_band) * dst_w + j] * dstRow[k];
        }
        dstRow[j] = sum * diagonal[j];
    }
    for (int j = dst_w - 2; j >= 0; --j) {
        int end = j + c_band;
        if (end > dst_w - 1) end = dst_w - 1;
        float sum = 0.0f;
        for (int k = end; k > j; --k) {
            sum += upper[(k - j - 1) * dst_w + j] * dstRow[k];
        }
        dstRow[j] -= sum;
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_descale_v(uint8_t *__restrict__ pDst, const int dstPitch,
    float *__restrict__ pVScratch, const int scratchPitchFloats,
    const float *__restrict__ pSrc, const int srcPitchFloats,
    const int src_h, const int dst_w, const int dst_h,
    const int c_band, const int weights_columns,
    const float *__restrict__ weights,
    const int *__restrict__ left_idx, const int *__restrict__ right_idx,
    const float *__restrict__ lower, const float *__restrict__ upper,
    const float *__restrict__ diagonal,
    const int writeIntegerOutput) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= dst_w) return;
    for (int j = 0; j < dst_h; ++j) {
        const int lj = left_idx[j];
        const int rj = right_idx[j];
        float sum = 0.0f;
        for (int k = lj; k < rj; ++k) {
            sum += weights[j * weights_columns + (k - lj)] * pSrc[k * srcPitchFloats + ix];
        }
        int start = j - c_band;
        if (start < 0) start = 0;
        for (int k = start; k < j; ++k) {
            sum -= lower[(k - j + c_band) * dst_h + j] * pVScratch[k * scratchPitchFloats + ix];
        }
        pVScratch[j * scratchPitchFloats + ix] = sum * diagonal[j];
    }
    for (int j = dst_h - 2; j >= 0; --j) {
        int end = j + c_band;
        if (end > dst_h - 1) end = dst_h - 1;
        float sum = 0.0f;
        for (int k = end; k > j; --k) {
            sum += upper[(k - j - 1) * dst_h + j] * pVScratch[k * scratchPitchFloats + ix];
        }
        pVScratch[j * scratchPitchFloats + ix] -= sum;
    }
    if (writeIntegerOutput) {
        for (int j = 0; j < dst_h; ++j) {
            float v = pVScratch[j * scratchPitchFloats + ix];
            v = clamp(v, 0.0f, 1.0f);
            Type *outPtr = (Type *)(pDst + j * dstPitch);
            outPtr[ix] = (Type)(v * (float)((1 << bit_depth) - 1) + 0.5f);
        }
    }
}

__global__ void kernel_rescale_h(float *__restrict__ pDst, const int dstPitchFloats,
    const float *__restrict__ pSrc, const int srcPitchFloats,
    const int src_w_recon, const int src_h,
    const int dst_w_descaled,
    const int weights_columns,
    const float *__restrict__ weights,
    const int *__restrict__ left_idx,
    const int *__restrict__ right_idx) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= src_w_recon || iy >= src_h) return;
    const int lj = left_idx[ix];
    const int rj = right_idx[ix];
    float sum = 0.0f;
    const float *srcRow = pSrc + iy * srcPitchFloats;
    for (int k = lj; k < rj; ++k) {
        sum += weights[ix * weights_columns + (k - lj)] * srcRow[k];
    }
    pDst[iy * dstPitchFloats + ix] = sum;
}

__global__ void kernel_rescale_v(float *__restrict__ pDst, const int dstPitchFloats,
    const float *__restrict__ pSrc, const int srcPitchFloats,
    const int src_w, const int src_h_recon,
    const int dst_h_descaled,
    const int weights_columns,
    const float *__restrict__ weights,
    const int *__restrict__ left_idx,
    const int *__restrict__ right_idx) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= src_w || iy >= src_h_recon) return;
    const int lj = left_idx[iy];
    const int rj = right_idx[iy];
    float sum = 0.0f;
    for (int k = lj; k < rj; ++k) {
        sum += weights[iy * weights_columns + (k - lj)] * pSrc[k * srcPitchFloats + ix];
    }
    pDst[iy * dstPitchFloats + ix] = sum;
}

template<typename Type, int bit_depth>
__global__ void kernel_compute_edge_weight(const uint8_t *__restrict__ pSrc, const int srcPitch,
    float *__restrict__ pWeights, const int weightsPitchFloats,
    const int src_w, const int src_h) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= src_w || iy >= src_h) return;
    if (ix == 0 || ix == src_w - 1 || iy == 0 || iy == src_h - 1) {
        pWeights[iy * weightsPitchFloats + ix] = 0.0f;
        return;
    }
    const Type *row_m = (const Type *)(pSrc + (iy - 1) * srcPitch);
    const Type *row_0 = (const Type *)(pSrc + (iy    ) * srcPitch);
    const Type *row_p = (const Type *)(pSrc + (iy + 1) * srcPitch);
    const float inv_max = 1.0f / (float)((1 << bit_depth) - 1);
    const float gx =
          (float)((int)row_m[ix + 1] - (int)row_m[ix - 1]) * inv_max
        + 2.0f * (float)((int)row_0[ix + 1] - (int)row_0[ix - 1]) * inv_max
        + (float)((int)row_p[ix + 1] - (int)row_p[ix - 1]) * inv_max;
    const float gy =
          (float)((int)row_p[ix - 1] - (int)row_m[ix - 1]) * inv_max
        + 2.0f * (float)((int)row_p[ix    ] - (int)row_m[ix    ]) * inv_max
        + (float)((int)row_p[ix + 1] - (int)row_m[ix + 1]) * inv_max;
    pWeights[iy * weightsPitchFloats + ix] = sqrtf(gx * gx + gy * gy);
}

template<typename Type, int bit_depth>
__global__ void kernel_descale_mse(float *__restrict__ pRowSums,
    const uint8_t *__restrict__ pOrig, const int origPitch,
    const float *__restrict__ pRecon, const int reconPitchFloats,
    const float *__restrict__ pWeights, const int weightsPitchFloats,
    const int width, const int height) {
    const int iy = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy >= height) return;
    const Type *origRow = (const Type *)(pOrig + iy * origPitch);
    const float *reconRow = pRecon + iy * reconPitchFloats;
    const float *wRow = pWeights + iy * weightsPitchFloats;
    const float inv_max = 1.0f / (float)((1 << bit_depth) - 1);
    const float delta = 4.0f / 255.0f;
    const float delta_sq = delta * delta;
    float sum = 0.0f;
    volatile float comp = 0.0f;
    for (int x = 0; x < width; ++x) {
        const float o = (float)origRow[x] * inv_max;
        const float r = reconRow[x];
        const float d = o - r;
        const float abs_d = fabsf(d);
        const float loss = (abs_d < delta) ? (d * d) : (2.0f * delta * abs_d - delta_sq);
        const float term = (wRow[x] + 0.1f) * loss;
        const float y = term - comp;
        const volatile float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    pRowSums[iy] = sum;
}

template<typename Type, int bit_depth>
static RGY_ERR launch_descale_h(RGYFrameInfo *pIntermediateFloat, const RGYFrameInfo *pInputPlane,
    const NVEncFilterDescaleCore& core, cudaStream_t stream) {
    dim3 blockSize(DESCALE_BLOCK);
    dim3 gridSize(divCeil(pInputPlane->height, DESCALE_BLOCK));
    kernel_descale_h<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (float *)pIntermediateFloat->ptr[0], pIntermediateFloat->pitch[0] / (int)sizeof(float),
        (const uint8_t *)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->height, core.dst_dim,
        core.c, core.weights_columns,
        (const float *)core.weights->ptr,
        (const int *)core.left_idx->ptr, (const int *)core.right_idx->ptr,
        (const float *)core.lower->ptr, (const float *)core.upper->ptr, (const float *)core.diagonal->ptr);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR launch_descale_v(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pIntermediateFloat,
    CUMemBuf *pVScratch, const NVEncFilterDescaleCore& core, cudaStream_t stream) {
    dim3 blockSize(DESCALE_BLOCK);
    dim3 gridSize(divCeil(pOutputPlane->width, DESCALE_BLOCK));
    kernel_descale_v<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (float *)pVScratch->ptr, pOutputPlane->width,
        (const float *)pIntermediateFloat->ptr[0], pIntermediateFloat->pitch[0] / (int)sizeof(float),
        core.src_dim, pOutputPlane->width, core.dst_dim,
        core.c, core.weights_columns,
        (const float *)core.weights->ptr,
        (const int *)core.left_idx->ptr, (const int *)core.right_idx->ptr,
        (const float *)core.lower->ptr, (const float *)core.upper->ptr, (const float *)core.diagonal->ptr,
        1);
    return err_to_rgy(cudaGetLastError());
}

NVEncFilterDescale::NVEncFilterDescale() : NVEncFilter(), m_cores(), m_intermediateH(), m_intermediateV(), m_intermediatePitchFloats{}, m_frameIdx(0) {
    m_name = _T("descale");
}

NVEncFilterDescale::~NVEncFilterDescale() {
    close();
}

RGY_ERR NVEncFilterDescale::buildForwardWeights(ProbeForwardWeights& fw,
    int src_dim_low, int dst_dim_high,
    VppDescaleKernel kernel, double b, double c_param,
    double shift, VppDescaleBorder border) {
    const int support = kernel_support(kernel);
    if (support <= 0 || src_dim_low <= 0 || dst_dim_high <= src_dim_low) {
        return RGY_ERR_INVALID_PARAM;
    }
    std::vector<double> dense;
    build_scaling_weights(kernel, support, src_dim_low, dst_dim_high,
        b, c_param, shift, (double)src_dim_low, border, dense);

    std::vector<int> lidx(dst_dim_high, 0), ridx(dst_dim_high, 0);
    int maxw = 0;
    for (int i = 0; i < dst_dim_high; ++i) {
        int lj = 0;
        for (int j = 0; j < src_dim_low; ++j) {
            if (dense[(size_t)i * src_dim_low + j] != 0.0) { lj = j; break; }
        }
        int rj = 0;
        for (int j = src_dim_low - 1; j >= 0; --j) {
            if (dense[(size_t)i * src_dim_low + j] != 0.0) { rj = j + 1; break; }
        }
        lidx[i] = lj;
        ridx[i] = rj;
        maxw = std::max(maxw, rj - lj);
    }
    fw.weights_columns = maxw;
    std::vector<float> packed((size_t)dst_dim_high * maxw, 0.0f);
    for (int i = 0; i < dst_dim_high; ++i) {
        for (int j = 0; j < ridx[i] - lidx[i]; ++j) {
            packed[(size_t)i * maxw + j] = (float)dense[(size_t)i * src_dim_low + lidx[i] + j];
        }
    }
    auto sts = upload_buf(fw.weights, packed.data(), packed.size() * sizeof(float));
    if (sts == RGY_ERR_NONE) sts = upload_buf(fw.left_idx, lidx.data(), lidx.size() * sizeof(int));
    if (sts == RGY_ERR_NONE) sts = upload_buf(fw.right_idx, ridx.data(), ridx.size() * sizeof(int));
    return sts;
}

namespace {
struct DescaleAutoCandidate {
    VppDescaleKernel kernel;
    float b;
    float c;
    const TCHAR *label;
};
static const DescaleAutoCandidate kDescaleAutoCandidates[] = {
    { VppDescaleKernel::Bilinear, 0.0f,    0.0f,    _T("bilinear")             },
    { VppDescaleKernel::Bicubic,  0.0f,    0.5f,    _T("bicubic(b=0,c=0.5)")   },
    { VppDescaleKernel::Bicubic,  1.f/3.f, 1.f/3.f, _T("bicubic(b=1/3,c=1/3)") },
    { VppDescaleKernel::Bicubic,  0.0f,    0.75f,   _T("bicubic(b=0,c=0.75)")  },
    { VppDescaleKernel::Spline16, 0.0f,    0.0f,    _T("spline16")             },
    { VppDescaleKernel::Spline36, 0.0f,    0.0f,    _T("spline36")             },
    { VppDescaleKernel::Spline64, 0.0f,    0.0f,    _T("spline64")             },
    { VppDescaleKernel::Lanczos2, 0.0f,    0.0f,    _T("lanczos2")             },
    { VppDescaleKernel::Lanczos3, 0.0f,    0.0f,    _T("lanczos3")             },
    { VppDescaleKernel::Lanczos4, 0.0f,    0.0f,    _T("lanczos4")             },
};
static const int kAutoCandCount = (int)(sizeof(kDescaleAutoCandidates) / sizeof(kDescaleAutoCandidates[0]));
static const int kCommonNativeHeights[] = { 360, 480, 486, 540, 576, 720, 810, 900, 1080 };
static const int kCommonNativeHeightsCount = (int)(sizeof(kCommonNativeHeights) / sizeof(kCommonNativeHeights[0]));

struct CoarseRepKernel { VppDescaleKernel kernel; float b; float c; };
static const CoarseRepKernel kCoarseRepKernels[] = {
    { VppDescaleKernel::Bicubic, 0.0f, 0.5f },
    { VppDescaleKernel::Lanczos3, 0.0f, 0.0f },
    { VppDescaleKernel::Spline36, 0.0f, 0.0f },
};
static const int kCoarseRepKernelsCount = (int)(sizeof(kCoarseRepKernels) / sizeof(kCoarseRepKernels[0]));

static inline int round_up_even(int x) {
    return (x + 1) & ~1;
}

static inline AVRational resolve_source_sar(AVFormatContext *fmtCtx, AVStream *videoStream,
    const TCHAR **outSrcName) {
    AVRational sar = { 0, 0 };
    const TCHAR *src = _T("none");
    if (videoStream) {
        if (videoStream->codecpar) {
            sar = videoStream->codecpar->sample_aspect_ratio;
            if (sar.num > 0 && sar.den > 0) { src = _T("codecpar"); goto done; }
        }
        sar = videoStream->sample_aspect_ratio;
        if (sar.num > 0 && sar.den > 0) { src = _T("stream"); goto done; }
        sar = av_guess_sample_aspect_ratio(fmtCtx, videoStream, nullptr);
        if (sar.num > 0 && sar.den > 0) { src = _T("av_guess"); goto done; }
        sar = AVRational{ 0, 0 };
    }
done:
    if (outSrcName) *outSrcName = src;
    return sar;
}

static inline int width_from_height(int src_w, int src_h, int height,
    AVFormatContext *fmtCtx, AVStream *videoStream) {
    if (src_h <= 0) return 0;
    AVRational dar;
    AVRational sar = resolve_source_sar(fmtCtx, videoStream, nullptr);
    if (sar.num > 0 && sar.den > 0) {
        dar = av_mul_q(sar, av_make_q(src_w, src_h));
        av_reduce(&dar.num, &dar.den, dar.num, dar.den, 65536);
    } else {
        dar = av_make_q(src_w, src_h);
    }
    static const AVRational kStandardDARs[] = {
        { 4,  3 },
        { 16, 9 },
        { 21, 9 },
        { 1,  1 },
        { 2,  1 },
    };
    const double darF = (double)dar.num / (double)dar.den;
    for (const auto& sd : kStandardDARs) {
        const double sdF = (double)sd.num / (double)sd.den;
        if (std::fabs(darF - sdF) / sdF < 0.01) {
            dar = sd;
            break;
        }
    }
    const int w = (int)std::lround((double)height * (double)dar.num / (double)dar.den);
    return round_up_even(w);
}

static tstring probe_label_for(VppDescaleKernel k, float b, float c) {
    switch (k) {
    case VppDescaleKernel::Bilinear: return _T("bilinear");
    case VppDescaleKernel::Bicubic:  return strsprintf(_T("bicubic(b=%.3f,c=%.3f)"), b, c);
    case VppDescaleKernel::Spline16: return _T("spline16");
    case VppDescaleKernel::Spline36: return _T("spline36");
    case VppDescaleKernel::Spline64: return _T("spline64");
    case VppDescaleKernel::Lanczos2: return _T("lanczos2");
    case VppDescaleKernel::Lanczos3: return _T("lanczos3");
    case VppDescaleKernel::Lanczos4: return _T("lanczos4");
    case VppDescaleKernel::Auto:     return _T("auto");
    }
    return _T("?");
}
}

RGY_ERR NVEncFilterDescale::scoreCandidates(std::vector<ProbeCandidate>& candidates,
    const std::vector<std::unique_ptr<CUMemBuf>>& lumaBufs,
    const std::vector<std::unique_ptr<CUMemBuf>>& edgeWeightsBufs,
    int src_w, int src_h, int src_pixel_bytes,
    bool symmetricForward) {
    if (lumaBufs.empty() || edgeWeightsBufs.size() != lumaBufs.size()) return RGY_ERR_INVALID_PARAM;
    const int src_pitch_bytes = src_w * src_pixel_bytes;
    const int edge_pitch_floats = src_w;
    auto alloc_buf = [](size_t bytes) {
        auto buf = std::make_unique<CUMemBuf>(bytes);
        return buf->alloc() == RGY_ERR_NONE ? std::move(buf) : nullptr;
    };

    int candIdx = 0;
    for (auto& c : candidates) {
        AddMessage(RGY_LOG_DEBUG, _T("probe: candidate %d/%d %s %dx%d\n"),
            candIdx + 1, (int)candidates.size(), c.label.c_str(), c.width, c.height);
        if (c.width <= 0 || c.height <= 0 || c.width >= src_w || c.height >= src_h) {
            c.mse = 1e30; candIdx++; continue;
        }

        NVEncFilterDescaleCore coreH, coreV;
        if (prepareCore(coreH, src_w, c.width, c.kernel, c.b, c.c, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE
            || prepareCore(coreV, src_h, c.height, c.kernel, c.b, c.c, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("probe: prepareCore failed for %s %dx%d, skipping.\n"),
                c.label.c_str(), c.width, c.height);
            c.mse = 1e30; candIdx++; continue;
        }
        ProbeForwardWeights fwH, fwV;
        const VppDescaleKernel fwKernel = symmetricForward ? c.kernel : VppDescaleKernel::Bicubic;
        const float fwB = symmetricForward ? c.b : 0.0f;
        const float fwC = symmetricForward ? c.c : 0.5f;
        if (buildForwardWeights(fwH, c.width, src_w, fwKernel, fwB, fwC, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE
            || buildForwardWeights(fwV, c.height, src_h, fwKernel, fwB, fwC, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("probe: buildForwardWeights failed for %s %dx%d, skipping.\n"),
                c.label.c_str(), c.width, c.height);
            c.mse = 1e30; candIdx++; continue;
        }

        auto bufDescaleH = alloc_buf((size_t)c.width * src_h * sizeof(float));
        auto bufDescaleV = alloc_buf((size_t)c.width * c.height * sizeof(float));
        auto bufRescaleH = alloc_buf((size_t)src_w * c.height * sizeof(float));
        auto bufReupV = alloc_buf((size_t)src_w * src_h * sizeof(float));
        auto bufRowSums = alloc_buf((size_t)src_h * sizeof(float));
        if (!bufDescaleH || !bufDescaleV || !bufRescaleH || !bufReupV || !bufRowSums) {
            AddMessage(RGY_LOG_WARN, _T("probe: scratch allocation failed for %s %dx%d, skipping.\n"),
                c.label.c_str(), c.width, c.height);
            c.mse = 1e30; candIdx++; continue;
        }

        double accumSSD = 0.0;
        for (size_t fi = 0; fi < lumaBufs.size(); ++fi) {
            const auto& luma_buf = lumaBufs[fi];
            const auto& edge_buf = edgeWeightsBufs[fi];
            const dim3 block1(DESCALE_BLOCK);
            if (src_pixel_bytes == 1) {
                kernel_descale_h<uint8_t, 8><<<divCeil(src_h, DESCALE_BLOCK), block1>>>(
                    (float *)bufDescaleH->ptr, c.width,
                    (const uint8_t *)luma_buf->ptr, src_pitch_bytes,
                    src_h, c.width,
                    coreH.c, coreH.weights_columns,
                    (const float *)coreH.weights->ptr, (const int *)coreH.left_idx->ptr, (const int *)coreH.right_idx->ptr,
                    (const float *)coreH.lower->ptr, (const float *)coreH.upper->ptr, (const float *)coreH.diagonal->ptr);
                kernel_descale_v<uint8_t, 8><<<divCeil(c.width, DESCALE_BLOCK), block1>>>(
                    (uint8_t *)bufDescaleV->ptr, c.width * src_pixel_bytes,
                    (float *)bufDescaleV->ptr, c.width,
                    (const float *)bufDescaleH->ptr, c.width,
                    src_h, c.width, c.height,
                    coreV.c, coreV.weights_columns,
                    (const float *)coreV.weights->ptr, (const int *)coreV.left_idx->ptr, (const int *)coreV.right_idx->ptr,
                    (const float *)coreV.lower->ptr, (const float *)coreV.upper->ptr, (const float *)coreV.diagonal->ptr,
                    0);
            } else {
                kernel_descale_h<uint16_t, 16><<<divCeil(src_h, DESCALE_BLOCK), block1>>>(
                    (float *)bufDescaleH->ptr, c.width,
                    (const uint8_t *)luma_buf->ptr, src_pitch_bytes,
                    src_h, c.width,
                    coreH.c, coreH.weights_columns,
                    (const float *)coreH.weights->ptr, (const int *)coreH.left_idx->ptr, (const int *)coreH.right_idx->ptr,
                    (const float *)coreH.lower->ptr, (const float *)coreH.upper->ptr, (const float *)coreH.diagonal->ptr);
                kernel_descale_v<uint16_t, 16><<<divCeil(c.width, DESCALE_BLOCK), block1>>>(
                    (uint8_t *)bufDescaleV->ptr, c.width * src_pixel_bytes,
                    (float *)bufDescaleV->ptr, c.width,
                    (const float *)bufDescaleH->ptr, c.width,
                    src_h, c.width, c.height,
                    coreV.c, coreV.weights_columns,
                    (const float *)coreV.weights->ptr, (const int *)coreV.left_idx->ptr, (const int *)coreV.right_idx->ptr,
                    (const float *)coreV.lower->ptr, (const float *)coreV.upper->ptr, (const float *)coreV.diagonal->ptr,
                    0);
            }
            dim3 block2(32, 8);
            kernel_rescale_h<<<dim3(divCeil(src_w, 32), divCeil(c.height, 8)), block2>>>(
                (float *)bufRescaleH->ptr, src_w,
                (const float *)bufDescaleV->ptr, c.width,
                src_w, c.height, c.width,
                fwH.weights_columns,
                (const float *)fwH.weights->ptr, (const int *)fwH.left_idx->ptr, (const int *)fwH.right_idx->ptr);
            kernel_rescale_v<<<dim3(divCeil(src_w, 32), divCeil(src_h, 8)), block2>>>(
                (float *)bufReupV->ptr, src_w,
                (const float *)bufRescaleH->ptr, src_w,
                src_w, src_h, c.height,
                fwV.weights_columns,
                (const float *)fwV.weights->ptr, (const int *)fwV.left_idx->ptr, (const int *)fwV.right_idx->ptr);
            if (src_pixel_bytes == 1) {
                kernel_descale_mse<uint8_t, 8><<<divCeil(src_h, DESCALE_BLOCK), block1>>>(
                    (float *)bufRowSums->ptr,
                    (const uint8_t *)luma_buf->ptr, src_pitch_bytes,
                    (const float *)bufReupV->ptr, src_w,
                    (const float *)edge_buf->ptr, edge_pitch_floats,
                    src_w, src_h);
            } else {
                kernel_descale_mse<uint16_t, 16><<<divCeil(src_h, DESCALE_BLOCK), block1>>>(
                    (float *)bufRowSums->ptr,
                    (const uint8_t *)luma_buf->ptr, src_pitch_bytes,
                    (const float *)bufReupV->ptr, src_w,
                    (const float *)edge_buf->ptr, edge_pitch_floats,
                    src_w, src_h);
            }
            auto err = err_to_rgy(cudaGetLastError());
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_WARN, _T("probe: kernel launch failed for %s: %s.\n"),
                    c.label.c_str(), get_err_mes(err));
                accumSSD = 1e30;
                break;
            }
            std::vector<float> rowSums(src_h);
            err = err_to_rgy(cudaMemcpy(rowSums.data(), bufRowSums->ptr, src_h * sizeof(float), cudaMemcpyDeviceToHost));
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_WARN, _T("probe: row sum readback failed for %s: %s.\n"),
                    c.label.c_str(), get_err_mes(err));
                accumSSD = 1e30;
                break;
            }
            double frameSSD = 0.0;
            for (float s : rowSums) frameSSD += (double)s;
            accumSSD += frameSSD;
        }
        c.mse = accumSSD / ((double)src_w * src_h * lumaBufs.size());
        AddMessage(RGY_LOG_DEBUG, _T("probe: candidate %d/%d %s %dx%d mse=%.6e\n"),
            candIdx + 1, (int)candidates.size(), c.label.c_str(), c.width, c.height, c.mse);
        candIdx++;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDescale::runResolutionSearch(NVEncFilterParamDescale *prm,
    std::vector<ProbeCandidate>& candidates,
    const std::vector<std::unique_ptr<CUMemBuf>>& lumaBufs,
    const std::vector<std::unique_ptr<CUMemBuf>>& edgeWeightsBufs,
    int src_w, int src_h, int src_pixel_bytes,
    AVFormatContext *fmtCtx, AVStream *videoStream) {
    int searchMin = prm->descale.search_min > 0 ? prm->descale.search_min : (int)(src_h * 0.48);
    int searchMax = prm->descale.search_max > 0 ? prm->descale.search_max : (int)(src_h * 0.85);
    if (searchMin < 2) searchMin = 2;
    if (searchMax >= src_h) searchMax = src_h - 1;
    if (searchMin > searchMax) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect: search range invalid (search_min=%d > search_max=%d).\n"),
            searchMin, searchMax);
        return RGY_ERR_INVALID_PARAM;
    }
    const int searchStep = std::max(1, prm->descale.search_step);
    const int coarseStep = std::max(2, searchStep * 8);
    const TCHAR *sarSrc = nullptr;
    const AVRational diagSar = resolve_source_sar(fmtCtx, videoStream, &sarSrc);
    AVRational diagDar = av_make_q(src_w, src_h);
    if (diagSar.num > 0 && diagSar.den > 0) {
        diagDar = av_mul_q(diagSar, av_make_q(src_w, src_h));
        av_reduce(&diagDar.num, &diagDar.den, diagDar.num, diagDar.den, 65536);
        AddMessage(RGY_LOG_DEBUG,
            _T("probe: SAR %d:%d from %s, derived DAR %d:%d for width calc.\n"),
            diagSar.num, diagSar.den, sarSrc, diagDar.num, diagDar.den);
    } else {
        AddMessage(RGY_LOG_DEBUG,
            _T("probe: SAR not found in any libav source; falling back to pixel ratio %d:%d.\n"),
            src_w, src_h);
    }
    std::set<int> loggedHeights;
    const bool sarValid = (diagSar.num > 0 && diagSar.den > 0);
    auto append_candidate = [&](VppDescaleKernel k, float b, float c, int height) {
        const int h = round_up_even(height);
        if (h < searchMin || h > searchMax) return;
        const int w = width_from_height(src_w, src_h, h, fmtCtx, videoStream);
        if (w <= 0 || w >= src_w) return;
        if (loggedHeights.insert(h).second) {
            if (sarValid) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("probe: width_from_height h=%d via SAR %d:%d (%s) -> DAR %d:%d -> width=%d.\n"),
                    h, diagSar.num, diagSar.den, sarSrc, diagDar.num, diagDar.den, w);
            } else {
                AddMessage(RGY_LOG_DEBUG,
                    _T("probe: width_from_height h=%d via pixel ratio %d:%d -> width=%d.\n"),
                    h, src_w, src_h, w);
            }
        }
        for (const auto& existing : candidates) {
            if (existing.kernel == k && existing.b == b && existing.c == c
                && existing.width == w && existing.height == h) return;
        }
        candidates.push_back(ProbeCandidate{ k, b, c, w, h, 0.0, probe_label_for(k, b, c) });
    };

    const size_t pass0Begin = candidates.size();
    for (int i = 0; i < kCommonNativeHeightsCount; ++i) {
        for (const auto& kv : kDescaleAutoCandidates) {
            append_candidate(kv.kernel, kv.b, kv.c, kCommonNativeHeights[i]);
        }
    }
    const size_t pass0End = candidates.size();
    if (pass0Begin == pass0End) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect: no common native heights fall within search range [%d, %d].\n"),
            searchMin, searchMax);
        return RGY_ERR_INVALID_PARAM;
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: Pass 0 - %d common-height candidates.\n"),
        (int)(pass0End - pass0Begin));
    auto err = scoreCandidates(candidates, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
    if (err != RGY_ERR_NONE) return err;

    const size_t pass1Begin = candidates.size();
    for (int h = searchMin; h <= searchMax; h += coarseStep) {
        for (int k = 0; k < kCoarseRepKernelsCount; ++k) {
            append_candidate(kCoarseRepKernels[k].kernel, kCoarseRepKernels[k].b, kCoarseRepKernels[k].c, h);
        }
    }
    const size_t pass1End = candidates.size();
    AddMessage(RGY_LOG_DEBUG, _T("probe: Pass 1 - %d coarse-stride candidates (step=%d).\n"),
        (int)(pass1End - pass1Begin), coarseStep);
    if (pass1End > pass1Begin) {
        std::vector<ProbeCandidate> tail(candidates.begin() + pass1Begin, candidates.end());
        err = scoreCandidates(tail, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
        if (err != RGY_ERR_NONE) return err;
        for (size_t i = 0; i < tail.size(); ++i) {
            candidates[pass1Begin + i].mse = tail[i].mse;
        }
    }

    int bestH = 0;
    double bestMse = 1e30;
    for (const auto& c : candidates) {
        if (std::isfinite(c.mse) && c.mse >= 0.0 && c.mse < bestMse) {
            bestMse = c.mse;
            bestH = c.height;
        }
    }
    if (bestH > 0) {
        const int loH = std::max(searchMin, bestH - coarseStep);
        const int hiH = std::min(searchMax, bestH + coarseStep);
        const size_t pass2Begin = candidates.size();
        for (int h = loH; h <= hiH; h += searchStep) {
            for (const auto& kv : kDescaleAutoCandidates) {
                append_candidate(kv.kernel, kv.b, kv.c, h);
            }
        }
        const size_t pass2End = candidates.size();
        AddMessage(RGY_LOG_DEBUG,
            _T("probe: Pass 2 - %d fine-refine candidates around best %dp.\n"),
            (int)(pass2End - pass2Begin), bestH);
        if (pass2End > pass2Begin) {
            std::vector<ProbeCandidate> tail(candidates.begin() + pass2Begin, candidates.end());
            err = scoreCandidates(tail, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
            if (err != RGY_ERR_NONE) return err;
            for (size_t i = 0; i < tail.size(); ++i) {
                candidates[pass2Begin + i].mse = tail[i].mse;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDescale::runProbe(NVEncFilterParamDescale *prm) {
    if (!prm) return RGY_ERR_NULL_PTR;
    if ((prm->descale.width > 0) != (prm->descale.height > 0)) {
        AddMessage(RGY_LOG_ERROR, _T("auto-detect needs both width= and height= set, or neither.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->inputFilePath.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("auto-detect requires a re-openable input file (got empty path).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto probeStart = std::chrono::steady_clock::now();
    const int src_w = prm->frameIn.width;
    const int src_h = prm->frameIn.height;
    const int dst_w = prm->descale.width;
    const int dst_h = prm->descale.height;
    const int bit_depth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int src_pixel_bytes = (bit_depth > 8) ? 2 : 1;

    std::string fileUtf8;
    if (tchar_to_string(prm->inputFilePath.c_str(), fileUtf8, CP_UTF8) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: failed to convert filename to utf-8.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (const auto protocol = unsupportedProbeProtocol(fileUtf8); protocol != nullptr) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect requires a re-openable input file, but input protocol is %s.\n")
            _T("    Please pass kernel=<concrete>,width=<int>,height=<int> explicitly.\n"),
            char_to_tstring(protocol).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    const int savedAvLogLevel = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    struct AvLogLevelRestorer { int prev; ~AvLogLevelRestorer() { av_log_set_level(prev); } } avGuard{ savedAvLogLevel };

    AVFormatContext *fmtCtxRaw = nullptr;
    if (avformat_open_input(&fmtCtxRaw, fileUtf8.c_str(), nullptr, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avformat_open_input failed.\n"));
        return RGY_ERR_FILE_OPEN;
    }
    std::unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> fmtGuard(
        fmtCtxRaw, RGYAVDeleter<AVFormatContext>(avformat_close_input));
    AVFormatContext *fmtCtx = fmtGuard.get();
    if (fmtCtx->pb != nullptr && !(fmtCtx->pb->seekable & AVIO_SEEKABLE_NORMAL)) {
        AddMessage(RGY_LOG_WARN,
            _T("source not seekable; kernel=auto requires seekable input.\n")
            _T("    Please pass kernel=<concrete> explicitly.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avformat_find_stream_info failed.\n"));
        return RGY_ERR_UNKNOWN;
    }
    const int videoIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoIdx < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: no video stream.\n"));
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    AVStream *vst = fmtCtx->streams[videoIdx];
    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) {
        AddMessage(RGY_LOG_ERROR, _T("probe: decoder not available for stream.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVCodecContext *codecCtxRaw = avcodec_alloc_context3(codec);
    if (!codecCtxRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> codecGuard(
        codecCtxRaw, RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    AVCodecContext *codecCtx = codecGuard.get();
    if (avcodec_parameters_to_context(codecCtx, vst->codecpar) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avcodec_parameters_to_context failed.\n"));
        return RGY_ERR_UNKNOWN;
    }
    codecCtx->time_base = vst->time_base;
    codecCtx->pkt_timebase = vst->time_base;
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avcodec_open2 failed.\n"));
        return RGY_ERR_UNKNOWN;
    }

    const int wantFrames = std::max(1, prm->descale.detect_frames);
    std::vector<std::vector<uint8_t>> lumaFrames;
    lumaFrames.reserve(wantFrames);
    AVPacket *pktRaw = av_packet_alloc();
    if (!pktRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> pktGuard(
        pktRaw, RGYAVDeleter<AVPacket>(av_packet_free));
    AVFrame *frameRaw = av_frame_alloc();
    if (!frameRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> frameGuard(
        frameRaw, RGYAVDeleter<AVFrame>(av_frame_free));

    int keyframeCaptures = 0;
    int nonKeyCaptures = 0;
    int totalDecoded = 0;
    bool requireKey = true;
    const int decodeAttemptCap = std::max(8, wantFrames * 4);
    auto captureOneFrame = [&]() -> bool {
        int rv = avcodec_receive_frame(codecCtx, frameGuard.get());
        if (rv != 0) return false;
        AVFrame *f = frameGuard.get();
        totalDecoded++;
#ifdef AV_FRAME_FLAG_KEY
        const bool isKey = (f->flags & AV_FRAME_FLAG_KEY) != 0;
#else
        const bool isKey = f->key_frame != 0;
#endif
        bool captured = false;
        if (f->width == src_w && f->height == src_h && (!requireKey || isKey)) {
            std::vector<uint8_t> luma((size_t)src_w * src_h * src_pixel_bytes);
            const int rowBytes = src_w * src_pixel_bytes;
            for (int y = 0; y < src_h; ++y) {
                memcpy(luma.data() + (size_t)y * rowBytes,
                    f->data[0] + (size_t)y * f->linesize[0],
                    rowBytes);
            }
            lumaFrames.push_back(std::move(luma));
            if (isKey) ++keyframeCaptures; else ++nonKeyCaptures;
            captured = true;
        }
        av_frame_unref(f);
        return captured;
    };

    int64_t streamDuration = vst->duration;
    if (streamDuration <= 0 && fmtCtx->duration > 0) {
        streamDuration = av_rescale_q(fmtCtx->duration, AVRational{ 1, AV_TIME_BASE }, vst->time_base);
    }
    const int64_t baseStartTs = (vst->start_time != AV_NOPTS_VALUE) ? vst->start_time : 0;
    const AVRational fr = vst->avg_frame_rate;
    auto frameToTs = [&](int frame_idx) -> int64_t {
        if (fr.num <= 0 || fr.den <= 0) return -1;
        return baseStartTs + av_rescale_q((int64_t)frame_idx, av_inv_q(fr), vst->time_base);
    };
    int64_t windowStartTs = baseStartTs;
    int64_t windowSpanTs = streamDuration > 0 ? (int64_t)(streamDuration * 0.80) : 0;
    int64_t windowOffsetTs = streamDuration > 0 ? (int64_t)(streamDuration * 0.10) : 0;
    bool useTrimWindow = false;
    if (prm->probeStartFrame > 0 && prm->probeEndFrame > prm->probeStartFrame) {
        const int64_t startTs = frameToTs(prm->probeStartFrame);
        const int64_t endTs = frameToTs(prm->probeEndFrame);
        if (startTs > 0 && endTs > startTs) {
            windowStartTs = startTs;
            windowOffsetTs = 0;
            windowSpanTs = endTs - startTs;
            useTrimWindow = true;
            AddMessage(RGY_LOG_DEBUG,
                _T("probe: sampling within --trim window [frame %d, frame %d).\n"),
                prm->probeStartFrame, prm->probeEndFrame);
        }
    }
    bool seekMode = (windowSpanTs > 0 && wantFrames > 1);
    int seekFailures = 0;
    const bool resSearch = (prm->descale.width <= 0 || prm->descale.height <= 0);
    tstring targetDesc = resSearch
        ? strsprintf(_T("heights %d-%d step %d"),
            prm->descale.search_min > 0 ? prm->descale.search_min : (src_h / 2),
            prm->descale.search_max > 0 ? prm->descale.search_max : (int)(src_h * 0.85),
            std::max(1, prm->descale.search_step))
        : strsprintf(_T("%dx%d"), dst_w, dst_h);
    AddMessage(RGY_LOG_INFO,
        useTrimWindow
            ? _T("auto-detect: probing %s over %d frames (window=[frame %d, frame %d)).\n")
            : _T("auto-detect: probing %s over %d frames (window=full file).\n"),
        targetDesc.c_str(), wantFrames, prm->probeStartFrame, prm->probeEndFrame);

    if (seekMode) {
        const int64_t startTs = windowStartTs + windowOffsetTs;
        for (int i = 0; i < wantFrames && (int)lumaFrames.size() < wantFrames; ++i) {
            const double frac = ((double)i + 0.5) / wantFrames;
            const int64_t target = startTs + (int64_t)((double)windowSpanTs * frac);
            if (av_seek_frame(fmtCtx, videoIdx, target, AVSEEK_FLAG_BACKWARD) < 0) {
                seekFailures++;
                continue;
            }
            avcodec_flush_buffers(codecCtx);
            bool gotFrame = false;
            for (int attempts = 0; attempts < 64 && !gotFrame; ++attempts) {
                int rd = av_read_frame(fmtCtx, pktGuard.get());
                if (rd < 0) break;
                if (pktGuard.get()->stream_index != videoIdx) {
                    av_packet_unref(pktGuard.get()); continue;
                }
                avcodec_send_packet(codecCtx, pktGuard.get());
                av_packet_unref(pktGuard.get());
                gotFrame = captureOneFrame();
            }
            if (requireKey && totalDecoded >= decodeAttemptCap && (int)lumaFrames.size() < wantFrames) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("probe: %d decoded frames without filling lumaFrames; relaxing keyframe requirement.\n"),
                    totalDecoded);
                requireKey = false;
            }
        }
        if (seekFailures > 0 && lumaFrames.empty()) {
            seekMode = false;
            av_seek_frame(fmtCtx, videoIdx, 0, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(codecCtx);
        }
    }
    if (!seekMode || (int)lumaFrames.size() < wantFrames) {
        while ((int)lumaFrames.size() < wantFrames) {
            int rd = av_read_frame(fmtCtx, pktGuard.get());
            if (rd < 0) break;
            if (pktGuard.get()->stream_index != videoIdx) {
                av_packet_unref(pktGuard.get()); continue;
            }
            avcodec_send_packet(codecCtx, pktGuard.get());
            av_packet_unref(pktGuard.get());
            while ((int)lumaFrames.size() < wantFrames && captureOneFrame()) {}
        }
        if (lumaFrames.empty()) {
            avcodec_send_packet(codecCtx, nullptr);
            while (captureOneFrame()) {}
        }
    }
    if (lumaFrames.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("probe: failed to decode any frames.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_INFO,
        _T("probe: captured %d frames (%d keyframes%s; %s; seek-failures=%d).\n"),
        (int)lumaFrames.size(), keyframeCaptures,
        nonKeyCaptures > 0 ? strsprintf(_T(" + %d non-key fallback"), nonKeyCaptures).c_str() : _T(""),
        seekMode ? (useTrimWindow ? _T("spread within --trim window") : _T("spread across full file")) : _T("sequential from frame 0"),
        seekFailures);

    AddMessage(RGY_LOG_DEBUG, _T("probe: uploading %d luma frames to GPU...\n"), (int)lumaFrames.size());
    std::vector<std::unique_ptr<CUMemBuf>> lumaBufs;
    lumaBufs.reserve(lumaFrames.size());
    for (auto& raw : lumaFrames) {
        std::unique_ptr<CUMemBuf> buf;
        auto sts = upload_buf(buf, raw.data(), raw.size());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("probe: failed to upload luma to GPU.\n"));
            return sts;
        }
        lumaBufs.push_back(std::move(buf));
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: upload complete (%zu MB total).\n"),
        (size_t)((lumaFrames.size() * (size_t)src_w * src_h * src_pixel_bytes) / (1024 * 1024)));

    AddMessage(RGY_LOG_DEBUG, _T("probe: computing edge weights for %d frames...\n"), (int)lumaBufs.size());
    std::vector<std::unique_ptr<CUMemBuf>> edgeWeightsBufs;
    edgeWeightsBufs.reserve(lumaBufs.size());
    const int src_pitch_bytes_local = src_w * src_pixel_bytes;
    for (size_t i = 0; i < lumaBufs.size(); ++i) {
        auto wbuf = std::make_unique<CUMemBuf>((size_t)src_w * src_h * sizeof(float));
        auto sts = wbuf->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("probe: failed to allocate edge weights buffer.\n"));
            return sts;
        }
        dim3 block(16, 8);
        dim3 grid(divCeil(src_w, 16), divCeil(src_h, 8));
        if (src_pixel_bytes == 1) {
            kernel_compute_edge_weight<uint8_t, 8><<<grid, block>>>(
                (const uint8_t *)lumaBufs[i]->ptr, src_pitch_bytes_local,
                (float *)wbuf->ptr, src_w, src_w, src_h);
        } else {
            kernel_compute_edge_weight<uint16_t, 16><<<grid, block>>>(
                (const uint8_t *)lumaBufs[i]->ptr, src_pitch_bytes_local,
                (float *)wbuf->ptr, src_w, src_w, src_h);
        }
        sts = err_to_rgy(cudaGetLastError());
        if (sts == RGY_ERR_NONE) sts = err_to_rgy(cudaDeviceSynchronize());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("probe: kernel_compute_edge_weight launch failed: %s.\n"), get_err_mes(sts));
            return sts;
        }
        edgeWeightsBufs.push_back(std::move(wbuf));
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: edge weights computed.\n"));

    std::vector<ProbeCandidate> candidates;
    const bool resolutionSearchMode = (prm->descale.width <= 0 || prm->descale.height <= 0);
    if (!resolutionSearchMode) {
        candidates.reserve(kAutoCandCount);
        for (const auto& kv : kDescaleAutoCandidates) {
            candidates.push_back(ProbeCandidate{ kv.kernel, kv.b, kv.c, dst_w, dst_h, 0.0,
                probe_label_for(kv.kernel, kv.b, kv.c) });
        }
        AddMessage(RGY_LOG_DEBUG, _T("probe: kernel-only mode, scoring %d candidates.\n"), (int)candidates.size());
        auto err = scoreCandidates(candidates, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
        if (err != RGY_ERR_NONE) return err;
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("probe: resolution-search mode.\n"));
        auto err = runResolutionSearch(prm, candidates, lumaBufs, edgeWeightsBufs,
            src_w, src_h, src_pixel_bytes, fmtCtx, vst);
        if (err != RGY_ERR_NONE) return err;
    }

    int nonFiniteCount = 0;
    for (auto& c : candidates) {
        if (!std::isfinite(c.mse) || c.mse < 0.0) {
            c.mse = 1e30;
            nonFiniteCount++;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("lock: sanity-loop done (%d clamped).\n"), nonFiniteCount);

    std::map<int, std::pair<size_t, double>> bestByHeight;
    std::map<int, std::vector<size_t>> heightToCandidates;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        if (c.mse >= 1e29) continue;
        auto it = bestByHeight.find(c.height);
        if (it == bestByHeight.end() || c.mse < it->second.second) {
            bestByHeight[c.height] = { i, c.mse };
        }
        heightToCandidates[c.height].push_back(i);
    }
    const bool useArgmaxRatio = resolutionSearchMode && bestByHeight.size() >= 2;
    size_t winnerIdx = 0;
    double winnerRatio = 0.0;
    double secondRatio = 0.0;
    const TCHAR *conf = _T("low");
    tstring gapDesc;
    double displayGap = 1.0;
    std::vector<std::tuple<int, double, double, double>> ratiosByHeight;
    bool snappedToStandard = false;

    if (useArgmaxRatio) {
        std::vector<std::pair<int, double>> hMse;
        for (const auto& kv : bestByHeight) hMse.push_back({ kv.first, kv.second.second });
        std::vector<double> smoothedMse(hMse.size(), 0.0);
        std::map<int, double> smoothedByHeight;
        for (size_t i = 0; i < hMse.size(); ++i) {
            const size_t lo = (i == 0) ? 0 : (i - 1);
            const size_t hi = std::min(hMse.size() - 1, i + 1);
            double sum = 0.0; int n = 0;
            for (size_t j = lo; j <= hi; ++j) { sum += hMse[j].second; ++n; }
            smoothedMse[i] = sum / (double)n;
            smoothedByHeight[hMse[i].first] = smoothedMse[i];
        }
        ratiosByHeight.push_back(std::make_tuple(hMse[0].first, hMse[0].second, smoothedMse[0], 0.0));
        int bestRatioHeight = hMse[0].first;
        std::map<int, double> ratioByHeight;
        ratioByHeight[hMse[0].first] = 0.0;
        for (size_t i = 1; i < hMse.size(); ++i) {
            const double r = (smoothedMse[i] > 0.0) ? smoothedMse[i - 1] / smoothedMse[i] : 0.0;
            ratiosByHeight.push_back(std::make_tuple(hMse[i].first, hMse[i].second, smoothedMse[i], r));
            ratioByHeight[hMse[i].first] = r;
            if (r > winnerRatio) {
                secondRatio = winnerRatio; winnerRatio = r; bestRatioHeight = hMse[i].first;
            } else if (r > secondRatio) {
                secondRatio = r;
            }
        }
        bool commonFallbackFired = false;
        const double decisiveness = (secondRatio > 0.0) ? (winnerRatio / secondRatio) : 1e30;
        if ((decisiveness < 1.05) && (winnerRatio < 1.10)) {
            std::vector<int> probedCommon;
            for (int k = 0; k < kCommonNativeHeightsCount; ++k) {
                if (smoothedByHeight.count(kCommonNativeHeights[k])) probedCommon.push_back(kCommonNativeHeights[k]);
            }
            std::sort(probedCommon.begin(), probedCommon.end());
            int bestCommonH = -1;
            double bestCommonRatio = 0.0;
            for (size_t i = 1; i < probedCommon.size(); ++i) {
                const double smHere = smoothedByHeight.at(probedCommon[i]);
                const double smPrev = smoothedByHeight.at(probedCommon[i - 1]);
                const double r = (smHere > 0.0) ? smPrev / smHere : 0.0;
                if (r > bestCommonRatio) { bestCommonRatio = r; bestCommonH = probedCommon[i]; }
            }
            if (bestCommonH > 0) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("lock: ambiguous peak (winner_ratio=%.4f, decisiveness=%.4f); common-height argmax-ratio fallback -> %dp (ratio=%.4f).\n"),
                    winnerRatio, decisiveness, bestCommonH, bestCommonRatio);
                bestRatioHeight = bestCommonH;
                commonFallbackFired = true;
            }
        }
        for (int k = 0; k < kCommonNativeHeightsCount; ++k) {
            const int common = kCommonNativeHeights[k];
            if (common == bestRatioHeight) break;
            if (std::abs(common - bestRatioHeight) > 8) continue;
            auto it = ratioByHeight.find(common);
            if (it == ratioByHeight.end() || winnerRatio <= 0.0) continue;
            if (it->second / winnerRatio >= 0.9) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("lock: snap from %dp (ratio=%.4f) to standard %dp (ratio=%.4f).\n"),
                    bestRatioHeight, winnerRatio, common, it->second);
                bestRatioHeight = common;
                snappedToStandard = true;
                break;
            }
        }
        winnerIdx = bestByHeight[bestRatioHeight].first;
        const double ratioGap = (secondRatio > 0.0) ? winnerRatio / secondRatio : 1e30;
        if (commonFallbackFired) conf = _T("low (common-height fallback)");
        else if (winnerRatio > 1.10 && ratioGap > 1.02) conf = _T("high");
        else if (winnerRatio > 1.05) conf = _T("medium");
        else conf = _T("low");
        displayGap = winnerRatio;
        gapDesc = commonFallbackFired ? _T("common-height fallback")
            : (snappedToStandard ? _T("argmax-ratio + snap to standard") : _T("argmax-ratio"));
    } else {
        double bestMse = 1e30;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].mse < bestMse) { bestMse = candidates[i].mse; winnerIdx = i; }
        }
        double secondMse = 1e30;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i != winnerIdx && candidates[i].mse < secondMse) secondMse = candidates[i].mse;
        }
        const double mseRatio = (candidates[winnerIdx].mse > 0.0) ? secondMse / candidates[winnerIdx].mse : 1e30;
        if (mseRatio > 10.0) conf = _T("high");
        else if (mseRatio > 1.02) conf = _T("medium");
        else conf = _T("low");
        displayGap = mseRatio;
        gapDesc = _T("argmin");
    }

    const ProbeCandidate& winner = candidates[winnerIdx];
    int lockedHeight = winner.height;
    int lockedWidth = winner.width;
    {
        static const int kStandardHeights[] = { 360, 480, 486, 540, 576, 720, 810, 900, 1080 };
        const bool highConfidence = (_tcsstr(conf, _T("high")) != nullptr);
        if (!highConfidence) {
            int nearestH = 0;
            int nearestDist = 11;
            for (int sh : kStandardHeights) {
                const int d = std::abs(sh - winner.height);
                if (d < nearestDist) { nearestDist = d; nearestH = sh; }
            }
            if (nearestH != 0 && nearestH != winner.height) {
                const int snappedW = width_from_height(src_w, src_h, nearestH, fmtCtx, vst);
                if (snappedW > 0 && snappedW < src_w) {
                    AddMessage(RGY_LOG_INFO,
                        _T("Stage 1 height snapped %d->%d (within 10px of standard).\n"),
                        winner.height, nearestH);
                    lockedHeight = nearestH;
                    lockedWidth = snappedW;
                }
            }
        }
    }

    AddMessage(RGY_LOG_INFO,
        _T("auto-detected %s at %dx%d, mse=%.6e, gap=%.4f (%s), confidence=%s after %d frames.\n"),
        winner.label.c_str(), lockedWidth, lockedHeight,
        winner.mse, displayGap, gapDesc.c_str(), conf, (int)lumaBufs.size());
    if (prm->descale.show_scores) {
        std::vector<ProbeCandidate> sorted = candidates;
        std::sort(sorted.begin(), sorted.end(),
            [](const ProbeCandidate& a, const ProbeCandidate& b) { return a.mse < b.mse; });
        const int dump_n = std::min((int)sorted.size(), 10);
        for (int i = 0; i < dump_n; ++i) {
            AddMessage(RGY_LOG_INFO, _T("  #%-2d %-26s %dx%d mse=%.6e\n"),
                i + 1, sorted[i].label.c_str(), sorted[i].width, sorted[i].height, sorted[i].mse);
        }
    }

    VppDescaleKernel stage2Kernel = winner.kernel;
    float stage2B = winner.b;
    float stage2C = winner.c;
    tstring stage2Label = winner.label;
    {
        std::vector<ProbeCandidate> stage2;
        stage2.reserve(kAutoCandCount);
        for (const auto& kv : kDescaleAutoCandidates) {
            stage2.push_back(ProbeCandidate{ kv.kernel, kv.b, kv.c,
                lockedWidth, lockedHeight, 0.0, probe_label_for(kv.kernel, kv.b, kv.c) });
        }
        auto s2err = scoreCandidates(stage2, lumaBufs, edgeWeightsBufs,
            src_w, src_h, src_pixel_bytes, true);
        if (s2err == RGY_ERR_NONE) {
            size_t bestIdx = 0;
            double bestMse = 1e30;
            for (size_t i = 0; i < stage2.size(); ++i) {
                if (std::isfinite(stage2[i].mse) && stage2[i].mse >= 0.0 && stage2[i].mse < bestMse) {
                    bestMse = stage2[i].mse; bestIdx = i;
                }
            }
            if (bestMse < 1e29) {
                stage2Kernel = stage2[bestIdx].kernel;
                stage2B = stage2[bestIdx].b;
                stage2C = stage2[bestIdx].c;
                stage2Label = stage2[bestIdx].label;
                AddMessage(RGY_LOG_INFO,
                    _T("Stage 2 (symmetric kernel pick at %dx%d): %s mse=%.6e (was %s).\n"),
                    lockedWidth, lockedHeight, stage2Label.c_str(), bestMse, winner.label.c_str());
            }
        } else {
            AddMessage(RGY_LOG_WARN,
                _T("Stage 2: scoring failed (%s); keeping Stage 1 kernel %s.\n"),
                get_err_mes(s2err), winner.label.c_str());
        }
    }

    prm->descale.kernel = stage2Kernel;
    prm->descale.b = stage2B;
    prm->descale.c = stage2C;
    prm->descale.width = lockedWidth;
    prm->descale.height = lockedHeight;
    prm->descale.autoDetect = false;

    const auto probeEnd = std::chrono::steady_clock::now();
    const double probeMs = std::chrono::duration<double, std::milli>(probeEnd - probeStart).count();
    AddMessage(RGY_LOG_INFO, _T("auto-detect: probe complete in %.1f ms.\n"), probeMs);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDescale::prepareCore(NVEncFilterDescaleCore& core, int src_dim, int dst_dim,
    VppDescaleKernel kernel, double b, double c_param, double shift, VppDescaleBorder border) {
    const int support = kernel_support(kernel);
    if (support <= 0 || src_dim <= 0 || dst_dim <= 0 || dst_dim >= src_dim) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid descale dimensions src=%d dst=%d support=%d (dst must be < src).\n"),
            src_dim, dst_dim, support);
        return RGY_ERR_INVALID_PARAM;
    }
    core.src_dim = src_dim;
    core.dst_dim = dst_dim;
    core.bandwidth = support * 4 - 1;
    core.c = core.bandwidth / 2;

    std::vector<double> dense;
    build_scaling_weights(kernel, support, dst_dim, src_dim, b, c_param, shift, (double)dst_dim, border, dense);

    std::vector<double> dense_t;
    transpose_matrix(src_dim, dst_dim, dense, dense_t);

    std::vector<int> lidx(dst_dim, 0), ridx(dst_dim, 0);
    for (int i = 0; i < dst_dim; ++i) {
        for (int j = 0; j < src_dim; ++j) {
            if (dense_t[(size_t)i * src_dim + j] != 0.0) { lidx[i] = j; break; }
        }
        for (int j = src_dim - 1; j >= 0; --j) {
            if (dense_t[(size_t)i * src_dim + j] != 0.0) { ridx[i] = j + 1; break; }
        }
    }

    std::vector<double> ata;
    multiply_sparse_matrices(dst_dim, src_dim, lidx, ridx, dense_t, dense, ata);
    banded_ldlt(dst_dim, core.bandwidth, ata);

    std::vector<double> lower_full;
    transpose_matrix(dst_dim, dst_dim, ata, lower_full);
    multiply_banded_with_diagonal(dst_dim, core.bandwidth, lower_full);

    std::vector<float> lower_packed, upper_packed, diagonal;
    pack_lower_upper_diag(dst_dim, core.bandwidth, lower_full, ata, lower_packed, upper_packed, diagonal);

    int maxw = 0;
    for (int i = 0; i < dst_dim; ++i) {
        maxw = std::max(maxw, ridx[i] - lidx[i]);
    }
    core.weights_columns = maxw;
    std::vector<float> weights_packed((size_t)dst_dim * maxw, 0.0f);
    for (int i = 0; i < dst_dim; ++i) {
        for (int j = 0; j < ridx[i] - lidx[i]; ++j) {
            weights_packed[(size_t)i * maxw + j] = (float)dense_t[(size_t)i * src_dim + lidx[i] + j];
        }
    }

    auto sts = upload_buf(core.weights, weights_packed.data(), weights_packed.size() * sizeof(float));
    if (sts == RGY_ERR_NONE) sts = upload_buf(core.left_idx, lidx.data(), lidx.size() * sizeof(int));
    if (sts == RGY_ERR_NONE) sts = upload_buf(core.right_idx, ridx.data(), ridx.size() * sizeof(int));
    if (sts == RGY_ERR_NONE) sts = upload_buf(core.lower, lower_packed.data(), lower_packed.size() * sizeof(float));
    if (sts == RGY_ERR_NONE) sts = upload_buf(core.upper, upper_packed.data(), upper_packed.size() * sizeof(float));
    if (sts == RGY_ERR_NONE) sts = upload_buf(core.diagonal, diagonal.data(), diagonal.size() * sizeof(float));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate descale core buffers: %s.\n"), get_err_mes(sts));
    }
    return sts;
}

RGY_ERR NVEncFilterDescale::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDescale>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid input dimensions.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->descale.width > 0 && prm->descale.height > 0
        && (prm->descale.width >= prm->frameIn.width || prm->descale.height >= prm->frameIn.height)) {
        AddMessage(RGY_LOG_ERROR,
            _T("target dimensions (%dx%d) must be strictly smaller than the input (%dx%d).\n"),
            prm->descale.width, prm->descale.height, prm->frameIn.width, prm->frameIn.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->descale.kernel == VppDescaleKernel::Auto || prm->descale.autoDetect) {
        auto probeErr = runProbe(prm.get());
        if (probeErr != RGY_ERR_NONE) return probeErr;
    }
    if (prm->descale.width <= 0 || prm->descale.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("--vpp-descale requires height= (and optionally kernel=, width=). For automatic kernel detection use auto=true.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    prm->frameOut = prm->frameIn;
    prm->frameOut.width = prm->descale.width;
    prm->frameOut.height = prm->descale.height;
    auto sts = AllocFrameBuf(prm->frameOut, 2);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate output frame buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const int plane_count = RGY_CSP_PLANES[prm->frameOut.csp];
    const auto luma_src_plane = getPlane(&prm->frameIn, RGY_PLANE_Y);
    const auto luma_dst_plane = getPlane(&prm->frameOut, RGY_PLANE_Y);
    sts = prepareCore(m_cores[0][0], luma_src_plane.width, luma_dst_plane.width,
        prm->descale.kernel, prm->descale.b, prm->descale.c, (double)prm->descale.src_left, prm->descale.border);
    if (sts != RGY_ERR_NONE) return sts;
    sts = prepareCore(m_cores[1][0], luma_src_plane.height, luma_dst_plane.height,
        prm->descale.kernel, prm->descale.b, prm->descale.c, (double)prm->descale.src_top, prm->descale.border);
    if (sts != RGY_ERR_NONE) return sts;
    if (plane_count > 1) {
        const auto chroma_src_plane = getPlane(&prm->frameIn, RGY_PLANE_U);
        const auto chroma_dst_plane = getPlane(&prm->frameOut, RGY_PLANE_U);
        sts = prepareCore(m_cores[0][1], chroma_src_plane.width, chroma_dst_plane.width,
            prm->descale.kernel, prm->descale.b, prm->descale.c, (double)prm->descale.src_left, prm->descale.border);
        if (sts != RGY_ERR_NONE) return sts;
        sts = prepareCore(m_cores[1][1], chroma_src_plane.height, chroma_dst_plane.height,
            prm->descale.kernel, prm->descale.b, prm->descale.c, (double)prm->descale.src_top, prm->descale.border);
        if (sts != RGY_ERR_NONE) return sts;
    }

    for (int i = 0; i < plane_count && i < (int)m_intermediateH.size(); ++i) {
        const auto src_plane = getPlane(&prm->frameIn, (RGY_PLANE)i);
        const auto dst_plane = getPlane(&prm->frameOut, (RGY_PLANE)i);
        m_intermediatePitchFloats[i] = dst_plane.width;
        const size_t bytes_h = (size_t)dst_plane.width * src_plane.height * sizeof(float);
        const size_t bytes_v = (size_t)dst_plane.width * dst_plane.height * sizeof(float);
        m_intermediateH[i] = std::make_unique<CUMemBuf>(bytes_h);
        m_intermediateV[i] = std::make_unique<CUMemBuf>(bytes_v);
        if ((sts = m_intermediateH[i]->alloc()) != RGY_ERR_NONE
            || (sts = m_intermediateV[i]->alloc()) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate descale intermediate buffers plane %d: %s.\n"), i, get_err_mes(sts));
            return sts;
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDescale::runHPlane(RGYFrameInfo *pIntermediateFloat, const RGYFrameInfo *pInputPlane,
    const NVEncFilterDescaleCore& core, cudaStream_t stream) {
    switch (RGY_CSP_DATA_TYPE[pInputPlane->csp]) {
    case RGY_DATA_TYPE_U8:  return launch_descale_h<uint8_t, 8>(pIntermediateFloat, pInputPlane, core, stream);
    case RGY_DATA_TYPE_U16: return launch_descale_h<uint16_t, 16>(pIntermediateFloat, pInputPlane, core, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR NVEncFilterDescale::runVPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pIntermediateFloat,
    CUMemBuf *pVScratch, const NVEncFilterDescaleCore& core, cudaStream_t stream) {
    switch (RGY_CSP_DATA_TYPE[pOutputPlane->csp]) {
    case RGY_DATA_TYPE_U8:  return launch_descale_v<uint8_t, 8>(pOutputPlane, pIntermediateFloat, pVScratch, core, stream);
    case RGY_DATA_TYPE_U16: return launch_descale_v<uint16_t, 16>(pOutputPlane, pIntermediateFloat, pVScratch, core, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR NVEncFilterDescale::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto& outFrame = m_frameBuf[(m_frameIdx++) % m_frameBuf.size()];
        ppOutputFrames[0] = &outFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const int plane_count = RGY_CSP_PLANES[ppOutputFrames[0]->csp];
    for (int i = 0; i < plane_count; ++i) {
        const bool isChroma = (i > 0);
        const auto& coreH = m_cores[0][isChroma ? 1 : 0];
        const auto& coreV = m_cores[1][isChroma ? 1 : 0];
        auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)i);
        auto dstPlane = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
        RGYFrameInfo intermediate{};
        intermediate.csp = RGY_CSP_Y_F32;
        intermediate.ptr[0] = (uint8_t *)m_intermediateH[i]->ptr;
        intermediate.pitch[0] = m_intermediatePitchFloats[i] * (int)sizeof(float);
        intermediate.width = dstPlane.width;
        intermediate.height = srcPlane.height;
        auto sts = runHPlane(&intermediate, &srcPlane, coreH, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = runVPlane(&dstPlane, &intermediate, m_intermediateV[i].get(), coreV, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterDescale::close() {
    for (auto& row : m_cores) {
        for (auto& core : row) {
            core.weights.reset();
            core.left_idx.reset();
            core.right_idx.reset();
            core.lower.reset();
            core.upper.reset();
            core.diagonal.reset();
        }
    }
    for (auto& buf : m_intermediateH) buf.reset();
    for (auto& buf : m_intermediateV) buf.reset();
    m_frameBuf.clear();
    m_frameIdx = 0;
}
