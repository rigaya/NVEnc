// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#pragma once

#include "NVEncFilterNnedi.h"
#include "rgy_cuda_util_kernel.h"

static constexpr int NNEDI_TILE_GROUPS_X = 32;
static constexpr int NNEDI_TILE_ROWS = 16;
static constexpr int NNEDI_TILE_PIXELS_X = NNEDI_TILE_GROUPS_X * 4;
static constexpr int NNEDI_TILE_MASK_COUNT = NNEDI_TILE_GROUPS_X * NNEDI_TILE_ROWS;
static constexpr int NNEDI_TILE_MAX_CANDIDATES = NNEDI_TILE_MASK_COUNT * 4;

static __device__ __forceinline__ int nnedi_clamp_int(const int v, const int lo, const int hi) {
    return min(max(v, lo), hi);
}

static __device__ __forceinline__ float nnedi_expf_device(const float f) {
    return __expf(min(f, 88.0f));
}

static __device__ __forceinline__ int nnedi_round_to_pixel_match_device(const float value) {
    return (int)(value + 0.5f);
}

static __device__ __forceinline__ int nnedi_prescreen_lane_count_device(const int mask) {
    return __popc((unsigned int)(mask & 15));
}

static __device__ __forceinline__ unsigned short nnedi_pack_tile_xy_device(const int x, const int y) {
    return (unsigned short)((y << 7) | x);
}

static __device__ __forceinline__ void nnedi_unpack_tile_xy_device(const unsigned short packed, int *x, int *y) {
    *x = (int)(packed & 127);
    *y = (int)(packed >> 7);
}

static __device__ __forceinline__ void nnedi_local_prefix512_device(int *prefix, const int tid, const int count) {
    prefix[tid] = count;
    __syncthreads();
    for (int offset = 1; offset < NNEDI_TILE_MASK_COUNT; offset <<= 1) {
        const int addend = (tid >= offset) ? prefix[tid - offset] : 0;
        __syncthreads();
        prefix[tid] += addend;
        __syncthreads();
    }
}

static __device__ __forceinline__ void nnedi_predictor_expand_tile_masks_device(
    unsigned short *candidateQueue,
    int *candidatePrefix,
    const uint8_t *candidateMask,
    const int maskOffset,
    const int tid
) {
    const int mask = (int)(candidateMask[maskOffset + tid] & 15);
    const int count = nnedi_prescreen_lane_count_device(mask);
    nnedi_local_prefix512_device(candidatePrefix, tid, count);

    int index = candidatePrefix[tid] - count;
    const int groupX = tid - (tid / NNEDI_TILE_GROUPS_X) * NNEDI_TILE_GROUPS_X;
    const int y = tid / NNEDI_TILE_GROUPS_X;
    const int x = groupX << 2;
    if (mask & 1) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 0, y);
    if (mask & 2) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 1, y);
    if (mask & 4) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 2, y);
    if (mask & 8) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 3, y);
    __syncthreads();
}

template<typename Type, int XDIA, int YDIA, int PRED_K>
static __device__ __forceinline__ void nnedi_load_patch_distributed_device(
    Type *patch,
    const uint8_t *pRef,
    const int refPitch,
    const int refOffset,
    const int x,
    const int y,
    const bool active,
    const int tx
) {
    for (int k = tx; k < PRED_K; k += 16) {
        const int row = k / XDIA;
        const int col = k - row * XDIA;
        Type v = (Type)0;
        if (active) {
            v = *(const Type *)(pRef + refOffset + (y + row) * refPitch + (x + col) * (int)sizeof(Type));
        }
        patch[k] = v;
    }
}

template<int PRED_K, int LOCAL_X>
static __device__ __forceinline__ void nnedi_predictor_finalize_patch_stats_u8_device(
    int *sumPart,
    int *sumsqPart,
    float *avg,
    float *stddev,
    float *invvar,
    const int row
) {
    int sumAll = 0;
    int sumsqAll = 0;
    int *sumPartRow = sumPart + row * LOCAL_X;
    int *sumsqPartRow = sumsqPart + row * LOCAL_X;
    for (int i = 0; i < LOCAL_X; i++) {
        sumAll += sumPartRow[i];
        sumsqAll += sumsqPartRow[i];
    }
    const float scale = 1.0f / (float)PRED_K;
    const float avg_ = (float)sumAll * scale;
    float stddev_ = (float)sumsqAll * scale - avg_ * avg_;
    float invvar_ = 0.0f;
    if (stddev_ <= 1.192092896e-07F) {
        stddev_ = 0.0f;
    } else {
        stddev_ = sqrtf(stddev_);
        invvar_ = 1.0f / stddev_;
    }
    avg[row] = avg_;
    stddev[row] = stddev_;
    invvar[row] = invvar_;
}

template<int PRED_K, int LOCAL_X>
static __device__ __forceinline__ void nnedi_predictor_finalize_patch_avg_device(
    int *sumPart,
    float *avg,
    const int row
) {
    int sumAll = 0;
    int *sumPartRow = sumPart + row * LOCAL_X;
    for (int i = 0; i < LOCAL_X; i++) {
        sumAll += sumPartRow[i];
    }
    avg[row] = (float)sumAll * (1.0f / (float)PRED_K);
}

template<int PRED_K, int LOCAL_X>
static __device__ __forceinline__ void nnedi_predictor_finalize_patch_stddev_device(
    float *varPart,
    float *stddev,
    float *invvar,
    const int row
) {
    const int partBase = row * LOCAL_X;
    float stddev_ = 0.0f;
    for (int i = 0; i < LOCAL_X; i++) {
        stddev_ += varPart[partBase + i];
    }
    stddev_ *= (1.0f / (float)PRED_K);
    float invvar_ = 0.0f;
    if (stddev_ <= 1.192092896e-07F) {
        stddev_ = 0.0f;
    } else {
        stddev_ = sqrtf(stddev_);
        invvar_ = 1.0f / stddev_;
    }
    stddev[row] = stddev_;
    invvar[row] = invvar_;
}

template<typename Type, int PRED_K, int NNS, int QUAL, int LOCAL_X>
static __device__ __forceinline__ float2 nnedi_predictor_lane_vote_device(
    const Type *patch,
    const float2 *weightsBody,
    const float2 *weightsBias,
    const int tx,
    const int q,
    const float invvar
) {
    static constexpr int PRED_GROUPS = NNS / LOCAL_X;
    static constexpr int PRED_BLOCK_FLOAT2_COUNT = PRED_K * LOCAL_X;
    static constexpr int PRED_QUAL_BODY_FLOAT2_COUNT = PRED_GROUPS * PRED_BLOCK_FLOAT2_COUNT;
    float weightedElliottVoteSum = 0.0f;
    float softmaxVoteWeightSum = 0.0f;
    const float2 *neuronBlockWeights = weightsBody + q * PRED_QUAL_BODY_FLOAT2_COUNT;
    const float2 *neuronBiasPtr = weightsBias + q * NNS + tx;
    for (int neuronGroup = 0; neuronGroup < PRED_GROUPS; neuronGroup++) {
        const float2 *sampleWeights = neuronBlockWeights + tx;
        const Type *patchPtr = patch;
        float2 weightedPatchSums = make_float2(0.0f, 0.0f);
        for (int sampleIndex = 0; sampleIndex < PRED_K; sampleIndex++) {
            const int patchPixelValue = (int)(*patchPtr);
            weightedPatchSums.x = fmaf((float)patchPixelValue, sampleWeights->x, weightedPatchSums.x);
            weightedPatchSums.y = fmaf((float)patchPixelValue, sampleWeights->y, weightedPatchSums.y);
            patchPtr++;
            sampleWeights += LOCAL_X;
        }

        const float2 neuronBias = *neuronBiasPtr;
        const float softmaxLogit = fmaf(weightedPatchSums.x, invvar, neuronBias.x);
        const float elliottInput = fmaf(weightedPatchSums.y, invvar, neuronBias.y);
        const float softmaxVoteWeight = nnedi_expf_device(softmaxLogit);
        weightedElliottVoteSum += softmaxVoteWeight * (elliottInput / (1.0f + fabsf(elliottInput)));
        softmaxVoteWeightSum += softmaxVoteWeight;
        neuronBlockWeights += PRED_BLOCK_FLOAT2_COUNT;
        neuronBiasPtr += LOCAL_X;
    }
    return make_float2(weightedElliottVoteSum, softmaxVoteWeightSum);
}

template<int LOCAL_X>
static __device__ __forceinline__ float nnedi_predictor_merge_votes_device(
    const float *weightedElliottVoteParts,
    const float *softmaxVoteWeightParts,
    const int row,
    const float avg,
    const float stddev,
    const float result
) {
    const int partBase = row * LOCAL_X;
    float weightedElliottVoteSum = 0.0f;
    float softmaxVoteWeightSum = 0.0f;
    for (int i = 0; i < LOCAL_X; i++) {
        weightedElliottVoteSum += weightedElliottVoteParts[partBase + i];
        softmaxVoteWeightSum += softmaxVoteWeightParts[partBase + i];
    }
    if (softmaxVoteWeightSum > 1.0e-10f) {
        return result + ((5.0f * weightedElliottVoteSum) / softmaxVoteWeightSum) * stddev + avg;
    }
    return result + avg;
}

template<typename Type, int QUAL>
static __device__ __forceinline__ void nnedi_predictor_write_result_device(
    uint8_t *pDst,
    const int dstPitch,
    const int dstOffset,
    const int x,
    const int y,
    const float result,
    const int valMin,
    const int valMax
) {
    const float scale = 1.0f / (float)QUAL;
    *(Type *)(pDst + dstOffset + y * dstPitch + x * (int)sizeof(Type)) =
        (Type)nnedi_clamp_int(nnedi_round_to_pixel_match_device(result * scale), valMin, valMax);
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA, int NNS, int QUAL>
__launch_bounds__(512, 1)
__global__ void kernel_nnedi_predictor_network_cuda(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstOffset,
    const uint8_t *__restrict__ pRef, const int refPitch, const int refOffset,
    const uint8_t *__restrict__ candidateMask, const int *__restrict__ numblocks,
    const float *__restrict__ weights,
    const int width4, const int height, const int valMin, const int valMax
) {
    static constexpr int LOCAL_X = 16;
    static constexpr int LOCAL_Y = 32;
    static constexpr int PRED_K = XDIA * YDIA;
    static constexpr int PRED_GROUPS = NNS / LOCAL_X;
    static constexpr int PRED_BLOCK_FLOAT2_COUNT = PRED_K * LOCAL_X;
    static constexpr int PRED_QUAL_BODY_FLOAT2_COUNT = PRED_GROUPS * PRED_BLOCK_FLOAT2_COUNT;
    static constexpr int PRED_BODY_FLOAT2_COUNT = QUAL * PRED_QUAL_BODY_FLOAT2_COUNT;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int maskOffset = bid * NNEDI_TILE_MASK_COUNT;
    const int tid = ty * LOCAL_X + tx;
    const int xbase = blockIdx.x * NNEDI_TILE_PIXELS_X;
    const int ybase = blockIdx.y * NNEDI_TILE_ROWS;

    __shared__ Type patchPixels[LOCAL_Y * PRED_K];
    __shared__ unsigned short candidateQueue[NNEDI_TILE_MAX_CANDIDATES];
    __shared__ int candidatePrefix[NNEDI_TILE_MASK_COUNT];
    __shared__ int sumPart[LOCAL_Y * LOCAL_X];
#if BIT_DEPTH <= 8
    __shared__ int sumsqPart[LOCAL_Y * LOCAL_X];
#else
    __shared__ float varPart[LOCAL_Y * LOCAL_X];
#endif
    __shared__ float avg[LOCAL_Y];
    __shared__ float stddev[LOCAL_Y];
    __shared__ float invvar[LOCAL_Y];
    __shared__ float weightedElliottVoteParts[LOCAL_Y * LOCAL_X];
    __shared__ float softmaxVoteWeightParts[LOCAL_Y * LOCAL_X];

    const int nb = numblocks[bid];
    if (nb <= 0) {
        return;
    }
    const float2 *weightsBody = (const float2 *)weights;
    const float2 *weightsBias = weightsBody + PRED_BODY_FLOAT2_COUNT;

    nnedi_predictor_expand_tile_masks_device(candidateQueue, candidatePrefix, candidateMask, maskOffset, tid);

    for (int b = 0; b < nb; b += LOCAL_Y) {
        int localX = 0;
        int localY = 0;
        const bool active = (b + ty < nb);
        if (active) {
            nnedi_unpack_tile_xy_device(candidateQueue[b + ty], &localX, &localY);
        }
        const int x = xbase + localX;
        const int y = ybase + localY;

        Type *patch = &patchPixels[ty * PRED_K];
        nnedi_load_patch_distributed_device<Type, XDIA, YDIA, PRED_K>(patch, pRef, refPitch, refOffset, x, y, active, tx);
        __syncthreads();

        float avgValue = 0.0f;
        float stddevValue = 0.0f;
        float invvarValue = 0.0f;
#if BIT_DEPTH <= 8
        int sum = 0;
        int sumsq = 0;
        for (int k = tx; k < PRED_K; k += LOCAL_X) {
            const int v = (int)patch[k];
            sum += v;
            sumsq += v * v;
        }
        sumPart[ty * LOCAL_X + tx] = sum;
        sumsqPart[ty * LOCAL_X + tx] = sumsq;
        __syncthreads();

        if (tx == 0) {
            nnedi_predictor_finalize_patch_stats_u8_device<PRED_K, LOCAL_X>(sumPart, sumsqPart, avg, stddev, invvar, ty);
        }
        __syncthreads();
        avgValue = avg[ty];
        stddevValue = stddev[ty];
        invvarValue = invvar[ty];
#else
        int sum = 0;
        for (int k = tx; k < PRED_K; k += LOCAL_X) {
            sum += (int)patch[k];
        }
        sumPart[ty * LOCAL_X + tx] = sum;
        __syncthreads();

        if (tx == 0) {
            nnedi_predictor_finalize_patch_avg_device<PRED_K, LOCAL_X>(sumPart, avg, ty);
        }
        __syncthreads();

        float sumsq = 0.0f;
        for (int k = tx; k < PRED_K; k += LOCAL_X) {
            const float diff = (float)patch[k] - avg[ty];
            sumsq = fmaf(diff, diff, sumsq);
        }
        varPart[ty * LOCAL_X + tx] = sumsq;
        __syncthreads();

        if (tx == 0) {
            nnedi_predictor_finalize_patch_stddev_device<PRED_K, LOCAL_X>(varPart, stddev, invvar, ty);
        }
        __syncthreads();
        avgValue = avg[ty];
        stddevValue = stddev[ty];
        invvarValue = invvar[ty];
#endif

        float result = 0.0f;
        const int partBase = ty * LOCAL_X;
        for (int q = 0; q < QUAL; q++) {
            const float2 vote = nnedi_predictor_lane_vote_device<Type, PRED_K, NNS, QUAL, LOCAL_X>(
                patch, weightsBody, weightsBias,
                tx, q, invvarValue);
            weightedElliottVoteParts[partBase + tx] = vote.x;
            softmaxVoteWeightParts[partBase + tx] = vote.y;
            __syncthreads();

            if (tx == 0) {
                result = nnedi_predictor_merge_votes_device<LOCAL_X>(
                    weightedElliottVoteParts, softmaxVoteWeightParts,
                    ty, avgValue, stddevValue, result);
            }
            __syncthreads();
        }

        if (tx == 0 && active) {
            nnedi_predictor_write_result_device<Type, QUAL>(pDst, dstPitch, dstOffset, x, y, result, valMin, valMax);
        }
        __syncthreads();
    }
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA, int NNS, int QUAL>
static RGY_ERR launchPredictorImpl(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax, cudaStream_t stream) {
    dim3 block(16, 32);
    dim3 grid((width4 + NNEDI_TILE_GROUPS_X - 1) / NNEDI_TILE_GROUPS_X,
        (height + NNEDI_TILE_ROWS - 1) / NNEDI_TILE_ROWS);
    kernel_nnedi_predictor_network_cuda<Type, BIT_DEPTH, XDIA, YDIA, NNS, QUAL><<<grid, block, 0, stream>>>(
        (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0] * 2, dstOffset,
        (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0], refEvalOffset,
        (const uint8_t *)workNNBuf->ptr, (const int *)numBlocksBuf->ptr,
        (const float *)predictorWeightBuf->ptr,
        width4, height, valMin, valMax);
    const auto err = cudaGetLastError();
    return (err == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(err);
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA, int QUAL>
static RGY_ERR launchPredictorNns(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax,
    int nns, cudaStream_t stream) {
    switch (nns) {
    case 16:  return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 16,  QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 32:  return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 32,  QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 64:  return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 64,  QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 128: return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 128, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 256: return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 256, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    default:  return RGY_ERR_INVALID_PARAM;
    }
}

template<typename Type, int BIT_DEPTH, int QUAL>
static RGY_ERR launchPredictorNsize(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax,
    int nsize, int nns, cudaStream_t stream) {
    switch (nsize) {
    case VPP_NNEDI_NSIZE_8x6:  return launchPredictorNns<Type, BIT_DEPTH,  8, 6, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_NSIZE_16x6: return launchPredictorNns<Type, BIT_DEPTH, 16, 6, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_NSIZE_32x6: return launchPredictorNns<Type, BIT_DEPTH, 32, 6, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_NSIZE_48x6: return launchPredictorNns<Type, BIT_DEPTH, 48, 6, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_NSIZE_8x4:  return launchPredictorNns<Type, BIT_DEPTH,  8, 4, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_NSIZE_16x4: return launchPredictorNns<Type, BIT_DEPTH, 16, 4, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_NSIZE_32x4: return launchPredictorNns<Type, BIT_DEPTH, 32, 4, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    default: return RGY_ERR_INVALID_PARAM;
    }
}
