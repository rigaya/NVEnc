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

#include <map>
#include "convert_csp.h"
#include "NVEncFilterIvtc.h"
#include "NVEncParam.h"

static const int IVTC_BLOCK_X = 32;
static const int IVTC_BLOCK_Y = 8;

#define BLOCK_COMB_THRESH 8
#define IVTC_W3F_SP0   5077
#define IVTC_W3F_SP1    981
#define IVTC_W3F_SHIFT   13
#define IVTC_W3F_ROUND (1 << (IVTC_W3F_SHIFT - 1))
#define IVTC_W3F_LF0   4309
#define IVTC_W3F_LF1    213
#define IVTC_W3F_HF0   5570
#define IVTC_W3F_HF1   3801
#define IVTC_W3F_HF2   1016

template<typename TypePixel>
__device__ __forceinline__ int ivtc_read_pix(const RGYFrameInfo frame, int x, int y) {
    x = clamp(x, 0, frame.width - 1);
    y = clamp(y, 0, frame.height - 1);
    const auto ptr = (const TypePixel *)((const uint8_t *)frame.ptr[0] + y * frame.pitch[0] + x * sizeof(TypePixel));
    return (int)ptr[0];
}

template<typename TypePixel>
__device__ __forceinline__ int ivtc_read_pix_same_parity(const RGYFrameInfo frame, int x, int y) {
    const int up = ivtc_read_pix<TypePixel>(frame, x, y - 1);
    const int dn = ivtc_read_pix<TypePixel>(frame, x, y + 1);
    return (up + dn + 1) >> 1;
}

template<typename TypePixel>
__device__ __forceinline__ int ivtc_pix_match(
    const RGYFrameInfo prev, const RGYFrameInfo cur, const RGYFrameInfo next,
    int x, int y, const int tff, const int match) {
    x = clamp(x, 0, cur.width - 1);
    y = clamp(y, 0, cur.height - 1);
    const int is_first_field_row = (y & 1) == (tff ? 0 : 1);
    const RGYFrameInfo src = (match == 1) ? (is_first_field_row ? cur : prev)
                           : (match == 2) ? (is_first_field_row ? next : cur)
                           : cur;
    return ivtc_read_pix<TypePixel>(src, x, y);
}

template<typename TypePixel, int bit_depth>
__global__ void kernel_ivtc_field_overlay(RGYFrameInfo dst, const RGYFrameInfo src, const int tff) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dst.width || iy >= dst.height) return;
    const int targetParity = tff ? 1 : 0;
    if ((iy & 1) == targetParity) {
        auto dstPix = (TypePixel *)((uint8_t *)dst.ptr[0] + iy * dst.pitch[0] + ix * sizeof(TypePixel));
        dstPix[0] = (TypePixel)ivtc_read_pix<TypePixel>(src, ix, iy);
    }
}

template<typename TypePixel, int bit_depth>
__global__ void kernel_ivtc_score_candidates(
    const RGYFrameInfo prev, const RGYFrameInfo cur, const RGYFrameInfo next,
    const int tff, const int nt, const int T, const int y0, const int y1, uint32_t *scores) {
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int ix  = blockIdx.x * blockDim.x + thx;
    const int iy  = blockIdx.y * blockDim.y + thy;
    const int tid = thy * IVTC_BLOCK_X + thx;
    const int WG_SIZE = IVTC_BLOCK_X * IVTC_BLOCK_Y;

    uint32_t mC = 0, mP = 0, mN = 0;
    uint32_t cC = 0, cP = 0, cN = 0;
    const int first_parity = tff ? 0 : 1;
    const int band_on = (y0 != 0 || y1 != 0);
    const int in_band = !band_on || (iy >= y0 && iy <= y1);

    if (in_band && ix < cur.width && (iy & 1) == first_parity && iy + 4 < cur.height) {
        #pragma unroll
        for (int m = 0; m < 3; m++) {
            const int v0 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy,     tff, m);
            const int v1 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy + 1, tff, m);
            const int v2 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy + 2, tff, m);
            const int v3 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy + 3, tff, m);
            const int v4 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy + 4, tff, m);
            const int interp1 = (v0 + v2 + 1) >> 1;
            const int interp2 = (v2 + v4 + 1) >> 1;
            const int diff = abs(v1 - interp1) + abs(v3 - interp2);
            const uint32_t diff_u = (diff > nt) ? (uint32_t)diff : 0u;
            const int d10 = v1 - v0;
            const int d12 = v1 - v2;
            const uint32_t comb_u = (uint32_t)((abs(d10) > T) && (abs(d12) > T) && ((d10 < 0) == (d12 < 0)));
            if (m == 0)      { mC = diff_u; cC = comb_u; }
            else if (m == 1) { mP = diff_u; cP = comb_u; }
            else             { mN = diff_u; cN = comb_u; }
        }
    }

    __shared__ uint32_t lred[IVTC_BLOCK_X * IVTC_BLOCK_Y * 6];
    lred[tid + 0 * WG_SIZE] = mC;
    lred[tid + 1 * WG_SIZE] = mP;
    lred[tid + 2 * WG_SIZE] = mN;
    lred[tid + 3 * WG_SIZE] = cC;
    lred[tid + 4 * WG_SIZE] = cP;
    lred[tid + 5 * WG_SIZE] = cN;
    __syncthreads();

    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int k = 0; k < 6; k++) {
                lred[tid + k * WG_SIZE] += lred[tid + s + k * WG_SIZE];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int wg_idx = blockIdx.y * gridDim.x + blockIdx.x;
        #pragma unroll
        for (int k = 0; k < 6; k++) {
            scores[wg_idx * 9 + k] = lred[k * WG_SIZE];
        }
        scores[wg_idx * 9 + 6] = (lred[3 * WG_SIZE] >= (uint32_t)BLOCK_COMB_THRESH) ? 1u : 0u;
        scores[wg_idx * 9 + 7] = (lred[4 * WG_SIZE] >= (uint32_t)BLOCK_COMB_THRESH) ? 1u : 0u;
        scores[wg_idx * 9 + 8] = (lred[5 * WG_SIZE] >= (uint32_t)BLOCK_COMB_THRESH) ? 1u : 0u;
    }
}

template<typename TypePixel, int bit_depth>
__global__ void kernel_ivtc_frame_diff(const RGYFrameInfo a, const RGYFrameInfo b, uint32_t *diffOut) {
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int ix  = blockIdx.x * blockDim.x + thx;
    const int iy  = blockIdx.y * blockDim.y + thy;
    const int tid = thy * IVTC_BLOCK_X + thx;
    const int WG_SIZE = IVTC_BLOCK_X * IVTC_BLOCK_Y;

    uint32_t d = 0;
    if (ix < a.width && iy < a.height) {
        d = (uint32_t)abs(ivtc_read_pix<TypePixel>(a, ix, iy) - ivtc_read_pix<TypePixel>(b, ix, iy));
    }
    __shared__ uint32_t lred[IVTC_BLOCK_X * IVTC_BLOCK_Y];
    lred[tid] = d;
    __syncthreads();
    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) lred[tid] += lred[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const int wg_idx = blockIdx.y * gridDim.x + blockIdx.x;
        diffOut[wg_idx] = lred[0];
    }
}

template<typename TypePixel, int bit_depth>
__global__ void kernel_ivtc_synthesize(
    RGYFrameInfo dst, const RGYFrameInfo prev, const RGYFrameInfo cur, const RGYFrameInfo next,
    const int tff, const int match, const int apply_blend, const int dthresh) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dst.width || iy >= dst.height) return;

    int out_val = 0;
    const int is_first_field_row = (iy & 1) == (tff ? 0 : 1);
    if (apply_blend && !is_first_field_row) {
        const int rowU = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy - 1, tff, match);
        const int rowL = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy + 1, tff, match);
        const int original = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy, tff, match);
        const int interpTwoTap = (rowU + rowL + 1) >> 1;
        const int residual = abs(original - interpTwoTap);
        if (dthresh > 0 && residual <= dthresh) {
            out_val = original;
        } else {
            const int prev_val = ivtc_read_pix<TypePixel>(prev, ix, iy);
            const int next_val = ivtc_read_pix<TypePixel>(next, ix, iy);
            const int motion_raw = max(abs(prev_val - original), abs(next_val - original));
            const int noise_floor = max(1, dthresh >> 3);
            const int motion = max(0, motion_raw - noise_floor);
            const int motion_thresh = dthresh * 2;
            const int hasFullCtx = (iy >= 3) && (iy < dst.height - 3);
            int spatial;
            if (hasFullCtx) {
                const int rowU3 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy - 3, tff, match);
                const int rowL3 = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy + 3, tff, match);
                spatial = (IVTC_W3F_SP0 * (rowU + rowL) - IVTC_W3F_SP1 * (rowU3 + rowL3) + IVTC_W3F_ROUND) >> IVTC_W3F_SHIFT;
            } else {
                spatial = interpTwoTap;
            }
            const int temporal = (prev_val + next_val + 2 * original + 2) >> 2;
            int blend_result;
            if (dthresh > 0 && motion_thresh > 0) {
                const int w = clamp(motion, 0, motion_thresh);
                const int w2 = (w * w + (motion_thresh >> 1)) / motion_thresh;
                blend_result = (temporal * (motion_thresh - w2) + spatial * w2 + (motion_thresh >> 1)) / motion_thresh;
            } else {
                blend_result = spatial;
            }
            const int rowMin = min(rowU, rowL);
            const int rowMax = max(rowU, rowL);
            const int max_eps = max(8, 8 << (bit_depth - 8));
            int epsilon = max(1, dthresh >> 1);
            if (epsilon > max_eps) epsilon = max_eps;
            const int pixMax = (1 << bit_depth) - 1;
            out_val = clamp(blend_result, max(0, rowMin - epsilon), min(pixMax, rowMax + epsilon));
        }
    } else {
        out_val = ivtc_pix_match<TypePixel>(prev, cur, next, ix, iy, tff, match);
    }
    auto ptr = (TypePixel *)((uint8_t *)dst.ptr[0] + iy * dst.pitch[0] + ix * sizeof(TypePixel));
    ptr[0] = (TypePixel)out_val;
}

template<typename TypePixel, int bit_depth>
__global__ void kernel_ivtc_bwdif_deint(
    RGYFrameInfo dst, const RGYFrameInfo prev2, const RGYFrameInfo prev, const RGYFrameInfo cur,
    const RGYFrameInfo next, const RGYFrameInfo next2, const int tff, const int scene_change, const int dthresh) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dst.width || iy >= dst.height) return;

    auto dstPix = (TypePixel *)((uint8_t *)dst.ptr[0] + iy * dst.pitch[0] + ix * sizeof(TypePixel));
    const int preservedParity = tff ? 0 : 1;
    const int needsInterp = ((iy & 1) != preservedParity);
    if (!needsInterp) {
        dstPix[0] = (TypePixel)ivtc_read_pix<TypePixel>(cur, ix, iy);
        return;
    }

    const int rowU = ivtc_read_pix<TypePixel>(cur, ix, iy - 1);
    const int rowL = ivtc_read_pix<TypePixel>(cur, ix, iy + 1);
    const int originalPix = ivtc_read_pix<TypePixel>(cur, ix, iy);
    const int interpTwoTap = (rowU + rowL + 1) >> 1;
    if (dthresh > 0 && abs(originalPix - interpTwoTap) <= dthresh) {
        dstPix[0] = (TypePixel)originalPix;
        return;
    }

    if (scene_change) {
        int sp;
        if ((iy >= 3) && (iy < dst.height - 3)) {
            const int rowU3 = ivtc_read_pix<TypePixel>(cur, ix, iy - 3);
            const int rowL3 = ivtc_read_pix<TypePixel>(cur, ix, iy + 3);
            sp = (IVTC_W3F_SP0 * (rowU + rowL) - IVTC_W3F_SP1 * (rowU3 + rowL3) + IVTC_W3F_ROUND) >> IVTC_W3F_SHIFT;
            const int rowMin = min(rowU, rowL);
            const int rowMax = max(rowU, rowL);
            const int max_eps = max(8, 8 << (bit_depth - 8));
            int epsilon = max(1, dthresh >> 1);
            if (epsilon > max_eps) epsilon = max_eps;
            const int pixMax = (1 << bit_depth) - 1;
            sp = clamp(sp, max(0, rowMin - epsilon), min(pixMax, rowMax + epsilon));
        } else {
            sp = interpTwoTap;
        }
        dstPix[0] = (TypePixel)sp;
        return;
    }

    const int p2_0 = ivtc_read_pix_same_parity<TypePixel>(prev2, ix, iy);
    const int n2_0 = ivtc_read_pix_same_parity<TypePixel>(next2, ix, iy);
    const int tAvg = (p2_0 + n2_0) >> 1;
    const int pUp = ivtc_read_pix<TypePixel>(prev, ix, iy - 1);
    const int pDn = ivtc_read_pix<TypePixel>(prev, ix, iy + 1);
    const int nUp = ivtc_read_pix<TypePixel>(next, ix, iy - 1);
    const int nDn = ivtc_read_pix<TypePixel>(next, ix, iy + 1);
    const int motA = abs(p2_0 - n2_0);
    const int motB = (abs(pUp - rowU) + abs(pDn - rowL)) >> 1;
    const int motC = (abs(nUp - rowU) + abs(nDn - rowL)) >> 1;
    int motion = max(motA >> 1, max(motB, motC));
    if (motion == 0) {
        dstPix[0] = (TypePixel)tAvg;
        return;
    }

    const int hasSpatBounds = (iy >= 2) && (iy < dst.height - 2);
    const int hasFullCtx = (iy >= 4) && (iy < dst.height - 4);
    const int p2_m2 = ivtc_read_pix_same_parity<TypePixel>(prev2, ix, iy - 2);
    const int p2_p2 = ivtc_read_pix_same_parity<TypePixel>(prev2, ix, iy + 2);
    const int n2_m2 = ivtc_read_pix_same_parity<TypePixel>(next2, ix, iy - 2);
    const int n2_p2 = ivtc_read_pix_same_parity<TypePixel>(next2, ix, iy + 2);
    int localMotion = motion;
    if (hasSpatBounds) {
        const int spreadU = ((p2_m2 + n2_m2) >> 1) - rowU;
        const int spreadL = ((p2_p2 + n2_p2) >> 1) - rowL;
        const int dU = tAvg - rowU;
        const int dL = tAvg - rowL;
        const int hiSet = max(dL, max(dU, min(spreadU, spreadL)));
        const int loSet = min(dL, min(dU, max(spreadU, spreadL)));
        localMotion = max(localMotion, max(loSet, -hiSet));
    }

    int spatial;
    if (hasFullCtx) {
        const int p2_m4 = ivtc_read_pix_same_parity<TypePixel>(prev2, ix, iy - 4);
        const int p2_p4 = ivtc_read_pix_same_parity<TypePixel>(prev2, ix, iy + 4);
        const int n2_m4 = ivtc_read_pix_same_parity<TypePixel>(next2, ix, iy - 4);
        const int n2_p4 = ivtc_read_pix_same_parity<TypePixel>(next2, ix, iy + 4);
        const int curU3 = ivtc_read_pix<TypePixel>(cur, ix, iy - 3);
        const int curD3 = ivtc_read_pix<TypePixel>(cur, ix, iy + 3);
        const int verticalEdge = abs(rowU - rowL);
        if (verticalEdge > motA) {
            const int hf = (IVTC_W3F_HF0 * (p2_0 + n2_0)
                          - IVTC_W3F_HF1 * (p2_m2 + n2_m2 + p2_p2 + n2_p2)
                          + IVTC_W3F_HF2 * (p2_m4 + n2_m4 + p2_p4 + n2_p4)) >> 2;
            spatial = (hf + IVTC_W3F_LF0 * (rowU + rowL) - IVTC_W3F_LF1 * (curU3 + curD3)) >> IVTC_W3F_SHIFT;
        } else {
            spatial = (IVTC_W3F_SP0 * (rowU + rowL) - IVTC_W3F_SP1 * (curU3 + curD3)) >> IVTC_W3F_SHIFT;
        }
    } else {
        spatial = interpTwoTap;
    }

    int interp;
    if (dthresh > 0) {
        const int temporal = (p2_0 + n2_0 + 2 * originalPix + 2) >> 2;
        const int motHybrid_raw = max(abs(p2_0 - originalPix), abs(n2_0 - originalPix));
        const int noise_floor_b = max(1, dthresh >> 3);
        const int motHybrid = max(0, motHybrid_raw - noise_floor_b);
        const int motion_thresh = dthresh * 2;
        const int w = clamp(motHybrid, 0, motion_thresh);
        const int w2 = (w * w + (motion_thresh >> 1)) / motion_thresh;
        interp = (temporal * (motion_thresh - w2) + spatial * w2 + (motion_thresh >> 1)) / motion_thresh;
    } else {
        interp = spatial;
    }
    interp = clamp(interp, tAvg - localMotion, tAvg + localMotion);
    interp = clamp(interp, 0, (1 << bit_depth) - 1);
    dstPix[0] = (TypePixel)interp;
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_ivtc_score_candidates_typed(const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, int tff, int nt, int T, int y0, int y1, uint32_t *scoreDev, cudaStream_t stream) {
    const dim3 blockSize(IVTC_BLOCK_X, IVTC_BLOCK_Y);
    const dim3 gridSize(divCeil(pCur->width, blockSize.x), divCeil(pCur->height, blockSize.y));
    kernel_ivtc_score_candidates<TypePixel, bit_depth><<<gridSize, blockSize, 0, stream>>>(*pPrev, *pCur, *pNext, tff, nt, T, y0, y1, scoreDev);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_ivtc_frame_diff_typed(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint32_t *diffDev, cudaStream_t stream) {
    const auto planeA = getPlane(pA, RGY_PLANE_Y);
    const auto planeB = getPlane(pB, RGY_PLANE_Y);
    const dim3 blockSize(IVTC_BLOCK_X, IVTC_BLOCK_Y);
    const dim3 gridSize(divCeil(planeA.width, blockSize.x), divCeil(planeA.height, blockSize.y));
    kernel_ivtc_frame_diff<TypePixel, bit_depth><<<gridSize, blockSize, 0, stream>>>(planeA, planeB, diffDev);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_ivtc_synthesize_frame_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, int tff, int match, int applyBlend, int dthresh, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pOutputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        auto planeOutput = getPlane(pOutputFrame, plane);
        const auto planePrev = getPlane(pPrev, plane);
        const auto planeCur  = getPlane(pCur,  plane);
        const auto planeNext = getPlane(pNext, plane);
        const dim3 blockSize(IVTC_BLOCK_X, IVTC_BLOCK_Y);
        const dim3 gridSize(divCeil(planeOutput.width, blockSize.x), divCeil(planeOutput.height, blockSize.y));
        kernel_ivtc_synthesize<TypePixel, bit_depth><<<gridSize, blockSize, 0, stream>>>(planeOutput, planePrev, planeCur, planeNext, tff, match, applyBlend, dthresh);
        auto sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_ivtc_bwdif_frame_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrev2, const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, const RGYFrameInfo *pNext2, int tff, int sceneChange, int dthresh, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pOutputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        auto planeOutput = getPlane(pOutputFrame, plane);
        const auto planePrev2 = getPlane(pPrev2, plane);
        const auto planePrev  = getPlane(pPrev,  plane);
        const auto planeCur   = getPlane(pCur,   plane);
        const auto planeNext  = getPlane(pNext,  plane);
        const auto planeNext2 = getPlane(pNext2, plane);
        const dim3 blockSize(IVTC_BLOCK_X, IVTC_BLOCK_Y);
        const dim3 gridSize(divCeil(planeOutput.width, blockSize.x), divCeil(planeOutput.height, blockSize.y));
        kernel_ivtc_bwdif_deint<TypePixel, bit_depth><<<gridSize, blockSize, 0, stream>>>(planeOutput, planePrev2, planePrev, planeCur, planeNext, planeNext2, tff, sceneChange, dthresh);
        auto sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_ivtc_field_overlay_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int tff, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pDst->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        auto planeDst = getPlane(pDst, plane);
        const auto planeSrc = getPlane(pSrc, plane);
        const dim3 blockSize(IVTC_BLOCK_X, IVTC_BLOCK_Y);
        const dim3 gridSize(divCeil(planeDst.width, blockSize.x), divCeil(planeDst.height, blockSize.y));
        kernel_ivtc_field_overlay<TypePixel, bit_depth><<<gridSize, blockSize, 0, stream>>>(planeDst, planeSrc, tff);
        auto sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR run_ivtc_score_candidates(const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, int tff, int nt, int T, int y0, int y1, uint32_t *scoreDev, cudaStream_t stream) {
    if (RGY_CSP_DATA_TYPE[pCur->csp] == RGY_DATA_TYPE_U8) {
        return run_ivtc_score_candidates_typed<uint8_t, 8>(pPrev, pCur, pNext, tff, nt, T, y0, y1, scoreDev, stream);
    }
    if (RGY_CSP_DATA_TYPE[pCur->csp] == RGY_DATA_TYPE_U16) {
        return run_ivtc_score_candidates_typed<uint16_t, 16>(pPrev, pCur, pNext, tff, nt, T, y0, y1, scoreDev, stream);
    }
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR run_ivtc_synthesize_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, int tff, int match, int applyBlend, int dthresh, cudaStream_t stream) {
    if (RGY_CSP_DATA_TYPE[pOutputFrame->csp] == RGY_DATA_TYPE_U8) {
        return run_ivtc_synthesize_frame_typed<uint8_t, 8>(pOutputFrame, pPrev, pCur, pNext, tff, match, applyBlend, dthresh, stream);
    }
    if (RGY_CSP_DATA_TYPE[pOutputFrame->csp] == RGY_DATA_TYPE_U16) {
        return run_ivtc_synthesize_frame_typed<uint16_t, 16>(pOutputFrame, pPrev, pCur, pNext, tff, match, applyBlend, dthresh, stream);
    }
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR run_ivtc_bwdif_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrev2, const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, const RGYFrameInfo *pNext2, int tff, int sceneChange, int dthresh, cudaStream_t stream) {
    if (RGY_CSP_DATA_TYPE[pOutputFrame->csp] == RGY_DATA_TYPE_U8) {
        return run_ivtc_bwdif_frame_typed<uint8_t, 8>(pOutputFrame, pPrev2, pPrev, pCur, pNext, pNext2, tff, sceneChange, dthresh, stream);
    }
    if (RGY_CSP_DATA_TYPE[pOutputFrame->csp] == RGY_DATA_TYPE_U16) {
        return run_ivtc_bwdif_frame_typed<uint16_t, 16>(pOutputFrame, pPrev2, pPrev, pCur, pNext, pNext2, tff, sceneChange, dthresh, stream);
    }
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR run_ivtc_frame_diff(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint32_t *diffDev, cudaStream_t stream) {
    if (RGY_CSP_DATA_TYPE[pA->csp] == RGY_DATA_TYPE_U8) {
        return run_ivtc_frame_diff_typed<uint8_t, 8>(pA, pB, diffDev, stream);
    }
    if (RGY_CSP_DATA_TYPE[pA->csp] == RGY_DATA_TYPE_U16) {
        return run_ivtc_frame_diff_typed<uint16_t, 16>(pA, pB, diffDev, stream);
    }
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR run_ivtc_field_overlay(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int tff, cudaStream_t stream) {
    if (RGY_CSP_DATA_TYPE[pDst->csp] == RGY_DATA_TYPE_U8) {
        return run_ivtc_field_overlay_typed<uint8_t, 8>(pDst, pSrc, tff, stream);
    }
    if (RGY_CSP_DATA_TYPE[pDst->csp] == RGY_DATA_TYPE_U16) {
        return run_ivtc_field_overlay_typed<uint16_t, 16>(pDst, pSrc, tff, stream);
    }
    return RGY_ERR_UNSUPPORTED;
}
