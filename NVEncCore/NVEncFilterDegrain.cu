// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
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

#include "NVEncFilterDegrain.h"

#include <cstdint>

#include "rgy_cuda_util.h"
#include "rgy_err.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static constexpr int DEGRAIN_BLOCK_X = 16;
static constexpr int DEGRAIN_BLOCK_Y = 16;
static constexpr float DEGRAIN_PI_F = 3.14159265358979323846f;
static constexpr int DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES = 3;
static constexpr int DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS = 8;
static constexpr int DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE = 32;
static constexpr int DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE_MAX = DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE * DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS;
static constexpr uint32_t DEGRAIN_MOTION_SEARCH_LARGE_COST = 0xffffffffu;
static constexpr int DEGRAIN_MOTION_SEARCH_COST_SHIFT = 10;
static constexpr int DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE = 1 << (DEGRAIN_MOTION_SEARCH_COST_SHIFT - 8);
static constexpr uint64_t DEGRAIN_MOTION_SEARCH_COST_ROUND = uint64_t(1) << (DEGRAIN_MOTION_SEARCH_COST_SHIFT - 1);

struct RGYDegrainMotionSearchVector {
    uint32_t score_primary;
    uint32_t sad_metric;
    int16_t pos_x;
    int16_t pos_y;
};

static_assert(sizeof(RGYDegrainMotionSearchVector) == RGYDegrainMotionSearchWorkspace::VECTOR_BYTES, "RGYDegrainMotionSearchVector size mismatch.");

struct RGYDegrainMotionSearchCandidate {
    uint32_t score_primary;
    uint32_t sad_metric;
    int16_t pos_x;
    int16_t pos_y;
};

struct RGYDegrainMotionSearchCandidateCost {
    uint32_t score_primary;
    uint32_t sad_metric;
    int16_t pos_x;
    int16_t pos_y;
};

struct RGYDegrainMotionSearchContext {
    int minX;
    int minY;
    int maxX;
    int maxY;
    int motionCostWeight;
};

__device__ __forceinline__ RGYDegrainMotionSearchVector degrainMotionSearchMakeVector(
    const int posX, const int posY, const uint32_t sadMetric, const uint32_t scorePrimary) {
    RGYDegrainMotionSearchVector vec;
    vec.score_primary = scorePrimary;
    vec.sad_metric = sadMetric;
    vec.pos_x = (int16_t)posX;
    vec.pos_y = (int16_t)posY;
    return vec;
}

__device__ __forceinline__ RGYDegrainMotionSearchCandidate degrainMotionSearchMakeCandidate(
    const int posX, const int posY, const uint32_t sadMetric, const uint32_t scorePrimary) {
    RGYDegrainMotionSearchCandidate candidate;
    candidate.score_primary = scorePrimary;
    candidate.sad_metric = sadMetric;
    candidate.pos_x = (int16_t)posX;
    candidate.pos_y = (int16_t)posY;
    return candidate;
}

__device__ __forceinline__ RGYDegrainMotionSearchCandidate degrainMotionSearchSavedVectorToCandidate(
    const RGYDegrainMotionSearchVector vec) {
    return degrainMotionSearchMakeCandidate(vec.pos_x, vec.pos_y, vec.sad_metric, vec.score_primary);
}

__device__ __forceinline__ RGYDegrainMotionSearchVector degrainMotionSearchCandidateToSavedVector(
    const RGYDegrainMotionSearchCandidate candidate) {
    return degrainMotionSearchMakeVector(candidate.pos_x, candidate.pos_y, candidate.sad_metric, candidate.score_primary);
}

__device__ __forceinline__ RGYDegrainMotionSearchVector degrainMotionSearchCandidateCostToSavedVector(
    const RGYDegrainMotionSearchCandidateCost candidateCost) {
    return degrainMotionSearchMakeVector(candidateCost.pos_x, candidateCost.pos_y, candidateCost.sad_metric, candidateCost.score_primary);
}

__device__ __forceinline__ RGYDegrainMotionSearchCandidateCost degrainMotionSearchMakeCandidateCost(
    const int posX, const int posY, const uint32_t sadMetric, const uint32_t scorePrimary) {
    RGYDegrainMotionSearchCandidateCost candidateCost;
    candidateCost.score_primary = scorePrimary;
    candidateCost.sad_metric = sadMetric;
    candidateCost.pos_x = (int16_t)posX;
    candidateCost.pos_y = (int16_t)posY;
    return candidateCost;
}

__device__ __forceinline__ int degrainMotionSearchVecZeroIndex(const int planeBase) {
    return planeBase;
}

__device__ __forceinline__ int degrainMotionSearchVecGlobalIndex(const int planeBase) {
    return planeBase + 1;
}

__device__ __forceinline__ int degrainMotionSearchVecCurrentIndex(const int planeBase, const int blockCount, const int block) {
    return planeBase + 2 + min(max(block, 0), max(blockCount - 1, 0));
}

__device__ __forceinline__ int degrainMotionSearchVecPrevIndex(const int planeBase, const int blockCount, const int block) {
    return planeBase + 2 + min(max(block, 0), max(blockCount - 1, 0));
}

__device__ __forceinline__ int degrainMotionSearchVecFinalIndex(const int finalBase, const int blockCount, const int block) {
    return finalBase + min(max(block, 0), max(blockCount - 1, 0));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainPixelMax();
template<>
__device__ __forceinline__ int degrainPixelMax<uint8_t>() { return 255; }
template<>
__device__ __forceinline__ int degrainPixelMax<uint16_t>() { return 65535; }

__device__ __forceinline__ int degrainClampInt(const int value, const int minValue, const int maxValue) {
    return min(max(value, minValue), maxValue);
}

template<typename TypePixel>
__device__ __forceinline__ TypePixel degrainClampPixel(const int value) {
    return (TypePixel)degrainClampInt(value, 0, degrainPixelMax<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int degrainPixelLoad(
    const uint8_t *src, const int pitch, const int width, const int height, const int x, const int y) {
    const int px = degrainClampInt(x, 0, width - 1);
    const int py = degrainClampInt(y, 0, height - 1);
    return (int)(*(const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

__device__ __forceinline__ int degrainMirrorCoord(const int value, const int size) {
    const int reflectedLow  = max(value, -value - 1);
    const int reflectedHigh = min(reflectedLow, 2 * size - 1 - value);
    return degrainClampInt(reflectedHigh, 0, size - 1);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainPixelLoadMirror(
    const uint8_t *src, const int pitch, const int width, const int height, const int x, const int y) {
    const int px = degrainMirrorCoord(x, width);
    const int py = degrainMirrorCoord(y, height);
    return (int)(*(const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

__device__ __forceinline__ int degrainFloorRshiftSigned(const int value, const int rshift) {
    if (rshift <= 0) {
        return value;
    }
    return value >= 0
        ? value >> rshift
        : -(((-value) + (1 << rshift) - 1) >> rshift);
}

__device__ __forceinline__ int degrainRoundRshiftSigned(const int value, const int rshift) {
    if (rshift <= 0) {
        return value;
    }
    return value >= 0
        ? (value + (1 << (rshift - 1))) >> rshift
        : -(((-value) + (1 << (rshift - 1))) >> rshift);
}

__device__ __forceinline__ int degrainPelRshift(const int pel) {
    if (pel <= 1) {
        return 0;
    }
    if (pel == 2) {
        return 1;
    }
    if (pel == 4) {
        return 2;
    }
    return 0;
}

__device__ __forceinline__ int degrainFloorDivPel(const int value, const int pel) {
    return pel <= 1 ? value : degrainFloorRshiftSigned(value, degrainPelRshift(pel));
}

__device__ __forceinline__ int degrainFloorModPel(const int value, const int base, const int pel) {
    return value - (base << degrainPelRshift(pel));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerV(
    const uint8_t *src, const int pitch, const int width, const int height, const int baseX, const int baseY) {
    const int s0 = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY - 2);
    const int s1 = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY - 1);
    const int s2 = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY);
    const int s3 = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY + 1);
    const int s4 = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY + 2);
    const int s5 = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY + 3);
    const int sum = s0 + 5 * (-s1 + (s2 << 2) + (s3 << 2) - s4) + s5;
    return degrainClampPixel<TypePixel>((sum + 16) >> 5);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerHFromSamples(
    const int s0, const int s1, const int s2, const int s3, const int s4, const int s5) {
    const int sum = s0 + 5 * (-s1 + (s2 << 2) + (s3 << 2) - s4) + s5;
    return degrainClampPixel<TypePixel>((sum + 16) >> 5);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerH(
    const uint8_t *src, const int pitch, const int width, const int height, const int baseX, const int baseY) {
    return degrainInterpHalfpelWienerHFromSamples<TypePixel>(
        degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX - 2, baseY),
        degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX - 1, baseY),
        degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, baseY),
        degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + 1, baseY),
        degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + 2, baseY),
        degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + 3, baseY));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerHV(
    const uint8_t *src, const int pitch, const int width, const int height, const int baseX, const int baseY) {
    return degrainInterpHalfpelWienerHFromSamples<TypePixel>(
        degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX - 2, baseY),
        degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX - 1, baseY),
        degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX, baseY),
        degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX + 1, baseY),
        degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX + 2, baseY),
        degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX + 3, baseY));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWeighted(
    const uint8_t *src, const int pitch, const int width, const int height,
    const int baseX, const int baseY, const int fracX, const int fracY, const int interpMode) {
    if (interpMode == 2) {
        if (fracX != 0 && fracY != 0) {
            return degrainInterpHalfpelWienerHV<TypePixel>(src, pitch, width, height, baseX, baseY);
        }
        if (fracX != 0) {
            return degrainInterpHalfpelWienerH<TypePixel>(src, pitch, width, height, baseX, baseY);
        }
        if (fracY != 0) {
            return degrainInterpHalfpelWienerV<TypePixel>(src, pitch, width, height, baseX, baseY);
        }
    }

    const int offsets[4] = { -1, 0, 1, 2 };
    int weightsX[4] = { 0, 1, 0, 0 };
    int weightsY[4] = { 0, 1, 0, 0 };
    int denomXShift = 0;
    int denomYShift = 0;

    if (fracX != 0) {
        if (interpMode == 2) {
            weightsX[0] = -1; weightsX[1] = 9; weightsX[2] = 9; weightsX[3] = -1;
            denomXShift = 4;
        } else {
            weightsX[0] = 1; weightsX[1] = 3; weightsX[2] = 3; weightsX[3] = 1;
            denomXShift = 3;
        }
    }
    if (fracY != 0) {
        if (interpMode == 2) {
            weightsY[0] = -1; weightsY[1] = 9; weightsY[2] = 9; weightsY[3] = -1;
            denomYShift = 4;
        } else {
            weightsY[0] = 1; weightsY[1] = 3; weightsY[2] = 3; weightsY[3] = 1;
            denomYShift = 3;
        }
    }

    int sum = 0;
    for (int iy = 0; iy < 4; iy++) {
        if (weightsY[iy] == 0) {
            continue;
        }
        for (int ix = 0; ix < 4; ix++) {
            if (weightsX[ix] == 0) {
                continue;
            }
            const int sample = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + offsets[ix], baseY + offsets[iy]);
            sum += sample * weightsX[ix] * weightsY[iy];
        }
    }
    return degrainClampPixel<TypePixel>(degrainRoundRshiftSigned(sum, denomXShift + denomYShift));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelFromSamples(
    const int s0, const int s1, const int s2, const int s3, const int s4, const int s5, const int interpMode) {
    if (interpMode == 2) {
        return degrainInterpHalfpelWienerHFromSamples<TypePixel>(s0, s1, s2, s3, s4, s5);
    }
    const int sum = s1 + 3 * s2 + 3 * s3 + s4;
    return degrainClampPixel<TypePixel>((sum + 4) >> 3);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerVMirror(
    const uint8_t *src, const int pitch, const int width, const int height, const int baseX, const int baseY) {
    const int sum =
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY - 2)
        - 5 *
            (degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY - 1)
            + 4 * degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY)
            + 4 * degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY + 1)
            - degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY + 2))
        + degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY + 3);
    return degrainClampPixel<TypePixel>((sum + 16) >> 5);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerHMirror(
    const uint8_t *src, const int pitch, const int width, const int height, const int baseX, const int baseY) {
    return degrainInterpHalfpelWienerHFromSamples<TypePixel>(
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX - 2, baseY),
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX - 1, baseY),
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY),
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX + 1, baseY),
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX + 2, baseY),
        degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX + 3, baseY));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWienerHVMirror(
    const uint8_t *src, const int pitch, const int width, const int height, const int baseX, const int baseY) {
    return degrainInterpHalfpelWienerHFromSamples<TypePixel>(
        degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX - 2, baseY),
        degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX - 1, baseY),
        degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX, baseY),
        degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX + 1, baseY),
        degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX + 2, baseY),
        degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX + 3, baseY));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpHalfpelWeightedMirror(
    const uint8_t *src, const int pitch, const int width, const int height,
    const int baseX, const int baseY, const int fracX, const int fracY, const int interpMode) {
    if (interpMode == 2) {
        if (fracX != 0 && fracY != 0) {
            return degrainInterpHalfpelWienerHVMirror<TypePixel>(src, pitch, width, height, baseX, baseY);
        }
        if (fracX != 0) {
            return degrainInterpHalfpelWienerHMirror<TypePixel>(src, pitch, width, height, baseX, baseY);
        }
        if (fracY != 0) {
            return degrainInterpHalfpelWienerVMirror<TypePixel>(src, pitch, width, height, baseX, baseY);
        }
    }

    const int offsets[4] = { -1, 0, 1, 2 };
    int weightsX[4] = { 0, 1, 0, 0 };
    int weightsY[4] = { 0, 1, 0, 0 };
    int denomXShift = 0;
    int denomYShift = 0;

    if (fracX != 0) {
        if (interpMode == 2) {
            weightsX[0] = -1; weightsX[1] = 9; weightsX[2] = 9; weightsX[3] = -1;
            denomXShift = 4;
        } else {
            weightsX[0] = 1; weightsX[1] = 3; weightsX[2] = 3; weightsX[3] = 1;
            denomXShift = 3;
        }
    }
    if (fracY != 0) {
        if (interpMode == 2) {
            weightsY[0] = -1; weightsY[1] = 9; weightsY[2] = 9; weightsY[3] = -1;
            denomYShift = 4;
        } else {
            weightsY[0] = 1; weightsY[1] = 3; weightsY[2] = 3; weightsY[3] = 1;
            denomYShift = 3;
        }
    }

    int sum = 0;
    for (int iy = 0; iy < 4; iy++) {
        if (weightsY[iy] == 0) {
            continue;
        }
        for (int ix = 0; ix < 4; ix++) {
            if (weightsX[ix] == 0) {
                continue;
            }
            const int sample = degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX + offsets[ix], baseY + offsets[iy]);
            sum += sample * weightsX[ix] * weightsY[iy];
        }
    }
    return degrainClampPixel<TypePixel>(degrainRoundRshiftSigned(sum, denomXShift + denomYShift));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpPel4H(
    const uint8_t *src, const int pitch, const int width, const int height,
    const int baseX, const int y, const int fracX, const int interpMode) {
    if (fracX == 0) {
        return degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, y);
    }

    const int halfPix = (interpMode == 2)
        ? degrainInterpHalfpelWienerH<TypePixel>(src, pitch, width, height, baseX, y)
        : degrainInterpHalfpelFromSamples<TypePixel>(
            degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX - 2, y),
            degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX - 1, y),
            degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX, y),
            degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + 1, y),
            degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + 2, y),
            degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + 3, y),
            interpMode);
    if (fracX == 2) {
        return halfPix;
    }

    const int side = degrainPixelLoad<TypePixel>(src, pitch, width, height, baseX + (fracX > 2 ? 1 : 0), y);
    return (side + halfPix + 1) >> 1;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainInterpPel4(
    const uint8_t *src, const int pitch, const int width, const int height,
    const int baseX, const int baseY, const int fracX, const int fracY, const int interpMode) {
    if ((fracX & 1) == 0 && (fracY & 1) == 0) {
        return degrainInterpHalfpelWeighted<TypePixel>(src, pitch, width, height, baseX, baseY, fracX >> 1, fracY >> 1, interpMode);
    }
    if (fracY == 0) {
        return degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY, fracX, interpMode);
    }

    const int halfPix = degrainInterpHalfpelFromSamples<TypePixel>(
        degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY - 2, fracX, interpMode),
        degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY - 1, fracX, interpMode),
        degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY,     fracX, interpMode),
        degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY + 1, fracX, interpMode),
        degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY + 2, fracX, interpMode),
        degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY + 3, fracX, interpMode),
        interpMode);
    if (fracY == 2) {
        return halfPix;
    }

    const int side = degrainInterpPel4H<TypePixel>(src, pitch, width, height, baseX, baseY + (fracY > 2 ? 1 : 0), fracX, interpMode);
    return (side + halfPix + 1) >> 1;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainPixelLoadPelMirror(
    const uint8_t *src, const int pitch, const int width, const int height,
    const int xPel, const int yPel, const int pel, const int subpelInterp) {
    if (pel <= 1) {
        return degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, xPel, yPel);
    }

    const int baseX = degrainFloorDivPel(xPel, pel);
    const int baseY = degrainFloorDivPel(yPel, pel);
    const int fracX = degrainFloorModPel(xPel, baseX, pel);
    const int fracY = degrainFloorModPel(yPel, baseY, pel);
    if (fracX == 0 && fracY == 0) {
        return degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX, baseY);
    }

    if (pel == 2 && subpelInterp >= 1) {
        return degrainInterpHalfpelWeightedMirror<TypePixel>(src, pitch, width, height, baseX, baseY, fracX, fracY, subpelInterp);
    }

    const int p00 = degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX,     baseY);
    const int p10 = degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX + 1, baseY);
    const int p01 = degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX,     baseY + 1);
    const int p11 = degrainPixelLoadMirror<TypePixel>(src, pitch, width, height, baseX + 1, baseY + 1);
    const int invX = pel - fracX;
    const int invY = pel - fracY;
    const int value = p00 * invX * invY
        + p10 * fracX * invY
        + p01 * invX * fracY
        + p11 * fracX * fracY;
    return degrainRoundRshiftSigned(value, degrainPelRshift(pel) << 1);
}

__device__ __forceinline__ int degrainMotionSearchRefX(const int blockX, const int step, const int dx, const int pel) {
    return blockX * step + degrainFloorDivPel(dx, pel);
}

__device__ __forceinline__ int degrainMotionSearchRefY(const int blockY, const int step, const int dy, const int pel) {
    return blockY * step + degrainFloorDivPel(dy, pel);
}

__device__ __forceinline__ int degrainMotionSearchRefFracX(const int dx, const int pel) {
    const int base = degrainFloorDivPel(dx, pel);
    return degrainFloorModPel(dx, base, pel);
}

__device__ __forceinline__ int degrainMotionSearchRefFracY(const int dy, const int pel) {
    const int base = degrainFloorDivPel(dy, pel);
    return degrainFloorModPel(dy, base, pel);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainMotionSearchRefSample(
    const uint8_t *ref,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int dx,
    const int dy,
    const int x,
    const int y,
    const int pel,
    const int subpelInterp) {
    if (pel <= 1) {
        return degrainPixelLoadMirror<TypePixel>(
            ref,
            refPitch,
            width,
            height,
            degrainMotionSearchRefX(blockX, step, dx, pel) + x,
            degrainMotionSearchRefY(blockY, step, dy, pel) + y);
    }
    return degrainPixelLoadPelMirror<TypePixel>(
        ref,
        refPitch,
        width,
        height,
        (blockX * step + x) * pel + dx,
        (blockY * step + y) * pel + dy,
        pel,
        subpelInterp);
}

__device__ __forceinline__ int degrainMotionSearchRefIsIntegerPel(const int dx, const int dy, const int pel) {
    return pel <= 1
        || (degrainMotionSearchRefFracX(dx, pel) == 0 && degrainMotionSearchRefFracY(dy, pel) == 0);
}

__device__ __forceinline__ int degrainPlaneScaleRshift(const int planeScale) {
    return planeScale > 1 ? 1 : 0;
}

__device__ __forceinline__ int degrainPlaneScaleX(const int planeScaleX) {
    return max(planeScaleX, 1);
}

__device__ __forceinline__ int degrainPlaneScaleY(const int planeScaleY) {
    return max(planeScaleY, 1);
}

__device__ __forceinline__ int degrainPlaneScaleRshiftX(const int planeScaleX) {
    return degrainPlaneScaleRshift(degrainPlaneScaleX(planeScaleX));
}

__device__ __forceinline__ int degrainPlaneScaleRshiftY(const int planeScaleY) {
    return degrainPlaneScaleRshift(degrainPlaneScaleY(planeScaleY));
}

template<typename TypePixel>
__device__ __forceinline__ uint32_t degrainMotionSearchCalcSadLuma(
    const uint8_t *cur,
    const uint8_t *ref,
    const int curPitch,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int dx,
    const int dy,
    const int blockSize,
    const int pel,
    const int subpelInterp) {
    const int srcX = blockX * step;
    const int srcY = blockY * step;
    const int refX = degrainMotionSearchRefX(blockX, step, dx, pel);
    const int refY = degrainMotionSearchRefY(blockY, step, dy, pel);
    uint32_t sad = 0u;
    for (int y = 0; y < blockSize; y++) {
        for (int x = 0; x < blockSize; x++) {
            const int srcValue = degrainPixelLoad<TypePixel>(cur, curPitch, width, height, srcX + x, srcY + y);
            const int refValue = degrainMotionSearchRefSample<TypePixel>(ref, refPitch, width, height, blockX, blockY, step, dx, dy, x, y, pel, subpelInterp);
            sad += (uint32_t)abs(srcValue - refValue);
        }
    }
    return sad;
}

template<typename TypePixel>
__device__ __forceinline__ uint32_t degrainMotionSearchCalcSadLumaPart(
    const uint8_t *cur,
    const uint8_t *ref,
    const int curPitch,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int dx,
    const int dy,
    const int tx,
    const int blockSize,
    const int pel,
    const int subpelInterp) {
    const int srcX = blockX * step;
    const int srcY = blockY * step;
    const int refX = degrainMotionSearchRefX(blockX, step, dx, pel);
    const int refY = degrainMotionSearchRefY(blockY, step, dy, pel);
    const int x = tx % blockSize;
    uint32_t sad = 0u;
    for (int y = tx / blockSize; y < blockSize; y += 8) {
        const int srcValue = degrainPixelLoad<TypePixel>(cur, curPitch, width, height, srcX + x, srcY + y);
        const int refValue = degrainMotionSearchRefSample<TypePixel>(ref, refPitch, width, height, blockX, blockY, step, dx, dy, x, y, pel, subpelInterp);
        sad += (uint32_t)abs(srcValue - refValue);
    }
    return sad;
}

__device__ __forceinline__ uint32_t degrainMotionSearchReduceGroup(const uint32_t value) {
    return value;
}

__device__ __forceinline__ uint32_t degrainMotionSearchReduceCandidates(const uint32_t value) {
    return value;
}

__device__ __forceinline__ int degrainMotionSearchSquaredDistance(
    const int ax, const int ay, const int bx, const int by) {
    const int motionOffsetX = ax - bx;
    const int motionOffsetY = ay - by;
    return motionOffsetX * motionOffsetX + motionOffsetY * motionOffsetY;
}

__device__ __forceinline__ int degrainMotionSearchMedianOfThree(const int a, const int b, const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

__device__ __forceinline__ RGYDegrainMotionSearchCandidate degrainMotionSearchConstrainCandidate(
    RGYDegrainMotionSearchCandidate candidate,
    const RGYDegrainMotionSearchContext *context) {
    const int maxX = context->maxX;
    const int maxY = context->maxY;
    const int minX = context->minX;
    const int minY = context->minY;
    if (maxX > minX && maxY > minY) {
        candidate.pos_x = (int16_t)degrainClampInt((int)candidate.pos_x, minX, maxX - 1);
        candidate.pos_y = (int16_t)degrainClampInt((int)candidate.pos_y, minY, maxY - 1);
    }
    return candidate;
}

__device__ __forceinline__ int degrainMotionSearchMotionInsideSearchWindow(
    const int motionOffsetX,
    const int motionOffsetY,
    const RGYDegrainMotionSearchContext *context) {
    const int maxX = context->maxX;
    const int maxY = context->maxY;
    const int minX = context->minX;
    const int minY = context->minY;
    return maxX > minX && maxY > minY
        && motionOffsetX >= minX && motionOffsetX < maxX
        && motionOffsetY >= minY && motionOffsetY < maxY;
}

__device__ __forceinline__ uint32_t degrainMotionSearchCalcMotionCost(
    const RGYDegrainMotionSearchCandidate candidate,
    const int seedDx,
    const int seedDy,
    const RGYDegrainMotionSearchContext *context) {
    const uint64_t accum = (uint64_t)max(context->motionCostWeight, 0)
        * (uint64_t)degrainMotionSearchSquaredDistance((int)candidate.pos_x, (int)candidate.pos_y, seedDx, seedDy);
    return (uint32_t)((accum + DEGRAIN_MOTION_SEARCH_COST_ROUND) >> DEGRAIN_MOTION_SEARCH_COST_SHIFT);
}

__device__ __forceinline__ uint32_t degrainMotionSearchScaledSadPenalty(
    const uint32_t sad,
    const int penalty) {
    const uint64_t accum = (uint64_t)sad * (uint64_t)max(penalty, 0);
    return (uint32_t)((accum + DEGRAIN_MOTION_SEARCH_COST_ROUND) >> DEGRAIN_MOTION_SEARCH_COST_SHIFT);
}

__device__ __forceinline__ int degrainMotionSearchInitialCostScale(
    const int candidateGroupIndex,
    const int level,
    const int zeroCandidateCostScale,
    const int frameAverageCandidateCostScale,
    const int newCandidateCostScale) {
    return (candidateGroupIndex == 0) ? ((level == 0) ? 0 : zeroCandidateCostScale * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE)
        : (candidateGroupIndex == 1) ? frameAverageCandidateCostScale * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE
        : (candidateGroupIndex == 2) ? 0
        : newCandidateCostScale * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE;
}

__device__ __forceinline__ RGYDegrainMotionSearchContext degrainMotionSearchMakeSearchContext(
    const RGYDegrainMotionSearchVector seed,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    const int blockSize,
    const int pel,
    const int pad,
    const int lowSadWeightScale,
    const int motionCostScale) {
    const int sourceBaseX = blockGridX * step;
    const int sourceBaseY = blockGridY * step;
    RGYDegrainMotionSearchContext context;
    context.maxX = pel * (width + pad - sourceBaseX - blockSize);
    context.maxY = pel * (height + pad - sourceBaseY - blockSize);
    context.minX = -pel * (sourceBaseX + pad);
    context.minY = -pel * (sourceBaseY + pad);
    const int sadHalf = (int)(seed.sad_metric >> 1);
    context.motionCostWeight = 0;
    if (blockGridY > 0 && lowSadWeightScale > 0) {
        const int64_t denomLL = (int64_t)lowSadWeightScale + (int64_t)sadHalf;
        const int64_t denom2 = denomLL * denomLL;
        const int motionCostWeight = (denom2 > 0)
            ? (int)(((int64_t)motionCostScale * (int64_t)lowSadWeightScale * (int64_t)lowSadWeightScale) / denom2)
            : motionCostScale;
        context.motionCostWeight = motionCostWeight * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE;
    }
    return context;
}

template<typename TypePixel>
__device__ __forceinline__ uint32_t degrainMotionSearchAccumulateLumaSadLane(
    const TypePixel *sourceBlockPixels,
    const uint8_t *referencePlane,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int motionOffsetX,
    const int motionOffsetY,
    const int sadLane,
    const int blockSize,
    const int pel,
    const int subpelInterp) {
    const int referenceX = degrainMotionSearchRefX(blockX, step, motionOffsetX, pel);
    const int referenceY = degrainMotionSearchRefY(blockY, step, motionOffsetY, pel);
    const int lanesPerRow = blockSize / 4;
    const int x = (sadLane % lanesPerRow) * 4;
    const int rowsPerLane = blockSize / lanesPerRow;
    const int firstLaneRow = sadLane / lanesPerRow;
    const int useFastPath = sizeof(TypePixel) == 1
        && degrainMotionSearchRefIsIntegerPel(motionOffsetX, motionOffsetY, pel)
        && referenceX >= 0 && referenceY >= 0
        && referenceX + blockSize <= width
        && referenceY + blockSize <= height;
    uint32_t sad = 0u;
    if (useFastPath) {
        for (int y = firstLaneRow; y < blockSize; y += rowsPerLane) {
            const int sourceBase = y * blockSize + x;
            const uint8_t *referenceLine = referencePlane + (referenceY + y) * refPitch + (referenceX + x) * (int)sizeof(TypePixel);
            const int sourceValue0 = (int)sourceBlockPixels[sourceBase + 0];
            const int sourceValue1 = (int)sourceBlockPixels[sourceBase + 1];
            const int sourceValue2 = (int)sourceBlockPixels[sourceBase + 2];
            const int sourceValue3 = (int)sourceBlockPixels[sourceBase + 3];
            const int referenceValue0 = (int)(*(const TypePixel *)(referenceLine + 0 * (int)sizeof(TypePixel)));
            const int referenceValue1 = (int)(*(const TypePixel *)(referenceLine + 1 * (int)sizeof(TypePixel)));
            const int referenceValue2 = (int)(*(const TypePixel *)(referenceLine + 2 * (int)sizeof(TypePixel)));
            const int referenceValue3 = (int)(*(const TypePixel *)(referenceLine + 3 * (int)sizeof(TypePixel)));
            sad += (uint32_t)abs(sourceValue0 - referenceValue0);
            sad += (uint32_t)abs(sourceValue1 - referenceValue1);
            sad += (uint32_t)abs(sourceValue2 - referenceValue2);
            sad += (uint32_t)abs(sourceValue3 - referenceValue3);
        }
    } else {
        for (int y = firstLaneRow; y < blockSize; y += rowsPerLane) {
            const int sourceBase = y * blockSize + x;
            const int sourceValue0 = (int)sourceBlockPixels[sourceBase + 0];
            const int sourceValue1 = (int)sourceBlockPixels[sourceBase + 1];
            const int sourceValue2 = (int)sourceBlockPixels[sourceBase + 2];
            const int sourceValue3 = (int)sourceBlockPixels[sourceBase + 3];
            const int referenceValue0 = degrainMotionSearchRefSample<TypePixel>(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 0, y, pel, subpelInterp);
            const int referenceValue1 = degrainMotionSearchRefSample<TypePixel>(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 1, y, pel, subpelInterp);
            const int referenceValue2 = degrainMotionSearchRefSample<TypePixel>(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 2, y, pel, subpelInterp);
            const int referenceValue3 = degrainMotionSearchRefSample<TypePixel>(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 3, y, pel, subpelInterp);
            sad += (uint32_t)abs(sourceValue0 - referenceValue0);
            sad += (uint32_t)abs(sourceValue1 - referenceValue1);
            sad += (uint32_t)abs(sourceValue2 - referenceValue2);
            sad += (uint32_t)abs(sourceValue3 - referenceValue3);
        }
    }
    return sad;
}

__device__ __forceinline__ uint32_t degrainMotionSearchSumCandidateSadLanes(
    uint32_t *candidateLaneSums,
    const uint32_t sad,
    const int candidateIsValid,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockSize) {
    const int partialBase = candidateGroupIndex * blockSize;
    if (candidateIsValid) {
        candidateLaneSums[partialBase + sadLane] = sad;
    }
    __syncthreads();

    for (int offset = blockSize >> 1; offset > 0; offset >>= 1) {
        if (candidateIsValid && sadLane < offset) {
            candidateLaneSums[partialBase + sadLane] += candidateLaneSums[partialBase + sadLane + offset];
        }
        __syncthreads();
    }
    return (candidateIsValid && sadLane == 0) ? candidateLaneSums[partialBase] : 0u;
}

__device__ __forceinline__ void degrainMotionSearchSelectLowestCandidateCost(
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    const int localThreadId,
    const int candidateCount) {
    if (localThreadId < 8 && localThreadId >= candidateCount) {
        candidateCosts[localThreadId] = degrainMotionSearchMakeCandidateCost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    __syncthreads();

    for (int stride = 1; stride < 8; stride <<= 1) {
        if (localThreadId < 8
            && (localThreadId + stride) < 8
            && (localThreadId & ((stride << 1) - 1)) == 0
            && candidateCosts[localThreadId + stride].score_primary < candidateCosts[localThreadId].score_primary) {
            candidateCosts[localThreadId] = candidateCosts[localThreadId + stride];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int degrainMotionSearchFindFirstMatchingCandidate(
    const RGYDegrainMotionSearchCandidate *candidate,
    const int candidateGroupIndex,
    const int candidateCount) {
    int canonical = candidateGroupIndex;
    if (candidateGroupIndex < candidateCount) {
        const RGYDegrainMotionSearchCandidate motionVector = candidate[candidateGroupIndex];
        for (int i = 0; i < candidateGroupIndex; i++) {
            if (candidate[i].pos_x == motionVector.pos_x && candidate[i].pos_y == motionVector.pos_y) {
                canonical = i;
                break;
            }
        }
    }
    return canonical;
}

__device__ __forceinline__ int2 degrainMotionSearchRefineWide6Offset(const int offsetIndex) {
    const int row = offsetIndex >> 1;
    const int side = offsetIndex & 1;
    const int y = (row - 1) * 2;
    const int x = (row == 1) ? (side * 4 - 2) : (side * 2 - 1);
    return make_int2(x, y);
}

__device__ __forceinline__ int2 degrainMotionSearchRefineSquareOffset(const int offsetIndex) {
    const int gridIndex = offsetIndex + ((offsetIndex >= 4) ? 1 : 0);
    return make_int2(gridIndex % 3 - 1, gridIndex / 3 - 1);
}

__device__ __forceinline__ void degrainMotionSearchRefineClearResults(
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    const int localThreadId) {
    if (localThreadId < 8) {
        candidateCosts[localThreadId] = degrainMotionSearchMakeCandidateCost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
}

__device__ __forceinline__ void degrainMotionSearchRefinePrepareOffsetCandidate(
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int candidateCount,
    const int baseX,
    const int baseY,
    const int2 offset) {
    if (localThreadId < candidateCount) {
        const int motionOffsetX = baseX + offset.x;
        const int motionOffsetY = baseY + offset.y;
        const RGYDegrainMotionSearchCandidate candidate = degrainMotionSearchMakeCandidate(motionOffsetX, motionOffsetY, 0u, 0u);
        const uint32_t motionCost = degrainMotionSearchCalcMotionCost(candidate, baseX, baseY, context);
        if (degrainMotionSearchMotionInsideSearchWindow(motionOffsetX, motionOffsetY, context) && motionCost < bestCandidateCost->score_primary) {
            candidateCosts[localThreadId].pos_x = (int16_t)motionOffsetX;
            candidateCosts[localThreadId].pos_y = (int16_t)motionOffsetY;
            candidateCosts[localThreadId].score_primary = motionCost;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void degrainMotionSearchRefinePrepareHexCandidates(
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int baseX,
    const int baseY) {
    degrainMotionSearchRefineClearResults(candidateCosts, localThreadId);
    degrainMotionSearchRefinePrepareOffsetCandidate(
        context, candidateCosts, bestCandidateCost, localThreadId, 6, baseX, baseY, degrainMotionSearchRefineWide6Offset(localThreadId));
}

__device__ __forceinline__ void degrainMotionSearchRefinePrepareSquareCandidates(
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int baseX,
    const int baseY) {
    degrainMotionSearchRefineClearResults(candidateCosts, localThreadId);
    degrainMotionSearchRefinePrepareOffsetCandidate(
        context, candidateCosts, bestCandidateCost, localThreadId, 8, baseX, baseY, degrainMotionSearchRefineSquareOffset(localThreadId));
}

__device__ __forceinline__ int degrainMotionSearchRefineHasValidCandidates(
    const RGYDegrainMotionSearchCandidateCost *candidateCosts,
    const int candidateCount) {
    int hasCandidateToEvaluate = 0;
    for (int i = 0; i < candidateCount; i++) {
        hasCandidateToEvaluate |= candidateCosts[i].score_primary != DEGRAIN_MOTION_SEARCH_LARGE_COST;
    }
    return hasCandidateToEvaluate;
}

__device__ __forceinline__ int degrainRoundFloatToInt(const float value) {
    return __float2int_rn(value);
}

__device__ __forceinline__ float degrainOverlapBlendCurve(const float phase) {
    const float c = cosf(DEGRAIN_PI_F * phase);
    return 0.5f + 0.5f * c;
}

__device__ __forceinline__ float degrainOverlapAxisGain(
    const int pos,
    const int blockSize,
    const int overlap,
    const int isFirst,
    const int isLast) {
    if (pos < 0 || pos >= blockSize) {
        return 0.0f;
    }
    if (overlap <= 0) {
        return 1.0f;
    }
    if (pos < overlap) {
        if (isFirst) {
            return 1.0f;
        }
        const float phase = ((float)pos + 0.5f) / (float)overlap;
        return degrainOverlapBlendCurve(phase);
    }
    if (pos >= blockSize - overlap) {
        if (isLast) {
            return 1.0f;
        }
        const float phase = ((float)(blockSize - pos) - 0.5f) / (float)overlap;
        return degrainOverlapBlendCurve(phase);
    }
    return 1.0f;
}

__device__ __forceinline__ float degrainWindowFactorRect2d(
    const int x,
    const int y,
    const int baseX,
    const int baseY,
    const int blockSizeX,
    const int blockSizeY,
    const int overlapX,
    const int overlapY,
    const int blockX,
    const int blockY,
    const int blocksX,
    const int blocksY) {
    const int localX = x - baseX;
    const int localY = y - baseY;
    if (localX < 0 || localX >= blockSizeX || localY < 0 || localY >= blockSizeY) {
        return 0.0f;
    }
    const float wx = degrainOverlapAxisGain(localX, blockSizeX, overlapX, blockX == 0, blockX == blocksX - 1);
    const float wy = degrainOverlapAxisGain(localY, blockSizeY, overlapY, blockY == 0, blockY == blocksY - 1);
    return wx * wy;
}

__device__ __forceinline__ int degrainRefIndex(const int block, const int refDirection, const int refs);

__device__ __forceinline__ int degrainReferenceIsValid(
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const int block,
    const int refDirection,
    const uint32_t thsad,
    const int directionDisabled,
    const int refs) {
    if (directionDisabled) {
        return 0;
    }
    const int clampedRefDirection = degrainClampInt(refDirection, 0, refs - 1);
    const int index = degrainRefIndex(block, clampedRefDirection, refs);
    return ((int)mv[index].refdir == clampedRefDirection) && (sad[index].sad < thsad);
}

__device__ __forceinline__ float degrainReferenceAffinityFromSad(
    const int sadLimit,
    const int blockSad) {
    if (sadLimit <= blockSad) {
        return 0.0f;
    }
    const float sadRatio = (float)blockSad / (float)sadLimit;
    const float sadRatio2 = sadRatio * sadRatio;
    return (1.0f - sadRatio2) / (1.0f + sadRatio2);
}

__device__ __forceinline__ float degrainReferenceMixAffinity(
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const int block,
    const int refDirection,
    const uint32_t thsad,
    const int directionDisabled,
    const int refs) {
    if (!degrainReferenceIsValid(mv, sad, block, refDirection, thsad, directionDisabled, refs)) {
        return 0.0f;
    }
    return degrainReferenceAffinityFromSad((int)thsad, (int)sad[degrainRefIndex(block, refDirection, refs)].sad);
}

__device__ __forceinline__ float degrainTemporalMixPriorCenter(const float *temporalMixPrior) {
    return temporalMixPrior[0];
}

__device__ __forceinline__ float degrainTemporalMixPriorRef(
    const float *temporalMixPrior,
    const int refDirection) {
    return temporalMixPrior[1 + refDirection];
}

__device__ __forceinline__ int degrainTraceFloatToQ8(const float value) {
    return degrainRoundFloatToInt(value * 256.0f);
}

__device__ __forceinline__ int degrainRefDirectionDisabled(const uint32_t disableMask, const int refDirection) {
    return ((disableMask >> refDirection) & 1u) != 0u;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainCompensatedSample(
    const uint8_t *ref,
    const int refPitch,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const int block,
    const int refDirection,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int index = degrainRefIndex(block, refDirection, refs);
    const RGYDegrainMV motion = mv[index];
    const int scaledDx = degrainFloorRshiftSigned((int)motion.dx, degrainPlaneScaleRshiftX(planeScaleX));
    const int scaledDy = degrainFloorRshiftSigned((int)motion.dy, degrainPlaneScaleRshiftY(planeScaleY));
    if (pel <= 1) {
        const int sampleX = x + scaledDx;
        const int sampleY = y + scaledDy;
        if ((uint32_t)sampleX < (uint32_t)width && (uint32_t)sampleY < (uint32_t)height) {
            return (int)(*(const TypePixel *)(ref + sampleY * refPitch + sampleX * (int)sizeof(TypePixel)));
        }
        return degrainPixelLoadMirror<TypePixel>(ref, refPitch, width, height, sampleX, sampleY);
    }
    return degrainPixelLoadPelMirror<TypePixel>(
        ref, refPitch, width, height,
        x * pel + scaledDx,
        y * pel + scaledDy,
        pel,
        subpelInterp);
}

__device__ __forceinline__ const uint8_t *degrainRefPlanePtrSamePitch(
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int refDirection) {
    switch (refDirection) {
    case 0: return refBackward1;
    case 1: return refForward1;
    case 2: return refBackward2;
    case 3: return refForward2;
    case 4: return refBackward3;
    case 5: return refForward3;
    case 6: return refBackward4;
    case 7: return refForward4;
    case 8: return refBackward5;
    default: return refForward5;
    }
}

template<typename TypePixel>
__device__ __forceinline__ int degrainCompensateBlockSample(
    const uint8_t *ref0,
    const uint8_t *ref,
    const int pitch,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const int block,
    const int refDirection,
    const uint32_t thsad,
    const int directionDisabled,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int useReference = degrainReferenceIsValid(mv, sad, block, refDirection, thsad, directionDisabled, refs);
    return useReference
        ? degrainCompensatedSample<TypePixel>(ref, pitch, width, height, mv, block, refDirection, planeScaleX, planeScaleY, x, y, refs, pel, subpelInterp)
        : degrainPixelLoad<TypePixel>(ref0, pitch, width, height, x, y);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainDegrainBlockSample(
    const uint8_t *cur,
    const int pitch,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const int block,
    const uint32_t thsad,
    const uint32_t disableMask,
    const float *temporalMixPrior,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int currentSample = degrainPixelLoad<TypePixel>(cur, pitch, width, height, x, y);
    const float sourceConfidenceRaw = degrainTemporalMixPriorCenter(temporalMixPrior);
    float referenceConfidenceRaw[RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS];
    float confidenceTotal = sourceConfidenceRaw;
    for (int referenceDirection = 0; referenceDirection < refs; referenceDirection++) {
        const float temporalMixPriorRef = degrainTemporalMixPriorRef(temporalMixPrior, referenceDirection);
        referenceConfidenceRaw[referenceDirection] = degrainReferenceMixAffinity(
            mv, sad, block, referenceDirection, thsad, degrainRefDirectionDisabled(disableMask, referenceDirection), refs) * temporalMixPriorRef;
        confidenceTotal += referenceConfidenceRaw[referenceDirection];
    }
    const float invTotal = (confidenceTotal > 0.0f) ? (1.0f / confidenceTotal) : 0.0f;
    float mixedValue = (float)currentSample * (sourceConfidenceRaw * invTotal);
    for (int referenceDirection = 0; referenceDirection < refs; referenceDirection++) {
        if (referenceConfidenceRaw[referenceDirection] <= 0.0f) {
            continue;
        }
        const float referenceMixNorm = referenceConfidenceRaw[referenceDirection] * invTotal;
        const uint8_t *referencePlane = degrainRefPlanePtrSamePitch(
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            referenceDirection);
        const float referenceSample = (float)degrainCompensatedSample<TypePixel>(
            referencePlane, pitch, width, height, mv, block, referenceDirection, planeScaleX, planeScaleY, x, y, refs, pel, subpelInterp);
        mixedValue = __fmaf_rn(referenceSample, referenceMixNorm, mixedValue);
    }
    return degrainClampPixel<TypePixel>(degrainRoundFloatToInt(mixedValue));
}

__device__ __forceinline__ int degrainTemporalMixPlanOffset(const int block, const int refs) {
    return block * (refs + 1);
}

__device__ __forceinline__ float degrainTemporalMixPlanSrc(
    const float *temporalMixPlan,
    const int planOffset) {
    return temporalMixPlan[planOffset];
}

__device__ __forceinline__ float degrainTemporalMixPlanRef(
    const float *temporalMixPlan,
    const int planOffset,
    const int refDirection) {
    return temporalMixPlan[planOffset + 1 + refDirection];
}

template<typename TypePixel, int refs>
__device__ __forceinline__ int degrainApplyTemporalMixPlanSamePitch(
    const uint8_t *cur,
    const int pitch,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const int block,
    const float *temporalMixPlan,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y,
    const int pel,
    const int subpelInterp) {
    const int srcSample = degrainPixelLoad<TypePixel>(cur, pitch, width, height, x, y);
    const int planOffset = degrainTemporalMixPlanOffset(block, refs);
    float value = (float)srcSample * degrainTemporalMixPlanSrc(temporalMixPlan, planOffset);
    for (int refDirection = 0; refDirection < refs; refDirection++) {
        const float referenceMixNorm = degrainTemporalMixPlanRef(temporalMixPlan, planOffset, refDirection);
        if (referenceMixNorm <= 0.0f) {
            continue;
        }
        const uint8_t *ref = degrainRefPlanePtrSamePitch(
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            refDirection);
        const float refSample = (float)degrainCompensatedSample<TypePixel>(
            ref, pitch, width, height, mv, block, refDirection, planeScaleX, planeScaleY, x, y, refs, pel, subpelInterp);
        value = __fmaf_rn(refSample, referenceMixNorm, value);
    }
    return degrainClampPixel<TypePixel>(degrainRoundFloatToInt(value));
}

__device__ __forceinline__ int degrainWindowedSampleContribution(
    const int sample,
    const float windowWeight) {
    return degrainRoundFloatToInt((float)sample * windowWeight);
}

using RGYDegrainWindowAccum = float;

__device__ __forceinline__ RGYDegrainWindowAccum degrainWindowAccumZero() {
    return 0.0f;
}

__device__ __forceinline__ void degrainAccumulateWindowedSample(
    RGYDegrainWindowAccum *sampleSum,
    RGYDegrainWindowAccum *weightSum,
    const int sample,
    const float windowWeight) {
    if (windowWeight > 0.0f) {
        *sampleSum = __fmaf_rn((float)sample, windowWeight, *sampleSum);
        *weightSum += windowWeight;
    }
}

__device__ __forceinline__ int degrainFinalizeWindowedSample(
    const RGYDegrainWindowAccum sampleSum,
    const RGYDegrainWindowAccum weightSum,
    const int fallback) {
    return (weightSum > 0.0f) ? degrainRoundFloatToInt(sampleSum / weightSum) : fallback;
}

__device__ __forceinline__ int degrainTraceWindowAccum(const RGYDegrainWindowAccum sampleSum) {
    return degrainRoundFloatToInt(sampleSum);
}

__device__ __forceinline__ void degrainAccumulateWeightedSampleFp32(
    float *sampleSum,
    float *weightSum,
    const int sample,
    const float weight) {
    if (weight > 0.0f) {
        *sampleSum = __fmaf_rn((float)sample, weight, *sampleSum);
        *weightSum += weight;
    }
}

__device__ __forceinline__ int degrainFinalizeWeightedSampleFp32(
    const float sampleSum,
    const float weightSum,
    const int fallback) {
    return (weightSum > 0.0f) ? degrainRoundFloatToInt(sampleSum / weightSum) : fallback;
}

template<typename TypePixel>
__device__ __forceinline__ void degrainMotionSearchRefineEvaluateCandidates(
    const TypePixel *sourceBlockPixels,
    const uint8_t *referencePlane,
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    uint32_t *candidateLaneSums,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int candidateCount,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int newCandidateCostScale) {
    const int candidateIsValid = candidateGroupIndex < candidateCount
        && candidateCosts[candidateGroupIndex].score_primary != DEGRAIN_MOTION_SEARCH_LARGE_COST;
    uint32_t sad = 0u;
    if (candidateIsValid) {
        sad = degrainMotionSearchAccumulateLumaSadLane<TypePixel>(
            sourceBlockPixels,
            referencePlane,
            refPitch,
            width,
            height,
            blockX,
            blockY,
            step,
            candidateCosts[candidateGroupIndex].pos_x,
            candidateCosts[candidateGroupIndex].pos_y,
            sadLane,
            blockSize,
            pel,
            subpelInterp);
    }
    sad = degrainMotionSearchSumCandidateSadLanes(candidateLaneSums, sad, candidateIsValid, sadLane, candidateGroupIndex, blockSize);

    if (candidateIsValid && sadLane == 0) {
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary += sad
            + degrainMotionSearchScaledSadPenalty(sad, newCandidateCostScale * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE);
    }
    __syncthreads();

    degrainMotionSearchSelectLowestCandidateCost(candidateCosts, localThreadId, candidateCount);

    if (localThreadId == 0 && candidateCosts[0].score_primary < bestCandidateCost->score_primary) {
        *bestCandidateCost = candidateCosts[0];
    }
    __syncthreads();
}

template<typename TypePixel>
__device__ __forceinline__ void degrainMotionSearchRefinePreparedCandidates(
    const TypePixel *sourceBlockPixels,
    const uint8_t *referencePlane,
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    uint32_t *candidateLaneSums,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height,
    const int candidateCount,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int newCandidateCostScale) {
    degrainMotionSearchRefineEvaluateCandidates<TypePixel>(
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, candidateCount, blockX, blockY, step, refPitch, width, height,
        blockSize, pel, subpelInterp, newCandidateCostScale);
}

template<typename TypePixel>
__device__ __forceinline__ void degrainMotionSearchRefineHex2(
    const TypePixel *sourceBlockPixels,
    const uint8_t *referencePlane,
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    uint32_t *candidateLaneSums,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int newCandidateCostScale) {
    degrainMotionSearchRefinePrepareHexCandidates(context, candidateCosts, bestCandidateCost, localThreadId, bestCandidateCost->pos_x, bestCandidateCost->pos_y);
    degrainMotionSearchRefinePreparedCandidates<TypePixel>(
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockX, blockY, step, refPitch, width, height, 6,
        blockSize, pel, subpelInterp, newCandidateCostScale);
}

template<typename TypePixel>
__device__ __forceinline__ void degrainMotionSearchRefineSquare8(
    const TypePixel *sourceBlockPixels,
    const uint8_t *referencePlane,
    const RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    uint32_t *candidateLaneSums,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int newCandidateCostScale) {
    degrainMotionSearchRefinePrepareSquareCandidates(context, candidateCosts, bestCandidateCost, localThreadId, bestCandidateCost->pos_x, bestCandidateCost->pos_y);
    degrainMotionSearchRefinePreparedCandidates<TypePixel>(
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockX, blockY, step, refPitch, width, height, 8,
        blockSize, pel, subpelInterp, newCandidateCostScale);
}

__device__ __forceinline__ RGYDegrainMotionSearchCandidate degrainMotionSearchLoadBaseCandidate(
    const RGYDegrainMotionSearchVector *vectors,
    const RGYDegrainMotionSearchContext *context,
    const int candidateSlot,
    const int planeBase,
    const int blockCount,
    const int block) {
    RGYDegrainMotionSearchVector motionVector = degrainMotionSearchMakeVector(0, 0, 0u, 0u);
    switch (candidateSlot) {
    case 0:
        motionVector = vectors[degrainMotionSearchVecZeroIndex(planeBase)];
        break;
    case 1:
        motionVector = vectors[degrainMotionSearchVecGlobalIndex(planeBase)];
        break;
    case 2:
        motionVector = vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block)];
        break;
    default:
        break;
    }
    return degrainMotionSearchConstrainCandidate(degrainMotionSearchSavedVectorToCandidate(motionVector), context);
}

template<typename TypePixel>
__device__ __forceinline__ void degrainMotionSearchSearchOneBlock(
    const uint8_t *sourcePlane,
    const uint8_t *referencePlane,
    RGYDegrainMotionSearchVector *vectors,
    const int pitch,
    const int width,
    const int height,
    const int contextWidth,
    const int contextHeight,
    const int planeBase,
    const int blockCount,
    const int step,
    const int block,
    const int blockGridX,
    const int blockGridY,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int localSize,
    TypePixel *sourceBlockPixels,
    RGYDegrainMotionSearchContext *context,
    RGYDegrainMotionSearchCandidate *candidate,
    RGYDegrainMotionSearchCandidateCost *candidateCosts,
    RGYDegrainMotionSearchCandidateCost *bestCandidateCost,
    uint32_t *candidateLaneSums,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int pad,
    const int motionCostScale,
    const int lowSadWeightScale,
    const int zeroCandidateCostScale,
    const int frameAverageCandidateCostScale,
    const int newCandidateCostScale,
    const int level) {
    const int sourceBaseX = blockGridX * step;
    const int sourceBaseY = blockGridY * step;
    for (int i = localThreadId; i < blockSize * blockSize; i += localSize) {
        const int x = i % blockSize;
        const int y = i / blockSize;
        sourceBlockPixels[i] = (TypePixel)degrainPixelLoad<TypePixel>(sourcePlane, pitch, width, height, sourceBaseX + x, sourceBaseY + y);
    }

    const RGYDegrainMotionSearchVector initialSeed = vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block)];
    if (localThreadId == 0) {
        *context = degrainMotionSearchMakeSearchContext(initialSeed, contextWidth, contextHeight, blockGridX, blockGridY, step,
            blockSize, pel, pad, lowSadWeightScale, motionCostScale);
    }
    __syncthreads();

    if (localThreadId < DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES) {
        candidate[localThreadId] = degrainMotionSearchLoadBaseCandidate(vectors, context, localThreadId, planeBase, blockCount, block);
    }
    if (localThreadId < DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS) {
        candidateCosts[localThreadId] = degrainMotionSearchMakeCandidateCost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    __syncthreads();

    const int candidateIsValid = candidateGroupIndex < DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES;
    RGYDegrainMotionSearchCandidate motionVector = degrainMotionSearchMakeCandidate(0, 0, 0u, 0u);
    if (candidateIsValid) {
        motionVector = candidate[candidateGroupIndex];
    }
    const int firstMatchingCandidateIndex = degrainMotionSearchFindFirstMatchingCandidate(
        candidate, candidateGroupIndex, DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES);
    const int candidateNeedsEvaluation = candidateIsValid && firstMatchingCandidateIndex == candidateGroupIndex;
    uint32_t sad = 0u;
    if (candidateNeedsEvaluation) {
        sad = degrainMotionSearchAccumulateLumaSadLane<TypePixel>(
            sourceBlockPixels,
            referencePlane,
            pitch,
            width,
            height,
            blockGridX,
            blockGridY,
            step,
            motionVector.pos_x,
            motionVector.pos_y,
            sadLane,
            blockSize,
            pel,
            subpelInterp);
    }
    sad = degrainMotionSearchSumCandidateSadLanes(candidateLaneSums, sad, candidateNeedsEvaluation, sadLane, candidateGroupIndex, blockSize);

    if (candidateNeedsEvaluation && sadLane == 0) {
        uint32_t cost = sad;
        if (candidateGroupIndex < 3) {
            cost += degrainMotionSearchScaledSadPenalty(
                sad,
                degrainMotionSearchInitialCostScale(candidateGroupIndex, level, zeroCandidateCostScale, frameAverageCandidateCostScale, newCandidateCostScale));
        } else {
            cost += degrainMotionSearchCalcMotionCost(motionVector, initialSeed.pos_x, initialSeed.pos_y, context);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    __syncthreads();

    if (candidateIsValid && !candidateNeedsEvaluation && sadLane == 0) {
        sad = candidateCosts[firstMatchingCandidateIndex].sad_metric;
        uint32_t cost = sad;
        if (candidateGroupIndex < 3) {
            cost += degrainMotionSearchScaledSadPenalty(
                sad,
                degrainMotionSearchInitialCostScale(candidateGroupIndex, level, zeroCandidateCostScale, frameAverageCandidateCostScale, newCandidateCostScale));
        } else {
            cost += degrainMotionSearchCalcMotionCost(motionVector, initialSeed.pos_x, initialSeed.pos_y, context);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    __syncthreads();

    degrainMotionSearchSelectLowestCandidateCost(candidateCosts, localThreadId, DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES);

    if (localThreadId == 0) {
        *bestCandidateCost = candidateCosts[0];
    }
    __syncthreads();

    degrainMotionSearchRefineHex2<TypePixel>(
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockGridX, blockGridY, step, pitch, width, height,
        blockSize, pel, subpelInterp, newCandidateCostScale);

    degrainMotionSearchRefineSquare8<TypePixel>(
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockGridX, blockGridY, step, pitch, width, height,
        blockSize, pel, subpelInterp, newCandidateCostScale);

    if (localThreadId == 0) {
        vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block)] =
            degrainMotionSearchCandidateCostToSavedVector(*bestCandidateCost);
    }
}

template<typename TypePixel>
__global__ void kernel_degrain_mv_search_parallel_cuda(
    const uint8_t *sourcePlane,
    const uint8_t *referencePlane,
    RGYDegrainMotionSearchVector *vectors,
    const int pitch,
    const int width,
    const int height,
    const int planeBase,
    const int blockCount,
    const int blocksX,
    const int blocksY,
    const int step,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int pad,
    const int motionCostScale,
    const int lowSadWeightScale,
    const int zeroCandidateCostScale,
    const int frameAverageCandidateCostScale,
    const int newCandidateCostScale,
    const int level) {
    const int localThreadId = (int)threadIdx.x;
    const int sadLane = localThreadId % blockSize;
    const int candidateGroupIndex = localThreadId / blockSize;
    const int localSize = (int)blockDim.x;
    const int block = (int)blockIdx.x;

    __shared__ TypePixel sourceBlockPixels[DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE * DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE];
    __shared__ RGYDegrainMotionSearchContext context;
    __shared__ RGYDegrainMotionSearchCandidate candidate[DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES];
    __shared__ RGYDegrainMotionSearchCandidateCost candidateCosts[DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS];
    __shared__ RGYDegrainMotionSearchCandidateCost bestCandidateCost;
    __shared__ uint32_t candidateLaneSums[DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE_MAX];

    if (sourcePlane == nullptr || referencePlane == nullptr || blocksX <= 0 || blocksY <= 0 || block >= blockCount) {
        return;
    }

    const int blockGridX = block % blocksX;
    const int blockGridY = block / blocksX;
    if (blockGridY >= blocksY) {
        return;
    }

    degrainMotionSearchSearchOneBlock<TypePixel>(
        sourcePlane,
        referencePlane,
        vectors,
        pitch,
        width,
        height,
        width,
        height,
        planeBase,
        blockCount,
        step,
        block,
        blockGridX,
        blockGridY,
        localThreadId,
        sadLane,
        candidateGroupIndex,
        localSize,
        sourceBlockPixels,
        &context,
        candidate,
        candidateCosts,
        &bestCandidateCost,
        candidateLaneSums,
        blockSize,
        pel,
        subpelInterp,
        pad,
        motionCostScale,
        lowSadWeightScale,
        zeroCandidateCostScale,
        frameAverageCandidateCostScale,
        newCandidateCostScale,
        level);
}

template<typename TypePixel>
__global__ void kernel_degrain_mv_spatial_refine_cuda(
    const uint8_t *sourcePlane,
    const uint8_t *referencePlane,
    RGYDegrainMotionSearchVector *vectors,
    const RGYDegrainMotionSearchVector *vectorsPrev,
    RGYDegrainMotionSearchVector *vectorsFinal,
    const int pitch,
    const int width,
    const int height,
    const int planeBase,
    const int finalBase,
    const int blockCount,
    const int blocksX,
    const int blocksY,
    const int step,
    const int blockSize,
    const int pel,
    const int subpelInterp,
    const int pad,
    const int motionCostScale,
    const int lowSadWeightScale,
    const int newCandidateCostScale) {
    const int localThreadId = (int)threadIdx.x;
    const int sadLane = localThreadId % blockSize;
    const int candidateGroupIndex = localThreadId / blockSize;
    const int localSize = (int)blockDim.x;
    const int block = (int)blockIdx.x;
    if (sourcePlane == nullptr || referencePlane == nullptr || block >= blockCount || blocksX <= 0 || blocksY <= 0) {
        return;
    }

    const int blockGridX = block % blocksX;
    const int blockGridY = block / blocksX;
    __shared__ TypePixel sourceBlockPixels[DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE * DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE];
    __shared__ RGYDegrainMotionSearchContext context;
    __shared__ RGYDegrainMotionSearchCandidate candidate[5];
    __shared__ RGYDegrainMotionSearchCandidateCost candidateCosts[DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS];
    __shared__ RGYDegrainMotionSearchCandidateCost bestCandidateCost;
    __shared__ int reusePreviousSad;
    __shared__ uint32_t candidateLaneSums[DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE_MAX];

    const int sourceBaseX = blockGridX * step;
    const int sourceBaseY = blockGridY * step;
    for (int i = localThreadId; i < blockSize * blockSize; i += localSize) {
        const int x = i % blockSize;
        const int y = i / blockSize;
        sourceBlockPixels[i] = (TypePixel)degrainPixelLoad<TypePixel>(sourcePlane, pitch, width, height, sourceBaseX + x, sourceBaseY + y);
    }

    const RGYDegrainMotionSearchVector initialSeed = vectorsPrev[degrainMotionSearchVecPrevIndex(planeBase, blockCount, block)];
    if (localThreadId == 0) {
        context = degrainMotionSearchMakeSearchContext(initialSeed, width, height, blockGridX, blockGridY, step,
            blockSize, pel, pad, lowSadWeightScale, motionCostScale);
    }
    __syncthreads();

    if (localThreadId == 0) {
        const RGYDegrainMotionSearchVector base = vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block)];
        const RGYDegrainMotionSearchCandidate baseCandidate = degrainMotionSearchSavedVectorToCandidate(base);
        bestCandidateCost.pos_x = base.pos_x;
        bestCandidateCost.pos_y = base.pos_y;
        bestCandidateCost.sad_metric = base.sad_metric;
        bestCandidateCost.score_primary = base.score_primary;
        candidate[0] = degrainMotionSearchConstrainCandidate(baseCandidate, &context);
        reusePreviousSad = candidate[0].pos_x == base.pos_x && candidate[0].pos_y == base.pos_y;
        candidate[1] = (blockGridX > 0)
            ? degrainMotionSearchConstrainCandidate(degrainMotionSearchSavedVectorToCandidate(vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block - 1)]), &context)
            : candidate[0];
        candidate[2] = (blockGridY > 0)
            ? degrainMotionSearchConstrainCandidate(degrainMotionSearchSavedVectorToCandidate(vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block - blocksX)]), &context)
            : candidate[0];
        candidate[3] = (blockGridX + 1 < blocksX && blockGridY + 1 < blocksY)
            ? degrainMotionSearchConstrainCandidate(degrainMotionSearchSavedVectorToCandidate(vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block + blocksX + 1)]), &context)
            : candidate[0];
        candidate[4].pos_x = (int16_t)degrainMotionSearchMedianOfThree(candidate[1].pos_x, candidate[2].pos_x, candidate[3].pos_x);
        candidate[4].pos_y = (int16_t)degrainMotionSearchMedianOfThree(candidate[1].pos_y, candidate[2].pos_y, candidate[3].pos_y);
        candidate[4].sad_metric = 0u;
        candidate[4].score_primary = 0u;
    }
    if (localThreadId < 5) {
        candidateCosts[localThreadId] = degrainMotionSearchMakeCandidateCost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    __syncthreads();

    const int candidateCount = 5;
    const int candidateIsValid = candidateGroupIndex < candidateCount;
    const int firstMatchingCandidateIndex = degrainMotionSearchFindFirstMatchingCandidate(candidate, candidateGroupIndex, candidateCount);
    const int candidateNeedsEvaluation = candidateIsValid && firstMatchingCandidateIndex == candidateGroupIndex;
    uint32_t sad = 0u;
    if (candidateNeedsEvaluation) {
        if (candidateGroupIndex == 0 && reusePreviousSad) {
            sad = (sadLane == 0) ? bestCandidateCost.sad_metric : 0u;
        } else {
            const RGYDegrainMotionSearchCandidate motionVector = candidate[candidateGroupIndex];
            sad = degrainMotionSearchAccumulateLumaSadLane<TypePixel>(
                sourceBlockPixels, referencePlane, pitch, width, height, blockGridX, blockGridY, step,
                motionVector.pos_x, motionVector.pos_y, sadLane, blockSize, pel, subpelInterp);
        }
    }
    sad = degrainMotionSearchSumCandidateSadLanes(candidateLaneSums, sad, candidateNeedsEvaluation, sadLane, candidateGroupIndex, blockSize);

    if (candidateNeedsEvaluation && sadLane == 0) {
        const RGYDegrainMotionSearchCandidate motionVector = candidate[candidateGroupIndex];
        uint32_t cost = sad;
        if (candidateGroupIndex > 0) {
            cost += degrainMotionSearchCalcMotionCost(motionVector, initialSeed.pos_x, initialSeed.pos_y, &context);
        } else {
            cost = min(bestCandidateCost.score_primary, sad);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    __syncthreads();

    if (candidateIsValid && !candidateNeedsEvaluation && sadLane == 0) {
        const RGYDegrainMotionSearchCandidate motionVector = candidate[candidateGroupIndex];
        sad = candidateCosts[firstMatchingCandidateIndex].sad_metric;
        uint32_t cost = sad;
        if (candidateGroupIndex > 0) {
            cost += degrainMotionSearchCalcMotionCost(motionVector, initialSeed.pos_x, initialSeed.pos_y, &context);
        } else {
            cost = min(bestCandidateCost.score_primary, sad);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    __syncthreads();

    degrainMotionSearchSelectLowestCandidateCost(candidateCosts, localThreadId, candidateCount);

    if (localThreadId == 0 && candidateCosts[0].score_primary < bestCandidateCost.score_primary) {
        bestCandidateCost = candidateCosts[0];
    }
    __syncthreads();

    degrainMotionSearchRefineSquare8<TypePixel>(
        sourceBlockPixels, referencePlane, &context, candidateCosts, candidateLaneSums, &bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockGridX, blockGridY, step, pitch, width, height,
        blockSize, pel, subpelInterp, newCandidateCostScale);

    if (localThreadId == 0) {
        vectorsFinal[degrainMotionSearchVecFinalIndex(finalBase, blockCount, block)] =
            degrainMotionSearchCandidateCostToSavedVector(bestCandidateCost);
    }
}

template<typename TypePixel>
__device__ __forceinline__ int degrainAnalysisLumaToFullRange(const int value, const int tvRange) {
    if (!tvRange) {
        return value;
    }
    const int pixelMax = degrainPixelMax<TypePixel>();
    if (pixelMax <= 255) {
        const int converted = ((value - 16) * pixelMax + (219 >> 1)) / 219;
        return degrainClampInt(converted, 0, pixelMax);
    }
    const int limitedScale = max((pixelMax + 1) >> 8, 1);
    const int limitedOffset = 16 * limitedScale;
    const int limitedRange = 219 * limitedScale;
    const int delta = value - limitedOffset;
    const int converted = delta + (delta * (pixelMax - limitedRange) + (limitedRange >> 1)) / limitedRange;
    return degrainClampInt(converted, 0, pixelMax);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainTemporalSmoothValue(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius) {
    int value = degrainPixelLoad<TypePixel>(srcCur, srcCurPitch, srcWidth, srcHeight, px, py);
    if (smoothRadius >= 2) {
        const int sum =
            degrainPixelLoad<TypePixel>(srcPrev2, srcPrev2Pitch, srcWidth, srcHeight, px, py)
          + 4 * degrainPixelLoad<TypePixel>(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
          + 6 * value
          + 4 * degrainPixelLoad<TypePixel>(srcNext, srcNextPitch, srcWidth, srcHeight, px, py)
          + degrainPixelLoad<TypePixel>(srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py);
        value = (sum + 8) >> 4;
    } else if (smoothRadius >= 1) {
        const int sum =
            degrainPixelLoad<TypePixel>(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
          + 2 * value
          + degrainPixelLoad<TypePixel>(srcNext, srcNextPitch, srcWidth, srcHeight, px, py);
        value = (sum + 2) >> 2;
    }
    return value;
}

__device__ __forceinline__ int degrainBlur3x3Weighted(
    const int p00, const int p10, const int p20,
    const int p01, const int p11, const int p21,
    const int p02, const int p12, const int p22) {
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

__device__ __forceinline__ int degrainEdgeSoftenCross(const int left, const int up, const int center, const int down, const int right) {
    return (left + up + 4 * center + down + right + 4) >> 3;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainSearchRefine1Blend(
    const int center, const int blur, const int edgeSoft, const int left, const int up, const int right, const int down) {
    const int edgeScale = max((degrainPixelMax<TypePixel>() + 31) / 32, 1);
    const int edgeStrength = abs(left - right) + abs(up - down) + abs(center - blur);
    const int edgeWeight = degrainClampInt((edgeStrength + (edgeScale >> 1)) / edgeScale, 0, 4);
    return (blur * (4 - edgeWeight) + edgeSoft * edgeWeight + 2) >> 2;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainSearchRefine1Value(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius) {
    const int p00 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius);
    const int p10 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py - 1, smoothRadius);
    const int p20 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius);
    const int p01 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py,     smoothRadius);
    const int p11 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py,     smoothRadius);
    const int p21 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py,     smoothRadius);
    const int p02 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius);
    const int p12 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py + 1, smoothRadius);
    const int p22 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius);
    const int blur = degrainBlur3x3Weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
    const int edgeSoft = degrainEdgeSoftenCross(p01, p10, p11, p12, p21);
    return degrainSearchRefine1Blend<TypePixel>(p11, blur, edgeSoft, p01, p10, p21, p12);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainAnalysisPrefilterValue(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius, const int searchRefine) {
    int value = degrainTemporalSmoothValue<TypePixel>(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    if (searchRefine >= 1) {
        value = degrainSearchRefine1Value<TypePixel>(
            srcPrev2, srcPrev2Pitch,
            srcPrev, srcPrevPitch,
            srcCur, srcCurPitch,
            srcNext, srcNextPitch,
            srcNext2, srcNext2Pitch,
            srcWidth, srcHeight,
            px, py, smoothRadius);
    }
    return value;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainRep0RepairValue(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius, const int searchRefine) {
    const int p0 = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py, smoothRadius, searchRefine);
    const int p1u = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py - 1, smoothRadius, searchRefine);
    const int p1d = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py + 1, smoothRadius, searchRefine);
    const int p2u = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py - 2, smoothRadius, searchRefine);
    const int p2d = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py + 2, smoothRadius, searchRefine);

    const int vertAvg = (p1u + p1d + 1) >> 1;
    const int nearDiff = abs(p1u - p1d);
    const int farDiff = abs(p2u - p2d);
    const int centerDiff = abs(p0 - vertAvg);
    const int flatness = nearDiff + (farDiff >> 1);
    const int threshold = max(flatness + max(degrainPixelMax<TypePixel>() / 64, 1), max(degrainPixelMax<TypePixel>() / 32, 1));
    if (centerDiff <= threshold) {
        return p0;
    }

    const int repair = (p0 + 3 * vertAvg + 2) >> 2;
    const int lo = min(min(p1u, p1d), min(p2u, p2d));
    const int hi = max(max(p1u, p1d), max(p2u, p2d));
    return degrainClampInt(repair, lo, hi);
}

template<typename TypePixel>
__global__ void kernel_degrain_temporal_smooth_luma_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int srcPitch, uint8_t *dst, const int dstPitch, const int width, const int height,
    const int tr0, const int searchRefine, const int rep0, const int tvRange) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    const int analysisValue = (rep0 >= 1)
        ? degrainRep0RepairValue<TypePixel>(prev2, srcPitch, prev, srcPitch, cur, srcPitch, next, srcPitch, next2, srcPitch, width, height, x, y, tr0, searchRefine)
        : degrainAnalysisPrefilterValue<TypePixel>(prev2, srcPitch, prev, srcPitch, cur, srcPitch, next, srcPitch, next2, srcPitch, width, height, x, y, tr0, searchRefine);
    *(TypePixel *)(dst + y * dstPitch + x * (int)sizeof(TypePixel)) =
        degrainClampPixel<TypePixel>(degrainAnalysisLumaToFullRange<TypePixel>(analysisValue, tvRange));
}

RGY_ERR launchNVEncDegrainTemporalSmoothLuma(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    const RGYFrameInfo &dst, const int tr0, const int searchRefine, const int rep0, const int tvRange, cudaStream_t stream) {
    const auto block = dim3(DEGRAIN_BLOCK_X, DEGRAIN_BLOCK_Y);
    const auto grid = dim3(divCeil(dst.width, DEGRAIN_BLOCK_X), divCeil(dst.height, DEGRAIN_BLOCK_Y));
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        kernel_degrain_temporal_smooth_luma_cuda<uint16_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
            dst.width, dst.height, tr0, searchRefine, rep0, tvRange);
    } else {
        kernel_degrain_temporal_smooth_luma_cuda<uint8_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
            dst.width, dst.height, tr0, searchRefine, rep0, tvRange);
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
__global__ void kernel_degrain_downsample_luma2x_cuda(
    const uint8_t *src,
    const int srcPitch,
    uint8_t *dst,
    const int dstPitch,
    const int srcWidth,
    const int srcHeight,
    const int dstWidth,
    const int dstHeight) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    const int sx = x * 2;
    const int sy = y * 2;
    const int wx[4] = { 1, 3, 3, 1 };
    const int wy[4] = { 1, 3, 3, 1 };

    int sum = 0;
    for (int ky = 0; ky < 4; ++ky) {
        const int py = sy + ky - 1;
        for (int kx = 0; kx < 4; ++kx) {
            const int px = sx + kx - 1;
            const int pix = degrainPixelLoad<TypePixel>(src, srcPitch, srcWidth, srcHeight, px, py);
            sum += pix * wy[ky] * wx[kx];
        }
    }
    *(TypePixel *)(dst + y * dstPitch + x * (int)sizeof(TypePixel)) = degrainClampPixel<TypePixel>((sum + 32) >> 6);
}

RGY_ERR launchNVEncDegrainDownsampleLuma2x(
    const RGYFrameInfo &src, const CUMemBuf &dst, const int dstPitch, const int dstWidth, const int dstHeight, cudaStream_t stream) {
    const auto block = dim3(DEGRAIN_BLOCK_X, DEGRAIN_BLOCK_Y);
    const auto grid = dim3(divCeil(dstWidth, DEGRAIN_BLOCK_X), divCeil(dstHeight, DEGRAIN_BLOCK_Y));
    if (RGY_CSP_BIT_DEPTH[src.csp] > 8) {
        kernel_degrain_downsample_luma2x_cuda<uint16_t><<<grid, block, 0, stream>>>(
            src.ptr[0], src.pitch[0], reinterpret_cast<uint8_t *>(dst.ptr), dstPitch,
            src.width, src.height, dstWidth, dstHeight);
    } else {
        kernel_degrain_downsample_luma2x_cuda<uint8_t><<<grid, block, 0, stream>>>(
            src.ptr[0], src.pitch[0], reinterpret_cast<uint8_t *>(dst.ptr), dstPitch,
            src.width, src.height, dstWidth, dstHeight);
    }
    return err_to_rgy(cudaGetLastError());
}

__device__ __forceinline__ int degrainPrimaryBlockIndex(const int x, const int y, const int blocksX, const int blocksY, const int step) {
    const int clampedStep = max(step, 1);
    const int blockX = min(x / clampedStep, blocksX - 1);
    const int blockY = min(y / clampedStep, blocksY - 1);
    return blockY * blocksX + blockX;
}

__device__ __forceinline__ int degrainDebugBorder(const int x, const int y, const int step) {
    const int clampedStep = max(step, 1);
    return (x % clampedStep) == 0 || (y % clampedStep) == 0;
}

__device__ __forceinline__ int degrainBlockOrigin(const int block, const int step) {
    return block * max(step, 1);
}

__device__ __forceinline__ int degrainIsCoveredPixel(const int x, const int y, const int coveredWidth, const int coveredHeight) {
    return x < coveredWidth && y < coveredHeight;
}

__device__ __forceinline__ int degrainRefIndex(const int block, const int refDirection, const int refs) {
    const int clampedRefDirection = degrainClampInt(refDirection, 0, refs - 1);
    return block * refs + clampedRefDirection;
}

__device__ __forceinline__ int degrainRenderConstBlockSize(const int blockSize) {
    return blockSize;
}

__device__ __forceinline__ int degrainRenderConstOverlap(const int overlap) {
    return overlap;
}

__device__ __forceinline__ int degrainRenderConstStep(const int step) {
    return step;
}

__device__ __forceinline__ int degrainRenderConstBlocksX(const int blocksX) {
    return blocksX;
}

__device__ __forceinline__ int degrainRenderConstBlocksY(const int blocksY) {
    return blocksY;
}

__device__ __forceinline__ int degrainRenderScaleCovered(const int covered, const int scale) {
    const int rshift = degrainPlaneScaleRshift(scale);
    return (covered + ((1 << rshift) - 1)) >> rshift;
}

__device__ __forceinline__ int degrainRenderScaleFloor(const int value, const int scale) {
    return value >> degrainPlaneScaleRshift(scale);
}

__device__ __forceinline__ int degrainRenderConstCoveredWidth(const int coveredWidth, const int scaleX) {
    return coveredWidth;
}

__device__ __forceinline__ int degrainRenderConstCoveredHeight(const int coveredHeight, const int scaleY) {
    return coveredHeight;
}

template<int refs>
__global__ void kernel_degrain_build_temporal_mix_plan_cuda(
    float *temporalMixPlan,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const float *temporalMixPrior,
    const int blockCount,
    const uint32_t thsad,
    const uint32_t disableMask) {
    const int block = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= blockCount) {
        return;
    }

    const float sourceConfidenceRaw = degrainTemporalMixPriorCenter(temporalMixPrior);
    float referenceConfidenceRaw[refs];
    float confidenceTotal = sourceConfidenceRaw;
    for (int referenceDirection = 0; referenceDirection < refs; referenceDirection++) {
        const float temporalMixPriorRef = degrainTemporalMixPriorRef(temporalMixPrior, referenceDirection);
        referenceConfidenceRaw[referenceDirection] = degrainReferenceMixAffinity(mv, sad, block, referenceDirection, thsad, degrainRefDirectionDisabled(disableMask, referenceDirection), refs) * temporalMixPriorRef;
        confidenceTotal += referenceConfidenceRaw[referenceDirection];
    }

    float referenceMixTotal = 0.0f;
    const float invTotal = (confidenceTotal > 0.0f) ? (1.0f / confidenceTotal) : 0.0f;
    const int planOffset = degrainTemporalMixPlanOffset(block, refs);
    for (int referenceDirection = 0; referenceDirection < refs; referenceDirection++) {
        float referenceMixNorm = 0.0f;
        if (referenceConfidenceRaw[referenceDirection] > 0.0f) {
            referenceMixNorm = referenceConfidenceRaw[referenceDirection] * invTotal;
            referenceMixTotal += referenceMixNorm;
        }
        temporalMixPlan[planOffset + 1 + referenceDirection] = referenceMixNorm;
    }
    const float sourceMixNorm = max(1.0f - referenceMixTotal, 0.0f);
    temporalMixPlan[planOffset] = sourceMixNorm;
}

template<typename TypePixel>
__global__ void kernel_degrain_overlap_plane_cuda(
    TypePixel *dst,
    const int dst_pitch,
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *ref0,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const float *temporalMixPrior,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const int modeType,
    const int refDirection,
    const uint32_t thsad,
    const uint32_t disableMask,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int x = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int y = (int)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrainPlaneScaleX(planeScaleX);
    const int scaleY = degrainPlaneScaleY(planeScaleY);
    const int renderBlockSize = degrainRenderConstBlockSize(blockSize);
    const int renderOverlap = degrainRenderConstOverlap(overlap);
    const int renderStep = degrainRenderConstStep(step);
    const int renderBlocksX = degrainRenderConstBlocksX(blocksX);
    const int renderBlocksY = degrainRenderConstBlocksY(blocksY);
    const int renderCoveredWidth = degrainRenderConstCoveredWidth(coveredWidth, scaleX);
    const int renderCoveredHeight = degrainRenderConstCoveredHeight(coveredHeight, scaleY);
    if (!degrainIsCoveredPixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrainRenderScaleFloor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrainRenderScaleFloor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrainRenderScaleFloor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrainRenderScaleFloor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrainRenderScaleFloor(renderStep, scaleX), 1);
    const int planeStepY = max(degrainRenderScaleFloor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && x < degrainBlockOrigin(primaryBlockX, planeStepX) + planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && y < degrainBlockOrigin(primaryBlockY, planeStepY) + planeOverlapY;
    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    RGYDegrainWindowAccum sampleSum = degrainWindowAccumZero();
    RGYDegrainWindowAccum weightSum = degrainWindowAccumZero();
    int sampleCount = 0;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int baseY = degrainBlockOrigin(blockY, planeStepY);
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int baseX = degrainBlockOrigin(blockX, planeStepX);
            const int localX = x - baseX;
            const int localY = y - baseY;
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY) {
                continue;
            }
            const float windowWeight = degrainWindowFactorRect2d(
                x, y,
                baseX, baseY,
                planeBlockSizeX, planeBlockSizeY,
                planeOverlapX, planeOverlapY,
                blockX, blockY,
                renderBlocksX, renderBlocksY);

            const int block = blockY * renderBlocksX + blockX;
            int sample = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
            if (modeType == 0 || modeType == 1) {
                if (refDirection < refs
                    && (((modeType == 0) && ((refDirection & 1) == 0))
                        || ((modeType == 1) && ((refDirection & 1) == 1)))) {
                    const uint8_t *ref = degrainRefPlanePtrSamePitch(
                        refBackward1, refForward1,
                        refBackward2, refForward2,
                        refBackward3, refForward3,
                        refBackward4, refForward4,
                        refBackward5, refForward5,
                        refDirection);
                    sample = degrainCompensateBlockSample<TypePixel>(
                        ref0, ref, cur_pitch,
                        width, height,
                        mv, sad,
                        block, refDirection, thsad, degrainRefDirectionDisabled(disableMask, refDirection),
                        planeScaleX, planeScaleY,
                        x, y,
                        refs, pel, subpelInterp);
                }
            } else {
                sample = degrainDegrainBlockSample<TypePixel>(
                    cur, cur_pitch,
                    refBackward1, refForward1,
                    refBackward2, refForward2,
                    refBackward3, refForward3,
                    refBackward4, refForward4,
                    refBackward5, refForward5,
                    width, height,
                    mv, sad,
                    block, thsad,
                    disableMask,
                    temporalMixPrior,
                    planeScaleX, planeScaleY,
                    x, y,
                    refs, pel, subpelInterp);
            }
            degrainAccumulateWindowedSample(&sampleSum, &weightSum, sample, windowWeight);
            sampleCount++;
        }
    }

    const int result = (sampleCount > 0) ? degrainFinalizeWindowedSample(sampleSum, weightSum, fallback) : fallback;
    dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(result);
}

RGY_ERR launchNVEncDegrainBuildTemporalMixPlan(
    CUMemBuf &temporalMixPlan, const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const int blockCount, const uint32_t thsad, const uint32_t disableMask, const int refs, cudaStream_t stream) {
    const int block = 256;
    const int grid = divCeil(blockCount, block);
#define LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN(REFS) do { \
    kernel_degrain_build_temporal_mix_plan_cuda<REFS><<<grid, block, 0, stream>>>( \
        reinterpret_cast<float *>(temporalMixPlan.ptr), \
        reinterpret_cast<const RGYDegrainMV *>(mv.ptr), \
        reinterpret_cast<const RGYDegrainSAD *>(sad.ptr), \
        reinterpret_cast<const float *>(temporalMixPrior.ptr), \
        blockCount, thsad, disableMask); \
} while (0)
    switch (refs) {
    case 2: LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN(2); break;
    case 4: LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN(4); break;
    case 6: LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN(6); break;
    case 8: LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN(8); break;
    case 10: LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN(10); break;
    default:
        return RGY_ERR_UNSUPPORTED;
    }
#undef LAUNCH_DEGRAIN_BUILD_TEMPORAL_MIX_PLAN
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainOverlapPlane(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *ref0,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const VppDegrainMode mode, const int refDirection,
    const uint32_t thsad, const uint32_t disableMask,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    if (pixelBytes > 1) {
        kernel_degrain_overlap_plane_cuda<uint16_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<uint16_t *>(dst), dstPitch,
            cur, curPitch, ref0,
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            reinterpret_cast<const float *>(temporalMixPrior.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            (mode == VppDegrainMode::MotionBack || mode == VppDegrainMode::MotionBack2) ? 0 : 1, refDirection, thsad, disableMask, refs, pel, subpelInterp);
    } else {
        kernel_degrain_overlap_plane_cuda<uint8_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<uint8_t *>(dst), dstPitch,
            cur, curPitch, ref0,
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            reinterpret_cast<const float *>(temporalMixPrior.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            (mode == VppDegrainMode::MotionBack || mode == VppDegrainMode::MotionBack2) ? 0 : 1, refDirection, thsad, disableMask, refs, pel, subpelInterp);
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
__device__ __forceinline__ void degrainCompensateOverlapPlaneRampGeneric(
    TypePixel *dst,
    const int dst_pitch,
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *ref0,
    const uint8_t *ref,
    const int refDirection,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint32_t thsad,
    const uint32_t disableMask,
    const float *windowRamp,
    const int originX,
    const int originY,
    const int compactTopLeftBorder,
    const int globalX,
    const int globalY,
    const int refs,
    const int pel,
    const int subpelInterp) {
    int x = originX + globalX;
    int y = originY + globalY;
    if (compactTopLeftBorder) {
        if (originX <= 0 || originY <= 0) {
            return;
        }
        const int compactScaleX = degrainPlaneScaleX(planeScaleX);
        const int compactScaleY = degrainPlaneScaleY(planeScaleY);
        const int compactStep = degrainRenderConstStep(step);
        const int compactBlocksX = degrainRenderConstBlocksX(blocksX);
        const int compactBlocksY = degrainRenderConstBlocksY(blocksY);
        const int compactPlaneStepX = max(degrainRenderScaleFloor(compactStep, compactScaleX), 1);
        const int compactPlaneStepY = max(degrainRenderScaleFloor(compactStep, compactScaleY), 1);
        const int interiorEndX = min(width, compactBlocksX * compactPlaneStepX);
        const int interiorEndY = min(height, compactBlocksY * compactPlaneStepY);
        const int lowerHeight = max(height - originY, 0);
        const int rightBorderWidth = max(width - interiorEndX, 0);
        const int bottomBorderWidth = max(interiorEndX - originX, 0);
        const int borderIndex = globalX;
        const int topBorderPixels = width * originY;
        const int leftBorderPixels = originX * lowerHeight;
        const int rightBorderPixels = rightBorderWidth * lowerHeight;
        if (borderIndex < topBorderPixels) {
            x = borderIndex % width;
            y = borderIndex / width;
        } else if (borderIndex < topBorderPixels + leftBorderPixels) {
            const int leftBorderIndex = borderIndex - topBorderPixels;
            x = leftBorderIndex % originX;
            y = originY + leftBorderIndex / originX;
        } else if (borderIndex < topBorderPixels + leftBorderPixels + rightBorderPixels) {
            const int rightBorderIndex = borderIndex - topBorderPixels - leftBorderPixels;
            x = interiorEndX + rightBorderIndex % rightBorderWidth;
            y = originY + rightBorderIndex / rightBorderWidth;
        } else {
            const int bottomBorderIndex = borderIndex - topBorderPixels - leftBorderPixels - rightBorderPixels;
            if (bottomBorderWidth <= 0) {
                return;
            }
            x = originX + bottomBorderIndex % bottomBorderWidth;
            y = interiorEndY + bottomBorderIndex / bottomBorderWidth;
        }
    }
    if (x >= width || y >= height) {
        return;
    }

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrainPlaneScaleX(planeScaleX);
    const int scaleY = degrainPlaneScaleY(planeScaleY);
    const int renderBlockSize = degrainRenderConstBlockSize(blockSize);
    const int renderOverlap = degrainRenderConstOverlap(overlap);
    const int renderStep = degrainRenderConstStep(step);
    const int renderBlocksX = degrainRenderConstBlocksX(blocksX);
    const int renderBlocksY = degrainRenderConstBlocksY(blocksY);
    const int renderCoveredWidth = degrainRenderConstCoveredWidth(coveredWidth, scaleX);
    const int renderCoveredHeight = degrainRenderConstCoveredHeight(coveredHeight, scaleY);
    if (!degrainIsCoveredPixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrainRenderScaleFloor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrainRenderScaleFloor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrainRenderScaleFloor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrainRenderScaleFloor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrainRenderScaleFloor(renderStep, scaleX), 1);
    const int planeStepY = max(degrainRenderScaleFloor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int primaryBaseX = degrainBlockOrigin(primaryBlockX, planeStepX);
    const int primaryBaseY = degrainBlockOrigin(primaryBlockY, planeStepY);
    const int primaryLocalX = x - primaryBaseX;
    const int primaryLocalY = y - primaryBaseY;
    const int primaryBlock = primaryBlockY * renderBlocksX + primaryBlockX;
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && primaryLocalX < planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && primaryLocalY < planeOverlapY;
    const float wxPrev = usePrevBlockX ? windowRamp[primaryLocalX] : 0.0f;
    const float wyPrev = usePrevBlockY ? windowRamp[planeOverlapX + primaryLocalY] : 0.0f;
    const float wx[2] = { 1.0f - wxPrev, wxPrev };
    const float wy[2] = { 1.0f - wyPrev, wyPrev };

    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int localXs[2] = { primaryLocalX, primaryLocalX + planeStepX };
    const int localYs[2] = { primaryLocalY, primaryLocalY + planeStepY };
    const int blockRows[2] = { primaryBlock, primaryBlock - renderBlocksX };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;
    const int directionDisabled = degrainRefDirectionDisabled(disableMask, refDirection);

    float sampleSum = 0.0f;
    float weightSum = 0.0f;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int localY = localYs[byIndex];
        const int blockRow = blockRows[byIndex];
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int localX = localXs[bxIndex];
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY
                || blockX < 0 || blockX >= renderBlocksX || blockY < 0 || blockY >= renderBlocksY) {
                continue;
            }
            const int block = blockRow - bxIndex;
            const int sample = degrainCompensateBlockSample<TypePixel>(
                ref0, ref, cur_pitch,
                width, height,
                mv, sad,
                block, refDirection, thsad, directionDisabled,
                planeScaleX, planeScaleY,
                x, y,
                refs, pel, subpelInterp);
            degrainAccumulateWeightedSampleFp32(&sampleSum, &weightSum, sample, wx[bxIndex] * wy[byIndex]);
        }
    }

    const int result = degrainFinalizeWeightedSampleFp32(sampleSum, weightSum, fallback);
    dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(result);
}

template<typename TypePixel>
__global__ void kernel_degrain_compensate_overlap_plane_ramp_cuda(
    TypePixel *dst,
    const int dst_pitch,
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *ref0,
    const uint8_t *ref,
    const int refDirection,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint32_t thsad,
    const uint32_t disableMask,
    const float *windowRamp,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int globalX = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int globalY = (int)blockIdx.y * blockDim.y + threadIdx.y;
    degrainCompensateOverlapPlaneRampGeneric<TypePixel>(
        dst, dst_pitch,
        cur, cur_pitch,
        ref0, ref, refDirection,
        width, height,
        mv, sad,
        blocksX, blocksY,
        blockSize, overlap, step,
        coveredWidth, coveredHeight,
        planeScaleX, planeScaleY,
        thsad, disableMask,
        windowRamp,
        0, 0, 0,
        globalX, globalY,
        refs, pel, subpelInterp);
}

RGY_ERR launchNVEncDegrainCompensateOverlapPlaneRamp(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *ref0, const uint8_t *ref,
    const int refDirection, const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const uint32_t thsad, const uint32_t disableMask,
    const CUMemBuf &windowRamp,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    if (pixelBytes > 1) {
        kernel_degrain_compensate_overlap_plane_ramp_cuda<uint16_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<uint16_t *>(dst), dstPitch,
            cur, curPitch, ref0, ref, refDirection,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            thsad, disableMask,
            reinterpret_cast<const float *>(windowRamp.ptr),
            refs, pel, subpelInterp);
    } else {
        kernel_degrain_compensate_overlap_plane_ramp_cuda<uint8_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<uint8_t *>(dst), dstPitch,
            cur, curPitch, ref0, ref, refDirection,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            thsad, disableMask,
            reinterpret_cast<const float *>(windowRamp.ptr),
            refs, pel, subpelInterp);
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
__global__ void kernel_degrain_degrain_overlap_plane_cuda(
    TypePixel *dst,
    const int dst_pitch,
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const float *temporalMixPrior,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint32_t thsad,
    const uint32_t disableMask,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int x = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int y = (int)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrainPlaneScaleX(planeScaleX);
    const int scaleY = degrainPlaneScaleY(planeScaleY);
    const int renderBlockSize = degrainRenderConstBlockSize(blockSize);
    const int renderOverlap = degrainRenderConstOverlap(overlap);
    const int renderStep = degrainRenderConstStep(step);
    const int renderBlocksX = degrainRenderConstBlocksX(blocksX);
    const int renderBlocksY = degrainRenderConstBlocksY(blocksY);
    const int renderCoveredWidth = degrainRenderConstCoveredWidth(coveredWidth, scaleX);
    const int renderCoveredHeight = degrainRenderConstCoveredHeight(coveredHeight, scaleY);
    if (!degrainIsCoveredPixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrainRenderScaleFloor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrainRenderScaleFloor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrainRenderScaleFloor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrainRenderScaleFloor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrainRenderScaleFloor(renderStep, scaleX), 1);
    const int planeStepY = max(degrainRenderScaleFloor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && x < degrainBlockOrigin(primaryBlockX, planeStepX) + planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && y < degrainBlockOrigin(primaryBlockY, planeStepY) + planeOverlapY;
    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    RGYDegrainWindowAccum sampleSum = degrainWindowAccumZero();
    RGYDegrainWindowAccum weightSum = degrainWindowAccumZero();
    int sampleCount = 0;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int baseY = degrainBlockOrigin(blockY, planeStepY);
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int baseX = degrainBlockOrigin(blockX, planeStepX);
            const int localX = x - baseX;
            const int localY = y - baseY;
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY) {
                continue;
            }
            const float windowWeight = degrainWindowFactorRect2d(
                x, y,
                baseX, baseY,
                planeBlockSizeX, planeBlockSizeY,
                planeOverlapX, planeOverlapY,
                blockX, blockY,
                renderBlocksX, renderBlocksY);

            const int block = blockY * renderBlocksX + blockX;
            int sample = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
            sample = degrainDegrainBlockSample<TypePixel>(
                cur, cur_pitch,
                refBackward1, refForward1,
                refBackward2, refForward2,
                refBackward3, refForward3,
                refBackward4, refForward4,
                refBackward5, refForward5,
                width, height,
                mv, sad,
                block, thsad,
                disableMask,
                temporalMixPrior,
                planeScaleX, planeScaleY,
                x, y,
                refs, pel, subpelInterp);
            degrainAccumulateWindowedSample(&sampleSum, &weightSum, sample, windowWeight);
            sampleCount++;
        }
    }

    const int result = (sampleCount > 0) ? degrainFinalizeWindowedSample(sampleSum, weightSum, fallback) : fallback;
    dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(result);
}

template<typename TypePixel, int refs>
__device__ __forceinline__ void degrainDegrainOverlapPlanePreweightedRampGeneric(
    TypePixel *dst,
    const int dst_pitch,
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const float *windowRamp,
    const float *temporalMixPlan,
    const int originX,
    const int originY,
    const int compactTopLeftBorder,
    const int globalX,
    const int globalY,
    const int pel,
    const int subpelInterp) {
    int x = originX + globalX;
    int y = originY + globalY;
    if (compactTopLeftBorder) {
        if (originX <= 0 || originY <= 0) {
            return;
        }
        const int compactScaleX = degrainPlaneScaleX(planeScaleX);
        const int compactScaleY = degrainPlaneScaleY(planeScaleY);
        const int compactStep = degrainRenderConstStep(step);
        const int compactBlocksX = degrainRenderConstBlocksX(blocksX);
        const int compactBlocksY = degrainRenderConstBlocksY(blocksY);
        const int compactPlaneStepX = max(degrainRenderScaleFloor(compactStep, compactScaleX), 1);
        const int compactPlaneStepY = max(degrainRenderScaleFloor(compactStep, compactScaleY), 1);
        const int interiorEndX = min(width, compactBlocksX * compactPlaneStepX);
        const int interiorEndY = min(height, compactBlocksY * compactPlaneStepY);
        const int lowerHeight = max(height - originY, 0);
        const int rightBorderWidth = max(width - interiorEndX, 0);
        const int bottomBorderWidth = max(interiorEndX - originX, 0);
        const int borderIndex = globalX;
        const int topBorderPixels = width * originY;
        const int leftBorderPixels = originX * lowerHeight;
        const int rightBorderPixels = rightBorderWidth * lowerHeight;
        if (borderIndex < topBorderPixels) {
            x = borderIndex % width;
            y = borderIndex / width;
        } else if (borderIndex < topBorderPixels + leftBorderPixels) {
            const int leftBorderIndex = borderIndex - topBorderPixels;
            x = leftBorderIndex % originX;
            y = originY + leftBorderIndex / originX;
        } else if (borderIndex < topBorderPixels + leftBorderPixels + rightBorderPixels) {
            const int rightBorderIndex = borderIndex - topBorderPixels - leftBorderPixels;
            x = interiorEndX + rightBorderIndex % rightBorderWidth;
            y = originY + rightBorderIndex / rightBorderWidth;
        } else {
            const int bottomBorderIndex = borderIndex - topBorderPixels - leftBorderPixels - rightBorderPixels;
            if (bottomBorderWidth <= 0) {
                return;
            }
            x = originX + bottomBorderIndex % bottomBorderWidth;
            y = interiorEndY + bottomBorderIndex / bottomBorderWidth;
        }
    }
    if (x >= width || y >= height) {
        return;
    }

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrainPlaneScaleX(planeScaleX);
    const int scaleY = degrainPlaneScaleY(planeScaleY);
    const int renderBlockSize = degrainRenderConstBlockSize(blockSize);
    const int renderOverlap = degrainRenderConstOverlap(overlap);
    const int renderStep = degrainRenderConstStep(step);
    const int renderBlocksX = degrainRenderConstBlocksX(blocksX);
    const int renderBlocksY = degrainRenderConstBlocksY(blocksY);
    const int renderCoveredWidth = degrainRenderConstCoveredWidth(coveredWidth, scaleX);
    const int renderCoveredHeight = degrainRenderConstCoveredHeight(coveredHeight, scaleY);
    if (!degrainIsCoveredPixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrainRenderScaleFloor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrainRenderScaleFloor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrainRenderScaleFloor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrainRenderScaleFloor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrainRenderScaleFloor(renderStep, scaleX), 1);
    const int planeStepY = max(degrainRenderScaleFloor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int primaryBaseX = degrainBlockOrigin(primaryBlockX, planeStepX);
    const int primaryBaseY = degrainBlockOrigin(primaryBlockY, planeStepY);
    const int primaryLocalX = x - primaryBaseX;
    const int primaryLocalY = y - primaryBaseY;
    const int primaryBlock = primaryBlockY * renderBlocksX + primaryBlockX;
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && primaryLocalX < planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && primaryLocalY < planeOverlapY;
    const float wxPrev = usePrevBlockX ? windowRamp[primaryLocalX] : 0.0f;
    const float wyPrev = usePrevBlockY ? windowRamp[planeOverlapX + primaryLocalY] : 0.0f;
    const float wx[2] = { 1.0f - wxPrev, wxPrev };
    const float wy[2] = { 1.0f - wyPrev, wyPrev };

    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int localXs[2] = { primaryLocalX, primaryLocalX + planeStepX };
    const int localYs[2] = { primaryLocalY, primaryLocalY + planeStepY };
    const int blockRows[2] = { primaryBlock, primaryBlock - renderBlocksX };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    float sampleSum = 0.0f;
    float weightSum = 0.0f;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int localY = localYs[byIndex];
        const int blockRow = blockRows[byIndex];
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int localX = localXs[bxIndex];
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY
                || blockX < 0 || blockX >= renderBlocksX || blockY < 0 || blockY >= renderBlocksY) {
                continue;
            }
            const int block = blockRow - bxIndex;
            const int sample = degrainApplyTemporalMixPlanSamePitch<TypePixel, refs>(
                cur, cur_pitch,
                refBackward1, refForward1,
                refBackward2, refForward2,
                refBackward3, refForward3,
                refBackward4, refForward4,
                refBackward5, refForward5,
                width, height,
                mv, block, temporalMixPlan,
                planeScaleX, planeScaleY,
                x, y,
                pel, subpelInterp);
            degrainAccumulateWeightedSampleFp32(&sampleSum, &weightSum, sample, wx[bxIndex] * wy[byIndex]);
        }
    }

    const int result = degrainFinalizeWeightedSampleFp32(sampleSum, weightSum, fallback);
    dst[y * dstPitch + x] = degrainClampPixel<TypePixel>(result);
}

template<typename TypePixel, int refs>
__global__ void kernel_degrain_degrain_overlap_plane_preweighted_ramp_cuda(
    TypePixel *dst,
    const int dst_pitch,
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const float *windowRamp,
    const float *temporalMixPlan,
    const int pel,
    const int subpelInterp) {
    const int globalX = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int globalY = (int)blockIdx.y * blockDim.y + threadIdx.y;
    degrainDegrainOverlapPlanePreweightedRampGeneric<TypePixel, refs>(
        dst, dst_pitch,
        cur, cur_pitch,
        refBackward1, refForward1,
        refBackward2, refForward2,
        refBackward3, refForward3,
        refBackward4, refForward4,
        refBackward5, refForward5,
        width, height,
        mv,
        blocksX, blocksY,
        blockSize, overlap, step,
        coveredWidth, coveredHeight,
        planeScaleX, planeScaleY,
        windowRamp, temporalMixPlan,
        0, 0, 0,
        globalX, globalY,
        pel, subpelInterp);
}

RGY_ERR launchNVEncDegrainDegrainOverlapPlane(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const uint32_t thsad, const uint32_t disableMask,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    if (pixelBytes > 1) {
        kernel_degrain_degrain_overlap_plane_cuda<uint16_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<uint16_t *>(dst), dstPitch,
            cur, curPitch,
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            reinterpret_cast<const float *>(temporalMixPrior.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            thsad, disableMask, refs, pel, subpelInterp);
    } else {
        kernel_degrain_degrain_overlap_plane_cuda<uint8_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<uint8_t *>(dst), dstPitch,
            cur, curPitch,
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            reinterpret_cast<const float *>(temporalMixPrior.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            thsad, disableMask, refs, pel, subpelInterp);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainDegrainOverlapPlanePreweightedRamp(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const CUMemBuf &windowRamp, const CUMemBuf &temporalMixPlan,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
#define LAUNCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE, REFS) do { \
    kernel_degrain_degrain_overlap_plane_preweighted_ramp_cuda<TYPE, REFS><<<grid, block, 0, stream>>>( \
        reinterpret_cast<TYPE *>(dst), dstPitch, \
        cur, curPitch, \
        refBackward1, refForward1, \
        refBackward2, refForward2, \
        refBackward3, refForward3, \
        refBackward4, refForward4, \
        refBackward5, refForward5, \
        width, height, \
        reinterpret_cast<const RGYDegrainMV *>(mv.ptr), \
        layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step, \
        coveredWidth, coveredHeight, planeScaleX, planeScaleY, \
        reinterpret_cast<const float *>(windowRamp.ptr), \
        reinterpret_cast<const float *>(temporalMixPlan.ptr), \
        pel, subpelInterp); \
} while (0)
#define SWITCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE) \
    switch (refs) { \
    case 2: LAUNCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE, 2); break; \
    case 4: LAUNCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE, 4); break; \
    case 6: LAUNCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE, 6); break; \
    case 8: LAUNCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE, 8); break; \
    case 10: LAUNCH_DEGRAIN_PREWEIGHTED_RAMP(TYPE, 10); break; \
    default: return RGY_ERR_UNSUPPORTED; \
    }
    if (pixelBytes > 1) {
        SWITCH_DEGRAIN_PREWEIGHTED_RAMP(uint16_t);
    } else {
        SWITCH_DEGRAIN_PREWEIGHTED_RAMP(uint8_t);
    }
#undef SWITCH_DEGRAIN_PREWEIGHTED_RAMP
#undef LAUNCH_DEGRAIN_PREWEIGHTED_RAMP
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
__global__ void kernel_degrain_pixel_trace_cuda(
    const uint8_t *cur,
    const int cur_pitch,
    const uint8_t *refBackward1,
    const uint8_t *refForward1,
    const uint8_t *refBackward2,
    const uint8_t *refForward2,
    const uint8_t *refBackward3,
    const uint8_t *refForward3,
    const uint8_t *refBackward4,
    const uint8_t *refForward4,
    const uint8_t *refBackward5,
    const uint8_t *refForward5,
    const int width,
    const int height,
    const RGYDegrainMV *mv,
    const RGYDegrainSAD *sad,
    const float *temporalMixPrior,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint32_t thsad,
    const uint32_t disableMask,
    const int targetX,
    const int targetY,
    int *trace,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int x = degrainClampInt(targetX, 0, max(width - 1, 0));
    const int y = degrainClampInt(targetY, 0, max(height - 1, 0));
    const int fallback = degrainPixelLoad<TypePixel>(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrainPlaneScaleX(planeScaleX);
    const int scaleY = degrainPlaneScaleY(planeScaleY);
    const int renderBlockSize = degrainRenderConstBlockSize(blockSize);
    const int renderOverlap = degrainRenderConstOverlap(overlap);
    const int renderStep = degrainRenderConstStep(step);
    const int renderBlocksX = degrainRenderConstBlocksX(blocksX);
    const int renderBlocksY = degrainRenderConstBlocksY(blocksY);
    const int renderCoveredWidth = degrainRenderConstCoveredWidth(coveredWidth, scaleX);
    const int renderCoveredHeight = degrainRenderConstCoveredHeight(coveredHeight, scaleY);
    const int covered = degrainIsCoveredPixel(x, y, renderCoveredWidth, renderCoveredHeight);

    const int planeBlockSizeX = max(degrainRenderScaleFloor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrainRenderScaleFloor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrainRenderScaleFloor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrainRenderScaleFloor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrainRenderScaleFloor(renderStep, scaleX), 1);
    const int planeStepY = max(degrainRenderScaleFloor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && x < degrainBlockOrigin(primaryBlockX, planeStepX) + planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && y < degrainBlockOrigin(primaryBlockY, planeStepY) + planeOverlapY;
    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    RGYDegrainWindowAccum sampleSum = degrainWindowAccumZero();
    RGYDegrainWindowAccum weightSum = degrainWindowAccumZero();
    int sampleCount = 0;
    int record = 0;
    for (int i = 0; i < 256; i++) {
        trace[i] = 0;
    }

    if (covered) {
        for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
            const int blockY = blockYs[byIndex];
            const int baseY = degrainBlockOrigin(blockY, planeStepY);
            for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
                const int blockX = blockXs[bxIndex];
                const int baseX = degrainBlockOrigin(blockX, planeStepX);
                const int localX = x - baseX;
                const int localY = y - baseY;
                if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY) {
                    continue;
                }
                const float windowWeight = degrainWindowFactorRect2d(
                    x, y,
                    baseX, baseY,
                    planeBlockSizeX, planeBlockSizeY,
                    planeOverlapX, planeOverlapY,
                    blockX, blockY,
                    renderBlocksX, renderBlocksY);

                const int block = blockY * renderBlocksX + blockX;
                const int srcSample = fallback;
                const float sourceConfidenceRaw = degrainTemporalMixPriorCenter(temporalMixPrior);
                float referenceConfidenceRaw[RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS];
                int referenceSample[RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS];
                int referenceValid[RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS];
                float confidenceTotal = sourceConfidenceRaw;
                for (int referenceDirection = 0; referenceDirection < refs; referenceDirection++) {
                    const float temporalMixPriorRef = degrainTemporalMixPriorRef(temporalMixPrior, referenceDirection);
                    referenceValid[referenceDirection] = degrainReferenceIsValid(mv, sad, block, referenceDirection, thsad, degrainRefDirectionDisabled(disableMask, referenceDirection), refs);
                    referenceConfidenceRaw[referenceDirection] = degrainReferenceMixAffinity(mv, sad, block, referenceDirection, thsad, degrainRefDirectionDisabled(disableMask, referenceDirection), refs) * temporalMixPriorRef;
                    confidenceTotal += referenceConfidenceRaw[referenceDirection];
                    referenceSample[referenceDirection] = 0;
                }
                const float invTotal = (confidenceTotal > 0.0f) ? (1.0f / confidenceTotal) : 0.0f;
                float mixedValue = (float)srcSample * (sourceConfidenceRaw * invTotal);
                float referenceMixTotal = 0.0f;
                float referenceMixNorm[RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS];
                for (int referenceDirection = 0; referenceDirection < refs; referenceDirection++) {
                    referenceMixNorm[referenceDirection] = (referenceConfidenceRaw[referenceDirection] > 0.0f) ? (referenceConfidenceRaw[referenceDirection] * invTotal) : 0.0f;
                    referenceMixTotal += referenceMixNorm[referenceDirection];
                    if (referenceMixNorm[referenceDirection] <= 0.0f) {
                        continue;
                    }
                    const uint8_t *referencePlane = degrainRefPlanePtrSamePitch(
                        refBackward1, refForward1,
                        refBackward2, refForward2,
                        refBackward3, refForward3,
                        refBackward4, refForward4,
                        refBackward5, refForward5,
                        referenceDirection);
                    referenceSample[referenceDirection] = degrainCompensatedSample<TypePixel>(referencePlane, cur_pitch, width, height, mv, block, referenceDirection, planeScaleX, planeScaleY, x, y, refs, pel, subpelInterp);
                    mixedValue = __fmaf_rn((float)referenceSample[referenceDirection], referenceMixNorm[referenceDirection], mixedValue);
                }
                const float sourceMixNorm = sourceConfidenceRaw * invTotal;
                const int sample = degrainClampPixel<TypePixel>(degrainRoundFloatToInt(mixedValue));
                const int contribution = degrainWindowedSampleContribution(sample, windowWeight);
                degrainAccumulateWindowedSample(&sampleSum, &weightSum, sample, windowWeight);
                sampleCount++;

                if (record < 4) {
                    const int out = 32 + record * 48;
                    trace[out + 0] = blockX;
                    trace[out + 1] = blockY;
                    trace[out + 2] = block;
                    trace[out + 3] = baseX;
                    trace[out + 4] = baseY;
                    trace[out + 5] = x - baseX;
                    trace[out + 6] = y - baseY;
                    trace[out + 7] = degrainTraceFloatToQ8(windowWeight);
                    trace[out + 8] = srcSample;
                    trace[out + 9] = sample;
                    trace[out + 10] = contribution;
                    trace[out + 11] = degrainTraceFloatToQ8(sourceMixNorm);
                    trace[out + 12] = degrainTraceFloatToQ8(referenceMixTotal);
                    trace[out + 13] = degrainTraceFloatToQ8(confidenceTotal);
                    trace[out + 14] = degrainTraceFloatToQ8(degrainTemporalMixPriorCenter(temporalMixPrior));
                    for (int refDirection = 0; refDirection < min(refs, 4); refDirection++) {
                        const int traceOffset = out + 15 + refDirection * 6;
                        const int motionIndex = degrainRefIndex(block, refDirection, refs);
                        trace[traceOffset + 0] = degrainTraceFloatToQ8(referenceMixNorm[refDirection]);
                        trace[traceOffset + 1] = referenceSample[refDirection];
                        trace[traceOffset + 2] = (int)mv[motionIndex].dx;
                        trace[traceOffset + 3] = (int)mv[motionIndex].dy;
                        trace[traceOffset + 4] = (int)sad[motionIndex].sad;
                        trace[traceOffset + 5] = referenceValid[refDirection];
                    }
                    record++;
                }
            }
        }
    }

    const int result = (covered && sampleCount > 0) ? degrainFinalizeWindowedSample(sampleSum, weightSum, fallback) : fallback;
    trace[0] = 0x4d435054;
    trace[1] = x;
    trace[2] = y;
    trace[3] = width;
    trace[4] = height;
    trace[5] = fallback;
    trace[6] = covered;
    trace[7] = scaleX;
    trace[8] = scaleY;
    trace[9] = planeBlockSizeX;
    trace[10] = planeBlockSizeY;
    trace[11] = planeOverlapX;
    trace[12] = planeOverlapY;
    trace[13] = planeStepX;
    trace[14] = primaryBlockX;
    trace[15] = primaryBlockY;
    trace[16] = blockCountX;
    trace[17] = blockCountY;
    trace[18] = degrainTraceWindowAccum(sampleSum);
    trace[19] = sampleCount;
    trace[20] = result;
    trace[21] = (int)thsad;
    trace[22] = (int)disableMask;
    trace[23] = renderBlocksX;
    trace[24] = renderBlocksY;
    trace[25] = record;
}

RGY_ERR launchNVEncDegrainPixelTrace(
    const uint8_t *cur, const int curPitch, const int pixelBytes,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const uint32_t thsad, const uint32_t disableMask,
    const int targetX, const int targetY,
    CUMemBuf &trace,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    if (pixelBytes > 1) {
        kernel_degrain_pixel_trace_cuda<uint16_t><<<1, 1, 0, stream>>>(
            cur, curPitch,
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            reinterpret_cast<const float *>(temporalMixPrior.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            thsad, disableMask, targetX, targetY,
            reinterpret_cast<int *>(trace.ptr),
            refs, pel, subpelInterp);
    } else {
        kernel_degrain_pixel_trace_cuda<uint8_t><<<1, 1, 0, stream>>>(
            cur, curPitch,
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            width, height,
            reinterpret_cast<const RGYDegrainMV *>(mv.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            reinterpret_cast<const float *>(temporalMixPrior.ptr),
            layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
            coveredWidth, coveredHeight, planeScaleX, planeScaleY,
            thsad, disableMask, targetX, targetY,
            reinterpret_cast<int *>(trace.ptr),
            refs, pel, subpelInterp);
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
__device__ __forceinline__ int degrainCenteredSignedValue(const int value, const int search, const int pel) {
    const int searchRange = max(search * pel, 1);
    const int clampedValue = degrainClampInt(value, -searchRange, searchRange);
    const int center = (degrainPixelMax<TypePixel>() + 1) >> 1;
    const int range = max(center - 1, 1);
    return degrainClampInt(center + (clampedValue * range) / searchRange, 0, degrainPixelMax<TypePixel>());
}

template<typename TypePixel>
__global__ void kernel_degrain_debug_mv_cuda(
    uint8_t *dst, const int dstPitch, const int width, const int height,
    const RGYDegrainMV *mv, const RGYDegrainSAD *sad,
    const int blocksX, const int blocksY, const int blockSize, const int overlap, const int step,
    const int coveredWidth, const int coveredHeight, const int refs, const int search, const int pel) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height || !degrainIsCoveredPixel(x, y, coveredWidth, coveredHeight)) {
        return;
    }

    const int block = degrainPrimaryBlockIndex(x, y, blocksX, blocksY, step);
    const int blockX = block % blocksX;
    const int blockY = block / blocksX;
    const int localX = degrainClampInt(x - degrainBlockOrigin(blockX, step), 0, blockSize - 1);
    const int localY = degrainClampInt(y - degrainBlockOrigin(blockY, step), 0, blockSize - 1);
    int refDirection = 0;
    int showDy = 0;
    if (refs <= 2) {
        refDirection = ((localY * 2) >= blockSize) ? min(1, refs - 1) : 0;
        showDy = (localX * 2) >= blockSize;
    } else {
        const int halfX = max(blockSize / 2, 1);
        const int halfY = max(blockSize / 2, 1);
        const int quadrantX = (localX >= halfX);
        const int quadrantY = (localY >= halfY);
        const int quadrantWidth = max(quadrantX ? (blockSize - halfX) : halfX, 1);
        const int localQuadrantX = quadrantX ? (localX - halfX) : localX;
        refDirection = degrainClampInt(quadrantY * 2 + quadrantX, 0, refs - 1);
        showDy = (localQuadrantX * 2) >= quadrantWidth;
    }
    (void)sad;
    (void)overlap;
    const RGYDegrainMV motion = mv[degrainRefIndex(block, refDirection, refs)];
    const int signedComponent = showDy ? (int)motion.dy : (int)motion.dx;
    const int value = degrainDebugBorder(x, y, step)
        ? degrainPixelMax<TypePixel>()
        : degrainCenteredSignedValue<TypePixel>(signedComponent, search, pel);
    *(TypePixel *)(dst + y * dstPitch + x * (int)sizeof(TypePixel)) = degrainClampPixel<TypePixel>(value);
}

template<typename TypePixel>
__global__ void kernel_degrain_debug_sad_cuda(
    uint8_t *dst, const int dstPitch, const int width, const int height,
    const RGYDegrainMV *mv, const RGYDegrainSAD *sad,
    const int blocksX, const int blocksY, const int blockSize, const int overlap, const int step,
    const int coveredWidth, const int coveredHeight, const int refs, const int search, const int pel) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height || !degrainIsCoveredPixel(x, y, coveredWidth, coveredHeight)) {
        return;
    }

    const int block = degrainPrimaryBlockIndex(x, y, blocksX, blocksY, step);
    const int blockX = block % blocksX;
    const int blockY = block / blocksX;
    const int localX = degrainClampInt(x - degrainBlockOrigin(blockX, step), 0, blockSize - 1);
    const int localY = degrainClampInt(y - degrainBlockOrigin(blockY, step), 0, blockSize - 1);
    int refDirection = 0;
    if (refs <= 2) {
        refDirection = ((localY * 2) >= blockSize) ? min(1, refs - 1) : 0;
    } else {
        const int halfX = max(blockSize / 2, 1);
        const int halfY = max(blockSize / 2, 1);
        refDirection = degrainClampInt((localY >= halfY) * 2 + (localX >= halfX), 0, refs - 1);
    }
    (void)overlap;
    (void)search;
    (void)pel;
    const int sadIndex = degrainRefIndex(block, refDirection, refs);
    const uint32_t sadMix = sad[sadIndex].sad + mv[sadIndex].sad;
    const int value = degrainDebugBorder(x, y, step)
        ? degrainPixelMax<TypePixel>()
        : min(degrainPixelMax<TypePixel>(), (int)(sadMix >> 4));
    *(TypePixel *)(dst + y * dstPitch + x * (int)sizeof(TypePixel)) = degrainClampPixel<TypePixel>(value);
}

RGY_ERR launchNVEncDegrainDebug(
    const RGYFrameInfo &dst, const VppDegrainMode mode, const CUMemBuf &mv, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout, const int pel, cudaStream_t stream) {
    const auto block = dim3(DEGRAIN_BLOCK_X, DEGRAIN_BLOCK_Y);
    const auto grid = dim3(divCeil(dst.width, DEGRAIN_BLOCK_X), divCeil(dst.height, DEGRAIN_BLOCK_Y));
    const auto *mvPtr = reinterpret_cast<const RGYDegrainMV *>(mv.ptr);
    const auto *sadPtr = reinterpret_cast<const RGYDegrainSAD *>(sad.ptr);
    if (RGY_CSP_BIT_DEPTH[dst.csp] > 8) {
        if (mode == VppDegrainMode::MV) {
            kernel_degrain_debug_mv_cuda<uint16_t><<<grid, block, 0, stream>>>(
                dst.ptr[0], dst.pitch[0], dst.width, dst.height, mvPtr, sadPtr,
                layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
                layout.coveredWidth, layout.coveredHeight, layout.temporalDirections, layout.search, pel);
        } else {
            kernel_degrain_debug_sad_cuda<uint16_t><<<grid, block, 0, stream>>>(
                dst.ptr[0], dst.pitch[0], dst.width, dst.height, mvPtr, sadPtr,
                layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
                layout.coveredWidth, layout.coveredHeight, layout.temporalDirections, layout.search, pel);
        }
    } else {
        if (mode == VppDegrainMode::MV) {
            kernel_degrain_debug_mv_cuda<uint8_t><<<grid, block, 0, stream>>>(
                dst.ptr[0], dst.pitch[0], dst.width, dst.height, mvPtr, sadPtr,
                layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
                layout.coveredWidth, layout.coveredHeight, layout.temporalDirections, layout.search, pel);
        } else {
            kernel_degrain_debug_sad_cuda<uint8_t><<<grid, block, 0, stream>>>(
                dst.ptr[0], dst.pitch[0], dst.width, dst.height, mvPtr, sadPtr,
                layout.blocksX, layout.blocksY, layout.blockSize, layout.overlap, layout.step,
                layout.coveredWidth, layout.coveredHeight, layout.temporalDirections, layout.search, pel);
        }
    }
    return err_to_rgy(cudaGetLastError());
}

__global__ void kernel_degrain_mv_seed_anchor_vectors_cuda(
    RGYDegrainMotionSearchVector *vectors,
    const int2 *frameAverageMV,
    const int planeBase,
    const int planeStride,
    const int planeCount,
    const int pel) {
    const int plane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (plane >= planeCount) {
        return;
    }
    const int base = planeBase + plane * planeStride;
    vectors[degrainMotionSearchVecZeroIndex(base)] = degrainMotionSearchMakeVector(0, 0, 0u, 0u);
    const int2 frameAverageVec = frameAverageMV ? frameAverageMV[plane] : make_int2(0, 0);
    vectors[degrainMotionSearchVecGlobalIndex(base)] = degrainMotionSearchMakeVector(
        frameAverageVec.x * pel,
        frameAverageVec.y * pel,
        0u,
        0u);
}

__global__ void kernel_degrain_mv_seed_zero_vectors_cuda(
    RGYDegrainMotionSearchVector *vectors,
    RGYDegrainMotionSearchVector *vectorsPrev,
    uint32_t *sads,
    const int planeBase,
    const int sadBase,
    const int blockCount) {
    const int block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= blockCount) {
        return;
    }
    const auto zero = vectors[degrainMotionSearchVecZeroIndex(planeBase)];
    vectors[degrainMotionSearchVecCurrentIndex(planeBase, blockCount, block)] = zero;
    vectorsPrev[degrainMotionSearchVecPrevIndex(planeBase, blockCount, block)] = zero;
    sads[sadBase + block] = zero.sad_metric;
}

__global__ void kernel_degrain_mv_expand_coarse_vectors_cuda(
    const RGYDegrainMotionSearchVector *srcVectorsFinal,
    RGYDegrainMotionSearchVector *dstVectors,
    RGYDegrainMotionSearchVector *dstVectorsPrev,
    uint32_t *dstSads,
    const int srcFinalBase,
    const int dstPlaneBase,
    const int dstSadBase,
    const int srcBlockCount,
    const int dstBlockCount,
    const int srcBlocksX,
    const int srcBlocksY,
    const int dstBlocksX) {
    const int block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= dstBlockCount) {
        return;
    }
    const int dstX = block % dstBlocksX;
    const int dstY = block / dstBlocksX;
    const int srcX = min(dstX >> 1, srcBlocksX - 1);
    const int srcY = min(dstY >> 1, srcBlocksY - 1);
    const int srcBlock = srcY * srcBlocksX + srcX;
    auto vec = srcVectorsFinal[degrainMotionSearchVecFinalIndex(srcFinalBase, srcBlockCount, srcBlock)];
    vec.pos_x <<= 1;
    vec.pos_y <<= 1;
    dstVectors[degrainMotionSearchVecCurrentIndex(dstPlaneBase, dstBlockCount, block)] = vec;
    dstVectorsPrev[degrainMotionSearchVecPrevIndex(dstPlaneBase, dstBlockCount, block)] = vec;
    dstSads[dstSadBase + block] = vec.sad_metric;
}

__global__ void kernel_degrain_mv_export_sad_cuda(
    RGYDegrainMotionSearchVector *vectorsFinal,
    uint32_t *sadsInternal,
    RGYDegrainMV *outputMotion,
    RGYDegrainSAD *outputSad,
    const int finalBase,
    const int sadBase,
    const int blockCount,
    const int outOffset,
    const int referenceDirection,
    const int refs) {
    const int block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= blockCount) {
        return;
    }

    auto finalVector = vectorsFinal[degrainMotionSearchVecFinalIndex(finalBase, blockCount, block)];
    const uint32_t finalSad = finalVector.sad_metric;
    finalVector.sad_metric = finalSad;
    finalVector.score_primary = finalSad;
    vectorsFinal[degrainMotionSearchVecFinalIndex(finalBase, blockCount, block)] = finalVector;
    sadsInternal[sadBase + block] = finalSad;

    const int outputIndex = outOffset + degrainRefIndex(block, referenceDirection, refs);
    if (outputMotion) {
        RGYDegrainMV exportedMotion;
        exportedMotion.dx = finalVector.pos_x;
        exportedMotion.dy = finalVector.pos_y;
        exportedMotion.sad = (uint16_t)min(finalSad, 65535u);
        exportedMotion.refdir = (uint16_t)referenceDirection;
        exportedMotion.flags = 0u;
        exportedMotion.reserved = finalSad;
        outputMotion[outputIndex] = exportedMotion;
    }
    if (outputSad) {
        RGYDegrainSAD exportedSad;
        exportedSad.sad = finalSad;
        exportedSad.srcAvg = 0u;
        exportedSad.refAvg = 0u;
        exportedSad.reserved = finalSad;
        outputSad[outputIndex] = exportedSad;
    }
}

RGY_ERR launchNVEncDegrainMotionSearchSeedAnchorVectors(
    CUMemBuf &vectors, const CUMemBuf &frameAverageMV, const int planeBase, const int planeStride,
    const int planeCount, const int pel, cudaStream_t stream) {
    const int block = 64;
    const int grid = divCeil(planeCount, block);
    kernel_degrain_mv_seed_anchor_vectors_cuda<<<grid, block, 0, stream>>>(
        reinterpret_cast<RGYDegrainMotionSearchVector *>(vectors.ptr),
        reinterpret_cast<const int2 *>(frameAverageMV.ptr),
        planeBase, planeStride, planeCount, pel);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainMotionSearchSeedZeroVectors(
    CUMemBuf &vectors, CUMemBuf &vectorsPrev, CUMemBuf &sads, const int planeBase,
    const int sadBase, const int blockCount, cudaStream_t stream) {
    const int block = 256;
    const int grid = divCeil(blockCount, block);
    kernel_degrain_mv_seed_zero_vectors_cuda<<<grid, block, 0, stream>>>(
        reinterpret_cast<RGYDegrainMotionSearchVector *>(vectors.ptr),
        reinterpret_cast<RGYDegrainMotionSearchVector *>(vectorsPrev.ptr),
        reinterpret_cast<uint32_t *>(sads.ptr),
        planeBase, sadBase, blockCount);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainMotionSearchExpandCoarseVectors(
    const CUMemBuf &srcVectorsFinal, CUMemBuf &dstVectors, CUMemBuf &dstVectorsPrev, CUMemBuf &dstSads,
    const int srcFinalBase, const int dstPlaneBase, const int dstSadBase, const int srcBlockCount,
    const int dstBlockCount, const int srcBlocksX, const int srcBlocksY, const int dstBlocksX, cudaStream_t stream) {
    const int block = 256;
    const int grid = divCeil(dstBlockCount, block);
    kernel_degrain_mv_expand_coarse_vectors_cuda<<<grid, block, 0, stream>>>(
        reinterpret_cast<const RGYDegrainMotionSearchVector *>(srcVectorsFinal.ptr),
        reinterpret_cast<RGYDegrainMotionSearchVector *>(dstVectors.ptr),
        reinterpret_cast<RGYDegrainMotionSearchVector *>(dstVectorsPrev.ptr),
        reinterpret_cast<uint32_t *>(dstSads.ptr),
        srcFinalBase, dstPlaneBase, dstSadBase, srcBlockCount, dstBlockCount, srcBlocksX, srcBlocksY, dstBlocksX);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainMotionSearchExportSad(
    CUMemBuf &vectorsFinal, CUMemBuf &sadsInternal, CUMemBuf *outputMotion, CUMemBuf *outputSad,
    const int finalBase, const int sadBase, const int blockCount, const int outOffset,
    const int referenceDirection, const int refs, cudaStream_t stream) {
    const int block = 256;
    const int grid = divCeil(blockCount, block);
    kernel_degrain_mv_export_sad_cuda<<<grid, block, 0, stream>>>(
        reinterpret_cast<RGYDegrainMotionSearchVector *>(vectorsFinal.ptr),
        reinterpret_cast<uint32_t *>(sadsInternal.ptr),
        outputMotion ? reinterpret_cast<RGYDegrainMV *>(outputMotion->ptr) : nullptr,
        outputSad ? reinterpret_cast<RGYDegrainSAD *>(outputSad->ptr) : nullptr,
        finalBase, sadBase, blockCount, outOffset, referenceDirection, refs);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainMotionSearchSearchParallel(
    const uint8_t *sourcePlane, const uint8_t *referencePlane, CUMemBuf &vectors,
    const int pitch, const int width, const int height, const int planeBase, const int blockCount,
    const RGYDegrainBlockLayout &layout, const int pixelBytes, const int pel, const int subpelInterp,
    const int pad, const int motionCostScale, const int lowSadWeightScale,
    const int zeroCandidateCostScale, const int frameAverageCandidateCostScale,
    const int newCandidateCostScale, const int level, cudaStream_t stream) {
    if (layout.blockSize <= 0 || layout.blockSize > DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int block = layout.blockSize * DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS;
    const int grid = blockCount;
    if (pixelBytes > 1) {
        kernel_degrain_mv_search_parallel_cuda<uint16_t><<<grid, block, 0, stream>>>(
            sourcePlane, referencePlane, reinterpret_cast<RGYDegrainMotionSearchVector *>(vectors.ptr),
            pitch, width, height, planeBase, blockCount, layout.blocksX, layout.blocksY, layout.step,
            layout.blockSize, pel, subpelInterp, pad, motionCostScale, lowSadWeightScale,
            zeroCandidateCostScale, frameAverageCandidateCostScale, newCandidateCostScale, level);
    } else {
        kernel_degrain_mv_search_parallel_cuda<uint8_t><<<grid, block, 0, stream>>>(
            sourcePlane, referencePlane, reinterpret_cast<RGYDegrainMotionSearchVector *>(vectors.ptr),
            pitch, width, height, planeBase, blockCount, layout.blocksX, layout.blocksY, layout.step,
            layout.blockSize, pel, subpelInterp, pad, motionCostScale, lowSadWeightScale,
            zeroCandidateCostScale, frameAverageCandidateCostScale, newCandidateCostScale, level);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainMotionSearchSpatialRefine(
    const uint8_t *sourcePlane, const uint8_t *referencePlane,
    CUMemBuf &vectors, const CUMemBuf &vectorsPrev, CUMemBuf &vectorsFinal,
    const int pitch, const int width, const int height, const int planeBase, const int finalBase,
    const int blockCount, const RGYDegrainBlockLayout &layout, const int pixelBytes,
    const int pel, const int subpelInterp, const int pad, const int motionCostScale,
    const int lowSadWeightScale, const int newCandidateCostScale, cudaStream_t stream) {
    if (layout.blockSize <= 0 || layout.blockSize > DEGRAIN_MOTION_SEARCH_MAX_BLOCK_SIZE) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int block = layout.blockSize * DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS;
    const int grid = blockCount;
    if (pixelBytes > 1) {
        kernel_degrain_mv_spatial_refine_cuda<uint16_t><<<grid, block, 0, stream>>>(
            sourcePlane, referencePlane,
            reinterpret_cast<RGYDegrainMotionSearchVector *>(vectors.ptr),
            reinterpret_cast<const RGYDegrainMotionSearchVector *>(vectorsPrev.ptr),
            reinterpret_cast<RGYDegrainMotionSearchVector *>(vectorsFinal.ptr),
            pitch, width, height, planeBase, finalBase, blockCount, layout.blocksX, layout.blocksY, layout.step,
            layout.blockSize, pel, subpelInterp, pad, motionCostScale, lowSadWeightScale, newCandidateCostScale);
    } else {
        kernel_degrain_mv_spatial_refine_cuda<uint8_t><<<grid, block, 0, stream>>>(
            sourcePlane, referencePlane,
            reinterpret_cast<RGYDegrainMotionSearchVector *>(vectors.ptr),
            reinterpret_cast<const RGYDegrainMotionSearchVector *>(vectorsPrev.ptr),
            reinterpret_cast<RGYDegrainMotionSearchVector *>(vectorsFinal.ptr),
            pitch, width, height, planeBase, finalBase, blockCount, layout.blocksX, layout.blocksY, layout.step,
            layout.blockSize, pel, subpelInterp, pad, motionCostScale, lowSadWeightScale, newCandidateCostScale);
    }
    return err_to_rgy(cudaGetLastError());
}
