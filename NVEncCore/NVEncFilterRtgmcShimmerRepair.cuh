#pragma once

#include "NVEncFilterRtgmcShimmerRepair.h"
#include "rgy_cuda_util_kernel.h"

#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

template<typename Type>
__device__ int rtgmc_read_pix(
    const uint8_t *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const Type *)(src + y * pitch + x * sizeof(Type)));
}

template<typename Type>
__device__ void rtgmc_write_pix(
    uint8_t *dst, int x, int y, const int pitch, const int value
) {
    Type *dstPix = (Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, (int)((sizeof(Type) == 2) ? 0xffff : 0xff));
}

template<typename Type>
__device__ int rtgmcShimmerRepairSignedToDiff(const int signedValue, const int rangeHalf, const int maxVal) {
    return clamp(signedValue + rangeHalf, 0, maxVal);
}

template<typename Type>
__device__ int rtgmcRepairDeltaCentered(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    return clamp(
        rtgmc_read_pix<Type>(reference, x, y, referencePitch, width, height)
            - rtgmc_read_pix<Type>(input, x, y, inputPitch, width, height)
            + rangeHalf,
        0,
        maxVal);
}

template<typename Type>
__device__ int rtgmcRepairVerticalWindow(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal,
    const int useMax
) {
    int value = rtgmcRepairDeltaCentered<Type>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    for (int dy = -2; dy <= 2; dy++) {
        const int sample = rtgmcRepairDeltaCentered<Type>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal);
        value = useMax ? max(value, sample) : min(value, sample);
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosVerticalContract(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairVerticalWindow<Type>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal, 0);
    if constexpr (THIN_LEVEL > 5) {
        for (int dy = -1; dy <= 1; dy++) {
            value = min(value, rtgmcRepairVerticalWindow<Type>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal, 0));
        }
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegVerticalExpand(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairVerticalWindow<Type>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal, 1);
    if constexpr (THIN_LEVEL > 5) {
        for (int dy = -1; dy <= 1; dy++) {
            value = max(value, rtgmcRepairVerticalWindow<Type>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal, 1));
        }
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosLocalContract(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairPosVerticalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr ((THIN_LEVEL % 3) == 0) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmcRepairPosVerticalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal);
        }
    }
    return min(center, (sum + 4) / 9);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegLocalExpand(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairNegVerticalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr ((THIN_LEVEL % 3) == 0) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmcRepairNegVerticalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal);
        }
    }
    return max(center, (sum + 4) / 9);
}

static __device__ __forceinline__ void rtgmcRepairSort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static __device__ __forceinline__ void rtgmcRepairSort2Desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

static __device__ __forceinline__ void rtgmcRepairSort8(int *v) {
    rtgmcRepairSort2    (&v[0], &v[1]); rtgmcRepairSort2Desc(&v[2], &v[3]); rtgmcRepairSort2    (&v[4], &v[5]); rtgmcRepairSort2Desc(&v[6], &v[7]);
    rtgmcRepairSort2    (&v[0], &v[2]); rtgmcRepairSort2    (&v[1], &v[3]); rtgmcRepairSort2Desc(&v[4], &v[6]); rtgmcRepairSort2Desc(&v[5], &v[7]);
    rtgmcRepairSort2    (&v[0], &v[1]); rtgmcRepairSort2    (&v[2], &v[3]); rtgmcRepairSort2Desc(&v[4], &v[5]); rtgmcRepairSort2Desc(&v[6], &v[7]);
    rtgmcRepairSort2    (&v[0], &v[4]); rtgmcRepairSort2    (&v[1], &v[5]); rtgmcRepairSort2    (&v[2], &v[6]); rtgmcRepairSort2    (&v[3], &v[7]);
    rtgmcRepairSort2    (&v[0], &v[2]); rtgmcRepairSort2    (&v[1], &v[3]); rtgmcRepairSort2    (&v[4], &v[6]); rtgmcRepairSort2    (&v[5], &v[7]);
    rtgmcRepairSort2    (&v[0], &v[1]); rtgmcRepairSort2    (&v[2], &v[3]); rtgmcRepairSort2    (&v[4], &v[5]); rtgmcRepairSort2    (&v[6], &v[7]);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosRankLimit(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr (THIN_LEVEL != 2 && THIN_LEVEL != 5) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int v[8] = {
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x - 1, y - 1, width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x    , y - 1, width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + 1, y - 1, width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x - 1, y    , width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + 1, y    , width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x - 1, y + 1, width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x    , y + 1, width, height, rangeHalf, maxVal),
        rtgmcRepairPosLocalContract<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + 1, y + 1, width, height, rangeHalf, maxVal)
    };
    rtgmcRepairSort8(v);
    return clamp(center, v[3], v[4]);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegRankLimit(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr (THIN_LEVEL != 2 && THIN_LEVEL != 5) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int v[8] = {
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x - 1, y - 1, width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x    , y - 1, width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + 1, y - 1, width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x - 1, y    , width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + 1, y    , width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x - 1, y + 1, width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x    , y + 1, width, height, rangeHalf, maxVal),
        rtgmcRepairNegLocalExpand<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + 1, y + 1, width, height, rangeHalf, maxVal)
    };
    rtgmcRepairSort8(v);
    return clamp(center, v[3], v[4]);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosVerticalRestore(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairPosRankLimit<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    for (int dy = -2; dy <= 2; dy++) {
        value = max(value, rtgmcRepairPosRankLimit<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal));
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegVerticalRestore(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairNegRankLimit<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    for (int dy = -2; dy <= 2; dy++) {
        value = min(value, rtgmcRepairNegRankLimit<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal));
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosRestoreWide(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairPosVerticalRestore<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr (THIN_LEVEL > 4) {
        for (int dy = -1; dy <= 1; dy++) {
            value = max(value, rtgmcRepairPosVerticalRestore<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal));
        }
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegRestoreWide(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairNegVerticalRestore<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr (THIN_LEVEL > 4) {
        for (int dy = -1; dy <= 1; dy++) {
            value = min(value, rtgmcRepairNegVerticalRestore<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y + dy, width, height, rangeHalf, maxVal));
        }
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosRestoreSoftOnce(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairPosRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmcRepairPosRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal);
        }
    }
    return max(center, (sum + 4) / 9);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegRestoreSoftOnce(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairNegRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmcRepairNegRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal);
        }
    }
    return min(center, (sum + 4) / 9);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosRestoreSoftTwice(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairPosRestoreSoftOnce<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmcRepairPosRestoreSoftOnce<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal);
        }
    }
    return max(center, (sum + 4) / 9);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegRestoreSoftTwice(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int center = rtgmcRepairNegRestoreSoftOnce<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmcRepairNegRestoreSoftOnce<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal);
        }
    }
    return min(center, (sum + 4) / 9);
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairPosRestoreArea(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairPosRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            value = max(value, rtgmcRepairPosRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal));
        }
    }
    return value;
}

template<typename Type, int THIN_LEVEL>
__device__ int rtgmcRepairNegRestoreArea(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairNegRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            value = min(value, rtgmcRepairNegRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height, rangeHalf, maxVal));
        }
    }
    return value;
}

template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
__device__ int rtgmcRepairPosLimit(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairPosRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr (PAD_LEVEL == 1 || PAD_LEVEL == 2) {
        value = (PAD_LEVEL == 1)
            ? rtgmcRepairPosRestoreSoftOnce<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal)
            : rtgmcRepairPosRestoreSoftTwice<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    } else if constexpr (PAD_LEVEL >= 3) {
        value = rtgmcRepairPosRestoreArea<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    }
    return value;
}

template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
__device__ int rtgmcRepairNegLimit(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int value = rtgmcRepairNegRestoreWide<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if constexpr (PAD_LEVEL == 1 || PAD_LEVEL == 2) {
        value = (PAD_LEVEL == 1)
            ? rtgmcRepairNegRestoreSoftOnce<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal)
            : rtgmcRepairNegRestoreSoftTwice<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    } else if constexpr (PAD_LEVEL >= 3) {
        value = rtgmcRepairNegRestoreArea<Type, THIN_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    }
    return value;
}

template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
__device__ int rtgmcRepairLimitedDelta(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    int diff = rtgmcRepairDeltaCentered<Type>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
    if (diff >= rangeHalf + 1) {
        const int upperEnvelope = rtgmcRepairPosLimit<Type, THIN_LEVEL, PAD_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
        diff = max(upperEnvelope, rangeHalf);
    } else if (diff <= rangeHalf - 1) {
        const int lowerEnvelope = rtgmcRepairNegLimit<Type, THIN_LEVEL, PAD_LEVEL>(input, inputPitch, reference, referencePitch, x, y, width, height, rangeHalf, maxVal);
        diff = min(lowerEnvelope, rangeHalf);
    }
    return clamp(diff, 0, maxVal);
}

template<typename Type>
__device__ int rtgmcRepairStageRead(
    const uint8_t *src, const int pitch, const int x, const int bufferY
) {
    return (int)(*(const Type *)(src + bufferY * pitch + x * sizeof(Type)));
}

template<typename Type>
__device__ void rtgmcRepairStageWrite(
    uint8_t *dst, const int pitch, const int x, const int bufferY, const int value
) {
    *(Type *)(dst + bufferY * pitch + x * sizeof(Type)) = (Type)value;
}

template<typename Type>
__global__ void kernel_rtgmc_shimmer_repair_stage_vertical(
    uint8_t *pVerticalContractPositive,
    uint8_t *pVerticalExpandNegative,
    const int stagePitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const int width,
    const int height,
    const int stageYOffset,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int bufferY = blockIdx.y * blockDim.y + threadIdx.y;
    const int stagedHeight = height + stageYOffset * 2;
    if (ix >= width || bufferY >= stagedHeight) return;

    const int logicalY = bufferY - stageYOffset;
    int positive = maxVal;
    int negative = 0;
    for (int dy = -2; dy <= 2; dy++) {
        const int sample = rtgmcRepairDeltaCentered<Type>(
            pInput, inputPitch, pReference, referencePitch,
            ix, logicalY + dy, width, height, rangeHalf, maxVal);
        positive = min(positive, sample);
        negative = max(negative, sample);
    }
    rtgmcRepairStageWrite<Type>(pVerticalContractPositive, stagePitch, ix, bufferY, positive);
    rtgmcRepairStageWrite<Type>(pVerticalExpandNegative, stagePitch, ix, bufferY, negative);
}

template<typename Type>
__global__ void kernel_rtgmc_shimmer_repair_stage_local(
    uint8_t *pLocalContractPositive,
    uint8_t *pLocalExpandNegative,
    const uint8_t *pVerticalContractPositive,
    const uint8_t *pVerticalExpandNegative,
    const int stagePitch,
    const int width,
    const int height,
    const int stageYOffset
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int bufferY = blockIdx.y * blockDim.y + threadIdx.y;
    const int stagedHeight = height + stageYOffset * 2;
    if (ix >= width || bufferY >= stagedHeight) return;

    const int logicalY = bufferY - stageYOffset;
    const int centerPositive = rtgmcRepairStageRead<Type>(pVerticalContractPositive, stagePitch, ix, bufferY);
    const int centerNegative = rtgmcRepairStageRead<Type>(pVerticalExpandNegative, stagePitch, ix, bufferY);
    int positive = centerPositive;
    int negative = centerNegative;
    if (ix > 0 && ix < width - 1 && logicalY > 0 && logicalY < height - 1) {
        int sumPositive = 0;
        int sumNegative = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sumPositive += rtgmcRepairStageRead<Type>(
                    pVerticalContractPositive, stagePitch, ix + dx, bufferY + dy);
                sumNegative += rtgmcRepairStageRead<Type>(
                    pVerticalExpandNegative, stagePitch, ix + dx, bufferY + dy);
            }
        }
        positive = min(centerPositive, (sumPositive + 4) / 9);
        negative = max(centerNegative, (sumNegative + 4) / 9);
    }
    rtgmcRepairStageWrite<Type>(pLocalContractPositive, stagePitch, ix, bufferY, positive);
    rtgmcRepairStageWrite<Type>(pLocalExpandNegative, stagePitch, ix, bufferY, negative);
}

template<typename Type>
__device__ int rtgmcRepairStagedLimitedDelta(
    const uint8_t *input, const int inputPitch,
    const uint8_t *reference, const int referencePitch,
    const uint8_t *localContractPositive,
    const uint8_t *localExpandNegative,
    const int stagePitch,
    const int x,
    const int y,
    const int width,
    const int height,
    const int stageYOffset,
    const int rangeHalf,
    const int maxVal
) {
    int diff = rtgmcRepairDeltaCentered<Type>(
        input, inputPitch, reference, referencePitch,
        x, y, width, height, rangeHalf, maxVal);
    if (diff >= rangeHalf + 1) {
        int upperEnvelope = rtgmcRepairStageRead<Type>(
            localContractPositive, stagePitch, x, y - 2 + stageYOffset);
        for (int dy = -1; dy <= 2; dy++) {
            upperEnvelope = max(upperEnvelope, rtgmcRepairStageRead<Type>(
                localContractPositive, stagePitch, x, y + dy + stageYOffset));
        }
        diff = max(upperEnvelope, rangeHalf);
    } else if (diff <= rangeHalf - 1) {
        int lowerEnvelope = rtgmcRepairStageRead<Type>(
            localExpandNegative, stagePitch, x, y - 2 + stageYOffset);
        for (int dy = -1; dy <= 2; dy++) {
            lowerEnvelope = min(lowerEnvelope, rtgmcRepairStageRead<Type>(
                localExpandNegative, stagePitch, x, y + dy + stageYOffset));
        }
        diff = min(lowerEnvelope, rangeHalf);
    }
    return clamp(diff, 0, maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_shimmer_repair_apply_staged(
    uint8_t *pDst, const int dstPitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const uint8_t *pLocalContractPositive,
    const uint8_t *pLocalExpandNegative,
    const int stagePitch,
    const int width,
    const int height,
    const int stageYOffset,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int inputValue = rtgmc_read_pix<Type>(pInput, ix, iy, inputPitch, width, height);
    const int mergedDiff = rtgmcRepairStagedLimitedDelta<Type>(
        pInput, inputPitch, pReference, referencePitch,
        pLocalContractPositive, pLocalExpandNegative, stagePitch,
        ix, iy, width, height, stageYOffset, rangeHalf, maxVal);
    rtgmc_write_pix<Type>(pDst, ix, iy, dstPitch,
        clamp(inputValue + mergedDiff - rangeHalf, 0, maxVal));
}

template<typename Type>
__global__ void kernel_rtgmc_shimmer_repair_apply_fused_staged(
    uint8_t *pDst, const int dstPitch,
    uint8_t *pCorrectionDelta, const int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    uint8_t *pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const uint8_t *pLocalContractPositive,
    const uint8_t *pLocalExpandNegative,
    const int stagePitch,
    const int width,
    const int height,
    const int stageYOffset,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int inputValue = rtgmc_read_pix<Type>(pInput, ix, iy, inputPitch, width, height);
    const int referenceValue = rtgmc_read_pix<Type>(pReference, ix, iy, referencePitch, width, height);
    const int signedDelta = referenceValue - inputValue;
    const int mergedDiff = rtgmcRepairStagedLimitedDelta<Type>(
        pInput, inputPitch, pReference, referencePitch,
        pLocalContractPositive, pLocalExpandNegative, stagePitch,
        ix, iy, width, height, stageYOffset, rangeHalf, maxVal);
    const int selectedSigned = mergedDiff - rangeHalf;
    const int positiveGateSigned = (signedDelta > 0 && selectedSigned > 0) ? selectedSigned : 0;
    const int negativeGateSigned = (signedDelta < 0 && selectedSigned < 0) ? selectedSigned : 0;

    rtgmc_write_pix<Type>(pCorrectionDelta, ix, iy, correctionDeltaPitch,
        rtgmcShimmerRepairSignedToDiff<Type>(signedDelta, rangeHalf, maxVal));
    rtgmc_write_pix<Type>(pPositiveCorrectionGate, ix, iy, positiveCorrectionGatePitch,
        rtgmcShimmerRepairSignedToDiff<Type>(positiveGateSigned, rangeHalf, maxVal));
    rtgmc_write_pix<Type>(pNegativeCorrectionGate, ix, iy, negativeCorrectionGatePitch,
        rtgmcShimmerRepairSignedToDiff<Type>(negativeGateSigned, rangeHalf, maxVal));
    rtgmc_write_pix<Type>(pDst, ix, iy, dstPitch,
        clamp(inputValue + selectedSigned, 0, maxVal));
}

template<typename Type>
__global__ void kernel_rtgmc_shimmer_repair_copy(
    uint8_t *pDst, const int dstPitch,
    const uint8_t *pSrc, const int srcPitch,
    const int width,
    const int height,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    rtgmc_write_pix<Type>(pDst, ix, iy, dstPitch, value);
    (void)maxVal;
}

template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
__global__ void kernel_rtgmc_shimmer_repair_apply(
    uint8_t *pDst, const int dstPitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const int width,
    const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int inputValue = rtgmc_read_pix<Type>(pInput, ix, iy, inputPitch, width, height);
    const int mergedDiff = rtgmcRepairLimitedDelta<Type, THIN_LEVEL, PAD_LEVEL>(
        pInput, inputPitch, pReference, referencePitch, ix, iy, width, height, rangeHalf, maxVal);

    rtgmc_write_pix<Type>(pDst, ix, iy, dstPitch,
        clamp(inputValue + mergedDiff - rangeHalf, 0, maxVal));
}

template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
__global__ void kernel_rtgmc_shimmer_repair_apply_fused(
    uint8_t *pDst, const int dstPitch,
    uint8_t *pCorrectionDelta, const int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    uint8_t *pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const int width,
    const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int inputValue = rtgmc_read_pix<Type>(pInput, ix, iy, inputPitch, width, height);
    const int referenceValue = rtgmc_read_pix<Type>(pReference, ix, iy, referencePitch, width, height);
    const int signedDelta = referenceValue - inputValue;
    const int mergedDiff = rtgmcRepairLimitedDelta<Type, THIN_LEVEL, PAD_LEVEL>(
        pInput, inputPitch, pReference, referencePitch, ix, iy, width, height, rangeHalf, maxVal);
    const int selectedSigned = mergedDiff - rangeHalf;
    const int positiveGateSigned = (signedDelta > 0 && selectedSigned > 0) ? selectedSigned : 0;
    const int negativeGateSigned = (signedDelta < 0 && selectedSigned < 0) ? selectedSigned : 0;

    rtgmc_write_pix<Type>(pCorrectionDelta, ix, iy, correctionDeltaPitch,
        rtgmcShimmerRepairSignedToDiff<Type>(signedDelta, rangeHalf, maxVal));
    rtgmc_write_pix<Type>(pPositiveCorrectionGate, ix, iy, positiveCorrectionGatePitch,
        rtgmcShimmerRepairSignedToDiff<Type>(positiveGateSigned, rangeHalf, maxVal));
    rtgmc_write_pix<Type>(pNegativeCorrectionGate, ix, iy, negativeCorrectionGatePitch,
        rtgmcShimmerRepairSignedToDiff<Type>(negativeGateSigned, rangeHalf, maxVal));
    rtgmc_write_pix<Type>(pDst, ix, iy, dstPitch,
        clamp(inputValue + selectedSigned, 0, maxVal));
}

#if !defined(NVENC_RTGMC_SHIMMER_REPAIR_BUILD_FUSED)
template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
static bool launchRtgmcShimmerRepairApplyByProfile(
    const int thinLevel, const int padLevel,
    const dim3 gridSize, const dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, const int dstPitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const int width, const int height,
    const int rangeHalf, const int maxVal) {
    if (thinLevel == THIN_LEVEL && padLevel == PAD_LEVEL) {
        kernel_rtgmc_shimmer_repair_apply<Type, THIN_LEVEL, PAD_LEVEL><<<gridSize, blockSize, 0, stream>>>(
            pDst, dstPitch,
            pInput, inputPitch,
            pReference, referencePitch,
            width, height,
            rangeHalf, maxVal);
        return true;
    }
    if constexpr (PAD_LEVEL < RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL) {
        return launchRtgmcShimmerRepairApplyByProfile<Type, THIN_LEVEL, PAD_LEVEL + 1>(
            thinLevel, padLevel, gridSize, blockSize, stream,
            pDst, dstPitch, pInput, inputPitch, pReference, referencePitch,
            width, height, rangeHalf, maxVal);
    } else if constexpr (THIN_LEVEL < RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL) {
        return launchRtgmcShimmerRepairApplyByProfile<Type, THIN_LEVEL + 1, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
            thinLevel, padLevel, gridSize, blockSize, stream,
            pDst, dstPitch, pInput, inputPitch, pReference, referencePitch,
            width, height, rangeHalf, maxVal);
    } else {
        return false;
    }
}
#endif

#if !defined(NVENC_RTGMC_SHIMMER_REPAIR_BUILD_APPLY)
template<typename Type, int THIN_LEVEL, int PAD_LEVEL>
static bool launchRtgmcShimmerRepairFusedByProfile(
    const int thinLevel, const int padLevel,
    const dim3 gridSize, const dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, const int dstPitch,
    uint8_t *pCorrectionDelta, const int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    uint8_t *pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    const int width, const int height,
    const int rangeHalf, const int maxVal) {
    if (thinLevel == THIN_LEVEL && padLevel == PAD_LEVEL) {
        kernel_rtgmc_shimmer_repair_apply_fused<Type, THIN_LEVEL, PAD_LEVEL><<<gridSize, blockSize, 0, stream>>>(
            pDst, dstPitch,
            pCorrectionDelta, correctionDeltaPitch,
            pPositiveCorrectionGate, positiveCorrectionGatePitch,
            pNegativeCorrectionGate, negativeCorrectionGatePitch,
            pInput, inputPitch,
            pReference, referencePitch,
            width, height,
            rangeHalf, maxVal);
        return true;
    }
    if constexpr (PAD_LEVEL < RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL) {
        return launchRtgmcShimmerRepairFusedByProfile<Type, THIN_LEVEL, PAD_LEVEL + 1>(
            thinLevel, padLevel, gridSize, blockSize, stream,
            pDst, dstPitch,
            pCorrectionDelta, correctionDeltaPitch,
            pPositiveCorrectionGate, positiveCorrectionGatePitch,
            pNegativeCorrectionGate, negativeCorrectionGatePitch,
            pInput, inputPitch, pReference, referencePitch,
            width, height, rangeHalf, maxVal);
    } else if constexpr (THIN_LEVEL < RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL) {
        return launchRtgmcShimmerRepairFusedByProfile<Type, THIN_LEVEL + 1, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
            thinLevel, padLevel, gridSize, blockSize, stream,
            pDst, dstPitch,
            pCorrectionDelta, correctionDeltaPitch,
            pPositiveCorrectionGate, positiveCorrectionGatePitch,
            pNegativeCorrectionGate, negativeCorrectionGatePitch,
            pInput, inputPitch, pReference, referencePitch,
            width, height, rangeHalf, maxVal);
    } else {
        return false;
    }
}

#endif

#if defined(NVENC_RTGMC_SHIMMER_REPAIR_KERNEL_ONLY)
#if defined(NVENC_RTGMC_SHIMMER_REPAIR_BUILD_APPLY)
bool NVENC_RTGMC_SHIMMER_REPAIR_STAGED_LAUNCH_NAME(
    const dim3 gridSize, const dim3 stagedGridSize, const dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, const int dstPitch,
    uint8_t *pCorrectionDelta, const int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    uint8_t *pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const uint8_t *pInput, const int inputPitch,
    const uint8_t *pReference, const int referencePitch,
    uint8_t *pVerticalContractPositive,
    uint8_t *pVerticalExpandNegative,
    uint8_t *pLocalContractPositive,
    uint8_t *pLocalExpandNegative,
    const int stagePitch,
    const int width, const int height, const int stageYOffset,
    const int rangeHalf, const int maxVal) {
    kernel_rtgmc_shimmer_repair_stage_vertical<NVENC_RTGMC_SHIMMER_REPAIR_TYPE><<<stagedGridSize, blockSize, 0, stream>>>(
        pVerticalContractPositive, pVerticalExpandNegative, stagePitch,
        pInput, inputPitch, pReference, referencePitch,
        width, height, stageYOffset, rangeHalf, maxVal);
    kernel_rtgmc_shimmer_repair_stage_local<NVENC_RTGMC_SHIMMER_REPAIR_TYPE><<<stagedGridSize, blockSize, 0, stream>>>(
        pLocalContractPositive, pLocalExpandNegative,
        pVerticalContractPositive, pVerticalExpandNegative, stagePitch,
        width, height, stageYOffset);
    const bool fused = pCorrectionDelta != nullptr
        && pPositiveCorrectionGate != nullptr
        && pNegativeCorrectionGate != nullptr;
    if (fused) {
        kernel_rtgmc_shimmer_repair_apply_fused_staged<NVENC_RTGMC_SHIMMER_REPAIR_TYPE><<<gridSize, blockSize, 0, stream>>>(
            pDst, dstPitch,
            pCorrectionDelta, correctionDeltaPitch,
            pPositiveCorrectionGate, positiveCorrectionGatePitch,
            pNegativeCorrectionGate, negativeCorrectionGatePitch,
            pInput, inputPitch, pReference, referencePitch,
            pLocalContractPositive, pLocalExpandNegative, stagePitch,
            width, height, stageYOffset, rangeHalf, maxVal);
    } else {
        kernel_rtgmc_shimmer_repair_apply_staged<NVENC_RTGMC_SHIMMER_REPAIR_TYPE><<<gridSize, blockSize, 0, stream>>>(
            pDst, dstPitch,
            pInput, inputPitch, pReference, referencePitch,
            pLocalContractPositive, pLocalExpandNegative, stagePitch,
            width, height, stageYOffset, rangeHalf, maxVal);
    }
    return true;
}

bool NVENC_RTGMC_SHIMMER_REPAIR_COPY_LAUNCH_NAME(
    const dim3 gridSize, const dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, const int dstPitch, const uint8_t *pSrc, const int srcPitch,
    const int width, const int height, const int maxVal) {
    kernel_rtgmc_shimmer_repair_copy<NVENC_RTGMC_SHIMMER_REPAIR_TYPE><<<gridSize, blockSize, 0, stream>>>(
        pDst, dstPitch, pSrc, srcPitch, width, height, maxVal);
    return true;
}

bool NVENC_RTGMC_SHIMMER_REPAIR_LAUNCH_NAME(
    const int thinLevel, const int padLevel, const dim3 gridSize, const dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, const int dstPitch, const uint8_t *pInput, const int inputPitch, const uint8_t *pReference, const int referencePitch,
    const int width, const int height, const int rangeHalf, const int maxVal) {
    return launchRtgmcShimmerRepairApplyByProfile<NVENC_RTGMC_SHIMMER_REPAIR_TYPE, RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
        thinLevel, padLevel, gridSize, blockSize, stream, pDst, dstPitch, pInput, inputPitch, pReference, referencePitch, width, height, rangeHalf, maxVal);
}
#else
bool NVENC_RTGMC_SHIMMER_REPAIR_LAUNCH_NAME(
    const int thinLevel, const int padLevel, const dim3 gridSize, const dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, const int dstPitch, uint8_t *pCorrectionDelta, const int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, const int positiveCorrectionGatePitch, uint8_t *pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const uint8_t *pInput, const int inputPitch, const uint8_t *pReference, const int referencePitch,
    const int width, const int height, const int rangeHalf, const int maxVal) {
    return launchRtgmcShimmerRepairFusedByProfile<NVENC_RTGMC_SHIMMER_REPAIR_TYPE, RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
        thinLevel, padLevel, gridSize, blockSize, stream, pDst, dstPitch, pCorrectionDelta, correctionDeltaPitch,
        pPositiveCorrectionGate, positiveCorrectionGatePitch, pNegativeCorrectionGate, negativeCorrectionGatePitch,
        pInput, inputPitch, pReference, referencePitch, width, height, rangeHalf, maxVal);
}
#endif
#endif

