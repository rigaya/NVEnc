#include "NVEncFilterRtgmcShimmerRepair.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

#include "rgy_cuda_util_kernel.h"

#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

namespace {
static constexpr int RTGMC_SHIMMER_REPAIR_BLOCK_X = 32;
static constexpr int RTGMC_SHIMMER_REPAIR_BLOCK_Y = 8;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT = 0;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_DELTA = 1;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE = 2;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE = 3;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_INPUT_TMP = 6;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_REF_TMP = 7;

static const char *rtgmcShimmerRepairTargetName(const RGYRtgmcShimmerRepairStage stage) {
    return (stage == RGYRtgmcShimmerRepairStage::PreRetouch) ? "rep1" : "rep2";
}

static const TCHAR *rtgmcShimmerRepairStageName(const RGYRtgmcShimmerRepairStage stage) {
    return (stage == RGYRtgmcShimmerRepairStage::PreRetouch) ? _T("pre-retouch") : _T("post-tr2");
}

static void rtgmcShimmerRepairLoadProfile(NVEncFilterParamRtgmcShimmerRepair *prm) {
    prm->repairProfile = rgy_rtgmc_repair_profile_from_levels(prm->repairThin, prm->repairPad);
}

static RGY_ERR rtgmcShimmerRepairWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto& waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            const auto sts = err_to_rgy(cudaStreamWaitEvent(stream, waitEvent(), 0));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR rtgmcShimmerRepairRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
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
}

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

__device__ void rtgmcRepairSort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

__device__ void rtgmcRepairSort2Desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

__device__ void rtgmcRepairSort8(int *v) {
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

tstring NVEncFilterParamRtgmcShimmerRepair::print() const {
    return strsprintf(_T("rtgmc-shimmer-repair: stage=%s repair-thin=%d repair-pad=%d process_chroma=%s"),
        rtgmcShimmerRepairStageName(stage), repairThin, repairPad, processChroma ? _T("true") : _T("false"));
}

NVEncFilterRtgmcShimmerRepair::NVEncFilterRtgmcShimmerRepair() :
    NVEncFilter(),
    m_buildOptions(),
    m_lumaDump(),
    m_lumaDumpPath(),
    m_lumaDumpStage("shimmer_corrected"),
    m_lumaDumpTarget(),
    m_lumaDumpMaxFrames(0),
    m_lumaDumpFrameCount(0),
    m_lumaDumpEnabled(false),
    m_lumaDumpHeaderWritten(false),
    m_lumaDumpFullYuv(false),
    m_useKernel(false) {
    m_name = _T("rtgmc-shimmer-repair");
}

NVEncFilterRtgmcShimmerRepair::~NVEncFilterRtgmcShimmerRepair() {
    close();
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcShimmerRepair> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair requires identical input/output csp and resolution.\n"));
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
    if (!rgy_rtgmc_repair_thin_level_is_valid(prm->repairThin)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair rep-thin must be 0-7.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!rgy_rtgmc_repair_pad_level_is_valid(prm->repairPad)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair rep-pad must be 0-3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcShimmerRepair> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm->repairProfile;
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D range_half=%d -D rtgmc_shimmer_repair_block_x=%d -D rtgmc_shimmer_repair_block_y=%d"
        " -D RTGMC_SHIMMER_REPAIR_THIN_LEVEL=%d -D RTGMC_SHIMMER_REPAIR_PAD_LEVEL=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf,
        RTGMC_SHIMMER_REPAIR_BLOCK_X,
        RTGMC_SHIMMER_REPAIR_BLOCK_Y,
        profile.thinRejectLevel,
        profile.restorePaddingLevel);
    AddMessage(RGY_LOG_DEBUG, _T("Using CUDA kernel for rtgmc-shimmer-repair: %s\n"),
        char_to_tstring(m_buildOptions).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::initLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcShimmerRepair &prm) {
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "shimmer_corrected";
    m_lumaDumpTarget.clear();
    m_lumaDumpFullYuv = false;
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }

    const char *dumpPathEnv = std::getenv("QSVENC_RTGMC_REP_YUV_DUMP_Y4M");
    if (dumpPathEnv != nullptr && dumpPathEnv[0] != '\0') {
        m_lumaDumpFullYuv = true;
    } else {
        dumpPathEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_Y4M");
    }
    if (dumpPathEnv == nullptr || dumpPathEnv[0] == '\0') {
        return RGY_ERR_NONE;
    }
    m_lumaDumpPath = dumpPathEnv;

    if (const char *stageEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_STAGE"); stageEnv != nullptr && stageEnv[0] != '\0') {
        m_lumaDumpStage = stageEnv;
        std::transform(m_lumaDumpStage.begin(), m_lumaDumpStage.end(), m_lumaDumpStage.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }
    if (m_lumaDumpStage != "correction_delta" && m_lumaDumpStage != "positive_correction_gate"
        && m_lumaDumpStage != "negative_correction_gate" && m_lumaDumpStage != "shimmer_corrected") {
        AddMessage(RGY_LOG_ERROR, _T("unsupported rtgmc rep luma dump stage: %s.\n"),
            char_to_tstring(m_lumaDumpStage).c_str());
        return RGY_ERR_INVALID_PARAM;
    }

    if (const char *targetEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_TARGET"); targetEnv != nullptr && targetEnv[0] != '\0') {
        m_lumaDumpTarget = targetEnv;
        std::transform(m_lumaDumpTarget.begin(), m_lumaDumpTarget.end(), m_lumaDumpTarget.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }
    const char *activeTarget = (prm.repairThin > 0) ? rtgmcShimmerRepairTargetName(prm.stage) : "";
    if (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != activeTarget) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc rep luma dump target %s skipped for inactive %s instance.\n"),
            char_to_tstring(m_lumaDumpTarget).c_str(), char_to_tstring(activeTarget).c_str());
        return RGY_ERR_NONE;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    if (bitdepth > 8) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc rep stage dump supports only 8bit input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_REP_YUV_DUMP_Y4M supports only 4:2:0 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (!m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frameInfo.csp] != 1) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_REP_LUMA_DUMP_Y4M supports only 4:2:0/Y8 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }

    const char *maxFrames = std::getenv(m_lumaDumpFullYuv
        ? "QSVENC_RTGMC_REP_YUV_DUMP_MAX_FRAMES"
        : "QSVENC_RTGMC_REP_LUMA_DUMP_MAX_FRAMES");
    if (maxFrames == nullptr || maxFrames[0] == '\0') {
        maxFrames = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_MAX_FRAMES");
    }
    if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const long parsed = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames && parsed > 0) {
            m_lumaDumpMaxFrames = (int)std::min<long>(parsed, std::numeric_limits<int>::max());
        }
    }

    m_lumaDump.open(m_lumaDumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open rtgmc rep luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpEnabled = true;
    AddMessage(RGY_LOG_INFO, _T("rtgmc rep %s dump enabled: %s (target=%s, stage=%s).\n"),
        m_lumaDumpFullYuv ? _T("yuv") : _T("luma"),
        char_to_tstring(m_lumaDumpPath).c_str(), char_to_tstring(activeTarget).c_str(), char_to_tstring(m_lumaDumpStage).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::dumpLumaFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_lumaDumpEnabled) {
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpMaxFrames > 0 && m_lumaDumpFrameCount >= m_lumaDumpMaxFrames) {
        return RGY_ERR_NONE;
    }
    if (frame == nullptr || frame->ptr[0] == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    if (bitdepth > 8 || (m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420)
        || (!m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frame->csp] != 1)) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc rep luma dump disabled by unsupported frame csp: %s.\n"),
            RGY_CSP_NAMES[frame->csp]);
        m_lumaDumpEnabled = false;
        return RGY_ERR_NONE;
    }

    auto sts = rtgmcShimmerRepairWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    CUFrameBuf hostFrame(frame->width, frame->height, frame->csp);
    hostFrame.frame.mem_type = RGY_MEM_TYPE_CPU;
    sts = hostFrame.allocHost();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc rep luma dump host buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = copyFrameAsync(&hostFrame.frame, frame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc rep luma dump frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = err_to_rgy(cudaStreamSynchronize(stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc rep luma dump read: %s.\n"), get_err_mes(sts));
        return sts;
    }

    if (!m_lumaDumpHeaderWritten) {
        m_lumaDump << "YUV4MPEG2 W" << hostFrame.frame.width << " H" << hostFrame.frame.height << " F30000:1001 Ip A0:0 C420jpeg\n";
        m_lumaDumpHeaderWritten = true;
    }
    m_lumaDump << "FRAME\n";
    for (int y = 0; y < hostFrame.frame.height; y++) {
        m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.frame.ptr[0] + (size_t)y * hostFrame.frame.pitch[0]), hostFrame.frame.width);
    }
    const int chromaWidth = (hostFrame.frame.width + 1) >> 1;
    const int chromaHeight = (hostFrame.frame.height + 1) >> 1;
    if (m_lumaDumpFullYuv) {
        for (int y = 0; y < chromaHeight; y++) {
            m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.frame.ptr[1] + (size_t)y * hostFrame.frame.pitch[1]), chromaWidth);
        }
        for (int y = 0; y < chromaHeight; y++) {
            m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.frame.ptr[2] + (size_t)y * hostFrame.frame.pitch[2]), chromaWidth);
        }
    } else {
        std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    }
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write rtgmc rep luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_lumaDumpEnabled || m_lumaDumpStage != stage || (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != target)) {
        return RGY_ERR_NONE;
    }
    return dumpLumaFrame(frame, stream, wait_events);
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairFused(
    RGYFrameInfo *pOutputFrame,
    RGYFrameInfo *pCorrectionDeltaFrame,
    RGYFrameInfo *pPositiveCorrectionGateFrame,
    RGYFrameInfo *pNegativeCorrectionGateFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcShimmerRepair &prm,
    int iplane, cudaStream_t stream) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto deltaPlane = getPlane(pCorrectionDeltaFrame, (RGY_PLANE)iplane);
    const auto positivePlane = getPlane(pPositiveCorrectionGateFrame, (RGY_PLANE)iplane);
    const auto negativePlane = getPlane(pNegativeCorrectionGateFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    const int bitdepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm.repairProfile;
    const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    const dim3 gridSize(divCeil(outPlane.width, blockSize.x), divCeil(outPlane.height, blockSize.y));
    const bool launched = (bitdepth <= 8)
        ? launchRtgmcShimmerRepairFusedByProfile<uint8_t, RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (uint8_t *)deltaPlane.ptr[0], deltaPlane.pitch[0],
            (uint8_t *)positivePlane.ptr[0], positivePlane.pitch[0],
            (uint8_t *)negativePlane.ptr[0], negativePlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal)
        : launchRtgmcShimmerRepairFusedByProfile<uint16_t, RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (uint8_t *)deltaPlane.ptr[0], deltaPlane.pitch[0],
            (uint8_t *)positivePlane.ptr[0], positivePlane.pitch[0],
            (uint8_t *)negativePlane.ptr[0], negativePlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal);
    if (!launched) {
        AddMessage(RGY_LOG_ERROR, _T("invalid rtgmc-shimmer-repair profile: thin=%d pad=%d.\n"),
            profile.thinRejectLevel, profile.restorePaddingLevel);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto err = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
            _T("kernel_rtgmc_shimmer_repair_apply_fused"), iplane, get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairApply(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcShimmerRepair &prm,
    int iplane, cudaStream_t stream) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    const int bitdepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm.repairProfile;
    const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    const dim3 gridSize(divCeil(outPlane.width, blockSize.x), divCeil(outPlane.height, blockSize.y));
    const bool launched = (bitdepth <= 8)
        ? launchRtgmcShimmerRepairApplyByProfile<uint8_t, RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal)
        : launchRtgmcShimmerRepairApplyByProfile<uint16_t, RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL, RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL>(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal);
    if (!launched) {
        AddMessage(RGY_LOG_ERROR, _T("invalid rtgmc-shimmer-repair profile: thin=%d pad=%d.\n"),
            profile.thinRejectLevel, profile.restorePaddingLevel);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto err = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
            _T("kernel_rtgmc_shimmer_repair_apply"), iplane, get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcShimmerRepair>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    rtgmcShimmerRepairLoadProfile(prm.get());

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcShimmerRepair>(m_param);
    if (m_useKernel
        && (!m_param
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->repairThin != prm->repairThin
        || prmPrev->repairPad != prm->repairPad)) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-shimmer-repair kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 8);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = initLumaDump(prm->frameOut, *prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcShimmerRepair &prm,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int repair = prm.repairThin;
    const char *target = (repair > 0) ? rtgmcShimmerRepairTargetName(prm.stage) : "";
    auto sts = rtgmcShimmerRepairWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto launchCopy = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, int iplane) -> RGY_ERR {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        if (RGY_CSP_BIT_DEPTH[dstFrame->csp] <= 8) {
            kernel_rtgmc_shimmer_repair_copy<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                255);
        } else {
            kernel_rtgmc_shimmer_repair_copy<uint16_t><<<gridSize, blockSize, 0, stream>>>(
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                65535);
        }
        const auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto err = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_shimmer_repair_copy"), iplane, get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    };
    auto dumpProcessedStages = [&](const char *stageTarget, RGYFrameInfo *deltaFrame, RGYFrameInfo *positiveGateFrame,
        RGYFrameInfo *negativeGateFrame, RGYFrameInfo *outputFrame, bool dumpNow) -> RGY_ERR {
        if (!dumpNow) {
            return RGY_ERR_NONE;
        }
        auto err = dumpStageFrame("correction_delta", deltaFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("positive_correction_gate", positiveGateFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("negative_correction_gate", negativeGateFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("shimmer_corrected", outputFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        return RGY_ERR_NONE;
    };

    for (int iplane = 0; iplane < planes; iplane++) {
        const bool processPlane = (iplane == 0 || prm.processChroma);
        if (!processPlane || repair == 0) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (m_lumaDumpEnabled && m_lumaDumpFullYuv && repair != 0) {
                err = launchCopy(&m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_DELTA]->frame, pInputFrame, iplane);
                if (err != RGY_ERR_NONE) return err;
                err = launchCopy(&m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE]->frame, pInputFrame, iplane);
                if (err != RGY_ERR_NONE) return err;
                err = launchCopy(&m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE]->frame, pInputFrame, iplane);
                if (err != RGY_ERR_NONE) return err;
            }
            continue;
        }

        RGYFrameInfo *correctionDelta = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_DELTA]->frame;
        RGYFrameInfo *positiveCorrectionGate = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE]->frame;
        RGYFrameInfo *negativeCorrectionGate = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE]->frame;

        const auto err = m_lumaDumpEnabled
            ? launchRtgmcShimmerRepairFused(
                pOutputFrame, correctionDelta, positiveCorrectionGate, negativeCorrectionGate,
                pInputFrame, pRefFrame, prm, iplane, stream)
            : launchRtgmcShimmerRepairApply(
                pOutputFrame, pInputFrame, pRefFrame, prm, iplane, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (iplane == 0 && !m_lumaDumpFullYuv) {
            sts = dumpProcessedStages(target, correctionDelta, positiveCorrectionGate, negativeCorrectionGate, pOutputFrame, true);
            if (sts != RGY_ERR_NONE) return sts;
        }
    }
    if (m_lumaDumpFullYuv) {
        RGY_ERR err = RGY_ERR_NONE;
        if (m_lumaDumpStage == "correction_delta") {
            err = dumpStageFrame("correction_delta", &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_DELTA]->frame, target, stream, {});
        } else if (m_lumaDumpStage == "positive_correction_gate") {
            err = dumpStageFrame("positive_correction_gate", &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE]->frame, target, stream, {});
        } else if (m_lumaDumpStage == "negative_correction_gate") {
            err = dumpStageFrame("negative_correction_gate", &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE]->frame, target, stream, {});
        } else if (m_lumaDumpStage == "shimmer_corrected") {
            err = dumpStageFrame("shimmer_corrected", pOutputFrame, target, stream, {});
        }
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return rtgmcShimmerRepairRecordEvent(stream, event);
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
    RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0] || !pRefFrame || !pRefFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (m_useKernel && !m_frameBuf.size()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load rtgmc-shimmer-repair kernel (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcShimmerRepair>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutFrame = m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT]->frame.mem_type);
        const auto refMemcpyKind = getCudaMemcpyKind(pRefFrame->mem_type, m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT]->frame.mem_type);
        if (memcpyKind == cudaMemcpyDeviceToDevice && refMemcpyKind == cudaMemcpyDeviceToDevice) {
            auto err = processFrame(&pOutFrame->frame, pInputFrame, pRefFrame, *prm, stream, wait_events, event);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            return RGY_ERR_NONE;
        }
    }

    if (m_useKernel) {
        auto pInputTmp = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_INPUT_TMP]->frame;
        auto pRefTmp = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_REF_TMP]->frame;
        auto waitErr = rtgmcShimmerRepairWaitEvents(stream, wait_events);
        if (waitErr != RGY_ERR_NONE) {
            return waitErr;
        }
        auto copyErr = m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_INPUT_TMP]->copyFrameAsync(pInputFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-shimmer-repair input frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        copyErr = m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_REF_TMP]->copyFrameAsync(pRefFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-shimmer-repair reference frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        auto err = processFrame(&pOutFrame->frame, pInputTmp, pRefTmp, *prm, stream, {}, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        return RGY_ERR_NONE;
    }

    auto waitErr = rtgmcShimmerRepairWaitEvents(stream, wait_events);
    if (waitErr != RGY_ERR_NONE) {
        return waitErr;
    }
    auto copyErr = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    return rtgmcShimmerRepairRecordEvent(stream, event);
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    return run_filter(pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, stream, {}, nullptr);
}

void NVEncFilterRtgmcShimmerRepair::close() {
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "shimmer_corrected";
    m_lumaDumpTarget.clear();
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFullYuv = false;
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
}
