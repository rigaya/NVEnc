// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU8TRP() {
    return getNVEncRtgmcSearchPrefilterLaunchFuncs<uint8_t, -1>();
}

RGY_ERR launchRtgmcSearchPrefilterDebugU8(
    const int debugStage, const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &dst,
    const uint32_t repairProfile, const int smoothRadius, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterDebugTyped<uint8_t>(
        debugStage, prev2, prev, cur, next, next2, dst, repairProfile, smoothRadius, stream);
}
