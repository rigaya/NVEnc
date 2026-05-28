// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU16TRP() {
    return getNVEncRtgmcSearchPrefilterLaunchFuncs<uint16_t, -1>();
}

RGY_ERR launchRtgmcSearchPrefilterDebugU16(
    const int debugStage, const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &dst,
    const uint32_t repairProfile, const int smoothRadius, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterDebugTyped<uint16_t>(
        debugStage, prev2, prev, cur, next, next2, dst, repairProfile, smoothRadius, stream);
}
