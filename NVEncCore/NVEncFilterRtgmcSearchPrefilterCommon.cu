// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

RGY_ERR launchRtgmcSearchPrefilterScenechangeU8(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    uint32_t *partial, const int groupCount, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterScenechangeTyped<uint8_t>(prev2, prev, cur, next, next2, partial, groupCount, stream);
}

RGY_ERR launchRtgmcSearchPrefilterScenechangeU16(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    uint32_t *partial, const int groupCount, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterScenechangeTyped<uint16_t>(prev2, prev, cur, next, next2, partial, groupCount, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRefine2TileU8(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterRefine2TileTyped<uint8_t>(motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRefine2TileU16(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterRefine2TileTyped<uint16_t>(motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchU8(
    const RGYFrameInfo &searchSmoothed3x3, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterEdgeSoftenedSearchTyped<uint8_t>(searchSmoothed3x3, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchU16(
    const RGYFrameInfo &searchSmoothed3x3, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterEdgeSoftenedSearchTyped<uint16_t>(searchSmoothed3x3, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3U8(
    const RGYFrameInfo &src, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSearchSmoothed3x3Typed<uint8_t>(src, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3U16(
    const RGYFrameInfo &src, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSearchSmoothed3x3Typed<uint16_t>(src, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendU8(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendTyped<uint8_t>(spatialGuide, motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendU16(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendTyped<uint16_t>(spatialGuide, motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU8(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedTyped<uint8_t>(spatialGuide, motionGuide, fieldGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU16(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedTyped<uint16_t>(spatialGuide, motionGuide, fieldGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchU8(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &spatialGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterStabilizedSearchTyped<uint8_t>(motionGuide, fieldGuide, spatialGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchU16(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &spatialGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterStabilizedSearchTyped<uint16_t>(motionGuide, fieldGuide, spatialGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRangeConvertU8(
    const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterRangeConvertTyped<uint8_t>(dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRangeConvertU16(
    const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterRangeConvertTyped<uint16_t>(dst, fullRangeMode, stream);
}
