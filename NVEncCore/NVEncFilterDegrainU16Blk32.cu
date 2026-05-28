#include "NVEncFilterDegrain.cuh"

RGY_ERR launchNVEncDegrainMotionSearchSearchParallelU16Blk32(
    const uint8_t *sourcePlane, const uint8_t *referencePlane, CUMemBuf &vectors,
    const int pitch, const int width, const int height, const int planeBase, const int blockCount,
    const RGYDegrainBlockLayout &layout, const int pel, const int subpelInterp,
    const int pad, const int motionCostScale, const int lowSadWeightScale,
    const int zeroCandidateCostScale, const int frameAverageCandidateCostScale,
    const int newCandidateCostScale, const int level, cudaStream_t stream) {
    return launchNVEncDegrainMotionSearchSearchParallelBlock<uint16_t, 32>(
        sourcePlane, referencePlane, vectors, pitch, width, height, planeBase, blockCount, layout,
        pel, subpelInterp, pad, motionCostScale, lowSadWeightScale, zeroCandidateCostScale,
        frameAverageCandidateCostScale, newCandidateCostScale, level, stream);
}

RGY_ERR launchNVEncDegrainMotionSearchSpatialRefineU16Blk32(
    const uint8_t *sourcePlane, const uint8_t *referencePlane,
    CUMemBuf &vectors, const CUMemBuf &vectorsPrev, CUMemBuf &vectorsFinal,
    const int pitch, const int width, const int height, const int planeBase, const int finalBase,
    const int blockCount, const RGYDegrainBlockLayout &layout,
    const int pel, const int subpelInterp, const int pad, const int motionCostScale,
    const int lowSadWeightScale, const int newCandidateCostScale, cudaStream_t stream) {
    return launchNVEncDegrainMotionSearchSpatialRefineBlock<uint16_t, 32>(
        sourcePlane, referencePlane, vectors, vectorsPrev, vectorsFinal, pitch, width, height,
        planeBase, finalBase, blockCount, layout, pel, subpelInterp, pad, motionCostScale,
        lowSadWeightScale, newCandidateCostScale, stream);
}
