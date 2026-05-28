#include "NVEncFilterNnedi.cuh"

RGY_ERR launchNVEncNnediPredictorU8Fast(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax,
    int nsize, int nns, cudaStream_t stream) {
    return launchPredictorNsize<uint8_t, 8, VPP_NNEDI_QUALITY_FAST>(
        dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf,
        dstOffset, refEvalOffset, width4, height, valMin, valMax, nsize, nns, stream);
}
