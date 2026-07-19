#pragma once

#include <array>
#include <fstream>
#include <vector>

#include "NVEncFilter.h"
#include "NVEncFilterRtgmcCommon.h"
#include "NVEncFilterRtgmcSearchPrefilter.h"

enum class RGYRtgmcShimmerRepairStage {
    PreRetouch,
    PostTR2,
};

bool launchRtgmcShimmerRepairApplyU8(
    int thinLevel, int padLevel, dim3 gridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch, const uint8_t *pInput, int inputPitch, const uint8_t *pReference, int referencePitch,
    int width, int height, int rangeHalf, int maxVal);
bool launchRtgmcShimmerRepairApplyU16(
    int thinLevel, int padLevel, dim3 gridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch, const uint8_t *pInput, int inputPitch, const uint8_t *pReference, int referencePitch,
    int width, int height, int rangeHalf, int maxVal);
bool launchRtgmcShimmerRepairCopyU8(
    dim3 gridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch, const uint8_t *pSrc, int srcPitch,
    int width, int height, int maxVal);
bool launchRtgmcShimmerRepairCopyU16(
    dim3 gridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch, const uint8_t *pSrc, int srcPitch,
    int width, int height, int maxVal);
bool launchRtgmcShimmerRepairFusedU8(
    int thinLevel, int padLevel, dim3 gridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch, uint8_t *pCorrectionDelta, int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, int positiveCorrectionGatePitch, uint8_t *pNegativeCorrectionGate, int negativeCorrectionGatePitch,
    const uint8_t *pInput, int inputPitch, const uint8_t *pReference, int referencePitch,
    int width, int height, int rangeHalf, int maxVal);
bool launchRtgmcShimmerRepairFusedU16(
    int thinLevel, int padLevel, dim3 gridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch, uint8_t *pCorrectionDelta, int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, int positiveCorrectionGatePitch, uint8_t *pNegativeCorrectionGate, int negativeCorrectionGatePitch,
    const uint8_t *pInput, int inputPitch, const uint8_t *pReference, int referencePitch,
    int width, int height, int rangeHalf, int maxVal);
bool launchRtgmcShimmerRepairStagedU8(
    dim3 gridSize, dim3 stagedGridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch,
    uint8_t *pCorrectionDelta, int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, int positiveCorrectionGatePitch,
    uint8_t *pNegativeCorrectionGate, int negativeCorrectionGatePitch,
    const uint8_t *pInput, int inputPitch, const uint8_t *pReference, int referencePitch,
    uint8_t *pVerticalContractPositive, uint8_t *pVerticalExpandNegative,
    uint8_t *pLocalContractPositive, uint8_t *pLocalExpandNegative, int stagePitch,
    int width, int height, int stageYOffset, int rangeHalf, int maxVal);
bool launchRtgmcShimmerRepairStagedU16(
    dim3 gridSize, dim3 stagedGridSize, dim3 blockSize, cudaStream_t stream,
    uint8_t *pDst, int dstPitch,
    uint8_t *pCorrectionDelta, int correctionDeltaPitch,
    uint8_t *pPositiveCorrectionGate, int positiveCorrectionGatePitch,
    uint8_t *pNegativeCorrectionGate, int negativeCorrectionGatePitch,
    const uint8_t *pInput, int inputPitch, const uint8_t *pReference, int referencePitch,
    uint8_t *pVerticalContractPositive, uint8_t *pVerticalExpandNegative,
    uint8_t *pLocalContractPositive, uint8_t *pLocalExpandNegative, int stagePitch,
    int width, int height, int stageYOffset, int rangeHalf, int maxVal);

class NVEncFilterParamRtgmcShimmerRepair : public NVEncFilterParam {
public:
    RGYRtgmcShimmerRepairStage stage;
    int repairThin;
    int repairPad;
    bool processChroma;
    RGYRtgmcRepairProfile repairProfile;

    NVEncFilterParamRtgmcShimmerRepair()
        : stage(RGYRtgmcShimmerRepairStage::PreRetouch),
          repairThin(0),
          repairPad(0),
          processChroma(true),
          repairProfile() {}
    virtual ~NVEncFilterParamRtgmcShimmerRepair() {}
    virtual tstring print() const override;
};

class NVEncFilterRtgmcShimmerRepair : public NVEncFilter {
public:
    NVEncFilterRtgmcShimmerRepair();
    virtual ~NVEncFilterRtgmcShimmerRepair();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

    RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcShimmerRepair> &prm);
    RGY_ERR buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcShimmerRepair> &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        const NVEncFilterParamRtgmcShimmerRepair &prm,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

private:
    RGY_ERR launchRtgmcShimmerRepairFused(
        RGYFrameInfo *pOutputFrame,
        RGYFrameInfo *pCorrectionDeltaFrame,
        RGYFrameInfo *pPositiveCorrectionGateFrame,
        RGYFrameInfo *pNegativeCorrectionGateFrame,
        const RGYFrameInfo *pInputFrame,
        const RGYFrameInfo *pRefFrame,
        const NVEncFilterParamRtgmcShimmerRepair &prm,
        int iplane, cudaStream_t stream);
    RGY_ERR launchRtgmcShimmerRepairApply(
        RGYFrameInfo *pOutputFrame,
        const RGYFrameInfo *pInputFrame,
        const RGYFrameInfo *pRefFrame,
        const NVEncFilterParamRtgmcShimmerRepair &prm,
        int iplane, cudaStream_t stream);
    RGY_ERR launchRtgmcShimmerRepairStaged(
        RGYFrameInfo *pOutputFrame,
        RGYFrameInfo *pCorrectionDeltaFrame,
        RGYFrameInfo *pPositiveCorrectionGateFrame,
        RGYFrameInfo *pNegativeCorrectionGateFrame,
        const RGYFrameInfo *pInputFrame,
        const RGYFrameInfo *pRefFrame,
        int iplane, cudaStream_t stream);

protected:
    std::string m_buildOptions;
    std::ofstream m_lumaDump;
    std::string m_lumaDumpPath;
    std::string m_lumaDumpStage;
    std::string m_lumaDumpTarget;
    int m_lumaDumpMaxFrames;
    int m_lumaDumpFrameCount;
    bool m_lumaDumpEnabled;
    bool m_lumaDumpHeaderWritten;
    bool m_lumaDumpFullYuv;
    bool m_useKernel;
    bool m_useStagedThin4Pad0;
    std::array<CUMemBuf, 4> m_stagedBuffers;
    int m_stagedPitch;
    int m_stagedHeight;

    RGY_ERR initLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcShimmerRepair &prm);
    RGY_ERR dumpLumaFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
};
