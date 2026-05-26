#pragma once

#include <fstream>
#include <vector>

#include "NVEncFilter.h"
#include "NVEncFilterRtgmcCommon.h"
#include "NVEncFilterRtgmcSearchPrefilter.h"

enum class RGYRtgmcShimmerRepairStage {
    PreRetouch,
    PostTR2,
};

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

    RGY_ERR initLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcShimmerRepair &prm);
    RGY_ERR dumpLumaFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
};
