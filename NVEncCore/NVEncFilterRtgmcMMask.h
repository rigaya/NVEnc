#pragma once

#include <string>

#include "NVEncFilter.h"
#include "NVEncFilterDegrainMV.h"

class NVEncFilterParamRtgmcMMask : public NVEncFilterParam {
public:
    int kind;
    int time;
    double ml;
    double gamma;

    NVEncFilterParamRtgmcMMask();
    virtual ~NVEncFilterParamRtgmcMMask() {}
    virtual tstring print() const override;
};

class NVEncFilterRtgmcMMask : public NVEncFilter {
public:
    NVEncFilterRtgmcMMask();
    virtual ~NVEncFilterRtgmcMMask();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    RGY_ERR run_filter(const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame, const RGYDegrainAnalyzeResult &analyzeResult,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcMMask> &prm);
    RGY_ERR buildKernel(const std::shared_ptr<NVEncFilterParamRtgmcMMask> &prm);
    RGY_ERR checkAnalyzeResult(const RGYDegrainAnalyzeResult &analyzeResult, const RGYFrameInfo *pSourceFrame);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame,
        const RGYDegrainAnalyzeResult &analyzeResult, const NVEncFilterParamRtgmcMMask &prm,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

    std::string m_buildOptions;
    bool m_useKernel;
};
