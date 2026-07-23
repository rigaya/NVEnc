// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "NVEncParam.h"

class NVEncFilterParamLensCorrection : public NVEncFilterParam {
public:
    VppLensCorrection lenscorrection;
    NVEncFilterParamLensCorrection() : lenscorrection() {};
    virtual ~NVEncFilterParamLensCorrection() {};
    virtual tstring print() const override;
};

class NVEncFilterLensCorrection : public NVEncFilter {
public:
    NVEncFilterLensCorrection();
    virtual ~NVEncFilterLensCorrection();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, float fillValue, cudaStream_t stream);
    RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
};
