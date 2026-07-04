// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "NVEncFilterDehalo.h"
#include "rgy_prm.h"

class NVEncFilterParamFineDehalo : public NVEncFilterParam {
public:
    VppFineDehalo finedehalo;

    NVEncFilterParamFineDehalo() : finedehalo() {};
    virtual ~NVEncFilterParamFineDehalo() {};
    virtual tstring print() const override;
};

class NVEncFilterFineDehalo : public NVEncFilter {
public:
    NVEncFilterFineDehalo();
    virtual ~NVEncFilterFineDehalo();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamFineDehalo> prm);
    RGY_ERR allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label);

    std::unique_ptr<NVEncFilterDehalo> m_dehalo;
    std::unique_ptr<CUFrameBuf> m_edges;
    std::unique_ptr<CUFrameBuf> m_strong;
    std::unique_ptr<CUFrameBuf> m_large;
    std::unique_ptr<CUFrameBuf> m_light;
    std::unique_ptr<CUFrameBuf> m_shrink;
    std::unique_ptr<CUFrameBuf> m_outside;
    std::unique_ptr<CUFrameBuf> m_morphTmp;
    std::unique_ptr<CUFrameBuf> m_shrMed;
};
