// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamHQDering : public NVEncFilterParam {
public:
    VppDering dering;

    NVEncFilterParamHQDering() : dering() {};
    virtual ~NVEncFilterParamHQDering() {};
    virtual tstring print() const override;
};

class NVEncFilterHQDering : public NVEncFilter {
public:
    NVEncFilterHQDering();
    virtual ~NVEncFilterHQDering();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamHQDering> prm);
    RGY_ERR allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label);

    std::unique_ptr<CUFrameBuf> m_edgeMask;
    std::unique_ptr<CUFrameBuf> m_ringMask;
    std::unique_ptr<CUFrameBuf> m_morphTmp;
    std::unique_ptr<CUFrameBuf> m_hBlurred;
    std::unique_ptr<CUFrameBuf> m_blurred;
};
