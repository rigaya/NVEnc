// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#pragma once

#include <vector>
#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamStab : public NVEncFilterParam {
public:
    VppStab stab;

    NVEncFilterParamStab() : stab() {};
    virtual ~NVEncFilterParamStab() {};
    virtual tstring print() const override;
};

class NVEncFilterStab : public NVEncFilter {
public:
    NVEncFilterStab();
    virtual ~NVEncFilterStab();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamStab> prm);
    RGY_ERR allocWorkBuf(std::unique_ptr<CUMemBuf>& buf, size_t bytes, const TCHAR *label);

    std::unique_ptr<CUMemBuf> m_srcReal;
    std::unique_ptr<CUMemBuf> m_curFreq;
    std::unique_ptr<CUMemBuf> m_prevFreq;
    std::unique_ptr<CUMemBuf> m_corrFreq;
    std::unique_ptr<CUMemBuf> m_corrReal;
    std::vector<float> m_corrHost;
    bool m_havePrev;
    float m_smoothShiftX;
    float m_smoothShiftY;
    bool m_haveSmoothing;
    int m_lowTrustFrames;
};
