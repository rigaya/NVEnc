// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "rgy_filter_nnedi_field.h"
#include "rgy_filter_nnedi_weights.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct RGYNnediParam {
    static constexpr uint32_t WEIGHTS_FILE_SIZE = 13574928u;

    bool enable;
    std::array<bool, 4> processPlane;
    VppNnediField field;
    VppNnediNSize nsize;
    int nns;
    VppNnediQuality quality;
    int prescreen;
    VppNnediErrorType errortype;
    int clamp;
    bool doubleHeight;
    tstring weightfile;

    RGYNnediParam();
    bool operator==(const RGYNnediParam& x) const;
    bool operator!=(const RGYNnediParam& x) const;
    tstring print() const;
};

struct RGYNnediNSizeDesc {
    int xdia;
    int ydia;
};

const RGYNnediNSizeDesc& rgy_nnedi_nsize_desc(int nsize);
int rgy_nnedi_nns_value(int nns);
int rgy_nnedi_nns_index(int nns);

class NVEncFilterParamNnedi : public NVEncFilterParam {
public:
    RGYNnediParam nnedi;
    std::pair<int, int> compute_capability;
    HMODULE hModule;
    rgy_rational<int> timebase;

    NVEncFilterParamNnedi();
    virtual ~NVEncFilterParamNnedi() {};
    virtual tstring print() const override { return nnedi.print(); };
};

class NVEncFilterNnedi : public NVEncFilter {
public:
    NVEncFilterNnedi();
    virtual ~NVEncFilterNnedi();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

    RGY_ERR validateParam(const RGYNnediParam& prm);
    std::shared_ptr<const std::vector<uint8_t>> readWeights(const tstring& weightFile, HMODULE hModule);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR initParams(const std::shared_ptr<NVEncFilterParamNnedi> prm);
    bool getInputTff(const RGYFrameInfo *frame) const;
    void setDoubleRateTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) const;
    RGY_ERR prepareFieldReference(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap, cudaStream_t stream);
    RGY_ERR classifyPixelsAndSeedOutput(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap, cudaStream_t stream);
    RGY_ERR resolveClassifiedPixels(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap, cudaStream_t stream);

    std::shared_ptr<const std::vector<uint8_t>> m_weights;
    RGYFilterNnediTransformedWeights m_transformedWeights;
    std::vector<std::unique_ptr<CUFrameBuf>> m_refBuf;
    std::unique_ptr<CUMemBuf> m_prescreenerWeightBuf;
    std::unique_ptr<CUMemBuf> m_predictorWeightBuf;
    std::vector<std::unique_ptr<CUMemBuf>> m_workNNBuf;
    std::vector<std::unique_ptr<CUMemBuf>> m_numBlocksBuf;
    int m_tileGroupsX;
    int m_tileRows;
    int m_predLocalX;
    int m_predLocalY;
    bool m_defaultTff;
};
