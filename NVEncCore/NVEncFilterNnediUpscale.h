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
#ifndef __NVENC_FILTER_NNEDI_UPSCALE_H__
#define __NVENC_FILTER_NNEDI_UPSCALE_H__

#include "NVEncFilter.h"
#include "NVEncFilterNnedi.h"
#include "rgy_prm.h"

class NVEncFilterTransform;

class NVEncFilterParamNnediUpscale : public NVEncFilterParam {
public:
    VppNnediUpscale nnediUpscale;
    std::pair<int, int> compute_capability;
    HMODULE hModule;
    rgy_rational<int> timebase;

    NVEncFilterParamNnediUpscale();
    virtual ~NVEncFilterParamNnediUpscale() {};
    virtual tstring print() const override { return nnediUpscale.print(); };
};

// 既存NNEDIの縦2倍処理を両軸へ順に適用するエッジ指向2倍拡大。
//
//   縦2倍 -> 転置 -> 縦2倍 -> 転置で戻す
//
// 既存フィルタを組み合わせ、ネットワーク・重み・調整値には手を加えない。
class NVEncFilterNnediUpscale : public NVEncFilter {
public:
    NVEncFilterNnediUpscale();
    virtual ~NVEncFilterNnediUpscale();
    // 内部補間は固定フィールドで現在フレームだけを拡大し、出力レートを変えない。
    virtual bool dependsOnHistory(const RGYParamVpp&) const override { return false; }
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
        int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR halfPixel(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream);
    RGY_ERR makeNnedi(std::unique_ptr<NVEncFilterNnedi>& target, const RGYFrameInfo& frameIn,
        const NVEncFilterParamNnediUpscale *prm, const TCHAR *label);
    RGY_ERR makeTranspose(std::unique_ptr<NVEncFilterTransform>& target, const RGYFrameInfo& frameIn,
        const NVEncFilterParamNnediUpscale *prm, const TCHAR *label);

    // 各軸に適用する縦2倍処理。
    std::unique_ptr<NVEncFilterNnedi> m_passV;
    std::unique_ptr<NVEncFilterNnedi> m_passH;
    std::unique_ptr<NVEncFilterTransform> m_transposeToH;
    std::unique_ptr<NVEncFilterTransform> m_transposeBack;
};

#endif //__NVENC_FILTER_NNEDI_UPSCALE_H__
