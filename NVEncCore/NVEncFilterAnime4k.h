// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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
#ifndef __NVENC_FILTER_ANIME4K_H__
#define __NVENC_FILTER_ANIME4K_H__

#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "rgy_prm.h"
#include <memory>

class NVEncFilterResize; // opt-in end-of-chain resize sub-filter (out_res=/resize=)

class NVEncFilterParamAnime4k : public NVEncFilterParam {
public:
    VppAnime4k anime4k;
    int sar[2] = { 0, 0 };  // input SAR (resolves a negative out_res= DAR-correctly)
    NVEncFilterParamAnime4k() : anime4k() {};
    virtual ~NVEncFilterParamAnime4k() {};
    virtual tstring print() const override;
};

// CUDA port of the GLSL Anime4K --vpp-anime4k-shader filter (bloc97 Anime4K v3.2, MIT).
// Increment 1 implements the base upscale chain (ani4k_original / ani4k_deblur):
// a 2x luma upscale via the Sobel + polynomial-refinement + directional-apply
// passes, plus geometric chroma resize. The darken_hq / thin_hq / dog* post-
// process modes and the joint-bilateral chroma path are added incrementally.
class NVEncFilterAnime4k : public NVEncFilter {
public:
    NVEncFilterAnime4k();
    virtual ~NVEncFilterAnime4k();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    // base Anime4K luma chain (sobel_x/y -> refine_x/y -> apply) on the Y plane.
    RGY_ERR runBaseChainY(RGYFrameInfo *pOutY, const RGYFrameInfo *pInY, cudaStream_t stream);
    // DoG alternative upscaler (dog_sharpen 1x / dog 2x).
    RGY_ERR runDogChain(RGYFrameInfo *pOutY, const RGYFrameInfo *pInY, cudaStream_t stream);
    // dtd composite (darken 1.8 -> thin 0.4 -> deblur 0.5, 2x).
    RGY_ERR runDtdChain(RGYFrameInfo *pOutY, const RGYFrameInfo *pInY, cudaStream_t stream);
    // geometric chroma resize of one chroma plane to the output chroma dims.
    RGY_ERR runChromaResize(RGYFrameInfo *pOutC, const RGYFrameInfo *pInC, cudaStream_t stream);
    // luma-guided joint-bilateral chroma upscale of one chroma plane.
    RGY_ERR runChromaJoint(RGYFrameInfo *pOutC, const RGYFrameInfo *pInC, const RGYFrameInfo *pSrcLumaY, cudaStream_t stream);
    // prefilter denoise of the input luma into m_prefilterPlane (before the base chain).
    RGY_ERR runPrefilterDenoise(const RGYFrameInfo *pInY, cudaStream_t stream);
    // post-process Y passes (operate on the output luma plane in place).
    RGY_ERR runDarkenY(RGYFrameInfo *pOutY, cudaStream_t stream);
    RGY_ERR runThinY(RGYFrameInfo *pOutY, cudaStream_t stream);
    RGY_ERR runDenoiseY(RGYFrameInfo *pOutY, cudaStream_t stream);
    RGY_ERR runClampHighlightsY(RGYFrameInfo *pOutY, const RGYFrameInfo *pSrcY, cudaStream_t stream);
    RGY_ERR runAntiringY(RGYFrameInfo *pOutY, const RGYFrameInfo *pSrcY, cudaStream_t stream);

    VppAnime4kMode m_mode;
    bool  m_isDogMode;
    int   m_inW, m_inH;
    int   m_outW, m_outH;
    int   m_scale;          // 1 or 2
    float m_strength;       // refine strength
    int   m_chromaMode;     // 0=spline36,1=bilinear,2=bicubic,3=lanczos3
    bool  m_chromaJoint;    // chroma_resize=joint (luma-guided)
    bool  m_doChroma;
    bool  m_doPrefilter;
    VppAnime4kDenoise m_prefilterDenoise;
    // post-process params (resolved from VppAnime4k at init).
    bool  m_doDarken;
    float m_darkenSigma;
    int   m_darkenRadius;
    bool  m_doThin;
    float m_thinSigma, m_thinRelstr;
    int   m_thinRadius;
    VppAnime4kDenoise m_denoise;
    float m_denoiseIntensity, m_denoiseSpatial, m_denoiseCurve, m_denoiseHistReg;
    bool  m_clampHighlights;
    float m_antiring;

    std::unique_ptr<CUMemBuf> m_scratchA;  // float4 ping-pong, outW*outH
    std::unique_ptr<CUMemBuf> m_scratchB;
    std::unique_ptr<CUMemBuf> m_clampStatsH; // float, src res (clamp_highlights horizontal max)
    std::unique_ptr<CUMemBuf> m_clampStats;  // float, src res (clamp_highlights 2d max)
    std::unique_ptr<CUMemBuf> m_chromaLumaLowres; // joint chroma: src-luma box-down to chroma res
    int m_chromaLowW, m_chromaLowH;
    std::unique_ptr<CUMemBuf> m_prefilterPlane;   // denoised input luma (input res)
    std::unique_ptr<CUMemBuf> m_prefilterRef;     // float4 ref for prefilter denoise (input res)
    std::unique_ptr<CUMemBuf> m_dtdSrcLuma;       // dtd 1x writable luma scratch (input res)

    std::unique_ptr<NVEncFilterResize> m_postResize; // out_res= end-of-chain resize
};

#endif //__NVENC_FILTER_ANIME4K_H__
