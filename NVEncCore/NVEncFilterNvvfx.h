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

#include <array>
#include "rgy_version.h"
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"

#if ENABLE_NVVFX
#pragma warning (push)
#pragma warning (disable: 4819)
#include "nvVideoEffects.h"
#pragma warning (pop)

using unique_nvvfx_handle = std::unique_ptr<std::remove_pointer<NvVFX_Handle>::type, decltype(&NvVFX_DestroyEffect)>;
#endif

class NVEncFilterNvvfxEffect : public NVEncFilter {
public:
    NVEncFilterNvvfxEffect();
    virtual ~NVEncFilterNvvfxEffect();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    RGY_ERR initEffect(const tstring& modelDir);
    virtual RGY_ERR checkParam(const NVEncFilterParam *param);
    virtual RGY_ERR setParam(const NVEncFilterParam *param);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual bool compareParam(const NVEncFilterParam *param) const = 0;
    bool compareModelDir(const tstring& modelDir) const;

#if ENABLE_NVVFX
    unique_nvvfx_handle m_effect;
    std::unique_ptr<NvCVImage> m_srcImg;
    std::unique_ptr<NvCVImage> m_dstImg;
#endif
    std::string m_effectName;
    int m_maxWidth;
    int m_maxHeight;
    std::unique_ptr<NVEncFilterCspCrop> m_srcCrop;
    std::unique_ptr<NVEncFilterCspCrop> m_dstCrop;
    std::unique_ptr<CUMemBuf> m_state;
    std::array<void *, 1> m_stateArray;
    uint32_t m_stateSizeInBytes;
};

class NVEncFilterParamNvvfx : public NVEncFilterParam {
public:
    tstring modelDir;
    std::pair<int, int> compute_capability;
    VideoVUIInfo vuiInfo;
    NVEncFilterParamNvvfx() : modelDir(), compute_capability(), vuiInfo() {};
    virtual ~NVEncFilterParamNvvfx() {};
};

class NVEncFilterParamNvvfxDenoise : public NVEncFilterParamNvvfx {
public:
    VppNvvfxDenoise nvvfxDenoise;
    NVEncFilterParamNvvfxDenoise() : nvvfxDenoise() {};
    virtual ~NVEncFilterParamNvvfxDenoise() {};
    virtual tstring print() const override;
};

class NVEncFilterParamNvvfxArtifactReduction : public NVEncFilterParamNvvfx {
public:
    VppNvvfxArtifactReduction nvvfxArtifactReduction;
    NVEncFilterParamNvvfxArtifactReduction() : nvvfxArtifactReduction() {};
    virtual ~NVEncFilterParamNvvfxArtifactReduction() {};
    virtual tstring print() const override;
};

class NVEncFilterParamNvvfxSuperRes : public NVEncFilterParamNvvfx {
public:
    VppNvvfxSuperRes nvvfxSuperRes;
    NVEncFilterParamNvvfxSuperRes() : nvvfxSuperRes() {};
    virtual ~NVEncFilterParamNvvfxSuperRes() {};
    virtual tstring print() const override;
};

class NVEncFilterParamNvvfxUpScaler : public NVEncFilterParamNvvfx {
public:
    VppNvvfxUpScaler nvvfxUpscaler;
    NVEncFilterParamNvvfxUpScaler() : nvvfxUpscaler() {};
    virtual ~NVEncFilterParamNvvfxUpScaler() {};
    virtual tstring print() const override;
};

class NVEncFilterNvvfxDenoise : public NVEncFilterNvvfxEffect {
public:
    NVEncFilterNvvfxDenoise();
    virtual ~NVEncFilterNvvfxDenoise();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setParam(const NVEncFilterParam *param) override;
    virtual bool compareParam(const NVEncFilterParam *param) const override;
};

class NVEncFilterNvvfxArtifactReduction : public NVEncFilterNvvfxEffect {
public:
    NVEncFilterNvvfxArtifactReduction();
    virtual ~NVEncFilterNvvfxArtifactReduction();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setParam(const NVEncFilterParam *param) override;
    virtual bool compareParam(const NVEncFilterParam *param) const override;
};

class NVEncFilterNvvfxSuperRes : public NVEncFilterNvvfxEffect {
public:
    NVEncFilterNvvfxSuperRes();
    virtual ~NVEncFilterNvvfxSuperRes();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setParam(const NVEncFilterParam *param) override;
    virtual bool compareParam(const NVEncFilterParam *param) const override;
};

class NVEncFilterNvvfxUpScaler : public NVEncFilterNvvfxEffect {
public:
    NVEncFilterNvvfxUpScaler();
    virtual ~NVEncFilterNvvfxUpScaler();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setParam(const NVEncFilterParam *param) override;
    virtual bool compareParam(const NVEncFilterParam *param) const override;
};
