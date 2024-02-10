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
#include <numeric>
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterNvvfx.h"
#include "NVEncFilterParam.h"
#include "rgy_filesystem.h"

char *g_nvVFXSDKPath = nullptr;

NVEncFilterNvvfxEffect::NVEncFilterNvvfxEffect() :
#if ENABLE_NVVFX
    m_effect(unique_nvvfx_handle(nullptr, NvVFX_DestroyEffect)),
    m_srcImg(),
    m_dstImg(),
#endif
    m_effectName(),
    m_maxWidth(std::numeric_limits<decltype(m_maxWidth)>::max()),
    m_maxHeight(std::numeric_limits<decltype(m_maxHeight)>::max()),
    m_state(),
    m_stateArray(),
    m_stateSizeInBytes(0) {

}

NVEncFilterNvvfxEffect::~NVEncFilterNvvfxEffect() {
    close();
}

void NVEncFilterNvvfxEffect::close() {
#if ENABLE_NVVFX
    m_effect.reset();
#endif
}

RGY_ERR NVEncFilterNvvfxEffect::initEffect(const tstring& modelDir) {
    AddMessage(RGY_LOG_DEBUG, _T("initEffect %s.\n"), m_effectName.c_str());
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters are not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    NvVFX_Handle effHandle = nullptr;
    auto err = err_to_rgy(NvVFX_CreateEffect(m_effectName.c_str(), &effHandle));
    if (err != RGY_ERR_NONE) {
        if (err == RGY_ERR_NVCV_LIBRARY) {
            // エラーチェック
            AddMessage(RGY_LOG_ERROR, _T("Failed load library for nvvfx.\n"));
            TCHAR path[8192] = { 0 };
            GetEnvironmentVariable(_T("NV_VIDEO_EFFECTS_PATH"), path, MAX_PATH);
            tstring dllPath;
            if (g_nvVFXSDKPath && g_nvVFXSDKPath[0]) {
                dllPath = char_to_tstring(g_nvVFXSDKPath);
            } else if (tstring(path) == _T("USE_APP_PATH")) {
                // NV_VIDEO_EFFECTS_PATH が USE_APP_PATH だとカレントディレクトリを探すらしい
                AddMessage(RGY_LOG_WARN, _T("env NV_VIDEO_EFFECTS_PATH = USE_APP_PATH.\n"));
                dllPath = _T("NVVideoEffects.dll");
            } else {
                memset(path, 0, sizeof(path));
                GetEnvironmentVariable(_T("ProgramFiles"), path, MAX_PATH);
                dllPath = PathCombineS(path, _T("NVIDIA Corporation\\NVIDIA Video Effects\\NVVideoEffects.dll"));
            }
            if (!rgy_file_exists(dllPath)) {
                AddMessage(RGY_LOG_ERROR, _T("target dll \"%s\" does not exist.\n"), dllPath.c_str());
                AddMessage(RGY_LOG_ERROR, _T("Please make sure you have downloaded and installed Video Effect models and runtime dependencies.\n"));
            } else {
                HMODULE nvEffectsDLLHandle = nullptr;
                if ((nvEffectsDLLHandle = RGY_LOAD_LIBRARY(dllPath.c_str())) == nullptr) {
                    AddMessage(RGY_LOG_ERROR, _T("target dll \"%s\" exists, but cannot be loaded.\n"), dllPath.c_str());
#if defined(_WIN32) || defined(_WIN64)
                    AddMessage(RGY_LOG_ERROR, _T("Please try installing VC runtime and try again.\n"));
#endif
                } else {
                    AddMessage(RGY_LOG_ERROR, _T("Unknwon error: target dll \"%s\" exists, and can be loaded, but error is caused.\n"));
                    RGY_FREE_LIBRARY(nvEffectsDLLHandle);
                    nvEffectsDLLHandle = nullptr;
                }
            }
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create effect %s: %s.\n"), m_effectName.c_str(), get_err_mes(err));
        }
        return RGY_ERR_INVALID_PARAM;
    }
    m_effect = unique_nvvfx_handle(effHandle, NvVFX_DestroyEffect);
    if (!modelDir.empty()) {
        AddMessage(RGY_LOG_DEBUG, _T("Set model dir \"%s\".\n"), modelDir.c_str());
        std::string model_dir = tchar_to_string(modelDir);
        err = err_to_rgy(NvVFX_SetString(m_effect.get(), NVVFX_MODEL_DIRECTORY, model_dir.c_str()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set model dir to \"%s\": %s.\n"), char_to_tstring(model_dir).c_str(), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    return RGY_ERR_NONE;
#endif
}

RGY_ERR NVEncFilterNvvfxEffect::checkParam(const NVEncFilterParam *param) {
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxEffect::setParam(const NVEncFilterParam *param) {
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxEffect::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters are not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<NVEncFilterParamNvvfx*>(pParam.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->compute_capability.first < 7) {
        AddMessage(RGY_LOG_ERROR, _T("NVVFX filters require Turing GPUs (CC:7.0) or later: current CC %d.%d.\n"), prm->compute_capability.first, prm->compute_capability.second);
        return RGY_ERR_UNSUPPORTED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("GPU CC: %d.%d.\n"),
        prm->compute_capability.first, prm->compute_capability.second);

    auto err = initEffect(prm->modelDir);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    if (false) {
        uint32_t maxInputWidth = 0;
        err = err_to_rgy(NvVFX_GetU32(m_effect.get(), NVVFX_MAX_INPUT_WIDTH, &maxInputWidth));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get max input width: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        if ((int)maxInputWidth < pParam->frameIn.width) {
            AddMessage(RGY_LOG_ERROR, _T("%s supports up to max width of %d, but input video has width of %d.\n"),
                m_sFilterName.c_str(),
                maxInputWidth, pParam->frameIn.width);
            return RGY_ERR_INVALID_PARAM;
        }

        uint32_t maxInputHeight = 0;
        err = err_to_rgy(NvVFX_GetU32(m_effect.get(), NVVFX_MAX_INPUT_HEIGHT, &maxInputHeight));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get max input height: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        if ((int)maxInputHeight < pParam->frameIn.height) {
            AddMessage(RGY_LOG_ERROR, _T("%s supports up to max height of %d, but input video has height of %d.\n"),
                m_sFilterName.c_str(),
                maxInputHeight, pParam->frameIn.height);
            return RGY_ERR_INVALID_PARAM;
        }
    } else {
        if (m_maxHeight < pParam->frameIn.height) {
            AddMessage(RGY_LOG_ERROR, _T("%s supports up to max height of %d, but input video has height of %d.\n"),
                m_sFilterName.c_str(),
                m_maxHeight, pParam->frameIn.height);
            return RGY_ERR_INVALID_PARAM;
        }
    }

    // C++コンストラクタでのメモリ確保ではなく、デフォルトコンストラクタでNvCVImageを作成した後、
    // NvCVImage_Allocでメモリ確保すること
    // そうしないと128で割り切れないwidthの場合にエラーが出る
    AddMessage(RGY_LOG_DEBUG, _T("Create nvvfx input image %dx%d.\n"), pParam->frameIn.width, pParam->frameIn.height);
    m_srcImg = std::make_unique<NvCVImage>();
    err = err_to_rgy(NvCVImage_Alloc(m_srcImg.get(), pParam->frameIn.width, pParam->frameIn.height,
        NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate nvvfx input image %dx%d: %s.\n"),
            pParam->frameIn.width, pParam->frameIn.height, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    AddMessage(RGY_LOG_DEBUG, _T("Create nvvfx output image %dx%d.\n"), pParam->frameOut.width, pParam->frameOut.height);
    m_dstImg = std::make_unique<NvCVImage>();
    err = err_to_rgy(NvCVImage_Alloc(m_dstImg.get(), pParam->frameOut.width, pParam->frameOut.height,
        NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate nvvfx output image %dx%d: %s.\n"),
            pParam->frameOut.width, pParam->frameOut.height, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    if (prm->vuiInfo.matrix == RGY_MATRIX_UNSPECIFIED) {
        prm->vuiInfo.matrix = (CspMatrix)COLOR_VALUE_AUTO_RESOLUTION;
    }
    prm->vuiInfo.apply_auto(prm->vuiInfo, pParam->frameIn.height);
    {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = RGY_CSP_BGR_F32;
        paramCrop->matrix = prm->vuiInfo.matrix;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.deivce_mem = true;
        paramCrop->frameOut.deivce_mem = true;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pPrintMes);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_srcCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcCrop->GetInputMessage().c_str());
    }
    {
        AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameIn.csp = RGY_CSP_BGR_F32;
        paramCrop->matrix = prm->vuiInfo.matrix;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.deivce_mem = true;
        paramCrop->frameOut.deivce_mem = true;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pPrintMes);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_dstCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstCrop->GetInputMessage().c_str());
    }

    AddMessage(RGY_LOG_DEBUG, _T("NVVFX_INPUT_IMAGE: set input image.\n"));
    err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_INPUT_IMAGE,  m_srcImg.get()));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set input image: %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    AddMessage(RGY_LOG_DEBUG, _T("NVVFX_OUTPUT_IMAGE: set output image.\n"));
    err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_OUTPUT_IMAGE, m_dstImg.get()));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set output image %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    err = checkParam(pParam.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }

    err = setParam(pParam.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }

    if (m_effectName == NVVFX_FX_DENOISING) {
        m_stateSizeInBytes = 0;
        err = err_to_rgy(NvVFX_GetU32(m_effect.get(), NVVFX_STATE_SIZE, &m_stateSizeInBytes));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get state size: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        m_state = std::make_unique<CUMemBuf>(m_stateSizeInBytes);
        err = err_to_rgy(m_state->alloc());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate buffer for state: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        cudaMemset(m_state->ptr, 0, m_stateSizeInBytes);
        m_stateArray[0] = m_state->ptr;
        err = err_to_rgy(NvVFX_SetObject(m_effect.get(), NVVFX_STATE, (void*)m_stateArray.data()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set state array: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    AddMessage(RGY_LOG_DEBUG, _T("Loading effect...\n"));
    err = err_to_rgy(NvVFX_Load(m_effect.get()));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to load effect: %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pParam->frameOut, 1);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    tstring info = m_sFilterName + _T(": ");
    if (m_srcCrop) {
        info += m_srcCrop->GetInputMessage() + _T("\n");
    }
    tstring nameBlank(m_sFilterName.length() + _tcslen(_T(": ")), _T(' '));
    info += tstring(INFO_INDENT) + nameBlank + pParam->print();
    if (m_dstCrop) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + nameBlank + m_dstCrop->GetInputMessage();
    }
    setFilterInfo(info);
    m_pParam = pParam;
    return sts;
#endif
}

RGY_ERR NVEncFilterNvvfxEffect::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (true) {
        RGYFrameInfo srcImgInfo = m_srcCrop->GetFilterParam()->frameOut;
        srcImgInfo.ptr = (uint8_t *)m_srcImg->pixels;
        srcImgInfo.pitch = m_srcImg->pitch;

        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { &srcImgInfo };
        RGYFrameInfo cropInput = *pInputFrame;
        auto sts_filter = m_srcCrop->filter(&cropInput, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, stream);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
    } else { // デバッグ用
        const auto planeY = getPlane(pInputFrame, RGY_PLANE_Y);
        const auto planeU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeV = getPlane(pInputFrame, RGY_PLANE_V);
        const int bitdepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
        const int pix_byte = bitdepth / 8;
        const auto nvcvPixFmt = RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420 ? NVCV_YUV420 : NVCV_YUV444;
        const auto nvcvElemType = bitdepth > 8 ? NVCV_U16 : NVCV_U8;
        sts = err_to_rgy(NvCVImage_TransferFromYUV(
            planeY.ptr, pix_byte, planeY.pitch,
            planeU.ptr, planeV.ptr, pix_byte, planeU.pitch,
            nvcvPixFmt, nvcvElemType, NVCV_PLANAR, NVCV_GPU,
            m_srcImg.get(), nullptr, 1.0f / (float)((1 << bitdepth) - 1), stream, nullptr));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to run NvCVImage_TransferFromYUV: %s.\n"), get_err_mes(sts));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    if (true) {
        sts = err_to_rgy(NvVFX_Run(m_effect.get(), 0));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to run filter: %s.\n"), get_err_mes(sts));
            return RGY_ERR_INVALID_PARAM;
        }
    } else { // デバッグ用
        sts = err_to_rgy(NvCVImage_Transfer(m_srcImg.get(), m_dstImg.get(), 1.f, stream, nullptr));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to run NvCVImage_Transfer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    {
        RGYFrameInfo dstImgInfo = m_dstCrop->GetFilterParam()->frameIn;
        dstImgInfo.ptr = (uint8_t *)m_dstImg->pixels;
        dstImgInfo.pitch = m_dstImg->pitch;
        RGYFrameInfo *outInfo[1] = { &dstImgInfo };
        auto sts_filter = m_dstCrop->filter(&dstImgInfo, ppOutputFrames, pOutputFrameNum, stream);
        if (outInfo[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
    }
    return RGY_ERR_NONE;
#endif
}

tstring NVEncFilterParamNvvfxDenoise::print() const {
    return nvvfxDenoise.print();
}

NVEncFilterNvvfxDenoise::NVEncFilterNvvfxDenoise() {
    m_sFilterName = _T("nvvfx-denoise");
    m_maxHeight = 1080;
#if ENABLE_NVVFX
    m_effectName = NVVFX_FX_DENOISING;
#endif
}

NVEncFilterNvvfxDenoise::~NVEncFilterNvvfxDenoise() {
    close();
}

RGY_ERR NVEncFilterNvvfxDenoise::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxDenoise*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxDenoise.strength < 0.0f || 1.0f < prm->nvvfxDenoise.strength) {
        AddMessage(RGY_LOG_ERROR, _T("strength should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxDenoise::setParam(const NVEncFilterParam *param) {
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxDenoise*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = err_to_rgy(NvVFX_SetF32(m_effect.get(), NVVFX_STRENGTH, prm->nvvfxDenoise.strength));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %.2f: %s.\n"), NVVFX_STRENGTH, prm->nvvfxDenoise.strength, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
#endif
}

tstring NVEncFilterParamNvvfxArtifactReduction::print() const {
    return nvvfxArtifactReduction.print();
}

NVEncFilterNvvfxArtifactReduction::NVEncFilterNvvfxArtifactReduction() {
    m_sFilterName = _T("nvvfx-artifact-reduction");
    m_maxHeight = 1080;
#if ENABLE_NVVFX
    m_effectName = NVVFX_FX_ARTIFACT_REDUCTION;
#endif
}

NVEncFilterNvvfxArtifactReduction::~NVEncFilterNvvfxArtifactReduction() {
    close();
}

RGY_ERR NVEncFilterNvvfxArtifactReduction::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxArtifactReduction*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxArtifactReduction.mode != 0 && prm->nvvfxArtifactReduction.mode != 1) {
        AddMessage(RGY_LOG_ERROR, _T("mode should be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxArtifactReduction::setParam(const NVEncFilterParam *param) {
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxArtifactReduction*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFX_MODE, prm->nvvfxArtifactReduction.mode));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %d: %s.\n"), NVVFX_MODE, prm->nvvfxArtifactReduction.mode, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
#endif
}

tstring NVEncFilterParamNvvfxSuperRes::print() const {
    return nvvfxSuperRes.print();
}

NVEncFilterNvvfxSuperRes::NVEncFilterNvvfxSuperRes() {
    m_sFilterName = _T("nvvfx-superres");
    m_maxHeight = 2160;
#if ENABLE_NVVFX
    m_effectName = NVVFX_FX_SUPER_RES;
#endif
}

NVEncFilterNvvfxSuperRes::~NVEncFilterNvvfxSuperRes() {
    close();
}

RGY_ERR NVEncFilterNvvfxSuperRes::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxSuperRes*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxSuperRes.strength < 0.0f || 1.0f < prm->nvvfxSuperRes.strength) {
        AddMessage(RGY_LOG_ERROR, _T("strength should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxSuperRes.mode != 0 && prm->nvvfxSuperRes.mode != 1) {
        AddMessage(RGY_LOG_ERROR, _T("mode should be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxSuperRes::setParam(const NVEncFilterParam *param) {
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxSuperRes*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFX_MODE, prm->nvvfxSuperRes.mode));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %d: %s.\n"), NVVFX_MODE, prm->nvvfxSuperRes.mode, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    err = err_to_rgy(NvVFX_SetF32(m_effect.get(), NVVFX_STRENGTH, prm->nvvfxSuperRes.strength));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %.2f: %s.\n"), NVVFX_STRENGTH, prm->nvvfxSuperRes.strength, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
#endif
}

tstring NVEncFilterParamNvvfxUpScaler::print() const {
    return nvvfxUpscaler.print();
}

NVEncFilterNvvfxUpScaler::NVEncFilterNvvfxUpScaler() {
    m_sFilterName = _T("nvvfx-upscaler");
#if ENABLE_NVVFX
    m_effectName = NVVFX_FX_SR_UPSCALE;
#endif
}

NVEncFilterNvvfxUpScaler::~NVEncFilterNvvfxUpScaler() {
    close();
}

RGY_ERR NVEncFilterNvvfxUpScaler::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxUpScaler*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxUpscaler.strength < 0.0f || 1.0f < prm->nvvfxUpscaler.strength) {
        AddMessage(RGY_LOG_ERROR, _T("strength should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxUpScaler::setParam(const NVEncFilterParam *param) {
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxUpScaler*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = err_to_rgy(NvVFX_SetF32(m_effect.get(), NVVFX_STRENGTH, prm->nvvfxUpscaler.strength));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %.2f: %s.\n"), NVVFX_STRENGTH, prm->nvvfxUpscaler.strength, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
#endif
}
