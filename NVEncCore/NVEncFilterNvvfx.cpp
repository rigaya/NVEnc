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
#include <algorithm>
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterNvvfx.h"
#include "NVEncFilterParam.h"
#include "rgy_filesystem.h"

char *g_nvVFXSDKPath = nullptr;

#if ENABLE_NVVFX
// Video Frame Generation (VFG) interface.
// Not every nvVideoEffects.h shipped with the SDK carries these definitions,
// so fall back to the documented names when the header does not provide them.
#ifndef NVVFX_FX_VIDEO_FRAME_GENERATION
#define NVVFX_FX_VIDEO_FRAME_GENERATION "VideoFrameGeneration"
#define NVVFXVIDEOFRAMEGENERATION_MODE       "Mode"
#define NVVFXVIDEOFRAMEGENERATION_FRAME_MULTIPLIER "FrameMultiplier"
#define NVVFXVIDEOFRAMEGENERATION_FRAME_INDEX      "FrameIndex"
#define NVVFXVIDEOFRAMEGENERATION_SHOT_CHANGE      "ShotChange"
#define NVVFXVIDEOFRAMEGENERATION_AUTOMATIC_SHOT_CHANGE_DETECTION_ENABLED "AutomaticShotChangeDetectionEnabled"
#endif // #ifndef NVVFX_FX_VIDEO_FRAME_GENERATION
#ifndef NVVFX_INPUT_WIDTH
#define NVVFX_INPUT_WIDTH  "InputWidth"
#endif
#ifndef NVVFX_INPUT_HEIGHT
#define NVVFX_INPUT_HEIGHT "InputHeight"
#endif
#endif // #if ENABLE_NVVFX

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
    AddMessage(RGY_LOG_DEBUG, _T("Clear state buf.\n"));
    m_state.reset();
#if ENABLE_NVVFX
    AddMessage(RGY_LOG_DEBUG, _T("Clear dst image buffer.\n"));
    m_dstImg.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Clear src image buffer.\n"));
    m_srcImg.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Close nvvfx effect handle.\n"));
    m_effect.reset();
#endif
    AddMessage(RGY_LOG_DEBUG, _T("Close src scp conversion filter.\n"));
    m_srcCrop.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Close dst scp conversion filter.\n"));
    m_dstCrop.reset();
}

bool NVEncFilterNvvfxEffect::compareModelDir(const tstring& modelDir) const {
    if (!m_param) return true;
    auto prm = dynamic_cast<const NVEncFilterParamNvvfx *>(m_param.get());
    if (prm->modelDir.length() == 0 && modelDir.length() == 0) return false;
    return !rgy_path_is_same(prm->modelDir, modelDir);
}

RGY_ERR NVEncFilterNvvfxEffect::initEffect(const tstring& modelDir) {
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters are not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    if (!m_effect) {
        AddMessage(RGY_LOG_DEBUG, _T("initEffect %s.\n"), m_effectName.c_str());
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
    }
    if (!modelDir.empty() && compareModelDir(modelDir)) {
        AddMessage(RGY_LOG_DEBUG, _T("Set model dir \"%s\".\n"), modelDir.c_str());
        std::string model_dir = tchar_to_string(modelDir);
        auto err = err_to_rgy(NvVFX_SetString(m_effect.get(), NVVFX_MODEL_DIRECTORY, model_dir.c_str()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set model dir to \"%s\": %s.\n"), char_to_tstring(model_dir).c_str(), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    return RGY_ERR_NONE;
#endif
}

RGY_ERR NVEncFilterNvvfxEffect::checkParam([[maybe_unused]] const NVEncFilterParam *param) {
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxEffect::setParam([[maybe_unused]] const NVEncFilterParam *param) {
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxEffect::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
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

    if (rgy_csp_has_alpha(pParam->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("nvvfx filters does not support alpha channel.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

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
                m_name.c_str(),
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
                m_name.c_str(),
                maxInputHeight, pParam->frameIn.height);
            return RGY_ERR_INVALID_PARAM;
        }
    } else {
        if (m_maxHeight < pParam->frameIn.height) {
            AddMessage(RGY_LOG_ERROR, _T("%s supports up to max height of %d, but input video has height of %d.\n"),
                m_name.c_str(),
                m_maxHeight, pParam->frameIn.height);
            return RGY_ERR_INVALID_PARAM;
        }
    }

    err = checkParam(pParam.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }

    if (!m_srcImg || (int)m_srcImg->width != pParam->frameIn.width || (int)m_srcImg->height != pParam->frameIn.height) {
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

        AddMessage(RGY_LOG_DEBUG, _T("NVVFX_INPUT_IMAGE: set input image.\n"));
        err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_INPUT_IMAGE, m_srcImg.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set input image: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    if (!m_dstImg || (int)m_dstImg->width != pParam->frameOut.width || (int)m_dstImg->height != pParam->frameOut.height) {
        // C++コンストラクタでのメモリ確保ではなく、デフォルトコンストラクタでNvCVImageを作成した後、
        // NvCVImage_Allocでメモリ確保すること
        // そうしないと128で割り切れないwidthの場合にエラーが出る
        AddMessage(RGY_LOG_DEBUG, _T("Create nvvfx output image %dx%d.\n"), pParam->frameOut.width, pParam->frameOut.height);
        m_dstImg = std::make_unique<NvCVImage>();
        err = err_to_rgy(NvCVImage_Alloc(m_dstImg.get(), pParam->frameOut.width, pParam->frameOut.height,
            NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate nvvfx output image %dx%d: %s.\n"),
                pParam->frameOut.width, pParam->frameOut.height, get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }

        AddMessage(RGY_LOG_DEBUG, _T("NVVFX_OUTPUT_IMAGE: set output image.\n"));
        err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_OUTPUT_IMAGE, m_dstImg.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set output image %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    if (prm->vuiInfo.matrix == RGY_MATRIX_UNSPECIFIED) {
        prm->vuiInfo.matrix = (CspMatrix)COLOR_VALUE_AUTO_RESOLUTION;
    }
    prm->vuiInfo.apply_auto(prm->vuiInfo, pParam->frameIn.height);
    if (!m_srcCrop
        || m_srcCrop->GetFilterParam()->frameIn.width  != pParam->frameIn.width
        || m_srcCrop->GetFilterParam()->frameIn.height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = RGY_CSP_BGR_F32;
        paramCrop->matrix = prm->vuiInfo.matrix;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_srcCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcCrop->GetInputMessage().c_str());
    }
    if (!m_dstCrop
        || m_dstCrop->GetFilterParam()->frameOut.width  != pParam->frameOut.width
        || m_dstCrop->GetFilterParam()->frameOut.height != pParam->frameOut.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameIn.csp = RGY_CSP_BGR_F32;
        paramCrop->matrix = prm->vuiInfo.matrix;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_dstCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstCrop->GetInputMessage().c_str());
    }

    err = setParam(pParam.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }

    if (m_effectName == NVVFX_FX_DENOISING) {
        uint32_t stateSizeInBytes = 0;
        err = err_to_rgy(NvVFX_GetU32(m_effect.get(), NVVFX_STATE_SIZE, &stateSizeInBytes));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get state size: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        if (m_stateSizeInBytes != stateSizeInBytes) {
            m_state = std::make_unique<CUMemBuf>(stateSizeInBytes);
            err = m_state->alloc();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to allocate buffer for state: %s.\n"), get_err_mes(err));
                return RGY_ERR_INVALID_PARAM;
            }
            cudaMemset(m_state->ptr, 0, stateSizeInBytes);
            m_stateArray[0] = m_state->ptr;
            err = err_to_rgy(NvVFX_SetObject(m_effect.get(), NVVFX_STATE, (void*)m_stateArray.data()));
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to set state array: %s.\n"), get_err_mes(err));
                return RGY_ERR_INVALID_PARAM;
            }
            m_stateSizeInBytes = stateSizeInBytes;
        }
    }
    if (compareModelDir(prm->modelDir) || compareParam(pParam.get())) {
        AddMessage(RGY_LOG_DEBUG, _T("Loading effect...\n"));
        err = err_to_rgy(NvVFX_Load(m_effect.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to load effect: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    sts = AllocFrameBuf(pParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    tstring info = m_name + _T(": ");
    if (m_srcCrop) {
        info += m_srcCrop->GetInputMessage() + _T("\n");
    }
    tstring nameBlank(m_name.length() + _tcslen(_T(": ")), _T(' '));
    info += tstring(INFO_INDENT) + nameBlank + pParam->print();
    if (m_dstCrop) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + nameBlank + m_dstCrop->GetInputMessage();
    }
    setFilterInfo(info);
    m_param = pParam;
    return sts;
#endif
}

RGY_ERR NVEncFilterNvvfxEffect::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (true) {
        RGYFrameInfo srcImgInfo = m_srcCrop->GetFilterParam()->frameOut;
        srcImgInfo.singleAlloc = true;
        srcImgInfo.ptr[0] = (uint8_t *)m_srcImg->pixels;
        srcImgInfo.pitch[0] = m_srcImg->pitch;

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
            planeY.ptr[0], pix_byte, planeY.pitch[0],
            planeU.ptr[0], planeV.ptr[0], pix_byte, planeU.pitch[0],
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
        dstImgInfo.singleAlloc = true;
        dstImgInfo.ptr[0] = (uint8_t *)m_dstImg->pixels;
        dstImgInfo.pitch[0] = m_dstImg->pitch;
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
    m_name = _T("nvvfx-denoise");
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

bool NVEncFilterNvvfxDenoise::compareParam(const NVEncFilterParam *param) const {
    if (!m_param) return true;
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxDenoise *>(m_param.get());
    if (!prm) return true;
    auto target = dynamic_cast<const NVEncFilterParamNvvfxDenoise *>(param);
    if (!target) return true;
    return prm->nvvfxDenoise != target->nvvfxDenoise;
};

tstring NVEncFilterParamNvvfxArtifactReduction::print() const {
    return nvvfxArtifactReduction.print();
}

NVEncFilterNvvfxArtifactReduction::NVEncFilterNvvfxArtifactReduction() {
    m_name = _T("nvvfx-artifact-reduction");
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

bool NVEncFilterNvvfxArtifactReduction::compareParam(const NVEncFilterParam *param) const {
    if (!m_param) return true;
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxArtifactReduction *>(m_param.get());
    if (!prm) return true;
    auto target = dynamic_cast<const NVEncFilterParamNvvfxArtifactReduction *>(param);
    if (!target) return true;
    return prm->nvvfxArtifactReduction != target->nvvfxArtifactReduction;
};

tstring NVEncFilterParamNvvfxUpScaler::print() const {
    return nvvfxUpscaler.print();
}

NVEncFilterNvvfxUpScaler::NVEncFilterNvvfxUpScaler() {
    m_name = _T("nvvfx-upscaler");
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

bool NVEncFilterNvvfxUpScaler::compareParam(const NVEncFilterParam *param) const {
    if (!m_param) return true;
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxUpScaler *>(m_param.get());
    if (!prm) return true;
    auto target = dynamic_cast<const NVEncFilterParamNvvfxUpScaler *>(param);
    if (!target) return true;
    return prm->nvvfxUpscaler != target->nvvfxUpscaler;
};

tstring NVEncFilterParamNvvfxFrameGen::print() const {
    return nvvfxFrameGen.print();
}

NVEncFilterNvvfxFrameGeneration::NVEncFilterNvvfxFrameGeneration() :
#if ENABLE_NVVFX
    m_prevImg(),
#endif
    m_outFrameBuf(),
    m_targetFps(),
    m_prevTimestamp(-1),
    m_inputFrames(0) {
    m_name = _T("nvvfx-framegen");
    // No m_maxHeight here on purpose: this filter overrides init(), so it never reaches the base
    // class check that reads m_maxHeight. VFG's real input limit depends on available GPU memory.
#if ENABLE_NVVFX
    m_effectName = NVVFX_FX_VIDEO_FRAME_GENERATION;
#endif
}

NVEncFilterNvvfxFrameGeneration::~NVEncFilterNvvfxFrameGeneration() {
    close();
}

int NVEncFilterNvvfxFrameGeneration::requiredOutputFrames() const {
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxFrameGen *>(m_param.get());
    return (prm) ? prm->nvvfxFrameGen.multiplier : 0;
}

void NVEncFilterNvvfxFrameGeneration::close() {
    m_outFrameBuf.clear();
#if ENABLE_NVVFX
    m_prevImg.reset();
#endif
    NVEncFilterNvvfxEffect::close();
}

RGY_ERR NVEncFilterNvvfxFrameGeneration::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxFrameGen*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxFrameGen.mode < 0 || 2 < prm->nvvfxFrameGen.mode) {
        AddMessage(RGY_LOG_ERROR, _T("mode should be 0 (low), 1 (medium) or 2 (high).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nvvfxFrameGen.multiplier < 2 || 8 < prm->nvvfxFrameGen.multiplier) {
        AddMessage(RGY_LOG_ERROR, _T("multiplier should be 2 - 8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // the model works on 640x360 or larger input
    const int shortSide = std::min(prm->frameIn.width, prm->frameIn.height);
    const int longSide  = std::max(prm->frameIn.width, prm->frameIn.height);
    if (longSide < 640 || shortSide < 360) {
        AddMessage(RGY_LOG_ERROR, _T("nvvfx-framegen requires at least 640x360 input, but input is %dx%d.\n"),
            prm->frameIn.width, prm->frameIn.height);
        return RGY_ERR_INVALID_PARAM;
    }
    // VFG is only available on Ada (and later) GPUs on Windows.
    if (prm->compute_capability.first < 8
        || (prm->compute_capability.first == 8 && prm->compute_capability.second < 9)) {
        AddMessage(RGY_LOG_ERROR, _T("nvvfx-framegen requires Ada GPUs (CC:8.9) or later: current CC %d.%d.\n"),
            prm->compute_capability.first, prm->compute_capability.second);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxFrameGeneration::setParam(const NVEncFilterParam *param) {
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxFrameGen*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // the model mode has to be set before loading the effect
    auto err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFXVIDEOFRAMEGENERATION_MODE, prm->nvvfxFrameGen.mode));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %d: %s.\n"), NVVFXVIDEOFRAMEGENERATION_MODE, prm->nvvfxFrameGen.mode, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    // the feature is enabled by default, so only touch it when it has to be turned off
    if (!prm->nvvfxFrameGen.autoShotChangeDetection) {
        err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFXVIDEOFRAMEGENERATION_AUTOMATIC_SHOT_CHANGE_DETECTION_ENABLED, 0));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to 0: %s.\n"), NVVFXVIDEOFRAMEGENERATION_AUTOMATIC_SHOT_CHANGE_DETECTION_ENABLED, get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    return RGY_ERR_NONE;
#endif
}

bool NVEncFilterNvvfxFrameGeneration::compareParam(const NVEncFilterParam *param) const {
    if (!m_param) return true;
    auto prm = dynamic_cast<const NVEncFilterParamNvvfxFrameGen *>(m_param.get());
    if (!prm) return true;
    auto target = dynamic_cast<const NVEncFilterParamNvvfxFrameGen *>(param);
    if (!target) return true;
    return prm->nvvfxFrameGen != target->nvvfxFrameGen;
};

RGY_ERR NVEncFilterNvvfxFrameGeneration::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters are not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<NVEncFilterParamNvvfxFrameGen*>(pParam.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (rgy_csp_has_alpha(pParam->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("nvvfx-framegen does not support alpha channel.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (interlaced(pParam->frameIn)) {
        AddMessage(RGY_LOG_ERROR, _T("nvvfx-framegen does not support interlaced input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (pParam->frameIn.width != pParam->frameOut.width
        || pParam->frameIn.height != pParam->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("nvvfx-framegen does not change the resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if ((sts = checkParam(pParam.get())) != RGY_ERR_NONE) {
        return sts;
    }
    AddMessage(RGY_LOG_DEBUG, _T("GPU CC: %d.%d.\n"),
        prm->compute_capability.first, prm->compute_capability.second);

    auto err = initEffect(prm->modelDir);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    // the mode (and the shot change detection flag) has to be set before loading the effect
    err = setParam(pParam.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }

    // VFG takes RGBA8 interleaved images. The base class provides m_srcImg and m_dstImg,
    // m_prevImg is added here for the second (previous frame) input.
    if (!m_prevImg || (int)m_prevImg->width != pParam->frameIn.width || (int)m_prevImg->height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create nvvfx previous frame image %dx%d.\n"), pParam->frameIn.width, pParam->frameIn.height);
        m_prevImg = std::make_unique<NvCVImage>();
        err = err_to_rgy(NvCVImage_Alloc(m_prevImg.get(), pParam->frameIn.width, pParam->frameIn.height,
            NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate nvvfx previous frame image %dx%d: %s.\n"),
                pParam->frameIn.width, pParam->frameIn.height, get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_INPUT_IMAGE_0, m_prevImg.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set previous frame image: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (!m_srcImg || (int)m_srcImg->width != pParam->frameIn.width || (int)m_srcImg->height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create nvvfx input image %dx%d.\n"), pParam->frameIn.width, pParam->frameIn.height);
        m_srcImg = std::make_unique<NvCVImage>();
        err = err_to_rgy(NvCVImage_Alloc(m_srcImg.get(), pParam->frameIn.width, pParam->frameIn.height,
            NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate nvvfx input image %dx%d: %s.\n"),
                pParam->frameIn.width, pParam->frameIn.height, get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_INPUT_IMAGE_1, m_srcImg.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set input image: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (!m_dstImg || (int)m_dstImg->width != pParam->frameIn.width || (int)m_dstImg->height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create nvvfx output image %dx%d.\n"), pParam->frameIn.width, pParam->frameIn.height);
        m_dstImg = std::make_unique<NvCVImage>();
        err = err_to_rgy(NvCVImage_Alloc(m_dstImg.get(), pParam->frameIn.width, pParam->frameIn.height,
            NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate nvvfx output image %dx%d: %s.\n"),
                pParam->frameIn.width, pParam->frameIn.height, get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
        err = err_to_rgy(NvVFX_SetImage(m_effect.get(), NVVFX_OUTPUT_IMAGE, m_dstImg.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set output image %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFX_INPUT_WIDTH, (uint32_t)pParam->frameIn.width));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set input width: %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFX_INPUT_HEIGHT, (uint32_t)pParam->frameIn.height));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set input height: %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFX_BATCH_SIZE, 1));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set batch size: %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    if (prm->vuiInfo.matrix == RGY_MATRIX_UNSPECIFIED) {
        prm->vuiInfo.matrix = (CspMatrix)COLOR_VALUE_AUTO_RESOLUTION;
    }
    prm->vuiInfo.apply_auto(prm->vuiInfo, pParam->frameIn.height);
    if (!m_srcCrop
        || m_srcCrop->GetFilterParam()->frameIn.width  != pParam->frameIn.width
        || m_srcCrop->GetFilterParam()->frameIn.height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = RGY_CSP_RGB32;
        paramCrop->matrix = prm->vuiInfo.matrix;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_srcCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcCrop->GetInputMessage().c_str());
    }
    if (!m_dstCrop
        || m_dstCrop->GetFilterParam()->frameOut.width  != pParam->frameOut.width
        || m_dstCrop->GetFilterParam()->frameOut.height != pParam->frameOut.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameIn.csp = RGY_CSP_RGB32;
        paramCrop->matrix = prm->vuiInfo.matrix;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_dstCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstCrop->GetInputMessage().c_str());
    }

    if (compareModelDir(prm->modelDir) || compareParam(pParam.get())) {
        AddMessage(RGY_LOG_DEBUG, _T("Loading effect...\n"));
        err = err_to_rgy(NvVFX_Load(m_effect.get()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to load effect: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    }

    sts = AllocFrameBuf(pParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // the multiplier mode outputs the input frame itself plus (multiplier - 1) generated
    // frame(s), so the frame rate is multiplied by the specified value.
    m_targetFps = pParam->baseFps * prm->nvvfxFrameGen.multiplier;
    m_prevTimestamp = -1;
    m_inputFrames = 0;

    tstring info = m_name + _T(": ");
    if (m_srcCrop) {
        info += m_srcCrop->GetInputMessage() + _T("\n");
    }
    tstring nameBlank(m_name.length() + _tcslen(_T(": ")), _T(' '));
    info += tstring(INFO_INDENT) + nameBlank + pParam->print();
    if (m_dstCrop) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + nameBlank + m_dstCrop->GetInputMessage();
    }
    setFilterInfo(info);
    pParam->baseFps = m_targetFps;
    m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
    m_param = pParam;
    return sts;
#endif
}

RGYFrameInfo *NVEncFilterNvvfxFrameGeneration::getNextOutFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    if (*pOutputFrameNum >= (int)m_outFrameBuf.size()) {
        auto uptr = std::make_unique<CUFrameBuf>(m_param->frameOut.width, m_param->frameOut.height, m_param->frameOut.csp);
        if (uptr->alloc() != RGY_ERR_NONE) {
            m_outFrameBuf.clear();
            return nullptr;
        }
        m_outFrameBuf.push_back(std::move(uptr));
    }
    const int outFrameIdx = (*pOutputFrameNum)++;
    auto ptr = &m_outFrameBuf[outFrameIdx]->frame;
    ppOutputFrames[outFrameIdx] = ptr;
    return ptr;
}

#if ENABLE_NVVFX
RGY_ERR NVEncFilterNvvfxFrameGeneration::convertToNvCVImage(const RGYFrameInfo *src, NvCVImage *dst, cudaStream_t stream) {
    RGYFrameInfo srcImgInfo = m_srcCrop->GetFilterParam()->frameOut;
    srcImgInfo.singleAlloc = true;
    srcImgInfo.ptr[0] = (uint8_t *)dst->pixels;
    srcImgInfo.pitch[0] = dst->pitch;

    int cropFilterOutputNum = 0;
    RGYFrameInfo *outInfo[1] = { &srcImgInfo };
    RGYFrameInfo cropInput = *src;
    auto sts_filter = m_srcCrop->filter(&cropInput, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, stream);
    if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcCrop->name().c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcCrop->name().c_str());
        return sts_filter;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNvvfxFrameGeneration::genFrame(RGYFrameInfo *outFrame, const RGYFrameInfo *frameProp,
    int frameIndex, int multiplier,
    int64_t genPts, int64_t genDuration, cudaStream_t stream) {
    // setting FrameMultiplier resets FrameIndex,
    // so the per-output selector has to be set after the multiplier every time.
    auto err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFXVIDEOFRAMEGENERATION_FRAME_MULTIPLIER, (uint32_t)multiplier));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %d: %s.\n"), NVVFXVIDEOFRAMEGENERATION_FRAME_MULTIPLIER, multiplier, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    err = err_to_rgy(NvVFX_SetU32(m_effect.get(), NVVFXVIDEOFRAMEGENERATION_FRAME_INDEX, (uint32_t)frameIndex));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set parameter %s to %d: %s.\n"), NVVFXVIDEOFRAMEGENERATION_FRAME_INDEX, frameIndex, get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }
    err = err_to_rgy(NvVFX_Run(m_effect.get(), 0));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to run filter: %s.\n"), get_err_mes(err));
        return RGY_ERR_INVALID_PARAM;
    }

    {
        RGYFrameInfo dstImgInfo = m_dstCrop->GetFilterParam()->frameIn;
        dstImgInfo.singleAlloc = true;
        dstImgInfo.ptr[0] = (uint8_t *)m_dstImg->pixels;
        dstImgInfo.pitch[0] = m_dstImg->pitch;
        RGYFrameInfo *outInfo[1] = { outFrame };
        int outFrameNum = 1;
        auto sts_filter = m_dstCrop->filter(&dstImgInfo, outInfo, &outFrameNum, stream);
        if (outInfo[0] == nullptr || outFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || outFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
    }
    copyFramePropWithoutRes(outFrame, frameProp);
    outFrame->timestamp = genPts;
    outFrame->duration  = genDuration;
    return RGY_ERR_NONE;
}
#endif // #if ENABLE_NVVFX

RGY_ERR NVEncFilterNvvfxFrameGeneration::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
#if !ENABLE_NVVFX
    AddMessage(RGY_LOG_ERROR, _T("nvvfx filters is not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    auto prm = dynamic_cast<NVEncFilterParamNvvfxFrameGen*>(m_param.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // multiplier mode (2 - 8) generates (multiplier - 1) intermediate frames per input pair.
    const int multiplier   = prm->nvvfxFrameGen.multiplier;
    const int nGenFrames   = multiplier - 1;

    *pOutputFrameNum = 0;

    // Pass the input frame through unchanged. Used for the very first frame (which cannot be
    // interpolated yet) and for the current frame of each interpolated pair.
    auto emitPassthrough = [&](int64_t timestamp, int64_t duration) -> RGY_ERR {
        auto outFrame = getNextOutFrame(ppOutputFrames, pOutputFrameNum);
        if (!outFrame) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate output frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        auto stsCopy = copyFrameAsync(outFrame, pInputFrame, stream);
        if (stsCopy != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(stsCopy));
            return stsCopy;
        }
        copyFramePropWithoutRes(outFrame, pInputFrame);
        outFrame->timestamp = timestamp;
        outFrame->duration  = duration;
        return RGY_ERR_NONE;
    };

    if (m_inputFrames++ == 0) {
        sts = convertToNvCVImage(pInputFrame, m_prevImg.get(), stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_prevTimestamp = (int64_t)pInputFrame->timestamp;
        // The first frame cannot be interpolated yet, but it must not keep the whole input
        // frame duration either: the generated frames that follow share the same input frame
        // interval, so the first output frame only covers the first output interval.
        const int64_t firstDuration = (int64_t)pInputFrame->duration / multiplier;
        return emitPassthrough((int64_t)pInputFrame->timestamp, firstDuration);
    }

    sts = convertToNvCVImage(pInputFrame, m_srcImg.get(), stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const int64_t prevTs = m_prevTimestamp;
    const int64_t currTs = (int64_t)pInputFrame->timestamp;
    const int64_t tsDiff = currTs - prevTs;
    int64_t lastPts = prevTs;

    // generate the intermediate frames
    for (int i = 1; i <= nGenFrames; i++) {
        const int64_t genPts = prevTs + (tsDiff * i) / multiplier;
        auto outFrame = getNextOutFrame(ppOutputFrames, pOutputFrameNum);
        if (!outFrame) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate output frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        sts = genFrame(outFrame, pInputFrame, i, multiplier,
            genPts, genPts - lastPts, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        lastPts = genPts;
    }

    // then the current frame itself
    sts = emitPassthrough(currTs, currTs - lastPts);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // Keep the current frame as the previous frame of the next iteration. This converts the same
    // input a second time rather than swapping m_srcImg/m_prevImg, because both images are bound
    // to the effect through NvVFX_SetImage() and re-binding them after NvVFX_Load() is not a
    // documented operation.
    sts = convertToNvCVImage(pInputFrame, m_prevImg.get(), stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_prevTimestamp = currTs;
    return RGY_ERR_NONE;
#endif
}
