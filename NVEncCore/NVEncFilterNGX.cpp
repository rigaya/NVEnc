﻿// -----------------------------------------------------------------------------------------
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
#include "NVEncFilterParam.h"
#include "NVEncFilterD3D11.h"
#include "NVEncFilterColorspace.h"
#include "NVEncFilterNGX.h"

tstring NVEncFilterParamNGXVSR::print() const {
    return ngxvsr.print();
}

tstring NVEncFilterParamNGXTrueHDR::print() const {
    return trueHDR.print();
}

#if ENABLE_NVSDKNGX
#include "rgy_device.h"

NVEncNVSDKNGXFuncs::NVEncNVSDKNGXFuncs() :
    hModule(),
    fcreate(nullptr),
    finit(nullptr),
    fdelete(nullptr),
    fprocFrame(nullptr) {
}
NVEncNVSDKNGXFuncs::~NVEncNVSDKNGXFuncs() {
    close();
}

RGY_ERR NVEncNVSDKNGXFuncs::load() {
    hModule = RGY_LOAD_LIBRARY(NVENC_NVSDKNGX_MODULENAME);
    if (!hModule) {
        return RGY_ERR_NULL_PTR;
    }

#define LOAD_PROC(proc, procName) { \
    proc = (decltype(procName) *)RGY_GET_PROC_ADDRESS(hModule, #procName); \
    if (!proc) { \
        close(); \
        return RGY_ERR_NULL_PTR; \
    } \
}

    LOAD_PROC(fcreate,           NVEncNVSDKNGXCreate);
    LOAD_PROC(finit,             NVEncNVSDKNGXInit);
    LOAD_PROC(fdelete,           NVEncNVSDKNGXDelete);
    LOAD_PROC(fprocFrame,        NVEncNVSDKNGXProcFrame);
#undef LOAD_PROC
    return RGY_ERR_NONE;
}
void NVEncNVSDKNGXFuncs::close() {
    if (hModule) {
        RGY_FREE_LIBRARY(hModule);
        hModule = nullptr;
    }
}

NVEncFilterNGX::NVEncFilterNGX() :
    m_func(),
    m_nvsdkNGX(unique_nvsdkngx_handle(nullptr, nullptr)),
    m_ngxCspIn(RGY_CSP_NA),
    m_ngxCspOut(RGY_CSP_NA),
    m_dxgiformatIn(DXGI_FORMAT_UNKNOWN),
    m_dxgiformatOut(DXGI_FORMAT_UNKNOWN),
    m_ngxFrameBufOut(),
    m_ngxTextIn(),
    m_ngxTextOut(),
    m_srcColorspace(),
    m_dstColorspace(),
    m_dx11(nullptr) {
    m_name = _T("nvsdk-ngx");
}
NVEncFilterNGX::~NVEncFilterNGX() {
    close();
}

RGY_ERR NVEncFilterNGX::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    sts = checkParam(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    if (rgy_csp_has_alpha(pParam->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("nfx filters does not support alpha channel.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    sts = initNGX(pParam, pPrintMes);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setNGXParam(pParam.get());

    sts = initCommon(pParam);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return sts;
}

RGY_ERR NVEncFilterNGX::initNGX(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = dynamic_cast<NVEncFilterParamNGX*>(pParam.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->compute_capability.first < 7) {
        AddMessage(RGY_LOG_ERROR, _T("NVSDK NGX filters require Turing GPUs (CC:7.0) or later: current CC %d.%d.\n"), prm->compute_capability.first, prm->compute_capability.second);
        return RGY_ERR_UNSUPPORTED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("GPU CC: %d.%d.\n"),
        prm->compute_capability.first, prm->compute_capability.second);

    m_dx11 = prm->dx11;

    if (!m_func) {
        m_func = std::make_unique<NVEncNVSDKNGXFuncs>();
        if ((sts = m_func->load()) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Cannot load dll %s: %s.\n"), NVENC_NVSDKNGX_MODULENAME, get_err_mes(sts));
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Loaded dll %s.\n"), NVENC_NVSDKNGX_MODULENAME);

        NVEncNVSDKNGXHandle ngxHandle = nullptr;
        if ((sts = m_func->fcreate(&ngxHandle, getNGXFeature())) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create NVSDK NGX handle: %s.\n"), get_err_mes(sts));
            // RGY_LOAD_LIBRARYを使用して、nvngx_xxx.dll がロードできるかを確認する
            // できなければ、エラーを返す
            auto hModule = RGY_LOAD_LIBRARY(NVENC_NVSDKNGX_DLL_NAME[getNGXFeature()]);
            if (hModule) {
                RGY_FREE_LIBRARY(hModule);
                hModule = nullptr;
            } else {
                AddMessage(RGY_LOG_ERROR, _T("%s is required but not found.\n"), NVENC_NVSDKNGX_DLL_NAME[getNGXFeature()]);
            }
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Created NVSDK NGX handle.\n"));

        m_nvsdkNGX = unique_nvsdkngx_handle(ngxHandle, m_func->fdelete);

        // CUDA専用初期化: deviceOrdinal/ctx/stream はここでは既定値(現在のコンテキスト)を渡す
        CUcontext currentContext;
        cuCtxGetCurrent(&currentContext);
        if ((sts = m_func->finit(m_nvsdkNGX.get(), -1, currentContext, cudaStreamPerThread)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to init NVSDK NGX library: %s.\n"), get_err_mes(sts));
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Initialized NVSDK NGX library.\n"));
    }
    return sts;
}

RGY_ERR NVEncFilterNGX::initCommon(shared_ptr<NVEncFilterParam> pParam) {
    RGY_ERR sts = RGY_ERR_NONE;
    auto prm = dynamic_cast<NVEncFilterParamNGX*>(pParam.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto inChromaFmt = RGY_CSP_CHROMA_FORMAT[pParam->frameIn.csp];
    VideoVUIInfo vui = prm->vui;
    if (inChromaFmt == RGY_CHROMAFMT_RGB || inChromaFmt == RGY_CHROMAFMT_RGB_PACKED) {
        vui.setIfUnsetUnknwonAuto(VideoVUIInfo().to(RGY_MATRIX_RGB).to(RGY_PRIM_BT709).to(RGY_TRANSFER_IEC61966_2_1));
    } else {
        vui.setIfUnsetUnknwonAuto(VideoVUIInfo().to((CspMatrix)COLOR_VALUE_AUTO_RESOLUTION).to((CspColorprim)COLOR_VALUE_AUTO_RESOLUTION).to((CspTransfer)COLOR_VALUE_AUTO_RESOLUTION));
    }
    vui.apply_auto(VideoVUIInfo(), pParam->frameIn.height);

    if (!m_srcColorspace
        || m_srcColorspace->GetFilterParam()->frameIn.width != pParam->frameIn.width
        || m_srcColorspace->GetFilterParam()->frameIn.height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input colorspace conversion filter.\n"));
        VppColorspace colorspace;
        colorspace.enable = true;
        // DXGI_FORMAT_R8G8B8A8_UNORM に合わせてRGBに変換する
        colorspace.convs.push_back(ColorspaceConv(vui, vui.to(RGY_MATRIX_RGB)));

        unique_ptr<NVEncFilterColorspace> filter(new NVEncFilterColorspace());
        shared_ptr<NVEncFilterParamColorspace> paramColorspace(new NVEncFilterParamColorspace());
        paramColorspace->frameIn = pParam->frameIn;
        paramColorspace->frameOut = paramColorspace->frameIn;
        paramColorspace->frameOut.csp = RGY_CSP_RGB_F32;
        paramColorspace->baseFps = pParam->baseFps;
        paramColorspace->colorspace = colorspace;
        paramColorspace->encCsp = paramColorspace->frameIn.csp;
        paramColorspace->VuiIn = prm->vui;
        paramColorspace->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramColorspace->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramColorspace->bOutOverwrite = false;
        sts = filter->init(paramColorspace, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_srcColorspace = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcColorspace->GetInputMessage().c_str());
    }
    if (!m_srcCrop
        || m_srcCrop->GetFilterParam()->frameIn.width != m_srcColorspace->GetFilterParam()->frameOut.width
        || m_srcCrop->GetFilterParam()->frameIn.height != m_srcColorspace->GetFilterParam()->frameOut.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = m_srcColorspace->GetFilterParam()->frameOut;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = m_ngxCspIn;
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
    // DX11テクスチャは不要

    // DX11テクスチャは不要

    if (!m_dstCrop
        || m_dstCrop->GetFilterParam()->frameOut.width != pParam->frameOut.width
        || m_dstCrop->GetFilterParam()->frameOut.height != pParam->frameOut.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameIn.csp = m_ngxCspOut;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->frameOut.csp = RGY_CSP_RGB_F32;
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

        m_ngxFrameBufOut = std::make_unique<CUFrameBuf>();
        sts = m_ngxFrameBufOut->alloc(m_dstCrop->GetFilterParam()->frameIn.width, m_dstCrop->GetFilterParam()->frameIn.height, m_ngxCspOut);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_DEBUG, _T("failed to allocate memory for ngx output: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    if (!m_dstColorspace
        || m_dstColorspace->GetFilterParam()->frameOut.width != m_dstCrop->GetFilterParam()->frameIn.width
        || m_dstColorspace->GetFilterParam()->frameOut.height != m_dstCrop->GetFilterParam()->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create output colorspace conversion filter.\n"));
        VppColorspace colorspace;
        colorspace.enable = true;
        if (getNGXFeature() == NVSDK_NVX_TRUEHDR) {
            // TrueHDRの出力DXGI_FORMAT_R16G16B16A16_FLOATはLinearRGBになっている
            // LinearRGBからBT.2020に変換する
            if (inChromaFmt == RGY_CHROMAFMT_RGB || inChromaFmt == RGY_CHROMAFMT_RGB_PACKED) {
                colorspace.convs.push_back(ColorspaceConv(vui.to(RGY_MATRIX_RGB).to(RGY_TRANSFER_LINEAR), vui.to(RGY_MATRIX_RGB).to(RGY_TRANSFER_ST2084).to(RGY_PRIM_BT2020)));
            } else {
                colorspace.convs.push_back(ColorspaceConv(vui.to(RGY_MATRIX_RGB).to(RGY_TRANSFER_LINEAR), vui.to(RGY_MATRIX_BT2020_NCL).to(RGY_TRANSFER_ST2084).to(RGY_PRIM_BT2020)));
            }
        } else {
            // DXGI_FORMAT_R8G8B8A8_UNORMのRGBから変換する
            colorspace.convs.push_back(ColorspaceConv(vui.to(RGY_MATRIX_RGB), vui));
        }

        unique_ptr<NVEncFilterColorspace> filter(new NVEncFilterColorspace());
        shared_ptr<NVEncFilterParamColorspace> paramColorspace(new NVEncFilterParamColorspace());
        paramColorspace->frameIn = m_dstCrop->GetFilterParam()->frameOut;
        paramColorspace->frameOut = pParam->frameOut;
        paramColorspace->baseFps = pParam->baseFps;
        paramColorspace->colorspace = colorspace;
        paramColorspace->encCsp = pParam->frameOut.csp;
        paramColorspace->VuiIn = prm->vui;
        paramColorspace->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramColorspace->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramColorspace->bOutOverwrite = false;
        sts = filter->init(paramColorspace, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_dstColorspace = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstColorspace->GetInputMessage().c_str());

        pParam->frameOut = m_dstColorspace->GetFilterParam()->frameOut;
    }

    if (m_frameBuf.size() == 0
        || !cmpFrameInfoCspResolution(&m_frameBuf[0]->frame, &pParam->frameOut)) {
        sts = AllocFrameBuf(pParam->frameOut, 2);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    const tstring nameBlank(m_name.length() + _tcslen(_T(": ")), _T(' '));
    tstring info = m_name + _T(": ");
    if (m_srcColorspace) {
        info += m_srcColorspace->GetInputMessage() + _T("\n");
    }
    if (m_srcCrop) {
        info += tstring(INFO_INDENT) + nameBlank + m_srcCrop->GetInputMessage() + _T("\n");
    }
    info += tstring(INFO_INDENT) + nameBlank + pParam->print() + _T("\n");
    if (m_dstCrop) {
        info += tstring(INFO_INDENT) + nameBlank + m_dstCrop->GetInputMessage() + _T("\n");
    }
    if (m_dstColorspace) {
        info += tstring(INFO_INDENT) + nameBlank + m_dstColorspace->GetInputMessage();
    }
    setFilterInfo(info);
    m_param = pParam;
    return sts;
}

RGY_ERR NVEncFilterNGX::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    auto prm = dynamic_cast<NVEncFilterParamNGX*>(m_param.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    //const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    //if (memcpyKind != cudaMemcpyDeviceToDevice) {
    //    AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
    //    return RGY_ERR_INVALID_PARAM;
    //}
    // pInputFrame -> srcColorspaceOut
    RGYFrameInfo *srcColorspaceOut = nullptr;
    {
        int filterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { nullptr };
        RGYFrameInfo cropInput = *pInputFrame;
        auto sts_filter = m_srcColorspace->filter(&cropInput, (RGYFrameInfo **)&outInfo, &filterOutputNum, stream);
        srcColorspaceOut = outInfo[0];
        if (srcColorspaceOut == nullptr || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcColorspace->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcColorspace->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(srcColorspaceOut, pInputFrame);
    }
    // srcColorspaceOut -> ngxFrameBufIn
    RGYFrameInfo *ngxFrameBufIn = nullptr;
    {
        int filterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { nullptr };
        auto sts_filter = m_srcCrop->filter(srcColorspaceOut, (RGYFrameInfo **)&outInfo, &filterOutputNum, stream);
        ngxFrameBufIn = outInfo[0];
        if (ngxFrameBufIn == nullptr || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(ngxFrameBufIn, pInputFrame);
    }
#if 1
    // CUDA APIへ移行: NVEncNVSDKNGX 側でD3D11テクスチャ<->cudaArray管理と評価を実施
    const NVEncNVSDKNGXRect rectDst = { 0, 0, m_ngxFrameBufOut->frame.width, m_ngxFrameBufOut->frame.height };
    const NVEncNVSDKNGXRect rectSrc = { 0, 0, ngxFrameBufIn->width,  ngxFrameBufIn->height };
    if (RGY_CSP_PLANES[ngxFrameBufIn->csp] != 1 || RGY_CSP_PLANES[m_ngxFrameBufOut->frame.csp] != 1) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp, must have only 1 plane.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const int srcBpp = (m_dxgiformatIn == DXGI_FORMAT_R8G8B8A8_UNORM) ? 4 : 4; // VSR/TrueHDR入力はRGBA8
    const int dstBpp = (m_dxgiformatOut == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 8 : 4;
    sts = m_func->fprocFrame(
        m_nvsdkNGX.get(),
        &rectDst,
        &rectSrc,
        getNGXParam(),
        ngxFrameBufIn->ptr[0], ngxFrameBufIn->pitch[0],
        m_ngxFrameBufOut->frame.ptr[0], m_ngxFrameBufOut->frame.pitch[0],
        srcBpp, dstBpp
    );
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to process frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
#else // for debug
    // cudaMemcpy2DAsyncでngxFrameBufInからm_ngxFrameBufOutにコピーする
    // ngxFrameBufIn -> m_ngxFrameBufOut
    {
        const int bytePerPix = getTextureBytePerPix(m_dxgiformatOut);
        if (bytePerPix == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported dxgiformat: %d.\n"), m_dxgiformatOut);
            return RGY_ERR_UNSUPPORTED;
        }
        sts = err_to_rgy(cudaMemcpy2DAsync(
            m_ngxFrameBufOut->frame.ptr[0], m_ngxFrameBufOut->frame.pitch[0],
            ngxFrameBufIn->ptr[0], ngxFrameBufIn->pitch[0],
            ngxFrameBufIn->width * bytePerPix, ngxFrameBufIn->height,
            cudaMemcpyDeviceToDevice, stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
#endif
    // m_ngxFrameBufOut -> dstCropOut
    RGYFrameInfo *dstCropOut = nullptr;
    {
        int filterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { nullptr };
        auto sts_filter = m_dstCrop->filter(&m_ngxFrameBufOut->frame, (RGYFrameInfo **)&outInfo, &filterOutputNum, stream);
        dstCropOut = outInfo[0];
        if (dstCropOut == nullptr || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcColorspace->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcColorspace->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(dstCropOut, pInputFrame);
    }
    // dstCropOut -> ppOutputFrames[0]
    {
        auto sts_filter = m_dstColorspace->filter(dstCropOut, ppOutputFrames, pOutputFrameNum, stream);
        if (ppOutputFrames[0] == nullptr || pOutputFrameNum[0] != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcColorspace->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || pOutputFrameNum[0] != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcColorspace->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], dstCropOut);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterNGX::close() {
    m_srcColorspace.reset();
    m_dstColorspace.reset();
    m_srcCrop.reset();
    m_dstCrop.reset();
    m_ngxTextIn.reset();
    m_ngxTextOut.reset();
    m_ngxFrameBufOut.reset();
    m_nvsdkNGX.reset();
    m_func.reset();
    m_frameBuf.clear();
    m_dx11 = nullptr;
}

NVEncFilterNGXVSR::NVEncFilterNGXVSR() : NVEncFilterNGX(), m_paramVSR() {
    m_name = _T("ngx-vsr");
}

NVEncFilterNGXVSR::~NVEncFilterNGXVSR() {
}

RGY_ERR NVEncFilterNGXVSR::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNGXVSR*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->ngxvsr.quality < 1 || 4 < prm->ngxvsr.quality) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid quality value %d, must be in the range of 1 to 4.\n"), prm->ngxvsr.quality);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterNGXVSR::setNGXParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNGXVSR*>(param);
    m_paramVSR.quality = prm->ngxvsr.quality;

    m_ngxCspIn = RGY_CSP_RGB32;
    m_ngxCspOut = RGY_CSP_RGB32;
    m_dxgiformatIn = DXGI_FORMAT_R8G8B8A8_UNORM;
    m_dxgiformatOut = DXGI_FORMAT_R8G8B8A8_UNORM;
}

NVEncFilterNGXTrueHDR::NVEncFilterNGXTrueHDR() : NVEncFilterNGX(), m_vuiOut(), m_paramTrueHDR() {
    m_name = _T("ngx-truehdr");
}

NVEncFilterNGXTrueHDR::~NVEncFilterNGXTrueHDR() {
}

RGY_ERR NVEncFilterNGXTrueHDR::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNGXTrueHDR*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->trueHDR.contrast < 0 || 200 < prm->trueHDR.contrast) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid contrast value %d, must be in the range of 0 to 200.\n"), prm->trueHDR.contrast);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->trueHDR.saturation < 0 || 200 < prm->trueHDR.saturation) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid saturation value %d, must be in the range of 0 to 200.\n"), prm->trueHDR.saturation);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->trueHDR.middleGray < 10 || 100 < prm->trueHDR.middleGray) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid middleGray value %d, must be in the range of 10 to 100.\n"), prm->trueHDR.middleGray);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->trueHDR.maxLuminance < 400 || 2000 < prm->trueHDR.maxLuminance) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid max luminance value %d, must be in the range of 400 to 2000.\n"), prm->trueHDR.maxLuminance);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterNGXTrueHDR::setNGXParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamNGXTrueHDR*>(param);
    m_vuiOut = prm->vui.to(RGY_MATRIX_BT2020_NCL).to(RGY_TRANSFER_ST2084).to(RGY_PRIM_BT2020);
    m_paramTrueHDR.contrast = prm->trueHDR.contrast;
    m_paramTrueHDR.saturation = prm->trueHDR.saturation;
    m_paramTrueHDR.middleGray = prm->trueHDR.middleGray;
    m_paramTrueHDR.maxLuminance = prm->trueHDR.maxLuminance;

    m_ngxCspIn = RGY_CSP_RGB32;
    m_ngxCspOut = RGY_CSP_RGBA_FP16_P;
    m_dxgiformatIn = DXGI_FORMAT_R8G8B8A8_UNORM;
    m_dxgiformatOut = DXGI_FORMAT_R16G16B16A16_FLOAT;
}

#else

NVEncFilterNGXVSR::NVEncFilterNGXVSR() : NVEncFilterDisabled() { m_name = _T("ngx-vsr"); }
NVEncFilterNGXVSR::~NVEncFilterNGXVSR() {};

NVEncFilterNGXTrueHDR::NVEncFilterNGXTrueHDR() : NVEncFilterDisabled() { m_name = _T("ngx-truehdr"); }
NVEncFilterNGXTrueHDR::~NVEncFilterNGXTrueHDR() { }
VideoVUIInfo NVEncFilterNGXTrueHDR::VuiOut() const { return VideoVUIInfo(); }

#endif //#if ENABLE_NVSDKNGX
