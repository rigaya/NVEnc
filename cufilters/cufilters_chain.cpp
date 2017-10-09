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

#pragma warning(push)
#pragma warning(disable: 4819)
//ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。
//データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#pragma warning(pop)
#include "cufilters_version.h"
#include "cufilters_chain.h"
#include "NVEncFilterDenoiseKnn.h"
#include "NVEncFilterDenoisePmd.h"
#include "NVEncFilterDeband.h"
#include "NVEncFilterUnsharp.h"
#include "NVEncFilterEdgelevel.h"

class cuFilterChainCtx {
    CUcontext& m_ctx;
public:
    cuFilterChainCtx(CUcontext& ctx) : m_ctx(ctx) {
        if (m_ctx) {
            cuCtxPushCurrent(m_ctx);
        }
    }
    ~cuFilterChainCtx() {
        if (m_ctx) {
            cuCtxPopCurrent(&m_ctx);
        }
    }
};

enum : uint32_t {
    CUFILTER_CHAIN_RESIZE    = 0x01,
    CUFILTER_CHAIN_KNN       = 0x02,
    CUFILTER_CHAIN_PMD       = 0x04,
    CUFILTER_CHAIN_UNSHARP   = 0x08,
    CUFILTER_CHAIN_EDGELEVEL = 0x10,
    CUFILTER_CHAIN_DEBAND    = 0x20,
};

cuFilterChainParam::cuFilterChainParam() :
    resizeEnable(false),
    resizeInterp(RESIZE_CUDA_SPLINE36),
    unsharp(),
    edgelevel(),
    knn(),
    pmd(),
    deband() {

}

uint32_t cuFilterChainParam::filter_enabled() const {
    uint32_t flags = 0x00;
    if (resizeEnable)     flags |= CUFILTER_CHAIN_RESIZE;
    if (knn.enable)       flags |= CUFILTER_CHAIN_KNN;
    if (pmd.enable)       flags |= CUFILTER_CHAIN_PMD;
    if (unsharp.enable)   flags |= CUFILTER_CHAIN_UNSHARP;
    if (edgelevel.enable) flags |= CUFILTER_CHAIN_EDGELEVEL;
    if (deband.enable)    flags |= CUFILTER_CHAIN_DEBAND;
    return flags;
}

cuFilterChain::cuFilterChain() :
    m_cuda_initilaized(false),
    m_prm(),
    m_nDeviceId(0),
    m_device(0),
    m_cuContextCurr(0),
    m_host(),
    m_dev(),
    m_vpFilters(),
    m_pLastFilterParam() {

}

cuFilterChain::~cuFilterChain() {
    {
        cuFilterChainCtx ctx(m_cuContextCurr);
        cudaThreadSynchronize();
    }
    m_cuContextCurr = 0;
    m_cuda_initilaized = false;
    m_pLastFilterParam.reset();
    m_vpFilters.clear();
    for (int i = 0; i < _countof(m_dev); i++) {
        m_dev[i].clear();
    }
    for (int i = 0; i < _countof(m_host); i++) {
        m_host[i].clear();
    }
}


void cuFilterChain::PrintMes(int logLevel, const TCHAR *format, ...) {
    if (logLevel < RGY_LOG_ERROR) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    MessageBoxA(NULL, buffer.data(), AUF_FULL_NAME, MB_OK | MB_ICONEXCLAMATION);
}

int cuFilterChain::init() {
    if (init_cuda(0)) {
        return 1;
    }

    m_convert_yc48_to_yuv444_16 = get_convert_csp_func(RGY_CSP_YC48, RGY_CSP_YUV444_16, false);
    if (m_convert_yc48_to_yuv444_16 == nullptr) {
        PrintMes(RGY_LOG_ERROR, _T("unsupported color format conversion, %s -> %s\n"), RGY_CSP_NAMES[RGY_CSP_YC48], RGY_CSP_NAMES[RGY_CSP_YUV444_16]);
        return 1;
    }
    m_convert_yuv444_16_to_yc48 = get_convert_csp_func(RGY_CSP_YUV444_16, RGY_CSP_YC48, false);
    if (m_convert_yuv444_16_to_yc48 == nullptr) {
        PrintMes(RGY_LOG_ERROR, _T("unsupported color format conversion, %s -> %s\n"), RGY_CSP_NAMES[RGY_CSP_YUV444_16], RGY_CSP_NAMES[RGY_CSP_YC48]);
        return 1;
    }
    return 0;
}

int cuFilterChain::init_cuda(int deviceId) {
    //ひとまず、これまでのすべてのエラーをflush
    auto cudaerr = cudaGetLastError();

    m_nDeviceId = deviceId;

    CUresult cuResult;
    if (CUDA_SUCCESS != (cuResult = cuInit(0))) {
        PrintMes(RGY_LOG_ERROR, _T("cuInit error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return 1;
    }
    PrintMes(RGY_LOG_DEBUG, _T("cuInit: Success.\n"));
    int deviceCount = 0;
    if (CUDA_SUCCESS != (cuResult = cuDeviceGetCount(&deviceCount))) {
        PrintMes(RGY_LOG_ERROR, _T("cuDeviceGetCount error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return 1;
    }
    PrintMes(RGY_LOG_DEBUG, _T("cuDeviceGetCount: Success.\n"));

    if (m_nDeviceId > deviceCount - 1) {
        PrintMes(RGY_LOG_ERROR, _T("Invalid Device Id = %d\n"), m_nDeviceId);
        return NV_ENC_ERR_INVALID_ENCODERDEVICE;
    }

    if (CUDA_SUCCESS != (cuResult = cuDeviceGet(&m_device, m_nDeviceId))) {
        PrintMes(RGY_LOG_ERROR, _T("cuDeviceGet error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return 1;
    }
    PrintMes(RGY_LOG_DEBUG, _T("cuDeviceGet: ID:%d.\n"), m_nDeviceId);

    if (CUDA_SUCCESS != (cuResult = cuCtxPopCurrent(&m_cuContextCurr))) {
        PrintMes(RGY_LOG_ERROR, _T("cuCtxPopCurrent error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(RGY_LOG_DEBUG, _T("cuCtxPopCurrent: Success.\n"));
    m_cuda_initilaized = true;
    return 0;
}

int cuFilterChain::allocate_buffer(const FrameInfo *pInputFrame, const FrameInfo *pOutputFrame) {
    auto cudaerr = cudaGetLastError();
    if (pInputFrame->width != m_host[0].frame.width
        || pInputFrame->height != m_host[0].frame.height) {
        if (m_host[0].frame.ptr) {
            cudaFreeHost(m_host[0].frame.ptr);
            m_host[0].frame.ptr = nullptr;
        }
        m_host[0].frame.width = pInputFrame->width;
        m_host[0].frame.height = pInputFrame->height;
        m_host[0].frame.picstruct = pInputFrame->picstruct;
        m_host[0].frame.flags = RGY_FRAME_FLAG_NONE;
        m_host[0].frame.csp = RGY_CSP_YUV444_16;
        m_host[0].frame.deivce_mem = false;
        m_host[0].frame.duration = 0;
        m_host[0].frame.timestamp = 0;

        m_dev[0].clear();
        m_dev[0].frame = m_host[0].frame;
        m_dev[0].frame.deivce_mem = true;
        auto cudaret = m_dev[0].alloc();
        if (cudaret != cudaSuccess) {
            return 1;
        }
        const auto infoEx = getFrameInfoExtra(&m_host[0].frame);
        if (infoEx.width_byte) {
            m_host[0].frame.pitch = ALIGN32(infoEx.width_byte);
            auto cudaret = cudaMallocHost(&m_host[0].frame.ptr, m_host[0].frame.pitch * infoEx.height_total);
            if (cudaret != cudaSuccess) {
                return 1;
            }
        }
    }
    if (pOutputFrame->width != m_host[1].frame.width
        || pOutputFrame->height != m_host[1].frame.height) {
        if (m_host[1].frame.ptr) {
            cudaFreeHost(m_host[1].frame.ptr);
            m_host[1].frame.ptr = nullptr;
        }
        m_host[1].frame.width = pOutputFrame->width;
        m_host[1].frame.height = pOutputFrame->height;
        m_host[1].frame.picstruct = pOutputFrame->picstruct;
        m_host[1].frame.flags = RGY_FRAME_FLAG_NONE;
        m_host[1].frame.csp = RGY_CSP_YUV444_16;
        m_host[1].frame.deivce_mem = false;
        m_host[1].frame.duration = 0;
        m_host[1].frame.timestamp = 0;

        m_dev[1].clear();
        m_dev[1].frame = m_host[1].frame;
        m_dev[1].frame.deivce_mem = true;
        auto cudaret = m_dev[1].alloc();
        if (cudaret != cudaSuccess) {
            return 1;
        }
        const auto infoEx = getFrameInfoExtra(&m_host[1].frame);
        if (infoEx.width_byte) {
            m_host[1].frame.pitch = ALIGN32(infoEx.width_byte);
            auto cudaret = cudaMallocHost(&m_host[1].frame.ptr, m_host[1].frame.pitch * infoEx.height_total);
            if (cudaret != cudaSuccess) {
                return 1;
            }
        }
    }
    m_host[0].frame.picstruct = pInputFrame->picstruct;
    m_host[1].frame.picstruct = pOutputFrame->picstruct;
    return 0;
}

int cuFilterChain::filter_chain_create(const FrameInfo *pInputFrame, const FrameInfo *pOutputFrame, bool reset) {
    cudaThreadSynchronize();
    if (reset) {
        m_vpFilters.clear();
    }

    int filter_idx = 0;
    FrameInfo inputFrame = *pInputFrame;
    inputFrame.ptr = nullptr;
    //GPUに上げる
    {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filterCrop(new NVEncFilterCspCrop());
            m_vpFilters.push_back(std::move(filterCrop));
        }
        shared_ptr<NVEncFilterParamCrop> param(new NVEncFilterParamCrop());
        param->frameIn = inputFrame;
        param->frameOut = param->frameIn;
        param->frameOut.csp = param->frameIn.csp;
        param->frameOut.deivce_mem = true;
        param->bOutOverwrite = false;
        memset(&param->crop, 0, sizeof(param->crop));
        param->bOutOverwrite = false;
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    //ノイズ除去 (knn)
    if (m_prm.knn.enable) {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterDenoiseKnn());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamDenoiseKnn> param(new NVEncFilterParamDenoiseKnn());
        param->knn = m_prm.knn;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->bOutOverwrite = false;
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    //ノイズ除去 (pmd)
    if (m_prm.pmd.enable) {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterDenoisePmd());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamDenoisePmd> param(new NVEncFilterParamDenoisePmd());
        param->pmd = m_prm.pmd;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->bOutOverwrite = false;
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }

    //リサイズ
    if (m_prm.resizeEnable) {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterResize());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamResize> param(new NVEncFilterParamResize());
        param->interp = (m_prm.resizeInterp != NPPI_INTER_UNDEFINED) ? m_prm.resizeInterp : RESIZE_CUDA_SPLINE36;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.width = pOutputFrame->width;
        param->frameOut.height = pOutputFrame->height;
        param->bOutOverwrite = false;
#if _M_IX86
        if (param->interp <= NPPI_INTER_MAX) {
            param->interp = RESIZE_CUDA_SPLINE36;
            PrintMes(RGY_LOG_WARN, _T("npp resize filters not supported in x86, switching to %s.\n"), get_chr_from_value(list_nppi_resize, param->interp));
        }
#endif
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    //unsharp
    if (m_prm.unsharp.enable) {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterUnsharp());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamUnsharp> param(new NVEncFilterParamUnsharp());
        param->unsharp.radius = m_prm.unsharp.radius;
        param->unsharp.weight = m_prm.unsharp.weight;
        param->unsharp.threshold = m_prm.unsharp.threshold;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->bOutOverwrite = false;
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    //edgelevel
    if (m_prm.edgelevel.enable) {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterEdgelevel());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamEdgelevel> param(new NVEncFilterParamEdgelevel());
        param->edgelevel = m_prm.edgelevel;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->bOutOverwrite = false;
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    //deband
    if (m_prm.deband.enable) {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterDeband());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamDeband> param(new NVEncFilterParamDeband());
        param->deband = m_prm.deband;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->bOutOverwrite = false;
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    //CPUに戻す
    {
        if (reset) {
            //フィルタチェーンに追加
            unique_ptr<NVEncFilter> filter(new NVEncFilterCspCrop());
            m_vpFilters.push_back(std::move(filter));
        }
        shared_ptr<NVEncFilterParamCrop> param(new NVEncFilterParamCrop());
        param->frameIn = inputFrame;
        param->frameOut = *pOutputFrame;
        param->frameOut.ptr = nullptr;
        param->frameOut.deivce_mem = false;
        param->bOutOverwrite = false;
        memset(&param->crop, 0, sizeof(param->crop));
        auto sts = m_vpFilters[filter_idx++]->init(param, nullptr);
        if (sts != NV_ENC_SUCCESS) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
    }
    return 0;
}

int cuFilterChain::proc(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const cuFilterChainParam& prm) {
    if (!m_cuda_initilaized) {
        return 1;
    }
    cuFilterChainCtx ctx(m_cuContextCurr);
    //解像度チェック、メモリ確保
    if (allocate_buffer(pInputFrame, pOutputFrame)) {
        return 1;
    }

    const bool recreate_filter_chain = prm.filter_enabled() != m_prm.filter_enabled();
    m_prm = prm;
    if (m_prm.filter_enabled() == 0) {
        return -1;
    }
    //フィルタチェーン更新
    if (filter_chain_create(&m_host[0].frame, &m_host[1].frame, recreate_filter_chain)) {
        return 1;
    }

    //YC48->YUV444(16bit)
    int crop[4] = { 0 };
    void *ptr_array[3];
    ptr_array[0] = m_host[0].frame.ptr + m_host[0].frame.pitch * m_host[0].frame.height * 0;
    ptr_array[1] = m_host[0].frame.ptr + m_host[0].frame.pitch * m_host[0].frame.height * 1;
    ptr_array[2] = m_host[0].frame.ptr + m_host[0].frame.pitch * m_host[0].frame.height * 2;
    m_convert_yc48_to_yuv444_16->func[0](
        ptr_array, (const void **)&pInputFrame->ptr,
        pInputFrame->width, pInputFrame->pitch, pInputFrame->pitch,
        m_host[0].frame.pitch, pInputFrame->height, m_host[0].frame.height, crop);
#if 1
    //フィルタチェーン実行
    auto frameInfo = m_host[0].frame;
    for (uint32_t ifilter = 0; ifilter < m_vpFilters.size() - 1; ifilter++) {
        int nOutFrames = 0;
        FrameInfo *outInfo[16] = { 0 };
        auto sts_filter = m_vpFilters[ifilter]->filter(&frameInfo, (FrameInfo **)&outInfo, &nOutFrames);
        if (sts_filter != NV_ENC_SUCCESS) {
            PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_vpFilters[ifilter]->name().c_str());
            return sts_filter;
        }
        if (nOutFrames > 1) {
            PrintMes(RGY_LOG_ERROR, _T("Currently only simple filters are supported.\n"));
            return NV_ENC_ERR_UNIMPLEMENTED;
        }
        frameInfo = *(outInfo[0]);
    }
    //最後のフィルタ
    auto& lastFilter = m_vpFilters[m_vpFilters.size()-1];
    int nOutFrames = 0;
    FrameInfo *outInfo[16] = { 0 };
    outInfo[0] = &m_host[1].frame;
    auto sts_filter = lastFilter->filter(&frameInfo, (FrameInfo **)&outInfo, &nOutFrames);
    if (sts_filter != NV_ENC_SUCCESS) {
        PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), lastFilter->name().c_str());
        return sts_filter;
    }
    cudaThreadSynchronize();
#else
    auto frameinfoEx = getFrameInfoExtra(&m_host[0].frame);
    memcpy(m_host[1].frame.ptr, m_host[0].frame.ptr, frameinfoEx.width_byte * frameinfoEx.height_total);
#endif

    //YUV444(16bit)->YC48
    ptr_array[0] = m_host[1].frame.ptr + m_host[1].frame.pitch * m_host[1].frame.height * 0;
    ptr_array[1] = m_host[1].frame.ptr + m_host[1].frame.pitch * m_host[1].frame.height * 1;
    ptr_array[2] = m_host[1].frame.ptr + m_host[1].frame.pitch * m_host[1].frame.height * 2;
    m_convert_yuv444_16_to_yc48->func[0](
        (void **)&pOutputFrame->ptr, (const void **)ptr_array,
        m_host[1].frame.width, m_host[1].frame.pitch, m_host[1].frame.pitch,
        pOutputFrame->pitch, m_host[1].frame.height, pOutputFrame->height, crop);

    return 0;
}
