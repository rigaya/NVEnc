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

#ifndef __NVENC_FILTER_NV_OPT_FLOW_H__
#define __NVENC_FILTER_NV_OPT_FLOW_H__
#include <array>
#include "rgy_version.h"
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "NVEncNVOFFRUC.h"

struct NVEncNVOFFRUCFuncs {
    HMODULE hModule;
    decltype(NVEncNVOFFRUCCreate) *fcreate;
    decltype(NVEncNVOFFRUCLoad)   *fload;
    decltype(NVEncNVOFFRUCDelete) *fdelete;
    decltype(NVEncNVOFFRUCCreateFURCHandle) *fcreateHandle;
    decltype(NVEncNVOFFRUCRegisterResource) *fregisterResource;
    decltype(NVEncNVOFFRUCCloseFURCHandle) *fcloseHandle;
    decltype(NVEncNVOFFRUCProc) *fproc;
    NVEncNVOFFRUCFuncs();
    ~NVEncNVOFFRUCFuncs();
    RGY_ERR load();
    void close();
};

using unique_fruc_handle = std::unique_ptr<std::remove_pointer<NVEncNVOFFRUCHandle>::type, decltype(NVEncNVOFFRUCFuncs::fcloseHandle)>;

class NVEncFilterParamNVOFFRUC : public NVEncFilterParam {
public:
    VppFruc fruc;
    rgy_rational<int> timebase;
    std::pair<int, int> compute_capability;
    NVEncFilterParamNVOFFRUC() : fruc(), timebase(), compute_capability() {};
    virtual ~NVEncFilterParamNVOFFRUC() {};
    virtual tstring print() const;
};

struct NVEncFilterFRUCHandle {
    unique_fruc_handle handle;
    int64_t prevFramePts;

    NVEncFilterFRUCHandle(unique_fruc_handle h) : handle(std::move(h)), prevFramePts(-1) { };
};

class NVEncFilterNVOFFRUC : public NVEncFilter {
public:
    NVEncFilterNVOFFRUC();
    virtual ~NVEncFilterNVOFFRUC();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    std::pair<RGY_ERR, unique_fruc_handle> createFRUCHandle();
    RGYFrameInfo *getNextOutFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);
    RGY_ERR setFirstFrame(NVEncFilterFRUCHandle& frucHandle, const CUFrameDevPtr *prev);
    RGY_ERR genFrame(const size_t frucHandleIdx, RGYFrameInfo *outFrame, const CUFrameDevPtr *prev, const CUFrameDevPtr *curr, const int64_t genPts, cudaStream_t stream);

    std::unique_ptr<NVEncNVOFFRUCFuncs> m_func;
    std::vector<NVEncFilterFRUCHandle> m_frucHandles;
    std::array<std::unique_ptr<CUFrameDevPtr>, 3> m_frucBuf;
    RGY_CSP m_frucCsp;

    int64_t m_prevTimestamp;
    rgy_rational<int> m_targetFps;
    rgy_rational<int> m_timebase;
    std::unique_ptr<NVEncFilterCspCrop> m_srcCrop;
    std::unique_ptr<NVEncFilterCspCrop> m_dstCrop;
    int m_inputFrames;
};

#endif //#ifndef __NVENC_FILTER_NV_OPT_FLOW_H__