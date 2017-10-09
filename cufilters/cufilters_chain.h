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

#ifndef __CUFILTER_CHAIN_H__
#define __CUFILTER_CHAIN_H__

#include <cstdint>
#include "NVEncParam.h"
#include "NVEncFilter.h"
#include "convert_csp.h"

struct cuFilterChainParam {
    bool resizeEnable;
    int resizeInterp;
    VppUnsharp unsharp;
    VppEdgelevel edgelevel;
    VppKnn knn;
    VppPmd pmd;
    VppDeband deband;

    cuFilterChainParam();
    uint32_t filter_enabled() const;
};

class cuFilterChain {
public:
    cuFilterChain();
    ~cuFilterChain();

    int init();
    std::string get_dev_name() const;
    int proc(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const cuFilterChainParam& prm);

private:
    int init_cuda(int deviceId);
    int allocate_buffer(const FrameInfo *pInputFrame, const FrameInfo *pOutputFrame);
    int filter_chain_create(const FrameInfo *pInputFrame, const FrameInfo *pOutputFrame, bool reset);
    void PrintMes(int logLevel, const TCHAR *format, ...);

    bool m_cuda_initilaized;
    cuFilterChainParam m_prm;
    int m_nDeviceId;
    CUdevice m_device;
    std::string m_deviceName;
    CUcontext m_cuContextCurr;
    CUFrameBuf m_host[2];
    CUFrameBuf m_dev[2];
    vector<unique_ptr<NVEncFilter>> m_vpFilters;
    shared_ptr<NVEncFilterParam>    m_pLastFilterParam;
    const ConvertCSP *m_convert_yc48_to_yuv444_16;
    const ConvertCSP *m_convert_yuv444_16_to_yc48;
};


#endif //__CUFILTER_CHAIN_H__
