// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#ifndef _AUO_NVENC_H_
#define _AUO_NVENC_H_

#include <Windows.h>
#include <stdio.h>
#include <vector>
#include "output.h"
#include "auo.h"
#include "auo_conf.h"
#include "auo_system.h"

#include "convertCSP.h"
#include "NVEncInput.h"
#include "NVEncCore.h"

typedef struct InputInfoAuo {
    const OUTPUT_INFO *oip;
    const SYSTEM_DATA *sys_dat;
    CONF_GUIEX *conf;
    PRM_ENC *pe;
    int *jitter;
    BOOL interlaced;
} InputInfoAuo;

typedef struct ConvCSPInfo {
    funcConvertCSP func[2];
    DWORD SIMD;
} ConvCSPInfo;

class AuoEncodeStatus : public EncodeStatus
{
public:
    AuoEncodeStatus();
    ~AuoEncodeStatus();
protected:
    virtual void UpdateDisplay(const TCHAR *mes, double progressPercent = 0.0) override;
    virtual void WriteLine(const TCHAR *mes) override;
    virtual int UpdateDisplay(double progressPercent = 0.0) override;
    virtual void SetPrivData(void *pPrivateData) override;

    InputInfoAuo m_auoData;
    std::chrono::system_clock::time_point m_tmLastLogUpdate;
};

class AuoInput : public NVEncBasicInput
{
private:
    const OUTPUT_INFO *oip;
    CONF_GUIEX *conf;
    PRM_ENC *pe;
    int frames;
    int *jitter;
    int m_iFrame;
    BOOL m_interlaced;
    BOOL m_pause;
public:
    AuoInput();
    ~AuoInput();
    virtual int Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) override;
    virtual int LoadNextFrame(void *dst, int dst_pitch) override;
    virtual void Close() override;
};

class CAuoLog : public CNVEncLog {
public:
    CAuoLog(const TCHAR *pLogFile, int log_level) : CNVEncLog(pLogFile, log_level) { };
    virtual void operator()(int logLevel, const TCHAR *format, ...) override;
};

class CAuoNvEnc : public NVEncCore
{
public:
    CAuoNvEnc();
    ~CAuoNvEnc();
protected:
    virtual NVENCSTATUS InitLog(const InEncodeVideoParam *inputParam) override;
    virtual NVENCSTATUS InitInput(InEncodeVideoParam *inputParam) override;
};

#endif //_AUO_NVENC_H_