//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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
    virtual void UpdateDisplay(const TCHAR *mes) override;
    virtual void WriteLine(const TCHAR *mes) override;
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
    uint32_t m_tmLastUpdate;
    BOOL m_pause;
public:
    AuoInput();
    ~AuoInput();
    virtual int Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus) override;
    virtual int LoadNextFrame(void *dst, int dst_pitch) override;
    virtual void Close() override;
};

class CAuoNvEnc : public NVEncCore
{
public:
    CAuoNvEnc();
    ~CAuoNvEnc();
protected:
    virtual int NVPrintf(FILE *fp, int logLevel, const TCHAR *format, ...) override;
    virtual NVENCSTATUS InitInput(InEncodeVideoParam *inputParam) override;
};

#endif //_AUO_NVENC_H_