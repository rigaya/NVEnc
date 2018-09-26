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
// --------------------------------------------------------------------------------------------

#ifndef __RGY_CAPTION_H__
#define __RGY_CAPTION_H__

#include "rgy_osdep.h"
#include "rgy_version.h"
#include "rgy_avutil.h"
#include "rgy_err.h"
#include "rgy_log.h"
#include "Caption.h"

enum {
    HLC_INVALID = 0,
    HLC_kigou   = 1,
    HLC_box     = 2,
    HLC_draw    = 3
};

#define CAPTIONF(x) \
    private: \
        x pf ## x; \
    public: \
        const x f_ ## x() { return pf ## x; };

class CaptionDLL {
public:
    CaptionDLL();
    ~CaptionDLL();
    RGY_ERR load();
    RGY_ERR init();
    bool unicode() const {
        return unicode_;
    }
private:
    std::unique_ptr<void, module_deleter> m_hModule;

    CAPTIONF(InitializeCP);
    CAPTIONF(InitializeUNICODECP);
    CAPTIONF(UnInitializeCP);
    CAPTIONF(AddTSPacketCP);
    CAPTIONF(ClearCP);
    CAPTIONF(GetTagInfoCP);
    CAPTIONF(GetCaptionDataCP);

    bool unicode_;
};

#undef CAPTIONF

enum STRING_SIZE {
    STR_SMALL = 0,  //SSZ
    STR_MEDIUM,     //MSZ
    STR_NORMAL,     //NSZ
    STR_MICRO,      //SZX 0x60
    STR_HIGH_W,     //SZX 0x41
    STR_WIDTH_W,    //SZX 0x44
    STR_W,          //SZX 0x45
    STR_SPECIAL_1,  //SZX 0x6B
    STR_SPECIAL_2,  //SZX 0x64
};

struct CAPTION_CHAR_DATA {
    std::string strDecode;
    DWORD wCharSizeMode;

    CLUT_DAT_DLL stCharColor;
    CLUT_DAT_DLL stBackColor;
    CLUT_DAT_DLL stRasterColor;

    BOOL bUnderLine;
    BOOL bShadow;
    BOOL bBold;
    BOOL bItalic;
    BYTE bFlushMode;
    BYTE bHLC;  //must ignore low 4bits

    WORD wCharW;
    WORD wCharH;
    WORD wCharHInterval;
    WORD wCharVInterval;
};

struct CAPTION_DATA {
    bool bClear;
    WORD wSWFMode;
    WORD wClientX;
    WORD wClientY;
    WORD wClientW;
    WORD wClientH;
    WORD wPosX;
    WORD wPosY;
    std::vector<CAPTION_CHAR_DATA> charList;
    DWORD dwWaitTime;
};

struct ASS_COLOR {
    uint8_t r, g, b, a;
};

struct LINE_STR {
    BYTE            outHLC;     //must ignore low 4bits
    ASS_COLOR       outCharColor;
    BOOL            outUnderLine;
    BOOL            outShadow;
    BOOL            outBold;
    BOOL            outItalic;
    BYTE            outFlushMode;
    int             outStrCount;
    std::string     str;
};

struct CAPTION_LINE {
    UINT            index;
    int64_t         pts;
    DWORD           startTime;
    DWORD           endTime;
    BYTE            outCharSizeMode;
    WORD            outCharW;
    WORD            outCharH;
    WORD            outCharHInterval;
    WORD            outCharVInterval;
    WORD            outPosX;
    WORD            outPosY;
    std::vector<LINE_STR> outStrings;
};

struct ass_setting_t {
    long        SWF0offset;
    long        SWF5offset;
    long        SWF7offset;
    long        SWF9offset;
    long        SWF11offset;
    TCHAR      *Comment1;
    TCHAR      *Comment2;
    TCHAR      *Comment3;
    long        PlayResX;
    long        PlayResY;
    TCHAR      *DefaultFontname;
    long        DefaultFontsize;
    TCHAR      *DefaultStyle;
    TCHAR      *BoxFontname;
    long        BoxFontsize;
    TCHAR      *BoxStyle;
    TCHAR      *RubiFontname;
    long        RubiFontsize;
    TCHAR      *RubiStyle;
};

struct c2a_ts {
    int64_t startPCR;
    int64_t lastPCR;
    int64_t lastPTS;
    int64_t basePCR;
    int64_t basePTS;
    int64_t correctTS;
    c2a_ts();
};

struct Caption2AssPrm {
    int     DelayTime;
    bool    keepInterval;
    BYTE    HLCmode;
    bool    srtornament;
    bool    norubi;
    int     LangType;
    int     detectLength;
    tstring ass_type;
    tstring FileName;
    tstring TargetFileName;
    int     readBufferSize;
    Caption2AssPrm();
};

struct PidInfo {
    uint16_t PMTPid;
    uint16_t CaptionPid;
    uint16_t PCRPid;
    PidInfo();
};

class Caption2Ass {
public:
    Caption2Ass();
    ~Caption2Ass();
    RGY_ERR init(std::shared_ptr<RGYLog> pLog);
    RGY_ERR proc(const uint8_t *data, const int64_t data_size);
private:
    void AddMessage(int log_level, const tstring& str) {
        if (!m_pLog || log_level < m_pLog->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pLog->write(log_level, (_T("cap: ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ...) {
        if (!m_pLog || log_level < m_pLog->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    std::vector<CAPTION_DATA> getCaptionDataList(uint8_t ucLangTag);
    RGY_ERR genCaption(int64_t pts);
    std::vector<AVSubtitle> genAss(uint32_t endTime);

    std::unique_ptr<CaptionDLL> m_dll;
    bool m_streamSync;
    rgy_stream m_stream;
    c2a_ts m_timestamp;
    Caption2AssPrm m_prm;
    PidInfo m_pid;
    std::list<LANG_TAG_INFO_DLL> m_langTagList;
    ass_setting_t m_ass;
    std::vector<std::unique_ptr<CAPTION_LINE>> m_capList;
    std::shared_ptr<RGYLog> m_pLog;
};

#endif //__RGY_CAPTION_H__
