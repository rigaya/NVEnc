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

enum C2AFormat {
    FORMAT_INVALID = 0,
    FORMAT_SRT     = 1,
    FORMAT_ASS     = 2,
    FORMAT_MAX
};

static const CX_DESC list_caption2ass[] = {
    { _T("invalid"), FORMAT_INVALID },
    { _T("srt"),     FORMAT_SRT },
    { _T("ass"),     FORMAT_ASS },
    { NULL, 0 }
};

enum {
    HLC_INVALID = 0,
    HLC_kigou   = 1,
    HLC_box     = 2,
    HLC_draw    = 3
};

static const CX_DESC list_caption2ass_hlc[] ={
    { _T("invalid"), HLC_INVALID },
    { _T("kigou"),   HLC_kigou },
    { _T("box"),     HLC_box },
    { _T("draw"),    HLC_draw },
    { NULL, 0 }
};

#if ENABLE_AVSW_READER

#include "Caption.h"

#define CAPTIONF(x) \
    private: \
        x pf ## x; \
    public: \
        const x f_ ## x() { return pf ## x; };

class CaptionDLL {
public:
    CaptionDLL();
    virtual ~CaptionDLL();
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
    int64_t         startTime;
    int64_t         endTime;
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
    int         SWF0offset;
    int         SWF5offset;
    int         SWF7offset;
    int         SWF9offset;
    int         SWF11offset;
    std::string Comment1;
    std::string Comment2;
    std::string Comment3;
    int         PlayResX;
    int         PlayResY;
    std::string DefaultFontname;
    int         DefaultFontsize;
    std::string DefaultStyle;
    std::string BoxFontname;
    int         BoxFontsize;
    std::string BoxStyle;
    std::string RubiFontname;
    int         RubiFontsize;
    std::string RubiStyle;

    ass_setting_t();
    void set(const tstring& inifile, int width, int height);
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
    bool    norubi;
    int     LangType;
    tstring ass_type;
    tstring FileName;
    Caption2AssPrm();
};

struct PidInfo {
    uint16_t PMTPid;
    uint16_t CaptionPid;
    uint16_t PCRPid;
    PidInfo();
};

struct SrtOut {
    bool ornament;
    int index;

    SrtOut();
};

class Caption2Ass {
public:
    Caption2Ass();
    virtual ~Caption2Ass();
    RGY_ERR init(std::shared_ptr<RGYLog> pLog, C2AFormat format);
    RGY_ERR proc(const uint8_t *data, const size_t data_size, std::vector<AVPacket>& subList);
    void close();
    bool enabled() const { return !!m_dll; };
    void setVidFirstKeyPts(int64_t pts) {
        m_vidFirstKeyPts = pts;
    }

    //assのヘッダを返す
    std::string assHeader() const;

    C2AFormat format() const { return m_format; }

    //入力データがtsかどうかの判定
    bool isTS(const uint8_t *data, const size_t data_size) const;

    //出力解像度の設定
    void setOutputResolution(int w, int h, int sar_x, int sar_y);

    //現在の設定を表示
    void printParam(int log_level);

    //内部データをリセット(seekが発生したときなどに使用する想定)
    void reset();
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
    std::vector<AVPacket> genCaption(int64_t pts);
    std::vector<AVPacket> genAss(int64_t endTime);
    std::vector<AVPacket> genSrt(int64_t endTime);

    std::unique_ptr<CaptionDLL> m_dll;
    C2AFormat m_format;
    bool m_streamSync;
    rgy_stream m_stream;
    c2a_ts m_timestamp;
    Caption2AssPrm m_prm;
    PidInfo m_pid;
    std::list<LANG_TAG_INFO_DLL> m_langTagList;
    ass_setting_t m_ass;
    std::vector<std::unique_ptr<CAPTION_LINE>> m_capList;
    std::shared_ptr<RGYLog> m_pLog;
    int64_t m_vidFirstKeyPts;
    int m_sidebarSize;
    SrtOut m_srt;
};

#endif //#if ENABLE_AVSW_READER

#endif //__RGY_CAPTION_H__
