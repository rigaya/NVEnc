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

#include <string>
#include <sstream>
#include "rgy_osdep.h"
#include "rgy_caption.h"
#include "packet_types.h"

#if ENABLE_AVSW_READER

#define TIMESTAMP_INVALID_VALUE     (-1LL)
#define WRAP_AROUND_VALUE           (1LL << 33)
#define WRAP_AROUND_CHECK_VALUE     ((1LL << 32) - 1)
#define PCR_MAXIMUM_INTERVAL        (100 * 90)

static const TCHAR *DEFAULT_FONT_NAME = _T("MS UI Gothic");
static const TCHAR *DEFAULT_STYLE      = _T("&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,15,0,1,2,2,1,10,10,10,0");
static const TCHAR *DEFAULT_BOX_STYLE  = _T("&HFFFFFFFF,&H000000FF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,0");
static const TCHAR *DEFAULT_RUBI_STYLE = _T("&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,10,0");

ass_setting_t::ass_setting_t() :
    SWF0offset(0),
    SWF5offset(0),
    SWF7offset(0),
    SWF9offset(0),
    SWF11offset(0),
    Comment1(""),
    Comment2(""),
    Comment3(""),
    PlayResX(1920),
    PlayResY(1080),
    DefaultFontname(tchar_to_string(DEFAULT_FONT_NAME)),
    DefaultFontsize(90),
    DefaultStyle(tchar_to_string(DEFAULT_STYLE)),
    BoxFontname(tchar_to_string(DEFAULT_FONT_NAME)),
    BoxFontsize(90),
    BoxStyle(tchar_to_string(DEFAULT_BOX_STYLE)),
    RubiFontname(tchar_to_string(DEFAULT_FONT_NAME)),
    RubiFontsize(50),
    RubiStyle(tchar_to_string(DEFAULT_RUBI_STYLE)) {

}
void ass_setting_t::set(const tstring& inifile, int width, int height) {
#if defined(_WIN32) || defined(_WIN64)
    SWF0offset = GetPrivateProfileInt(_T("SWFModeOffset"), _T("SWF0offset"), 0, inifile.c_str());
    SWF5offset = GetPrivateProfileInt(_T("SWFModeOffset"), _T("SWF5offset"), 0, inifile.c_str());
    SWF7offset = GetPrivateProfileInt(_T("SWFModeOffset"), _T("SWF7offset"), 0, inifile.c_str());
    SWF9offset = GetPrivateProfileInt(_T("SWFModeOffset"), _T("SWF9offset"), 0, inifile.c_str());
    SWF11offset = GetPrivateProfileInt(_T("SWFModeOffset"), _T("SWF11offset"), 0, inifile.c_str());

    static const TCHAR *KEY = (double)width  / (double)height > 1.5 ? _T("Default") : _T("Default43");
    TCHAR buffer[1024];
    GetPrivateProfileString(KEY, _T("Comment1"), _T(""), buffer, _countof(buffer), inifile.c_str()); Comment1 = tchar_to_string(buffer, CP_UTF8);
    GetPrivateProfileString(KEY, _T("Comment2"), _T(""), buffer, _countof(buffer), inifile.c_str()); Comment2 = tchar_to_string(buffer, CP_UTF8);
    GetPrivateProfileString(KEY, _T("Comment3"), _T(""), buffer, _countof(buffer), inifile.c_str()); Comment3 = tchar_to_string(buffer, CP_UTF8);
    PlayResX = GetPrivateProfileInt(KEY, _T("PlayResX"), 0, inifile.c_str());
    PlayResY = GetPrivateProfileInt(KEY, _T("PlayResY"), 0, inifile.c_str());

    GetPrivateProfileString(KEY, _T("DefaultFontname"), DEFAULT_FONT_NAME, buffer, _countof(buffer), inifile.c_str()); DefaultFontname = tchar_to_string(buffer, CP_UTF8);
    DefaultFontsize = GetPrivateProfileInt(KEY, _T("DefaultFontsize"), 0, inifile.c_str());
    GetPrivateProfileString(KEY, _T("DefaultStyle"), DEFAULT_STYLE, buffer, _countof(buffer), inifile.c_str()); DefaultStyle = tchar_to_string(buffer, CP_UTF8);

    GetPrivateProfileString(KEY, _T("BoxFontname"), DEFAULT_FONT_NAME, buffer, _countof(buffer), inifile.c_str()); BoxFontname = tchar_to_string(buffer, CP_UTF8);
    BoxFontsize = GetPrivateProfileInt(KEY, _T("BoxFontsize"), 0, inifile.c_str());
    GetPrivateProfileString(KEY, _T("BoxStyle"), DEFAULT_BOX_STYLE, buffer, _countof(buffer), inifile.c_str()); BoxStyle = tchar_to_string(buffer, CP_UTF8);

    GetPrivateProfileString(KEY, _T("RubiFontname"), DEFAULT_FONT_NAME, buffer, _countof(buffer), inifile.c_str()); RubiFontname = tchar_to_string(buffer, CP_UTF8);
    RubiFontsize = GetPrivateProfileInt(KEY, _T("RubiFontsize"), 0, inifile.c_str());
    GetPrivateProfileString(KEY, _T("RubiStyle"), DEFAULT_RUBI_STYLE, buffer, _countof(buffer), inifile.c_str()); RubiStyle = tchar_to_string(buffer, CP_UTF8);
#endif //#if defined(_WIN32) || defined(_WIN64)
}

static const unsigned char utf8_bom[3] = { 0xEF, 0xBB, 0xBF };

static const char HiraTable[][3] = {
    "ぁ", "あ", "ぃ", "い", "ぅ", "う", "ぇ",
    "え", "ぉ", "お", "か", "が", "き", "ぎ", "く",
    "ぐ", "け", "げ", "こ", "ご", "さ", "ざ", "し",
    "じ", "す", "ず", "せ", "ぜ", "そ", "ぞ", "た",
    "だ", "ち", "ぢ", "っ", "つ", "づ", "て", "で",
    "と", "ど", "な", "に", "ぬ", "ね", "の", "は",
    "ば", "ぱ", "ひ", "び", "ぴ", "ふ", "ぶ", "ぷ",
    "へ", "べ", "ぺ", "ほ", "ぼ", "ぽ", "ま", "み",
    "む", "め", "も", "ゃ", "や", "ゅ", "ゆ", "ょ",
    "よ", "ら", "り", "る", "れ", "ろ", "ゎ", "わ",
    "ゐ", "ゑ", "を", "ん", "　", "　", "　", "ゝ",
    "ゞ", "ー", "。", "「", "」", "、", "・",
    "！", "”", "＃", "＄", "％", "＆", "’",
    "（", "）", "＊", "＋", "，", "－", "．", "／",
    "０", "１", "２", "３", "４", "５", "６", "７",
    "８", "９", "：", "；", "＜", "＝", "＞", "？",
    "＠", "Ａ", "Ｂ", "Ｃ", "Ｄ", "Ｅ", "Ｆ", "Ｇ",
    "Ｈ", "Ｉ", "Ｊ", "Ｋ", "Ｌ", "Ｍ", "Ｎ", "Ｏ",
    "Ｐ", "Ｑ", "Ｒ", "Ｓ", "Ｔ", "Ｕ", "Ｖ", "Ｗ",
    "Ｘ", "Ｙ", "Ｚ", "［", "￥", "］", "＾", "＿",
    "‘", "ａ", "ｂ", "ｃ", "ｄ", "ｅ", "ｆ", "ｇ",
    "ｈ", "ｉ", "ｊ", "ｋ", "ｌ", "ｍ", "ｎ", "ｏ",
    "ｐ", "ｑ", "ｒ", "ｓ", "ｔ", "ｕ", "ｖ", "ｗ",
    "ｘ", "ｙ", "ｚ", "｛", "｜", "｝", "￣"
};

static const char HalfHiraTable[][3] = {
    "ぁ", "あ", "ぃ", "い", "ぅ", "う", "ぇ",
    "え", "ぉ", "お", "か", "が", "き", "ぎ", "く",
    "ぐ", "け", "げ", "こ", "ご", "さ", "ざ", "し",
    "じ", "す", "ず", "せ", "ぜ", "そ", "ぞ", "た",
    "だ", "ち", "ぢ", "っ", "つ", "づ", "て", "で",
    "と", "ど", "な", "に", "ぬ", "ね", "の", "は",
    "ば", "ぱ", "ひ", "び", "ぴ", "ふ", "ぶ", "ぷ",
    "へ", "べ", "ぺ", "ほ", "ぼ", "ぽ", "ま", "み",
    "む", "め", "も", "ゃ", "や", "ゅ", "ゆ", "ょ",
    "よ", "ら", "り", "る", "れ", "ろ", "ゎ", "わ",
    "ゐ", "ゑ", "を", "ん", " ", " ", " ", "ゝ",
    "ゞ", "ｰ", "｡", "｢", "｣", "､", "･",
    "!", "\"", "#", "$", "%", "&", "'",
    "(", ")", "*", "+", ", ", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z", "[", "￥", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "{", "|", "}", "￣"
};

static const char KanaTable[][3] = {
    "ァ", "ア", "ィ", "イ", "ゥ", "ウ", "ェ",
    "エ", "ォ", "オ", "カ", "ガ", "キ", "ギ", "ク",
    "グ", "ケ", "ゲ", "コ", "ゴ", "サ", "ザ", "シ",
    "ジ", "ス", "ズ", "セ", "ゼ", "ソ", "ゾ", "タ",
    "ダ", "チ", "ヂ", "ッ", "ツ", "ヅ", "テ", "デ",
    "ト", "ド", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ",
    "バ", "パ", "ヒ", "ビ", "ピ", "フ", "ブ", "プ",
    "ヘ", "ベ", "ペ", "ホ", "ボ", "ポ", "マ", "ミ",
    "ム", "メ", "モ", "ャ", "ヤ", "ュ", "ユ", "ョ",
    "ヨ", "ラ", "リ", "ル", "レ", "ロ", "ヮ", "ワ",
    "ヰ", "ヱ", "ヲ", "ン", "ヴ", "ヵ", "ヶ", "ヽ",
    "ヾ", "ー", "。", "「", "」", "、", "・"
};

static const char HalfKanaTable[][3] = {
    "ｧ", "ｱ", "ｨ", "ｲ", "ｩ", "ｳ", "ｪ",
    "ｴ", "ｫ", "ｵ", "ｶ", "ｶﾞ", "ｷ", "ｷﾞ", "ｸ",
    "ｸﾞ", "ｹ", "ｹﾞ", "ｺ", "ｺﾞ", "ｻ", "ｻﾞ", "ｼ",
    "ｼﾞ", "ｽ", "ｽﾞ", "ｾ", "ｾﾞ", "ｿ", "ｿﾞ", "ﾀ",
    "ﾀﾞ", "ﾁ", "ﾁﾞ", "ｯ", "ﾂ", "ﾂﾞ", "ﾃ", "ﾃﾞ",
    "ﾄ", "ﾄﾞ", "ﾅ", "ﾆ", "ﾇ", "ﾈ", "ﾉ", "ﾊ",
    "ﾊﾞ", "ﾊﾟ", "ﾋ", "ﾋﾞ", "ﾋﾟ", "ﾌ", "ﾌﾞ", "ﾌﾟ",
    "ﾍ", "ﾍﾞ", "ﾍﾟ", "ﾎ", "ﾎﾞ", "ﾎﾟ", "ﾏ", "ﾐ",
    "ﾑ", "ﾒ", "ﾓ", "ｬ", "ﾔ", "ｭ", "ﾕ", "ｮ",
    "ﾖ", "ﾗ", "ﾘ", "ﾙ", "ﾚ", "ﾛ", "ヮ", "ﾜ",
    "ヰ", "ヱ", "ｦ", "ﾝ", "ｳﾞ", "ヵ", "ヶ", "ヽ",
    "ヾ", "ｰ", "｡", "｢", "｣", "､", "･"
};

std::string GetHalfChar(std::string key) {
    CHAR ret[STRING_BUFFER_SIZE] = { 0 };
    BOOL bMatch = FALSE;

    // マッチしない文字は、そのまま使用
    const char *_p = key.c_str();
    char *p = (char *)key.c_str();

    while (p < _p + key.size()) {
        for (int i = 0; i < sizeof(HiraTable) / sizeof(HiraTable[0]) && p < _p + key.size(); i++) {
            bMatch = FALSE;
            if (memcmp(p, HiraTable[i], 2) == 0) {
                strcat_s( ret, STRING_BUFFER_SIZE, HalfHiraTable[i] );
                p += 2;
                bMatch = TRUE;
                i = -1;
            }
        }

        for (int i = 0; i < sizeof(KanaTable) / sizeof(KanaTable[0]) && p < _p + key.size(); i++) {
            bMatch = FALSE;
            if (memcmp(p, KanaTable[i], 2) == 0) {
                strcat_s(ret, STRING_BUFFER_SIZE, HalfKanaTable[i]);
                p += 2;
                bMatch = TRUE;
                i = -1;
            }
        }

        if (p < _p + key.size()) {
            strncat_s(ret, STRING_BUFFER_SIZE, p, 2);
            p += 2;
        }
    }

    return ret;
}


struct HALFCHAR_INFO {
    int     char_nums;
    int     point_nums;
    int     half_nums;
};

static int count_utf8_length(const unsigned char *string, HALFCHAR_INFO *hc) {
    int len         = 0;
    int point_count = 0;
    int half_count  = 0;

    while (*string) {
        if (string[0] == 0x00)
            break;

        if (string[0] < 0x1f || string[0] == 0x7f)
            // 制御コード
            ;
        else if (string[0] <= 0x7f) {
            // 1バイト文字
            ++len;              // 半角
            ++half_count;
        } else if (string[0] <= 0xbf)
            // 文字の続き
            ;
        else if (string[0] <= 0xdf) {
            // 2バイト文字
            len += 2;
            if (string[0] == 0xc2 && string[1] == 0xa5) {
                --len;          // 半角の￥
                ++half_count;
            }
        } else if (string[0] <= 0xef) {
            // 3バイト文字
            len += 2;
            if (string[0] == 0xe2 && string[1] == 0x80 && string[2] == 0xbe) {
                --len;          // 半角の￣
                ++half_count;
            } else if (string[0] == 0xef) {
                if (string[1] == 0xbd) {
                    if (string[2] >= 0xa1 && string[2] <= 0xbf) {
                        --len;  // 半角カナ 「。」～「ソ」
                        ++half_count;
                    }
                } else if (string[1] == 0xbe) {
                    if (string[2] >= 0x80 && string[2] <= 0x9f) {
                        --len;  // 半角カナ 「タ」～「゜」
                        ++half_count;
                    }
                    if (string[2] == 0x9e || string[2] == 0x9f) {
                        ++point_count;  // 濁点・半濁点をカウント
                        --half_count;
                    }
                }
            }
        } else if (string[0] <= 0xf7)
            // 4バイト文字
            len += 2;
        else if (string[0] <= 0xfb)
            // 5バイト文字
            len += 2;
        else if (string[0] <= 0xfd)
            // 6バイト文字
            len += 2;
        else
            // 使われていない範囲
            ;

        ++string;
    }

    if (hc) {
        hc->char_nums  = len;
        hc->point_nums = point_count;
        hc->half_nums  = half_count;
    }
    return len;
}

static int FindStartOffset(rgy_stream& st) {
    const char *const start_ptr = (char *)st.data();
    const char * fin_ptr = start_ptr + st.size();
    for (const char *ptr = start_ptr; ptr < fin_ptr; ptr++) {
        if (ptr[0] == 'G' && ptr[188] == 'G') {
            st.add_offset(ptr - start_ptr);
            return 0;
        }
    }
    return 1;
}

static int resync(void *pbPacket, rgy_stream& st) {
    char *ptr = (char *)memchr(pbPacket, 'G', 188);
    if (!ptr) {
        for (int i = 0; i < 20; i++) {
            if (st.size() < 188) {
                return 1;
            }
            memcpy(pbPacket, st.data(), 188);
            st.add_offset(188);
            ptr = (char *)memchr(pbPacket, 'G', 188);
            if (ptr != nullptr) {
                break;
            }
        }
    }
    if (!ptr) {
        return 1;
    }
    auto pos = ptr - (char *)pbPacket;
    st.add_offset(-(188 - pos));
    return 0;
}

static int64_t GetPTS(uint8_t *pbPacket) {
    int64_t PTS = TIMESTAMP_INVALID_VALUE;
    // Get PTS in PES Header(00 00 01 BD)
    for (int i = 4; i < 188 - 10; i++) {
        if (   pbPacket[i + 0] == 0x00
            && pbPacket[i + 1] == 0x00
            && pbPacket[i + 2] == 0x01
            && pbPacket[i + 3] == 0xBD) {

            uint8_t *pData = &pbPacket[i + 9];

            PTS = (int64_t)(((uint32_t)(*pData) & 0xE) >> 1) << 30;
            pData++;

            PTS += (uint32_t)(*pData) << 22;
            pData++;

            PTS += (uint32_t)((uint32_t)(*pData) >> 1) << 15;
            pData++;

            PTS += (uint32_t)(*pData) << 7;
            pData++;

            PTS += (uint32_t)(*pData) >> 1;

            //PTS = PTS / 90;

            break;
        }
    }
    return PTS;
}

static void parse_PAT(uint8_t *pbPacket, USHORT *PMTPid) {
    PAT_HEADER *pat = (PAT_HEADER *)(pbPacket + sizeof(_Packet_Header) + 1);

    for (int i = 0; i < (188 - 13) / 4; i++) {
        uint16_t wProgramID = swap16(pat->PMT_Array[i].program_id);
        uint16_t wPID       = swap16(pat->PMT_Array[i].PID) & 0x1FFF;
        if (wProgramID == 0xFFFF)
            break;

        if (wProgramID != 0 && *PMTPid == 0)    //the first PMTPid found
            *PMTPid = wPID;
    }
}

static void parse_PMT(uint8_t *pbPacket, USHORT *PCRPid, USHORT *CaptionPid) {
    PMT_HEADER *pmt = (PMT_HEADER *)(pbPacket + sizeof(_Packet_Header) + 1);

    if (*PCRPid == 0)
        *PCRPid = swap16(pmt->pcrpid) & 0x1FFF;

    int length = swap16(pmt->program_info_length) & 0x0FFF;
    uint8_t *pData = (uint8_t *)&pmt->program_info_length + 2;
    pData += length;    //read thrugh program_info

    while (pData < pbPacket + 184) {
        PMT_PID_Desc *pmt_pid = (PMT_PID_Desc *)&pData[0];

        if (pmt_pid->StreamTypeID == 0x6) {
            bool bcomponent_tag = false;
            int iDescLen = swap16(pmt_pid->DescLen) & 0x0FFF;
            for (int i = 0; i < iDescLen -2; i++) {
                if (pData[i + 5] == 0x52 && pData[i + 6] == 0x01 && pData[i + 7] == 0x30) {
                    bcomponent_tag = true;
                    break;
                }
            }
            if (bcomponent_tag) {
                *CaptionPid = (swap16(pmt_pid->EsPID) & 0x1FFF);
                break;
            }
        }
        pData += ((swap16(pmt_pid->DescLen) & 0x0FFF) + 5);
    }
}

static void parse_Packet_Header(Packet_Header *packet_header, uint8_t *pbPacket) {
    _Packet_Header *packet = (_Packet_Header *)pbPacket;

    packet_header->Sync             = packet->Sync;
    packet_header->TsErr            = (swap16(packet->PID) >> 15) & 0x01;
    packet_header->PayloadStartFlag = (swap16(packet->PID) >> 14) & 0x01;
    packet_header->Priority         = (swap16(packet->PID) >> 13) & 0x01;
    packet_header->PID              = (swap16(packet->PID) & 0x1FFF);
    packet_header->Scramble         = (packet->Counter & 0xC0) >> 6;
    packet_header->AdaptFlag        = (packet->Counter & 0x20) >> 5;
    packet_header->PayloadFlag      = (packet->Counter & 0x10) >> 4;
    packet_header->Counter          = (packet->Counter & 0x0F);
}

CaptionDLL::CaptionDLL() :
    m_hModule(),
    pfInitializeCP(nullptr),
    pfInitializeUNICODECP(nullptr),
    pfUnInitializeCP(nullptr),
    pfAddTSPacketCP(nullptr),
    pfClearCP(nullptr),
    pfGetTagInfoCP(nullptr),
    pfGetCaptionDataCP(nullptr),
    unicode_(false) {

}
CaptionDLL::~CaptionDLL() {
    m_hModule.reset();
}
RGY_ERR CaptionDLL::load() {
    m_hModule = std::unique_ptr<void, module_deleter>(LoadLibrary(_T("Caption.dll")), module_deleter());
    if (!m_hModule) {
        return RGY_ERR_INVALID_CALL;
    }
#define LOADADDR(x) (pf ## x = (x)GetProcAddress((HMODULE)m_hModule.get(), #x))

    if (   LOADADDR(InitializeCP) == NULL
        || LOADADDR(UnInitializeCP) == NULL
        || LOADADDR(AddTSPacketCP) == NULL
        || LOADADDR(ClearCP) == NULL
        || LOADADDR(GetTagInfoCP) == NULL
        || LOADADDR(GetCaptionDataCP) == NULL) {
        return RGY_ERR_INVALID_CALL;
    }

#undef LOADADDR
    return RGY_ERR_NONE;
}

RGY_ERR CaptionDLL::init() {
    pfInitializeUNICODECP = (InitializeUNICODECP)(GetProcAddress((HMODULE)m_hModule.get(), "InitializeUNICODE"));
    if (pfInitializeUNICODECP) {
        unicode_ = true;
        return (pfInitializeUNICODECP() != NO_ERR) ? RGY_ERR_INVALID_CALL : RGY_ERR_NONE;
    } else {
        unicode_ = false;
        return (pfInitializeCP() != NO_ERR) ? RGY_ERR_INVALID_CALL : RGY_ERR_NONE;
    }
}

c2a_ts::c2a_ts() :
    startPCR(TIMESTAMP_INVALID_VALUE),
    lastPCR(TIMESTAMP_INVALID_VALUE),
    lastPTS(TIMESTAMP_INVALID_VALUE),
    basePCR(0),
    basePTS(0),
    correctTS(0) {
}

PidInfo::PidInfo() :
    PMTPid(0),
    CaptionPid(0),
    PCRPid(0) {
}

SrtOut::SrtOut() :
    ornament(true),
    index(0) {
}

Caption2AssPrm::Caption2AssPrm() :
    DelayTime(0),
    keepInterval(true),
    HLCmode(HLC_kigou),
    norubi(false),
    LangType(1),
    ass_type(),
    FileName() {
}

Caption2Ass::Caption2Ass() :
    m_dll(),
    m_format(FORMAT_INVALID),
    m_streamSync(false),
    m_stream(),
    m_timestamp(),
    m_prm(),
    m_pid(),
    m_langTagList(),
    m_ass(),
    m_capList(),
    m_pLog(),
    m_vidFirstKeyPts(0),
    m_sidebarSize(0),
    m_srt() {
    m_stream.init();
}
Caption2Ass::~Caption2Ass() {
    close();
}

void Caption2Ass::close() {
    m_pLog.reset();
}

RGY_ERR Caption2Ass::init(std::shared_ptr<RGYLog> pLog, C2AFormat format) {
    m_pLog = pLog;
    m_dll.reset(new CaptionDLL());
    auto ret = m_dll->load();
    if (ret != RGY_ERR_NONE) {
        m_dll.reset();
        AddMessage(RGY_LOG_ERROR, _T("Failed to load Caption.dll.\n"));
        return ret;
    }

    ret = m_dll->init();
    if (ret != RGY_ERR_NONE) {
        m_dll.reset();
        AddMessage(RGY_LOG_ERROR, _T("Failed to init Caption.dll.\n"));
        return ret;
    }
    m_format = format;
    if (m_format != FORMAT_ASS && m_format != FORMAT_SRT) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid format specified.\n"));
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    return RGY_ERR_NONE;
}

std::vector<CAPTION_DATA> Caption2Ass::getCaptionDataList(uint8_t ucLangTag) {
    std::vector<CAPTION_DATA> captionList;
    CAPTION_DATA_DLL *pListDll = nullptr;
    DWORD count = 0;
    int ret = m_dll->f_GetCaptionDataCP()(ucLangTag, &pListDll, &count);
    if (ret == TRUE) {
        captionList.reserve(count);
        for (DWORD i = 0; i < count; i++) {
            CAPTION_DATA data;
            data.bClear = !!pListDll[i].bClear;
            data.wSWFMode = pListDll[i].wSWFMode;
            data.wClientX = pListDll[i].wClientX;
            data.wClientY = pListDll[i].wClientY;
            data.wClientW = pListDll[i].wClientW;
            data.wClientH = pListDll[i].wClientH;
            data.wPosX = pListDll[i].wPosX;
            data.wPosY = pListDll[i].wPosY;
            data.dwWaitTime = pListDll[i].dwWaitTime;

            data.charList.reserve(pListDll[i].dwListCount);
            for (DWORD j = 0; j < pListDll[i].dwListCount; j++) {
                CAPTION_CHAR_DATA charData;
                charData.strDecode = pListDll[i].pstCharList[j].pszDecode;
                charData.wCharSizeMode = pListDll[i].pstCharList[j].wCharSizeMode;
                charData.stCharColor = pListDll[i].pstCharList[j].stCharColor;
                charData.stBackColor = pListDll[i].pstCharList[j].stBackColor;
                charData.stRasterColor = pListDll[i].pstCharList[j].stRasterColor;
                charData.bUnderLine = pListDll[i].pstCharList[j].bUnderLine;
                charData.bShadow = pListDll[i].pstCharList[j].bShadow;
                charData.bBold = pListDll[i].pstCharList[j].bBold;
                charData.bItalic = pListDll[i].pstCharList[j].bItalic;
                charData.bFlushMode = pListDll[i].pstCharList[j].bFlushMode;
                charData.bHLC = pListDll[i].pstCharList[j].bHLC;
                charData.wCharW = pListDll[i].pstCharList[j].wCharW;
                charData.wCharH = pListDll[i].pstCharList[j].wCharH;
                charData.wCharHInterval = pListDll[i].pstCharList[j].wCharHInterval;
                charData.wCharVInterval = pListDll[i].pstCharList[j].wCharVInterval;
                data.charList.push_back(charData);
            }
            captionList.push_back(data);
        }
    }
    return std::move(captionList);
}

//assのヘッダを返す
std::string Caption2Ass::assHeader() const {
    std::stringstream ss;
    ss << "[Script Info]" << std::endl;
    ss << "; " << m_ass.Comment1 << std::endl;
    ss << "; " << m_ass.Comment2 << std::endl;
    ss << "; " << m_ass.Comment3 << std::endl;
    ss << "Title: Default Aegisub file" << std::endl;
    ss << "ScriptType: v4.00+" << std::endl;
    ss << "WrapStyle: 0" << std::endl;
    ss << "PlayResX: " << m_ass.PlayResX << std::endl;
    ss << "PlayResY: " << m_ass.PlayResY << std::endl;
    ss << "ScaledBorderAndShadow: yes" << std::endl;
    ss << "Video Aspect Ratio: 0" << std::endl;
    ss << "Video Zoom: 6" << std::endl;
    ss << "Video Position: 0" << std::endl;
    ss <<  std::endl;
    ss << "[V4+ Styles]" << std::endl;
    ss << "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding" << std::endl;
    ss << "Style: Default," << m_ass.DefaultFontname << "," << m_ass.DefaultFontsize << "," << m_ass.DefaultStyle << std::endl;
    ss << "Style: Box," << m_ass.BoxFontname << "," << m_ass.BoxFontsize << "," << m_ass.BoxStyle << std::endl;
    ss << "Style: Rubi," << m_ass.RubiFontname << "," << m_ass.RubiFontsize << "," << m_ass.RubiStyle << std::endl;
    ss << "//" << std::endl;
    return ss.str();
}

//内部データをリセット(seekが発生したときなどに使用する想定)
void Caption2Ass::reset() {
    m_streamSync = false;
    m_stream.clear();
    m_timestamp = c2a_ts();
    m_pid = PidInfo();
    m_langTagList.clear();
    m_capList.clear();
    m_dll->init();
}

//入力データがtsかどうかの判定
bool Caption2Ass::isTS(const uint8_t *data, const size_t data_size) const {
    rgy_stream st;
    st.append(data, data_size);
    return FindStartOffset(st) == 0;
}

void Caption2Ass::setOutputResolution(int w, int h, int sar_x, int sar_y) {
    m_ass.PlayResX = w;
    m_ass.PlayResY = h;
#if defined(_WIN32) || defined(_WIN64)
    TCHAR buf[1024] = { 0 };
    GetModuleFileName(NULL, buf, _countof(buf));
    auto exeDir = PathRemoveFileSpecFixed(buf);
    auto list = get_file_list(_T("Caption2Ass*.ini"), exeDir.second);
    if (list.size() > 0) {
        AddMessage(RGY_LOG_INFO, _T("load ass settings from \"%s\""), list[0].c_str());
        m_ass.set(list[0], w, h);
    }
#endif
    if (sar_x > 0 && sar_y > 0) {
        if (sar_x > sar_y) {
            m_ass.PlayResX = m_ass.PlayResX * sar_x / sar_y;
        } else {
            m_ass.PlayResY = m_ass.PlayResY * sar_y / sar_x;
        }
    }
    m_sidebarSize = 0;
    AddMessage(RGY_LOG_DEBUG, _T("PlayResX: %d, PlayResY: %d, m_sidebarSize: %d.\n"), m_ass.PlayResX, m_ass.PlayResY, m_sidebarSize);
}

void Caption2Ass::printParam(int log_level) {
    if (!m_pLog || log_level < m_pLog->getLogLevel()) {
        return;
    }
    AddMessage(log_level, _T("caption2ass:   %s\n"), get_chr_from_value(list_caption2ass, m_format));
    AddMessage(log_level, _T(" DelayTime:    %d\n"), m_prm.DelayTime);
    AddMessage(log_level, _T(" keepInterval: %s\n"), m_prm.keepInterval ? _T("yes") : _T("no"));
    AddMessage(log_level, _T(" HLCmode:      %d\n"), get_chr_from_value(list_caption2ass_hlc, m_prm.HLCmode));
    AddMessage(log_level, _T(" norubi:       %s\n"), m_prm.norubi ? _T("yes") : _T("no"));
    AddMessage(log_level, _T(" LangType:     %d\n"), m_prm.LangType);
    AddMessage(log_level, _T(" ass_type:     %s\n"), m_prm.ass_type.c_str());
    AddMessage(log_level, _T(" sidebarSize:  %d\n"), m_sidebarSize);
    AddMessage(log_level, _T(" srt ornament: %s\n"), m_srt.ornament ? _T("yes") : _T("no"));
}

RGY_ERR Caption2Ass::proc(const uint8_t *data, const size_t data_size, std::vector<AVPacket>& subList) {
    m_stream.append(data, data_size);

    if (m_streamSync) {
        if (!FindStartOffset(m_stream)) {
            return RGY_ERR_UNKNOWN;
        }
        m_streamSync = true;
    }

    bool bPrintPMT = true;
    uint8_t pbPacket[188 * 2 + 4] = { 0 };
    uint32_t packetCount = 0;

    while (m_stream.size() >= 188) {
        memcpy(pbPacket, m_stream.data(), 188);
        m_stream.add_offset(188);
        packetCount++;

        Packet_Header packet;
        parse_Packet_Header(&packet, &pbPacket[0]);

        if (packet.Sync != 'G') {
            if (!resync(pbPacket, m_stream)) {
                return RGY_ERR_UNKNOWN;
            }
            continue;
        }

        if (packet.TsErr)
            continue;

        // PAT
        if (packet.PID == 0 && (m_pid.PMTPid == 0 || bPrintPMT)) {
            parse_PAT(&pbPacket[0], &(m_pid.PMTPid));
            bPrintPMT = false;

            continue; // next packet
        }

        // PMT
        if (m_pid.PMTPid != 0 && packet.PID == m_pid.PMTPid) {
            if (pbPacket[5] != 0x02 || (pbPacket[6] & 0xf0) != 0xb0)
                /*--------------------------------------------------
                * pbPacket[5]  (8)  table_id
                * pbPacket[6]  (1)  section_syntax_indicator
                *              (1)  '0'
                *              (2)  reserved '11'
                *------------------------------------------------*/
                continue;   // next packet

            parse_PMT(&pbPacket[0], &(m_pid.PCRPid), &(m_pid.CaptionPid));

            if (m_timestamp.lastPTS == TIMESTAMP_INVALID_VALUE) {
                AddMessage(RGY_LOG_TRACE, _T("PMT, PCR, Caption : %04x, %04x, %04x\n"), m_pid.PMTPid, m_pid.PCRPid, m_pid.CaptionPid);
            }

            continue; // next packet
        }

        // PCR
        if (m_pid.PCRPid != 0 && packet.PID == m_pid.PCRPid) {
            uint32_t bADP = (((uint32_t)pbPacket[3] & 0x30) >> 4);
            if (!(bADP & 0x2))
                continue; // next packet

            uint32_t bAF = (uint32_t)pbPacket[5];
            if (!(bAF & 0x10))
                continue; // next packet

            // Get PCR.
            /*     90kHz           27MHz
            *  +--------+-------+-------+
            *  | 33 bits| 6 bits| 9 bits|
            *  +--------+-------+-------+
            */
            int64_t PCR_base =
                  ((int64_t)pbPacket[ 6] << 25)
                | ((int64_t)pbPacket[ 7] << 17)
                | ((int64_t)pbPacket[ 8] <<  9)
                | ((int64_t)pbPacket[ 9] <<  1)
                | ((int64_t)pbPacket[10] >>  7);
            int64_t PCR_ext =
                 ((int64_t)(pbPacket[10] & 0x01) << 8)
                |  (int64_t)pbPacket[11];
            int64_t PCR = PCR_base + PCR_ext / 300;

            if (m_timestamp.lastPTS == TIMESTAMP_INVALID_VALUE) {
                AddMessage(RGY_LOG_TRACE, _T("PCR, startPCR, lastPCR, basePCR : %11lld, %11lld, %11lld, %11lld\n"),
                    PCR, m_timestamp.startPCR, m_timestamp.lastPCR, m_timestamp.basePCR);
            }

            // Check startPCR.
            if (m_timestamp.startPCR == TIMESTAMP_INVALID_VALUE) {
                m_timestamp.startPCR  = PCR;
                m_timestamp.correctTS = m_prm.DelayTime;
            } else {
                int64_t checkTS = 0;
                // Check wrap-around.
                if (PCR < m_timestamp.lastPCR) {
                    AddMessage(RGY_LOG_DEBUG, _T("====== PCR less than lastPCR ======\n"));
                    AddMessage(RGY_LOG_DEBUG, _T("PCR, startPCR, lastPCR, basePCR : %11lld, %11lld, %11lld, %11lld\n"),
                        PCR, m_timestamp.startPCR, m_timestamp.lastPCR, m_timestamp.basePCR);
                    m_timestamp.basePCR += WRAP_AROUND_VALUE;
                    checkTS = WRAP_AROUND_VALUE;
                }
                // Check drop packet. (This is even if the CM cut.)
                checkTS += PCR;
                if (checkTS > m_timestamp.lastPCR) {
                    checkTS -= m_timestamp.lastPCR;
                    if (!(m_prm.keepInterval) && (checkTS > PCR_MAXIMUM_INTERVAL)) {
                        m_timestamp.correctTS -= checkTS - (PCR_MAXIMUM_INTERVAL >> 2);
                    }
                }
            }

            // Update lastPCR.
            m_timestamp.lastPCR = PCR;

            continue; // next packet
        }

        // Caption
        if (m_pid.CaptionPid != 0 && packet.PID == m_pid.CaptionPid) {

            int64_t PTS = 0;

            if (packet.PayloadStartFlag) {
#if 0
                // FIXME: Check PTS flag in PES Header.
                // [example]
                //if (!(packet.pts_flag))
                //    continue;
#endif

                // Get Caption PTS.
                PTS = GetPTS(pbPacket);
                AddMessage(RGY_LOG_TRACE, _T("PTS, lastPTS, basePTS, startPCR : %11lld, %11lld, %11lld, %11lld    "),
                    PTS, m_timestamp.lastPTS, m_timestamp.basePTS, m_timestamp.startPCR);

                // Check skip.
                if (PTS == TIMESTAMP_INVALID_VALUE || m_timestamp.startPCR == TIMESTAMP_INVALID_VALUE) {
                    //if (log->active)
                    //    AddMessage(RGY_LOG_TRACE, "Skip 1st caption\n");
                    continue;
                }

                // Check wrap-around.
                // [case]
                //   lastPCR:  Detection on the 1st packet.             [1st PCR  >>> w-around >>> 1st PTS]
                //   lastPTS:  Detection on the packet of 2nd or later. [prev PTS >>> w-around >>> now PTS]
                int64_t checkTS = (m_timestamp.lastPTS == TIMESTAMP_INVALID_VALUE) ? m_timestamp.lastPCR : m_timestamp.lastPTS;
                if ((PTS < checkTS) && ((checkTS - PTS) >(WRAP_AROUND_CHECK_VALUE))) {
                    m_timestamp.basePTS += WRAP_AROUND_VALUE;
                }

                // Update lastPTS.
                m_timestamp.lastPTS = PTS;

            } else {
                AddMessage(RGY_LOG_TRACE, _T("PTS, lastPTS, basePTS, startPCR : %11lld, %11lld, %11lld, %11lld    "),
                    PTS, m_timestamp.lastPTS, m_timestamp.basePTS, m_timestamp.startPCR);

                // Check skip.
                if (m_timestamp.lastPTS == TIMESTAMP_INVALID_VALUE || m_timestamp.startPCR == TIMESTAMP_INVALID_VALUE) {
                    AddMessage(RGY_LOG_TRACE, _T("Skip 2nd caption\n"));
                    continue;
                }

                // Get Caption PTS from 1st caption.
                PTS = m_timestamp.lastPTS;
            }

            // Correct PTS for output.
            PTS += m_timestamp.basePTS + m_timestamp.correctTS;

            rgy_time time((PTS > m_timestamp.startPCR) ? (PTS - m_timestamp.startPCR) / 90 : 0);
            if (packet.PayloadStartFlag) {
                AddMessage(RGY_LOG_TRACE, _T("%s Caption Time: %01d:%02d:%02d.%03d\n"),
                    ((packet.PayloadStartFlag) ? _T("1st") : _T("2nd")), time.h, time.m, time.s, time.ms);
            }

            auto ret = m_dll->f_AddTSPacketCP()((uint8_t *)pbPacket);
            if (ret == CHANGE_VERSION) {
                LANG_TAG_INFO_DLL *ptrListDll;
                DWORD count;
                if ((ret = m_dll->f_GetTagInfoCP()(&ptrListDll, &count)) == TRUE) {
                    m_langTagList.clear();
                    for (DWORD i = 0; i < count; i++) {
                        m_langTagList.push_back(ptrListDll[i]);
                    }
                }
            } else if (ret == NO_ERR_CAPTION) {
                vector_cat(subList, genCaption(PTS));
            }
        }
    }
    return RGY_ERR_NONE;
}

std::vector<AVPacket> Caption2Ass::genCaption(int64_t PTS) {
    std::vector<AVPacket> subList;
    int   workCharSizeMode  = 0;
    int   workCharW         = 0;
    int   workCharH         = 0;
    int   workCharHInterval = 0;
    int   workCharVInterval = 0;
    int   workPosX          = 0;
    int   workPosY          = 0;
    BYTE  workHLC           = HLC_kigou;    //must ignore low 4bits
    WORD  wLastSWFMode      = 999;
    int   offsetPosX        = 0;
    int   offsetPosY        = 0;
    double ratioX = 1.0;
    double ratioY = 1.0;

    // Get language tag.
    uint32_t ucLangTag = m_prm.LangType - 1; // m_dll->f_GetLangTagCP()(m_prm.LangType);

    // Output
    std::vector<CAPTION_DATA> captionList = getCaptionDataList((ucLangTag < m_langTagList.size()) ? (unsigned char)ucLangTag : 0);

    int addSpaceNum = 0;
    std::unique_ptr<CAPTION_LINE> pCapLine;
    for (auto itcap = captionList.begin(); itcap != captionList.end(); itcap++) {
        if (itcap->bClear && !pCapLine) {
            // 字幕のスキップをチェック
            if ((PTS + itcap->dwWaitTime * 90) <= m_timestamp.startPCR) {
                AddMessage(RGY_LOG_DEBUG, _T("%d Caption skip\n"), captionList.size());
            } else {
                //endTimeはstartTime同様、startPCRを基準とする
                int64_t endTime = (PTS + itcap->dwWaitTime * 90) - m_timestamp.startPCR;
                std::vector<AVPacket> pkts;
                switch (m_format) {
                case FORMAT_ASS: pkts = genAss(endTime); break;
                case FORMAT_SRT: pkts = genSrt(endTime); break;
                default: break;
                }
                vector_cat(subList, pkts);
            }
            m_capList.clear();
            continue;
        }

        AddMessage(RGY_LOG_TRACE,
            _T("  SWFMode        : %4d\n")
            _T("  Client X:Y:W:H : %4d\t%4d\n")
            _T("  Pos    X:Y     : %4d\t%4d\n"),
            itcap->wSWFMode,
            itcap->wClientX, itcap->wClientY, itcap->wClientW, itcap->wClientH,
            itcap->wPosX, itcap->wPosY);

        int wPosX = itcap->wPosX;
        int wPosY = itcap->wPosY;

        if (itcap->wSWFMode != wLastSWFMode) {
            wLastSWFMode = itcap->wSWFMode;
            static const auto resolution = make_array<std::pair<int, int>>(
                std::make_pair(1920, 1080),
                std::make_pair( 720,  480),
                std::make_pair(1280,  720),
                std::make_pair( 960,  540)
                );
            const int index = (wLastSWFMode ==  5) ? 0
                            : (wLastSWFMode ==  9) ? 1
                            : (wLastSWFMode == 11) ? 2
                            :                        3;
            ratioX = (double)(m_ass.PlayResX) / (double)(resolution[index].first);
            ratioY = (double)(m_ass.PlayResY) / (double)(resolution[index].second);
        }
        if (m_dll->unicode()) {
            if ((wPosX < 2000) || (wPosY < 2000)) {
                offsetPosX = itcap->wClientX;
                offsetPosY = itcap->wClientY;
            } else {
                offsetPosX = 0;
                offsetPosY = 0;
                wPosX -= 2000;
                wPosY -= 2000;
            }
        }

        auto it2 = itcap->charList.begin();

        if (itcap->charList.size() > 0 && !pCapLine) {
            workCharSizeMode  = it2->wCharSizeMode;
            workCharW         = it2->wCharW;
            workCharH         = it2->wCharH;
            workCharHInterval = it2->wCharHInterval;
            workCharVInterval = it2->wCharVInterval;
            // Calculate offsetPos[X/Y].
            if (!(m_dll->unicode())) {
                int amariPosX = 0;
                int amariPosY = 0;
                if (wLastSWFMode == 9) {
                    amariPosX = wPosX % 18;
                    amariPosY = wPosY % 15;
                } else {
                    amariPosX = wPosX % ((workCharW + workCharHInterval) / 2);
                    amariPosY = wPosY % ((workCharH + workCharVInterval) / 2);
                }
                if ((amariPosX == 0) || (amariPosY == 0)) {
                    offsetPosX = itcap->wClientX;
                    offsetPosY = itcap->wClientY +10;
                } else {
                    offsetPosX = 0;
                    offsetPosY = 0;
                }
            }
            // Calculate workPos[X/Y].
            int   y_swf_offset = 0;
            double x_ratio     = ratioX;
            double y_ratio     = ratioY;
            switch (wLastSWFMode) {
            case 0:
                y_swf_offset = m_ass.SWF0offset;
                break;
            case 5:
                y_swf_offset = m_ass.SWF5offset /* - 0 */;
                break;
            case 7:
                y_swf_offset = m_ass.SWF7offset /* + 0 */;
                break;
            case 9:
                y_swf_offset = m_ass.SWF9offset + ((m_dll->unicode()) ? 0 : -50);
                break;
            case 11:
                y_swf_offset = m_ass.SWF11offset /* - 0 */;
                break;
            default:
                x_ratio = y_ratio = 1.0;
                break;
            }
            workPosX = (int)((wPosX + offsetPosX) * x_ratio);
            workPosY = (int)((wPosY + offsetPosY + y_swf_offset) * y_ratio);
            // Correction for workPosX.
            workPosX = (workPosX > m_sidebarSize) ? workPosX - m_sidebarSize : 0;

            if (!(m_dll->unicode()) && (it2->wCharSizeMode == STR_SMALL))
                workPosY += (int)(10 * ratioY);
        }

        if (!pCapLine) {
            pCapLine.reset(new CAPTION_LINE());
        }

        int outStrW = 0;
        for (; it2 != itcap->charList.end(); it2++) {
            LINE_STR lineStr;

            int strCount          = 0;
            unsigned char workucR = it2->stCharColor.ucR;
            unsigned char workucG = it2->stCharColor.ucG;
            unsigned char workucB = it2->stCharColor.ucB;
            BOOL workUnderLine    = it2->bUnderLine;
            BOOL workShadow       = it2->bShadow;
            BOOL workBold         = it2->bBold;
            BOOL workItalic       = it2->bItalic;
            BYTE workFlushMode    = it2->bFlushMode;
            workHLC               = (it2->bHLC != 0) ? m_prm.HLCmode : it2->bHLC;

            if (!(m_dll->unicode()) && (it2->wCharSizeMode == STR_MEDIUM))
                // 全角 -> 半角
                it2->strDecode = GetHalfChar(it2->strDecode);

            const auto loglevel = RGY_LOG_TRACE;
            if (loglevel >= m_pLog->getLogLevel()) {
                AddMessage(loglevel, _T("pts: %11lld\n"), PTS);
                if (it2->bUnderLine)
                    AddMessage(loglevel, _T("  UnderLine : on\n"));
                if (it2->bBold)
                    AddMessage(loglevel, _T("  Bold : on\n"));
                if (it2->bItalic)
                    AddMessage(loglevel, _T("  Italic : on\n"));
                if (it2->bHLC != 0)
                    AddMessage(loglevel, _T("  HLC : on\n"));
                AddMessage(loglevel, _T("  Color : %#.X   "), it2->stCharColor);
                AddMessage(loglevel, _T("  Char M,W,H,HI,VI : %4d, %4d, %4d, %4d, %4d   "),
                    it2->wCharSizeMode, it2->wCharW, it2->wCharH, it2->wCharHInterval, it2->wCharVInterval);
                AddMessage(loglevel, _T("  %s\n"), char_to_tstring(it2->strDecode, (m_dll->unicode()) ? CP_UTF8 : 932).c_str());
            }

            std::string str_utf8;
            for (int i = 0; i < addSpaceNum; i++) {
                str_utf8 += " ";
            }
            addSpaceNum = 0;
            if (/*(m_prm.format == FORMAT_TAW) || */ (m_dll->unicode())) {
                str_utf8 += it2->strDecode;
            } else {
                auto wstr = char_to_wstring(it2->strDecode, 932);
                str_utf8 += wstring_to_string(wstr, CP_UTF8);
            }
            if (it2->wCharSizeMode != STR_SMALL) {
                HALFCHAR_INFO hc = { 0 };
                strCount = count_utf8_length((unsigned char *)str_utf8.c_str(), &hc);
                int char_nums = it2->wCharSizeMode == STR_MEDIUM ? hc.char_nums - (hc.char_nums - hc.half_nums) / 2 : hc.char_nums;
                outStrW += (char_nums - hc.point_nums) * (workCharW + workCharHInterval) / 2;
            }

            // Push back the caption strings.
            lineStr.outHLC               = workHLC;
            lineStr.outCharColor.a       = 0x00;
            lineStr.outCharColor.r       = workucR;
            lineStr.outCharColor.g       = workucG;
            lineStr.outCharColor.b       = workucB;
            lineStr.outUnderLine         = workUnderLine;
            lineStr.outShadow            = workShadow;
            lineStr.outBold              = workBold;
            lineStr.outItalic            = workItalic;
            lineStr.outFlushMode         = workFlushMode;
            lineStr.outStrCount          = strCount;
            lineStr.str                  = str_utf8;

            pCapLine->outStrings.push_back(lineStr);
        }

        if (pCapLine->outStrings.empty()) {
            pCapLine.reset();
            continue;
        }

        // Push back the caption lines.
        bool bPushBack = true;
        if (workCharSizeMode != STR_SMALL) {
            auto next = itcap + 1;
            for (; next != captionList.end(); next++) {
                if (next->bClear) {
                    continue;
                }
                if (itcap->wPosY == next->wPosY && itcap->dwWaitTime == next->dwWaitTime) {
                    auto it3 = next->charList.begin();
                    int diffPosX = next->wPosX - (itcap->wPosX + outStrW);
                    if (it3->wCharSizeMode != STR_SMALL && diffPosX >= 0) {
                        bPushBack = false;
                        if (diffPosX > 0) {
                            addSpaceNum = diffPosX * 2 / (workCharW + workCharHInterval);
                        }
                    }
                }
                break;
            }
        }
        if (bPushBack) {
            pCapLine->index            = 0;     //useless
            pCapLine->pts              = PTS;
            //startTimeはstartPCRを基準とし、デバッグ用に用いる
            pCapLine->startTime        = (PTS > m_timestamp.startPCR) ? (DWORD)(PTS - m_timestamp.startPCR) : 0;
            //endTimeは後で設定する
            pCapLine->endTime          = 0;
            pCapLine->outCharSizeMode  = (BYTE)workCharSizeMode;
            pCapLine->outCharW         = (WORD)(workCharW * ratioX);
            pCapLine->outCharH         = (WORD)(workCharH * ratioY);
            pCapLine->outCharHInterval = (WORD)(workCharHInterval * ratioX);
            pCapLine->outCharVInterval = (WORD)(workCharVInterval * ratioY);
            pCapLine->outPosX          = (WORD)workPosX;
            pCapLine->outPosY          = (WORD)workPosY;

            m_capList.push_back(std::move(pCapLine));
        }
    }
    return subList;
}

std::vector<AVPacket> Caption2Ass::genAss(int64_t endTime) {
    std::vector<AVPacket> assLines;
    auto it = m_capList.begin();
    for (int i = 0; it != m_capList.end(); it++, i++) {
        (*it)->endTime = endTime;

        rgy_time ts((uint32_t)((*it)->startTime / 90));
        rgy_time te((uint32_t)((*it)->endTime / 90));

        AVPacket pkt;
        av_init_packet(&pkt);
        //muxerには生のptsを伝達する
        pkt.pts = (*it)->pts;
        pkt.dts = (*it)->pts;
        //startTime, endTimeは、startPCRを基準としているのでその差分は有効
        pkt.duration = (*it)->endTime - (*it)->startTime;

        auto it2 = (*it)->outStrings.begin();
        UINT outCharColor = it2->outCharColor.b << 16
            | it2->outCharColor.g << 8
            | it2->outCharColor.r;

        if (((*it)->outCharSizeMode != STR_SMALL) && ((it2->outHLC == HLC_box) || (it2->outHLC == HLC_draw))) {
            BYTE outHLC  = it2->outHLC;
            int iHankaku = it2->outStrCount;
            for (auto it3 = it2 + 1; it3 != (*it)->outStrings.end(); it3++) {
                if (outHLC != it3->outHLC)
                    break;
                iHankaku += it3->outStrCount;
            }
            // Output HLC.
            if (outHLC == HLC_box) {
                int iBoxPosX = (*it)->outPosX + (iHankaku * (((*it)->outCharW + (*it)->outCharHInterval) / 4)) - ((*it)->outCharHInterval / 4);
                int iBoxPosY = (*it)->outPosY + ((*it)->outCharVInterval / 2);
                int iBoxScaleX = (iHankaku + 1) * 50;
                int iBoxScaleY = 100 * ((*it)->outCharH + (*it)->outCharVInterval) / (*it)->outCharH;
                //std::string str = strsprintf("0,%01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d,Box,,0000,0000,0000,,{\\pos(%d,%d)\\fscx%d\\fscy%d\\3c&H%06x&}",
                //    ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10, iBoxPosX, iBoxPosY, iBoxScaleX, iBoxScaleY, outCharColor);
                std::string str = strsprintf("0,0,Box,,0000,0000,0000,,{\\pos(%d,%d)\\fscx%d\\fscy%d\\3c&H%06x&}",
                    iBoxPosX, iBoxPosY, iBoxScaleX, iBoxScaleY, outCharColor);
                static uint8_t utf8box[] = { 0xE2, 0x96, 0xA0 };
                AVPacket pkt2;
                av_init_packet(&pkt2);
                av_packet_copy_props(&pkt2, &pkt);
                uint8_t *ptr = (uint8_t *)av_strdup(str.c_str());
                av_packet_from_data(&pkt2, ptr, (int)str.length());
                assLines.push_back(pkt2);
                AddMessage(RGY_LOG_DEBUG, _T("pts: %11lld, dur: %6lld, %01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d, %s\n"),
                    pkt2.pts, pkt2.duration,
                    ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10,
                    char_to_tstring(str, CP_UTF8).c_str());
            } else { /* outHLC == HLC_draw */
                int iBoxPosX = (*it)->outPosX + (iHankaku * (((*it)->outCharW + (*it)->outCharHInterval) / 4));
                int iBoxPosY = (*it)->outPosY + ((*it)->outCharVInterval / 4);
                int iBoxScaleX = iHankaku * 55;
                int iBoxScaleY = 100;   //*((*it)->outCharH + (*it)->outCharVInterval) / (*it)->outCharH;
                //auto str = strsprintf("0,%01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d,Box,,0000,0000,0000,,{\\pos(%d,%d)\\3c&H%06x&\\p1}m 0 0 l %d 0 %d %d 0 %d{\\p0}",
                //    ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10, iBoxPosX, iBoxPosY, outCharColor, iBoxScaleX, iBoxScaleX, iBoxScaleY, iBoxScaleY);
                auto str = strsprintf("0,0,Box,,0000,0000,0000,,{\\pos(%d,%d)\\3c&H%06x&\\p1}m 0 0 l %d 0 %d %d 0 %d{\\p0}",
                    iBoxPosX, iBoxPosY, outCharColor, iBoxScaleX, iBoxScaleX, iBoxScaleY, iBoxScaleY);
                AVPacket pkt2;
                av_init_packet(&pkt2);
                av_packet_copy_props(&pkt2, &pkt);
                uint8_t *ptr = (uint8_t *)av_strdup(str.c_str());
                av_packet_from_data(&pkt2, ptr, (int)str.length());
                assLines.push_back(pkt2);
                AddMessage(RGY_LOG_DEBUG, _T("pts: %11lld, dur: %6lld, %01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d, %s\n"),
                    pkt2.pts, pkt2.duration,
                    ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10,
                    char_to_tstring(str, CP_UTF8).c_str());
            }
        }
        //std::string str = strsprintf("0,%01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d,%s,,0000,0000,0000,,{\\pos(%d,%d)",
        //    ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10,
        //    ((*it)->outCharSizeMode == STR_SMALL) ? "Rubi" : "Default",
        //    (*it)->outPosX, (*it)->outPosY);
        std::string str = strsprintf("0,0,%s,,0000,0000,0000,,{\\pos(%d,%d)",
            ((*it)->outCharSizeMode == STR_SMALL) ? "Rubi" : "Default",
            (*it)->outPosX, (*it)->outPosY);

        if (outCharColor != 0x00ffffff)
            str += strsprintf("\\c&H%06x&", outCharColor);
        if (it2->outUnderLine)
            str += "\\u1";
        if (it2->outBold)
            str += "\\b1";
        if (it2->outItalic)
            str += "\\i1";
        str += "}";

        if (((*it)->outCharSizeMode == STR_SMALL) && (m_prm.norubi)) {
            str += "\\N";
        } else {
            BOOL bHLC = FALSE;
            // Output strings.
            while (1) {
                if (!bHLC && ((*it)->outCharSizeMode != STR_SMALL) && (it2->outHLC == HLC_kigou)) {
                    str += "[";
                    bHLC = TRUE;
                }

                str += it2->str;

                auto prev = it2;
                ++it2;
                if (it2 == (*it)->outStrings.end())
                    break;

                if (bHLC && (((*it)->outCharSizeMode == STR_SMALL) || (it2->outHLC != HLC_kigou))) {
                    str += "]";
                    bHLC = FALSE;
                }

                UINT prevCharColor = outCharColor;
                outCharColor = it2->outCharColor.b << 16
                    | it2->outCharColor.g << 8
                    | it2->outCharColor.r;

                if (prevCharColor != outCharColor
                    /* || prev->outCharColor.ucAlpha != it2->outCharColor.ucAlpha */
                    || prev->outUnderLine != it2->outUnderLine
                    || prev->outBold      != it2->outBold
                    || prev->outItalic    != it2->outItalic) {
                    str += strsprintf("{");
                    if (prevCharColor != outCharColor
                        /* || prev->outCharColor.ucAlpha != it2->outCharColor.ucAlpha */)
                        str += strsprintf("\\c&H%06x&", outCharColor);
                    if (prev->outUnderLine != it2->outUnderLine)
                        str += strsprintf("\\u%d", it2->outUnderLine ? 1 : 0);
                    if (prev->outBold != it2->outBold)
                        str += strsprintf("\\b%d", it2->outBold ? 1 : 0);
                    if (prev->outItalic != it2->outItalic)
                        str += strsprintf("\\i%d", it2->outItalic ? 1 : 0);
                    str += strsprintf("}");
                }
            }
            if (bHLC)
                str += "]";
            str += "\\N";
        }
        uint8_t *ptr = (uint8_t *)av_strdup(str.c_str());
        av_packet_from_data(&pkt, ptr, (int)str.length());
        assLines.push_back(pkt);
        AddMessage(RGY_LOG_DEBUG, _T("pts: %11lld, dur: %6lld, %01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d, %s\n"),
            pkt.pts, pkt.duration,
            ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10,
            char_to_tstring(str, CP_UTF8).c_str());
    }
    return assLines;
}

std::vector<AVPacket> Caption2Ass::genSrt(int64_t endTime) {
    AVPacket pkt;
    av_init_packet(&pkt);
    rgy_time ts, te;
    std::string str;
    bool bNoSRT = true;
    auto it = m_capList.begin();
    for (int i = 0; it != m_capList.end(); it++, i++) {
        (*it)->endTime = endTime;

        if (i == 0) {
            (*it)->endTime = endTime;
            pkt.pts = (*it)->pts;
            pkt.dts = (*it)->pts;
            pkt.duration = (*it)->endTime - (*it)->startTime;

            ts = rgy_time((uint32_t)((*it)->startTime / 90));
            te = rgy_time((uint32_t)((*it)->endTime / 90));
            //str += strsprintf("%d\r\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\r\n",
            //    m_srt.index, ts.h, ts.m, ts.s, ts.ms, te.h, te.m, te.s, te.ms);
        }

        // ふりがな Skip
        if ((*it)->outCharSizeMode == STR_SMALL)
            continue;
        bNoSRT = false;

        auto it2 = (*it)->outStrings.begin();
        bool italic = false, bold = false, underLine = false, charColor = false;
        auto ornament_start = [&](vector<LINE_STR>::iterator& s) {
            italic    = (s)->outItalic    != FALSE;
            bold      = (s)->outBold      != FALSE;
            underLine = (s)->outUnderLine != FALSE;
            charColor = ((s)->outCharColor.r != 0xff
                      || (s)->outCharColor.g != 0xff
                      || (s)->outCharColor.b != 0xff);

            if ((s)->outItalic)    str += "<i>";
            if ((s)->outBold)      str += "<b>";
            if ((s)->outUnderLine) str += "<u>";
            if (charColor) {
                str += strsprintf("<font color=\"#%02x%02x%02x\">",
                    (s)->outCharColor.r, (s)->outCharColor.g, (s)->outCharColor.b);
            }
        };
        auto ornament_end = [&]() {
            if (italic)    str += "</i>";
            if (bold)      str += "</b>";
            if (underLine) str += "</u>";
            if (charColor) str += "</font>";
        };
        if (m_srt.ornament) {
            ornament_start(it2);
        }

        BOOL bHLC = FALSE;
        // Output strings.
        while (true) {
            if (!bHLC && (it2->outHLC != 0)) {
                str += strsprintf("[");
                bHLC = TRUE;
            }

            str += it2->str;

            ++it2;
            if (it2 == (*it)->outStrings.end())
                break;

            if (bHLC && (it2->outHLC == 0)) {
                str += "]";
                bHLC = FALSE;
            }

            if (m_srt.ornament) {
                ornament_end();
                // Next ornament.
                ornament_start(it2);
            }
        }
        if (bHLC)
            str += "]";
        if (m_srt.ornament) {
            ornament_end();
        }
        //str += "\r\n";
    }

    //if (m_capList.size() > 0) {
    //    if (bNoSRT)
    //        str += "\r\n";
    //    str += "\r\n";
    //    m_srt.index++;
    //}
    AddMessage(RGY_LOG_DEBUG, _T("pts: %11lld, dur: %6lld, %01d:%02d:%02d.%02d,%01d:%02d:%02d.%02d, %s\n"),
        pkt.pts, pkt.duration,
        ts.h, ts.m, ts.s, ts.ms / 10, te.h, te.m, te.s, te.ms / 10,
        char_to_tstring(str, CP_UTF8).c_str());
    uint8_t *ptr = (uint8_t *)av_strdup(str.c_str());
    av_packet_from_data(&pkt, ptr, (int)str.length());

    std::vector<AVPacket> assLines;
    assLines.push_back(pkt);
    return assLines;
}

#endif //#if ENABLE_AVSW_READER
