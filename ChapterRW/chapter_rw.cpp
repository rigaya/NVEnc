//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx/2.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <tinyxml2.h>

#if defined(_WIN32) || defined(_WIN64)
#include <mlang.h>
#include <shlwapi.h>
#pragma comment (lib, "shlwapi.lib")
#else
#include "rgy_osdep.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <iconv.h>
#endif //#if defined(_WIN32) || defined(_WIN64)

#ifdef _UNICODE
typedef wchar_t TCHAR;
#define _tfopen   _wfopen
#define _tfopen_s _wfopen_s
#define tremove _wremove
#define trename _wrename
#define _T(x) (L ## x)
#else
typedef char TCHAR;
#define _tfopen   fopen
#define _tfopen_s fopen_s
#define tremove remove
#define trename rename
#define _T(x) (x)
#endif

#include "chapter_rw.h"

using std::vector;
using std::string;
using std::wstring;
using std::unique_ptr;

#if defined(_WIN32) || defined(_WIN64)
struct iunknown_deleter {
    void operator()(IUnknown *ptr) const {
        ptr->Release();
        CoUninitialize();
    }
};
#endif //#if defined(_WIN32) || defined(_WIN64)
struct fp_deleter {
    void operator()(FILE *fp) const {
        fclose(fp);
    }
};

//BOM文字リスト
static const int MAX_UTF8_CHAR_LENGTH = 6;
static const uint8_t UTF8_BOM[]     = { 0xEF, 0xBB, 0xBF };
static const uint8_t UTF16_LE_BOM[] = { 0xFF, 0xFE };
static const uint8_t UTF16_BE_BOM[] = { 0xFE, 0xFF };

//ボム文字かどうか、コードページの判定
static uint32_t check_bom(const void* chr) {
    if (chr == nullptr) return CODE_PAGE_UNSET;
    if (memcmp(chr, UTF16_LE_BOM, sizeof(UTF16_LE_BOM)) == 0) return CODE_PAGE_UTF16_LE;
    if (memcmp(chr, UTF16_BE_BOM, sizeof(UTF16_BE_BOM)) == 0) return CODE_PAGE_UTF16_BE;
    if (memcmp(chr, UTF8_BOM,     sizeof(UTF8_BOM))     == 0) return CODE_PAGE_UTF8;
    return CODE_PAGE_UNSET;
}

static BOOL isJis(const void *str, uint32_t size_in_byte) {
    static const uint8_t ESCAPE[][7] = {
        //先頭に比較すべきバイト数
        { 3, 0x1B, 0x28, 0x42, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x28, 0x4A, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x28, 0x49, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x24, 0x40, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x24, 0x42, 0x00, 0x00, 0x00 },
        { 6, 0x1B, 0x26, 0x40, 0x1B, 0x24, 0x42 },
        { 4, 0x1B, 0x24, 0x28, 0x44, 0x00, 0x00 },
        { 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } //終了
    };
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (*chr > 0x7F)
            return FALSE;
        for (int i = 0; ESCAPE[i][0]; i++) {
            if (str_fin - chr > ESCAPE[i][0] &&
                memcmp(chr, &ESCAPE[i][1], ESCAPE[i][0]) == 0)
                return TRUE;
        }
    }
    return FALSE;
}

static uint32_t isUTF16(const void *str, uint32_t size_in_byte) {
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (chr[0] == 0x00 && str_fin - chr > 1 && chr[1] <= 0x7F)
            return ((chr - (const uint8_t *)str) % 2 == 1) ? CODE_PAGE_UTF16_LE : CODE_PAGE_UTF16_BE;
    }
    return CODE_PAGE_UNSET;
}

static BOOL isASCII(const void *str, uint32_t size_in_byte) {
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (*chr == 0x1B || *chr >= 0x80)
            return FALSE;
    }
    return TRUE;
}

static uint32_t jpn_check(const void *str, uint32_t size_in_byte) {
    int score_sjis = 0;
    int score_euc = 0;
    int score_utf8 = 0;
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if ((0x81 <= chr[0] && chr[0] <= 0x9F) ||
            (0xE0 <= chr[0] && chr[0] <= 0xFC) ||
            (0x40 <= chr[1] && chr[1] <= 0x7E) ||
            (0x80 <= chr[1] && chr[1] <= 0xFC)) {
            score_sjis += 2; chr++;
        }
    }
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if ((0xC0 <= chr[0] && chr[0] <= 0xDF) &&
            (0x80 <= chr[1] && chr[1] <= 0xBF)) {
            score_utf8 += 2; chr++;
        } else if (
            str_fin - chr > 2 &&
            (0xE0 <= chr[0] && chr[0] <= 0xEF) &&
            (0x80 <= chr[1] && chr[1] <= 0xBF) &&
            (0x80 <= chr[2] && chr[2] <= 0xBF)) {
            score_utf8 += 3; chr++;
        }
    }
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if (((0xA1 <= chr[0] && chr[0] <= 0xFE) && (0xA1 <= chr[1] && chr[1] <= 0xFE)) ||
            (chr[0] == 0x8E                     && (0xA1 <= chr[1] && chr[1] <= 0xDF))) {
            score_euc += 2; chr++;
        } else if (
            str_fin - chr > 2 &&
            chr[0] == 0x8F &&
            (0xA1 <= chr[1] && chr[1] <= 0xFE) &&
            (0xA1 <= chr[2] && chr[2] <= 0xFE)) {
            score_euc += 3; chr += 2;
        }
    }
    if (score_sjis > score_euc && score_sjis > score_utf8)
        return CODE_PAGE_SJIS;
    if (score_utf8 > score_euc && score_utf8 > score_sjis)
        return CODE_PAGE_UTF8;
    if (score_euc > score_sjis && score_euc > score_utf8)
        return CODE_PAGE_EUC_JP;
    return CODE_PAGE_UNSET;
}

static uint32_t get_code_page(const void *str, uint32_t size_in_byte) {
    uint32_t ret = CODE_PAGE_UNSET;
    if ((ret = check_bom(str)) != CODE_PAGE_UNSET)
        return ret;

    if (isJis(str, size_in_byte))
        return CODE_PAGE_JIS;

    if ((ret = isUTF16(str, size_in_byte)) != CODE_PAGE_UNSET)
        return ret;

    if (isASCII(str, size_in_byte))
        return CODE_PAGE_US_ASCII;

    return jpn_check(str, size_in_byte);
}

static BOOL fix_ImulL_WesternEurope(uint32_t *code_page) {
    //IMultiLanguage2 の DetectInputCodepage はよく西ヨーロッパ言語と誤判定しやがる
    if (*code_page == CODE_PAGE_WEST_EUROPE)
        *code_page = CODE_PAGE_SJIS;
    return TRUE;
}

#pragma warning (push)
#pragma warning (disable: 4100)
#if defined(_WIN32) || defined(_WIN64)
static unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    uint32_t flags = (codepage == CP_UTF8) ? 0 : WC_NO_BEST_FIT_CHARS;
    int multibyte_length = WideCharToMultiByte(codepage, flags, wstr, -1, nullptr, 0, nullptr, nullptr);
    str.resize(multibyte_length, 0);
    if (0 == WideCharToMultiByte(codepage, flags, wstr, -1, &str[0], multibyte_length, nullptr, nullptr)) {
        str.clear();
        return 0;
    }
    return multibyte_length;
}
#else
static unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    auto ic = iconv_open("UTF-8", "wchar_t"); //to, from
    auto input_len = wcslen(wstr);
    auto output_len = input_len * 4;
    str.resize(output_len, 0);
    char *outbuf = &str[0];
    iconv(ic, (char **)&wstr, &input_len, &outbuf, &output_len);
    return output_len;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

static std::string wstring_to_string(const wchar_t *wstr, uint32_t codepage) {
    std::string str;
    wstring_to_string(wstr, str, codepage);
    return str;
}

static std::string wstring_to_string(const std::wstring& wstr, uint32_t codepage) {
    std::string str;
    wstring_to_string(wstr.c_str(), str, codepage);
    return str;
}

#if defined(_WIN32) || defined(_WIN64)
static unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    int widechar_length = MultiByteToWideChar(codepage, 0, str, -1, nullptr, 0);
    wstr.resize(widechar_length, 0);
    if (0 == MultiByteToWideChar(codepage, 0, str, -1, &wstr[0], (int)wstr.size())) {
        wstr.clear();
        return 0;
    }
    return widechar_length;
}
#else
static unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    auto ic = iconv_open("wchar_t", "UTF-8"); //to, from
    auto input_len = strlen(str);
    std::vector<char> buf(input_len + 1);
    strcpy(buf.data(), str);
    auto output_len = input_len;
    wstr.resize(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = (char *)&wstr[0];
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    return output_len;
}
#endif //#if defined(_WIN32) || defined(_WIN64)
static std::wstring char_to_wstring(const char *str, uint32_t codepage) {
    std::wstring wstr;
    char_to_wstring(wstr, str, codepage);
    return wstr;
}
static std::wstring char_to_wstring(const std::string& str, uint32_t codepage) {
    std::wstring wstr;
    char_to_wstring(wstr, str.c_str(), codepage);
    return wstr;
}

static std::string strsprintf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    const size_t len = _vscprintf(format, args) + 1;

    std::vector<char> buffer(len, 0);
    vsprintf_s(buffer.data(), buffer.size(), format, args);
    va_end(args);
    std::string retStr = std::string(buffer.data());
    return retStr;
}

static std::wstring strsprintf(const WCHAR* format, ...) {
    va_list args;
    va_start(args, format);
    const size_t len = _vscwprintf(format, args) + 1;

    std::vector<WCHAR> buffer(len, 0);
    vswprintf_s(buffer.data(), buffer.size(), format, args);
    va_end(args);
    std::wstring retStr = std::wstring(buffer.data());
    return retStr;
}

static std::string str_replace(std::string str, const std::string& from, const std::string& to) {
    std::string::size_type pos = 0;
    while(pos = str.find(from, pos), pos != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return std::move(str);
}

#if defined(_WIN32) || defined(_WIN64)
static std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to) {
    std::wstring::size_type pos = 0;
    while (pos = str.find(from, pos), pos != std::wstring::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return std::move(str);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#pragma warning (pop)

static std::vector<std::wstring> split(const std::wstring &str, const std::wstring &delim) {
    std::vector<std::wstring> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::wstring::npos != (found = str.find(delim, current))) {
        res.push_back(std::wstring(str, current, found - current));
        current = found + delimlen;
    }
    res.push_back(std::wstring(str, current, str.size() - current));
    return res;
}

static std::vector<std::string> split(const std::string &str, const std::string &delim) {
    std::vector<std::string> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::string::npos != (found = str.find(delim, current))) {
        res.push_back(std::string(str, current, found - current));
        current = found + delimlen;
    }
    res.push_back(std::string(str, current, str.size() - current));
    return res;
}

ChapterRW::ChapterRW() {
    init();
}

ChapterRW::~ChapterRW() {
    close();
}

void ChapterRW::init() {
    close();

    m_filepath = nullptr;
    m_chapter_type = CHAP_TYPE_UNKNOWN;
    m_code_page = CODE_PAGE_UNSET;
    m_duration = 0.0;
}

void ChapterRW::close() {
    if (chapters.size()) {
        std::vector<unique_ptr<ChapData>>().swap(chapters);
    }
}

int ChapterRW::file_chapter_type() {
    return m_chapter_type;
}

uint32_t ChapterRW::file_code_page() {
    return m_code_page;
}

 int ChapterRW::read_file(const TCHAR *filepath, uint32_t code_page, double duration) {
    init();
    this->m_filepath = filepath;
    this->m_code_page = code_page;
    this->m_duration = duration;
    return read_file();
}

int ChapterRW::write_chapter_apple_header(std::ostream& ostream) {
    int sts = AUO_CHAP_ERR_NONE;
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    ostream.write((const char *)UTF8_BOM, sizeof(UTF8_BOM));
    ostream << wstring_to_string(
    L"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\r\n"
    L"<TextStream version=\"1.1\">\r\n"
    L"<TextStreamHeader>\r\n"
    L"<TextSampleDescription>\r\n"
    L"</TextSampleDescription>\r\n"
    L"</TextStreamHeader>\r\n", CP_UTF8);
    return sts;
}

//終端 長さも指定する(しないとわけのわからんdurationになる)
int ChapterRW::write_chapter_apple_foot(std::ostream& ostream) {
    int sts = AUO_CHAP_ERR_NONE;
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_NULL_PTR;
    } else {
        int duration_ms = (int)(m_duration * 1000.0 + 0.5);
        if (m_duration <= 0 && 0 < chapters.size()) {
            duration_ms = chapters.back()->get_ms() + 1;
        }

        ostream << wstring_to_string(strsprintf(
            L"<TextSample sampleTime=\"%02d:%02d:%02d.%03d\" text=\"\" />\r\n</TextStream>",
            duration_ms / (60*60*1000),
            (duration_ms % (60*60*1000)) / (60*1000),
            (duration_ms % (60*1000)) / 1000,
            duration_ms % 1000
            ), CODE_PAGE_UTF8);
    }
    return sts;
}

int ChapterRW::write_chapter_apple(const TCHAR *out_filepath) {
    int sts = AUO_CHAP_ERR_NONE;
    std::ofstream ostream(out_filepath, std::ios::out | std::ios::binary);
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_FILE_OPEN;
    } else {
        write_chapter_apple_header(ostream);
        for (const auto& chap : chapters) {
            ostream << wstring_to_string(strsprintf(
                L"<TextSample sampleTime=\"%02d:%02d:%02d.%03d\">%s</TextSample>\r\n",
                chap->h, chap->m, chap->s, chap->ms, chap->name.c_str()  ), CODE_PAGE_UTF8);
        }
        write_chapter_apple_foot(ostream);
        ostream.close();
    }
    return sts;
}

int ChapterRW::write_chapter_nero(const TCHAR *out_filepath, bool utf8) {
    int sts = AUO_CHAP_ERR_NONE;
    std::ofstream ostream(out_filepath, std::ios::out | std::ios::binary);
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_FILE_OPEN;
    } else {
        if (utf8)
            ostream.write((const char *)UTF8_BOM, sizeof(UTF8_BOM));

        const uint32_t output_codepage = (utf8) ? CODE_PAGE_UTF8 : CODE_PAGE_SJIS;
        for (uint32_t i = 0; i < chapters.size(); i++) {
            const auto& chap = chapters[i];
            static const char * const KEY_BASE = "CHAPTER";
            static const char * const KEY_NAME = "NAME";
            auto chap_name_char = wstring_to_string(chap->name, output_codepage);

            ostream << strsprintf("%s%02d=%02d:%02d:%02d.%03d\r\n", KEY_BASE, i+1, chap->h, chap->m, chap->s, chap->ms);
            ostream << strsprintf("%s%02d%s=%s\r\n", KEY_BASE, i+1, KEY_NAME, chap_name_char.c_str());
        }
        ostream.close();
    }
    return sts;
}

int ChapterRW::write_file(const TCHAR *out_filepath, ChapType out_chapter_type, bool nero_in_utf8) {
    if (CHAP_TYPE_UNKNOWN == out_chapter_type) {
        out_chapter_type = m_chapter_type;
    } else if (CHAP_TYPE_ANOTHER == out_chapter_type) {
        out_chapter_type = (CHAP_TYPE_NERO == m_chapter_type) ? CHAP_TYPE_APPLE : CHAP_TYPE_NERO;
    }
    return (CHAP_TYPE_NERO == out_chapter_type) ? write_chapter_nero(out_filepath, nero_in_utf8) : write_chapter_apple(out_filepath);
}

int ChapterRW::overwrite_file(ChapType out_chapter_type, bool nero_in_utf8) {
    int sts = AUO_CHAP_ERR_NONE;
    std::basic_string<TCHAR> temp_file = m_filepath;
    temp_file += _T(".tmp");
    for (int i = 0; PathFileExists(temp_file.c_str()); i++) {
        temp_file = strsprintf(_T("%s.tmp%d"), m_filepath, i);
    }
    sts = write_file(temp_file.c_str(), out_chapter_type, nero_in_utf8);
    if (AUO_CHAP_ERR_NONE != sts) {
        tremove(temp_file.c_str());
    } else {
        tremove(m_filepath);
        trename(temp_file.c_str(), m_filepath);
    }
    return sts;
}

#if !(defined(_WIN32) || defined(_WIN64))
typedef struct DetectEncodingInfo {
    uint32_t nCodePage;
} DetectEncodingInfo;
#endif

uint32_t ChapterRW::check_code_page(vector<char>& src, uint32_t orig_code_page) {
    DetectEncodingInfo dEnc = { 0 };
    int denc_count = 1;
    int src_buf_len = (int)src.size();

#if defined(_WIN32) || defined(_WIN64)
    unique_ptr<IMultiLanguage2, iunknown_deleter> pIMultiLang;
    {
        CoInitialize(NULL);
        IMultiLanguage2 *pImul = nullptr;
        if (S_OK == CoCreateInstance(CLSID_CMultiLanguage, NULL, CLSCTX_INPROC_SERVER, IID_IMultiLanguage2, (void**)&pImul) && pImul != nullptr) {
            pIMultiLang.reset(pImul);
        }
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    if (   CODE_PAGE_UNSET == orig_code_page //指定があればスキップ
        && CODE_PAGE_UNSET == (dEnc.nCodePage = get_code_page(&src[0], (uint32_t)src.size())) //まず主に日本語をチェック
#if defined(_WIN32) || defined(_WIN64)
        && (!pIMultiLang || S_OK != pIMultiLang->DetectInputCodepage(MLDETECTCP_NONE, 0, &src[0], &src_buf_len, &dEnc, &denc_count)) //IMultiLanguage2で判定してみる
        && TRUE != fix_ImulL_WesternEurope(&dEnc.nCodePage)
#endif //#if defined(_WIN32) || defined(_WIN64)
    ) {
        return CODE_PAGE_UNSET;
    }

    return dEnc.nCodePage;
}

int ChapterRW::get_unicode_data(wstring& wchar_data, vector<char>& src) {
    int sts = AUO_CHAP_ERR_NONE;
    wchar_data.resize(src.size(), 0);

    if (m_code_page == CODE_PAGE_UTF16_LE) {
        memcpy(&wchar_data[0], &src[0], src.size());
        wchar_data.resize(src.size() / sizeof(wchar_t));
    } else {
        int start_index = 0;
        if (CODE_PAGE_UTF8 == m_code_page && 0 == memcmp(&src[0], UTF8_BOM, sizeof(UTF8_BOM)))
            start_index = sizeof(UTF8_BOM);

        if (0 == char_to_wstring(wchar_data, &src[start_index], m_code_page)) {
            sts = AUO_CHAP_ERR_CONVERTION;
        }
    }
    return sts;
}

int ChapterRW::get_unicode_data_from_file(wstring& wchar_data) {
    using std::ios;
    using std::ifstream;
    int sts = AUO_CHAP_ERR_NONE;
    //ファイルを一気に読み込み
    vector<char> file_data;
    ifstream inputFile(m_filepath, ios::in | ios::binary);
    if (!inputFile.good()) {
        sts = AUO_CHAP_ERR_FILE_OPEN;
    } else {
        file_data.resize((size_t)inputFile.seekg(0, ios::end).tellg() + 1, '\0');
        inputFile.seekg(0, ios::beg).read(&file_data[0], static_cast<std::streamsize>(file_data.size()));

        if (0 == file_data.size()) {
            sts = AUO_CHAP_ERR_FILE_OPEN;
        //文字コード判定
        } else if (CODE_PAGE_UNSET == (m_code_page = check_code_page(file_data, m_code_page))) {
            sts = AUO_CHAP_ERR_CP_DETECT;
        } else {
            //文字コード変換
            sts = get_unicode_data(wchar_data, file_data);
        }
    }
    return sts;
}

ChapType ChapterRW::check_chap_type(const wstring& data) {
    int sts = AUO_CHAP_ERR_NONE;
    if (0 == data.length()) {
        sts = CHAP_TYPE_UNKNOWN;
    }
    auto pos = data.find(L"CHAPTER");
    auto first_pos = pos;
    if (   string::npos != pos
        && string::npos != (pos = first_pos = data.find(L"=", pos))
        && string::npos != (pos = data.find_first_of(L"\r\n", pos))
        && string::npos != (pos = data.find_first_not_of(L"\r\n", pos))
        && string::npos != data.substr(pos).find(data.substr(0, first_pos))) {
        return CHAP_TYPE_NERO;
    }
    if (   string::npos != data.find(L"<TextStream")
        && string::npos != data.find(L"<TextSample")) {
        return CHAP_TYPE_APPLE;
    }
    if (string::npos != data.find(L"<Chapters")
        && string::npos != data.find(L"<EditionEntry")
        && string::npos != data.find(L"<ChapterAtom")
        && string::npos != data.find(L"<ChapterTimeStart")) {
        return CHAP_TYPE_MATROSKA;
    }

    return CHAP_TYPE_UNKNOWN;
}

ChapType ChapterRW::check_chap_type_from_file() {
    //文字コード変換してファイル内容を取得
    if (AUO_CHAP_ERR_NONE != get_unicode_data_from_file(m_wchar_filedata)) {
        m_chapter_type = CHAP_TYPE_UNKNOWN;
    } else {
        m_chapter_type = check_chap_type(m_wchar_filedata);
    }
    return m_chapter_type;
}

int ChapterRW::read_chapter_nero() {
    int sts = AUO_CHAP_ERR_NONE;
    if (0 == m_wchar_filedata.length()) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    //行単位に分解
    const wchar_t *delim = (string::npos != m_wchar_filedata.find(L"\n")) ? L"\n" : L"\r"; //適切な改行コードを見つける
    auto pw_line = split(m_wchar_filedata, delim);

    //読み取り
    static const wchar_t * const CHAP_KEY = L"CHAPTER";
    const wchar_t *pw_key[] = { CHAP_KEY, nullptr, nullptr }; // 時間行, 名前行, ダミー
    wchar_t *pw_data[2] = { nullptr, nullptr };
    const uint32_t total_lines = (uint32_t)pw_line.size();
    for (uint32_t i = 0; i < total_lines && !sts; i++) {
        //末尾の改行空白を取り除く
        pw_line[i] = pw_line[i].substr(0, pw_line[i].find_last_not_of(L"\r\n ") + 1);
        if (wcslen(pw_line[i].c_str()) == 0) {
            break;
        }
        if (string::npos == pw_line[i].find(pw_key[i&1], 0, wcslen(pw_key[i&1])))
            return AUO_CHAP_ERR_INVALID_FMT;
        pw_key[(i&1) + 1] = &pw_line[i][0];//CHAPTER KEY名を保存
        pw_data[i&1] = wcschr(&pw_line[i][0], L'=');
        *pw_data[i&1] = L'\0'; //CHAPTER KEY名を一つの文字列として扱えるように
        pw_data[i&1]++; //データは'='の次から
        if (i&1) {
            //読み取り
            unique_ptr<ChapData> chap(new ChapData());
            if (   4 != swscanf_s(pw_data[0], L"%d:%d:%d:%d", &chap->h, &chap->m, &chap->s, &chap->ms)
                && 4 != swscanf_s(pw_data[0], L"%d:%d:%d.%d", &chap->h, &chap->m, &chap->s, &chap->ms)
                && 4 != swscanf_s(pw_data[0], L"%d:%d.%d.%d", &chap->h, &chap->m, &chap->s, &chap->ms)
                && 4 != swscanf_s(pw_data[0], L"%d.%d.%d.%d", &chap->h, &chap->m, &chap->s, &chap->ms)) {
                sts = AUO_CHAP_ERR_INVALID_FMT;
            } else {
                chap->name = pw_data[1];
                chapters.push_back(std::move(chap));
            }
        }
    }
    return sts;
}

int ChapterRW::read_chapter_apple() {
    int sts = AUO_CHAP_ERR_NONE;
    if (nullptr == m_filepath) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    static const char * const ELEMENT_NAME = "TextSample";
    static const char * const ATTRIBUTE_NAME = "sampleTime";

    unique_ptr<FILE, fp_deleter> fp;
    {
        FILE *fp_tmp = NULL;
        if (_tfopen_s(&fp_tmp, m_filepath, _T("rb"))) {
            return AUO_CHAP_ERR_FILE_OPEN;
        }
        fp.reset(fp_tmp);
    }
    tinyxml2::XMLDocument xml;
    if (tinyxml2::XML_NO_ERROR != xml.LoadFile(fp.get())) {
        sts = AUO_CHAP_ERR_INIT_READ_STREAM;
    } else {
        auto elemTextStream = xml.FirstChildElement("TextStream");
        for (auto element = elemTextStream->FirstChildElement(ELEMENT_NAME); element != nullptr; element = element->NextSiblingElement(ELEMENT_NAME)) {
            int time[4] = { 0 };
            auto pSampleTime = element->Attribute(ATTRIBUTE_NAME);
            if (   4 != sscanf_s(pSampleTime, "%d:%d:%d:%d", &time[0], &time[1], &time[2], &time[3])
                && 4 != sscanf_s(pSampleTime, "%d:%d:%d.%d", &time[0], &time[1], &time[2], &time[3])
                && 4 != sscanf_s(pSampleTime, "%d:%d.%d.%d", &time[0], &time[1], &time[2], &time[3])
                && 4 != sscanf_s(pSampleTime, "%d.%d.%d.%d", &time[0], &time[1], &time[2], &time[3])) {
                return AUO_CHAP_ERR_PARSE_XML;
            }
            unique_ptr<ChapData> chap(new ChapData());
            chap->h  = time[0];
            chap->m  = time[1];
            chap->s  = time[2];
            chap->ms = time[3];
            m_duration = std::max<decltype(m_duration)>(m_duration, chap->get_ms());
            if (element->GetText()) {
                chap->name = char_to_wstring(element->GetText(), CP_UTF8);
                chapters.push_back(std::move(chap));
            }
        }
    }

    return sts;
}

int ChapterRW::read_chapter_matroska_chapter_atom(tinyxml2::XMLElement *elem, int& addedCount) {
    static const char *const CHAPTER_ATOM         = "ChapterAtom";
    static const char *const CHAPTER_TIME_START   = "ChapterTimeStart";
    static const char *const CHAPTER_TIME_END     = "ChapterTimeEnd";
    static const char *const CHAPTER_DISPLAY      = "ChapterDisplay";
    static const char *const CHAPTER_STRING       = "ChapterString";
    static const char *const CHAPTER_FLAG_HIDDEN  = "ChapterFlagHidden";
    static const char *const CHAPTER_FLAG_ENABLED = "ChapterFlagEnabled";

    for (auto chapterAtom = elem->FirstChildElement(CHAPTER_ATOM); chapterAtom != nullptr; chapterAtom = chapterAtom->NextSiblingElement(CHAPTER_ATOM)) {
        int count = 0;
        int ret = read_chapter_matroska_chapter_atom(chapterAtom, count);
        if (ret != AUO_CHAP_ERR_NONE) {
            return ret;
        }
        if (count == 0) {
            auto flagHidden = chapterAtom->FirstChildElement(CHAPTER_FLAG_HIDDEN);
            auto flagEnabled = chapterAtom->FirstChildElement(CHAPTER_FLAG_ENABLED);
            const bool hidden  = ( flagHidden  && strtol(flagHidden->GetText(), nullptr, 10) != 0);
            const bool enabled = (!flagEnabled || strtol(flagEnabled->GetText(), nullptr, 10) != 0);
            if (!hidden && enabled) {
                auto chapterTimeStart = chapterAtom->FirstChildElement(CHAPTER_TIME_START);
                if (chapterTimeStart == nullptr) {
                    return AUO_CHAP_ERR_PARSE_XML;
                }
                auto chapterDisplay = chapterAtom->FirstChildElement(CHAPTER_DISPLAY);

                auto timeStart = chapterTimeStart->GetText();
                int time[4] = { 0 };
                if (   4 != sscanf_s(timeStart, "%d:%d:%d:%d", &time[0], &time[1], &time[2], &time[3])
                    && 4 != sscanf_s(timeStart, "%d:%d:%d.%d", &time[0], &time[1], &time[2], &time[3])
                    && 4 != sscanf_s(timeStart, "%d:%d.%d.%d", &time[0], &time[1], &time[2], &time[3])
                    && 4 != sscanf_s(timeStart, "%d.%d.%d.%d", &time[0], &time[1], &time[2], &time[3])) {
                    return AUO_CHAP_ERR_PARSE_XML;
                }
                unique_ptr<ChapData> chap(new ChapData());
                chap->h  = time[0];
                chap->m  = time[1];
                chap->s  = time[2];
                chap->ms = time[3];
                m_duration = std::max<decltype(m_duration)>(m_duration, chap->get_ms());
                if (chapterDisplay && chapterDisplay->FirstChildElement(CHAPTER_STRING)) {
                    chap->name = char_to_wstring(chapterDisplay->FirstChildElement(CHAPTER_STRING)->GetText(), CP_UTF8);
                    chapters.push_back(std::move(chap));
                }
            }
            count++;
        }
        auto chapterTimeEnd = chapterAtom->FirstChildElement(CHAPTER_TIME_END);
        if (chapterTimeEnd) {
            auto timeEnd = chapterTimeEnd->GetText();
            int time[4] = { 0 };
            if (   4 == sscanf_s(timeEnd, "%d:%d:%d:%d", &time[0], &time[1], &time[2], &time[3])
                || 4 == sscanf_s(timeEnd, "%d:%d:%d.%d", &time[0], &time[1], &time[2], &time[3])
                || 4 == sscanf_s(timeEnd, "%d:%d.%d.%d", &time[0], &time[1], &time[2], &time[3])
                || 4 == sscanf_s(timeEnd, "%d.%d.%d.%d", &time[0], &time[1], &time[2], &time[3])) {
                ChapData chap;
                chap.h  = time[0];
                chap.m  = time[1];
                chap.s  = time[2];
                chap.ms = time[3];
                m_duration = std::max<decltype(m_duration)>(m_duration, chap.get_ms());
            }
        }
        addedCount += count;
    }
    return AUO_CHAP_ERR_NONE;
}

int ChapterRW::read_chapter_matroska() {
    int sts = AUO_CHAP_ERR_NONE;
    if (nullptr == m_filepath) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    static const char *const ROOT = "Chapters";
    static const char *const EDITION_ENTRY = "EditionEntry";

    unique_ptr<FILE, fp_deleter> fp;
    {
        FILE *fp_tmp = NULL;
        if (_tfopen_s(&fp_tmp, m_filepath, _T("rb"))) {
            return AUO_CHAP_ERR_FILE_OPEN;
        }
        fp.reset(fp_tmp);
    }
    tinyxml2::XMLDocument xml;
    if (tinyxml2::XML_NO_ERROR != xml.LoadFile(fp.get())) {
        sts = AUO_CHAP_ERR_INIT_READ_STREAM;
    } else {
        auto root = xml.FirstChildElement(ROOT);
        for (auto element = root->FirstChildElement(EDITION_ENTRY); element != nullptr; element = root->NextSiblingElement(EDITION_ENTRY)) {
            int count = 0;
            if ((sts = read_chapter_matroska_chapter_atom(element, count)) != AUO_CHAP_ERR_NONE) {
                return sts;
            }
        }
    }
    return sts;
}

int ChapterRW::read_chapter() {
    int sts = AUO_CHAP_ERR_NONE;
    if (0 == m_wchar_filedata.length()) {
        if (CHAP_TYPE_UNKNOWN == check_chap_type_from_file()) {
            return AUO_CHAP_ERR_CP_DETECT;
        }
    }
    switch (m_chapter_type) {
    case CHAP_TYPE_NERO:
        return read_chapter_nero();
    case CHAP_TYPE_APPLE:
        return read_chapter_apple();
    case CHAP_TYPE_MATROSKA:
        return read_chapter_matroska();
    case CHAP_TYPE_ANOTHER:
    case CHAP_TYPE_UNKNOWN:
    default:
        return AUO_CHAP_ERR_INVALID_FMT;
    }
}

int ChapterRW::read_file() {
    return read_chapter();
}

void ChapterRW::add_dummy_chap_zero_pos() {
    if (chapters.size()) {
        if (0 != (chapters[0]->h | chapters[0]->m | chapters[0]->s | chapters[0]->ms)) {
            unique_ptr<ChapData> chap(new ChapData);
            chap->name = L"";
            chapters.insert(chapters.begin(), std::move(chap));
        }
    }
}

void ChapterRW::delay_chapter(int delay_ms) {
    for (uint32_t i = 0; i < chapters.size(); i++) {
        uint32_t chap_time_ms = chapters[i]->get_ms();
        if (chap_time_ms) {
            chapters[i]->set_ms(chap_time_ms + delay_ms);
        }
    }
}
