// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#include <fstream>
#include <iostream>
#include <algorithm>
#include "rgy_osdep.h"
#if defined(_WIN32) || defined(_WIN64)
#include <mlang.h>
#include <shlwapi.h>
#pragma comment (lib, "shlwapi.lib")
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <iconv.h>
#endif //#if defined(_WIN32) || defined(_WIN64)

#include <tinyxml2.h>

#include "rgy_chapter.h"
#include "rgy_codepage.h"
#include "rgy_util.h"


#if defined(_WIN32) || defined(_WIN64)
struct iunknown_deleter {
    void operator()(IUnknown *ptr) const {
        ptr->Release();
        CoUninitialize();
    }
};

static BOOL fix_ImulL_WesternEurope(uint32_t *code_page) {
    //IMultiLanguage2 の DetectInputCodepage はよく西ヨーロッパ言語と誤判定しやがる
    if (*code_page == CODE_PAGE_WEST_EUROPE)
        *code_page = CODE_PAGE_SJIS;
    return TRUE;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

static const int MAX_UTF8_CHAR_LENGTH = 6;
static const uint8_t UTF8_BOM[] = { 0xEF, 0xBB, 0xBF };

ChapterRW::ChapterRW() :
    m_filedata(),
    m_filepath(),
    m_chapter_type(CHAP_TYPE_UNKNOWN),
    m_code_page(0),
    m_duration(0.0),
    chapters() {
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
    chapters.clear();
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
    ostream <<
    "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\r\n"
    "<TextStream version=\"1.1\">\r\n"
    "<TextStreamHeader>\r\n"
    "<TextSampleDescription>\r\n"
    "</TextSampleDescription>\r\n"
    "</TextStreamHeader>\r\n";
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

        ostream << strsprintf(
            "<TextSample sampleTime=\"%02d:%02d:%02d.%03d\" text=\"\" />\r\n</TextStream>",
            duration_ms / (60*60*1000),
            (duration_ms % (60*60*1000)) / (60*1000),
            (duration_ms % (60*1000)) / 1000,
            duration_ms % 1000
            );
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
            ostream << strsprintf(
                "<TextSample sampleTime=\"%02d:%02d:%02d.%03d\">%s</TextSample>\r\n",
                chap->h, chap->m, chap->s, chap->ms, chap->name.c_str() );
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

        const uint32_t output_codepage = (utf8) ? CODE_PAGE_UTF8 : CP_THREAD_ACP;
        for (uint32_t i = 0; i < chapters.size(); i++) {
            const auto& chap = chapters[i];
            static const char * const KEY_BASE = "CHAPTER";
            static const char * const KEY_NAME = "NAME";

            ostream << strsprintf("%s%02d=%02d:%02d:%02d.%03d\r\n", KEY_BASE, i+1, chap->h, chap->m, chap->s, chap->ms);
            ostream << strsprintf("%s%02d%s=%s\r\n", KEY_BASE, i+1, KEY_NAME, char_to_string(output_codepage, chap->name.c_str(), CODE_PAGE_UTF8).c_str());
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
        _tremove(temp_file.c_str());
    } else {
        _tremove(m_filepath);
        _trename(temp_file.c_str(), m_filepath);
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

int ChapterRW::get_unicode_data(std::string& data, vector<char>& src) {
    int sts = AUO_CHAP_ERR_NONE;
    data.clear();

    if (m_code_page == CODE_PAGE_UTF16_LE) {
        data = wstring_to_string((wchar_t *)src.data(), CODE_PAGE_UTF8);
    } else {
        int start_index = 0;
        if (CODE_PAGE_UTF8 == m_code_page && 0 == memcmp(&src[0], UTF8_BOM, sizeof(UTF8_BOM)))
            start_index = sizeof(UTF8_BOM);

        data = char_to_string(CODE_PAGE_UTF8, &src[start_index], m_code_page);
    }
    return sts;
}

int ChapterRW::get_unicode_data_from_file(std::string& data) {
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
            sts = get_unicode_data(data, file_data);
        }
    }
    return sts;
}

ChapType ChapterRW::check_chap_type(const std::string& data) {
    int sts = AUO_CHAP_ERR_NONE;
    if (0 == data.length()) {
        sts = CHAP_TYPE_UNKNOWN;
    }
    auto pos = data.find("CHAPTER");
    auto first_pos = pos;
    if (   std::string::npos != pos
        && std::string::npos != (pos = first_pos = data.find("=", pos))
        && std::string::npos != (pos = data.find_first_of("\r\n", pos))
        && std::string::npos != (pos = data.find_first_not_of("\r\n", pos))
        && std::string::npos != data.substr(pos).find(data.substr(0, first_pos))) {
        return CHAP_TYPE_NERO;
    }
    if (   std::string::npos != data.find("<TextStream")
        && std::string::npos != data.find("<TextSample")) {
        return CHAP_TYPE_APPLE;
    }
    if (std::string::npos != data.find("<Chapters")
        && std::string::npos != data.find("<EditionEntry")
        && std::string::npos != data.find("<ChapterAtom")
        && std::string::npos != data.find("<ChapterTimeStart")) {
        return CHAP_TYPE_MATROSKA;
    }

    return CHAP_TYPE_UNKNOWN;
}

ChapType ChapterRW::check_chap_type_from_file() {
    //文字コード変換してファイル内容を取得
    if (AUO_CHAP_ERR_NONE != get_unicode_data_from_file(m_filedata)) {
        m_chapter_type = CHAP_TYPE_UNKNOWN;
    } else {
        m_chapter_type = check_chap_type(m_filedata);
    }
    return m_chapter_type;
}

int ChapterRW::read_chapter_nero() {
    int sts = AUO_CHAP_ERR_NONE;
    if (0 == m_filedata.length()) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    //行単位に分解
    const char *delim = (std::string::npos != m_filedata.find("\n")) ? "\n" : "\r"; //適切な改行コードを見つける
    auto pw_line = split(m_filedata, delim);

    //読み取り
    static const char * const CHAP_KEY = "CHAPTER";
    const char *pw_key[] = { CHAP_KEY, nullptr, nullptr }; // 時間行, 名前行, ダミー
    char *pw_data[2] = { nullptr, nullptr };
    const uint32_t total_lines = (uint32_t)pw_line.size();
    for (uint32_t i = 0; i < total_lines && !sts; i++) {
        //末尾の改行空白を取り除く
        pw_line[i] = pw_line[i].substr(0, pw_line[i].find_last_not_of("\r\n ") + 1);
        if (strlen(pw_line[i].c_str()) == 0) {
            break;
        }
        if (std::string::npos == pw_line[i].find(pw_key[i&1], 0, strlen(pw_key[i&1])))
            return AUO_CHAP_ERR_INVALID_FMT;
        pw_key[(i&1) + 1] = &pw_line[i][0];//CHAPTER KEY名を保存
        pw_data[i&1] = strchr(&pw_line[i][0], '=');
        *pw_data[i&1] = '\0'; //CHAPTER KEY名を一つの文字列として扱えるように
        pw_data[i&1]++; //データは'='の次から
        if (i&1) {
            //読み取り
            unique_ptr<ChapData> chap(new ChapData());
            if (   4 != sscanf_s(pw_data[0], "%d:%d:%d:%d", &chap->h, &chap->m, &chap->s, &chap->ms)
                && 4 != sscanf_s(pw_data[0], "%d:%d:%d.%d", &chap->h, &chap->m, &chap->s, &chap->ms)
                && 4 != sscanf_s(pw_data[0], "%d:%d.%d.%d", &chap->h, &chap->m, &chap->s, &chap->ms)
                && 4 != sscanf_s(pw_data[0], "%d.%d.%d.%d", &chap->h, &chap->m, &chap->s, &chap->ms)) {
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
                chap->name = element->GetText();
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
                    chap->name = chapterDisplay->FirstChildElement(CHAPTER_STRING)->GetText();
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
    if (0 == m_filedata.length()) {
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
            chap->name = "";
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
