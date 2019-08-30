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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>

#include <objbase.h>
#pragma comment(lib, "ole32.lib")
#include <mlang.h>
#include <xmllite.h>
#pragma comment (lib, "xmllite.lib")
#include <shlwapi.h>
#pragma comment (lib, "shlwapi.lib")

#include "auo.h"
#include "auo_util.h"
#include "auo_chapter.h"

using std::vector;
using std::string;
using std::wstring;

chapter_file::chapter_file() {
    CoInitialize(NULL);
    init();
}

chapter_file::~chapter_file() {
    close();
    CoUninitialize();
}

void chapter_file::init() {
    close();

    pImul = nullptr;
    filepath = nullptr;
    sts = AUO_CHAP_ERR_NONE;
    chapter_type = CHAP_TYPE_UNKNOWN;
    code_page = CODE_PAGE_UNSET;
    duration = 0.0;
}

void chapter_file::close() {
    if (chapters.size()) {
        std::vector<std::unique_ptr<chapter>>().swap(chapters);
    }
    if (nullptr != pImul) {
        pImul->Release();
    }
    pImul = nullptr;
}

int chapter_file::file_chapter_type() {
    return chapter_type;
}

DWORD chapter_file::file_code_page() {
    return code_page;
}

int chapter_file::get_result() {
    return sts;
}

 int chapter_file::read_file(const char *chap_filepath, DWORD chap_code_page, double vid_duration) {
    init();
    this->filepath = chap_filepath;
    this->code_page = chap_code_page;
    this->duration = vid_duration;
    return read_file();
}

string chapter_file::to_utf8(wstring string) {
    DWORD encMode = 0;
    std::string string_utf8;
    string_utf8.resize(string.length() * MAX_UTF8_CHAR_LENGTH, '\0');
    UINT dst_len = (UINT)string_utf8.size();
    if (S_OK != pImul->ConvertString(&encMode, CODE_PAGE_UTF16_LE, CODE_PAGE_UTF8, (BYTE *)&string[0], nullptr, (BYTE *)&string_utf8[0], &dst_len)) {
        string_utf8 = "";
    } else if (dst_len) {
        string_utf8.resize(dst_len);
    }
    return string_utf8;
}

int chapter_file::write_chapter_apple_header(std::ostream& ostream) {
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    ostream.write((const char *)UTF8_BOM, sizeof(UTF8_BOM));
    ostream << to_utf8(
    L"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\r\n"
    L"<TextStream version=\"1.1\">\r\n"
    L"<TextStreamHeader>\r\n"
    L"<TextSampleDescription>\r\n"
    L"</TextSampleDescription>\r\n"
    L"</TextStreamHeader>\r\n");
    return sts;
}

//終端 長さも指定する(しないとわけのわからんdurationになる)
int chapter_file::write_chapter_apple_foot(std::ostream& ostream) {
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_NULL_PTR;
    } else {
        int duration_ms = (int)(duration * 1000.0 + 0.5);
        if (duration <= 0 && 0 < chapters.size()) {
            duration_ms = chapters.back()->get_ms() + 1;
        }

        ostream << to_utf8(strprintf(
            L"<TextSample sampleTime=\"%02d:%02d:%02d.%03d\" text=\"\" />\r\n</TextStream>",
            duration_ms / (60*60*1000),
            (duration_ms % (60*60*1000)) / (60*1000),
            (duration_ms % (60*1000)) / 1000,
            duration_ms % 1000
            ));
    }
    return sts;
}

int chapter_file::write_chapter_apple(const char *out_filepath) {
    std::ofstream ostream(out_filepath, std::ios::out | std::ios::binary);
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_FILE_OPEN;
    } else {
        write_chapter_apple_header(ostream);
        for (const auto& chap : chapters) {
            ostream << to_utf8(strprintf(
                L"<TextSample sampleTime=\"%02d:%02d:%02d.%03d\">%s</TextSample>\r\n",
                chap->h, chap->m, chap->s, chap->ms, chap->name.c_str()  ));
        }
        write_chapter_apple_foot(ostream);
        ostream.close();
    }
    return sts;
}

int chapter_file::write_chapter_nero(const char *out_filepath, bool utf8) {
    std::ofstream ostream(out_filepath, std::ios::out | std::ios::binary);
    if (!ostream.good()) {
        sts = AUO_CHAP_ERR_FILE_OPEN;
    } else {
        if (utf8)
            ostream.write((const char *)UTF8_BOM, sizeof(UTF8_BOM));

        const DWORD output_codepage = (utf8) ? CODE_PAGE_UTF8 : CODE_PAGE_SJIS;
        std::vector<char> char_buffer;
        for (DWORD i = 0; i < chapters.size(); i++) {
            const auto& chap = chapters[i];
            static const char * const KEY_BASE = "CHAPTER";
            static const char * const KEY_NAME = "NAME";
            const DWORD chapter_name_length = chap->name.length() + 1;
            if (char_buffer.size() < chapter_name_length * 4)
                char_buffer.resize(chapter_name_length * 4);
            std::fill(char_buffer.begin(), char_buffer.end(), 0);

            DWORD encMode = 0;
            UINT buf_len_in_byte = char_buffer.size();
            if (S_OK != pImul->ConvertString(&encMode, CODE_PAGE_UTF16_LE, output_codepage, (BYTE *)chap->name.c_str(), nullptr, (BYTE *)&char_buffer[0], &buf_len_in_byte))
                return AUO_CHAP_ERR_CONVERTION;

            ostream << strprintf("%s%02d=%02d:%02d:%02d.%03d\r\n", KEY_BASE, i+1, chap->h, chap->m, chap->s, chap->ms);
            ostream << strprintf("%s%02d%s=%s\r\n", KEY_BASE, i+1, KEY_NAME, &char_buffer[0]);
        }
        ostream.close();
    }
    return sts;
}

int chapter_file::write_file(const char *out_filepath, int out_chapter_type, bool nero_in_utf8) {
    if (CHAP_TYPE_UNKNOWN == out_chapter_type) {
        out_chapter_type = chapter_type;
    } else if (CHAP_TYPE_ANOTHER == out_chapter_type) {
        out_chapter_type = (CHAP_TYPE_NERO == chapter_type) ? CHAP_TYPE_APPLE : CHAP_TYPE_NERO;
    }
    sts = (CHAP_TYPE_NERO == out_chapter_type) ? write_chapter_nero(out_filepath, nero_in_utf8) : write_chapter_apple(out_filepath);
    return sts;
}

int chapter_file::overwrite_file(int out_chapter_type, bool nero_in_utf8) {
    std::string temp_file = filepath;
    temp_file += ".tmp";
    for (int i = 0; PathFileExists(temp_file.c_str()); i++) {
        temp_file = strprintf("%s.tmp%d", filepath, i);
    }
    sts = write_file(temp_file.c_str(), out_chapter_type, nero_in_utf8);
    if (AUO_CHAP_ERR_NONE != sts) {
        remove(temp_file.c_str());
    } else {
        remove(filepath);
        rename(temp_file.c_str(), filepath);
    }
    return sts;
}

DWORD chapter_file::check_code_page(vector<char>& src, DWORD orig_code_page) {
    DetectEncodingInfo dEnc = { 0 };
    int denc_count = 1;
    int src_buf_len = src.size();
    if (   CODE_PAGE_UNSET == orig_code_page //指定があればスキップ
        && CODE_PAGE_UNSET == (dEnc.nCodePage = get_code_page(&src[0], src.size())) //まず主に日本語をチェック
        && S_OK != pImul->DetectInputCodepage(MLDETECTCP_NONE, 0, &src[0], &src_buf_len, &dEnc, &denc_count) //IMultiLanguage2で判定してみる
        && TRUE != fix_ImulL_WesternEurope(&dEnc.nCodePage))
        return CODE_PAGE_UNSET;

    return dEnc.nCodePage;
}

int chapter_file::get_unicode_data(wstring& wchar_data, vector<char>& src) {
    wchar_data.resize(src.size(), 0);

    if (code_page == CODE_PAGE_UTF16_LE) {
        memcpy(&wchar_data[0], &src[0], src.size());
        wchar_data.resize(src.size() / sizeof(wchar_t));
    } else {
        int start_index = 0;
        if (CODE_PAGE_UTF8 == code_page && 0 == memcmp(&src[0], UTF8_BOM, sizeof(UTF8_BOM)))
            start_index = sizeof(UTF8_BOM);

        DWORD encMode = 0;
        UINT src_size = src.size() - start_index;
        UINT dst_output_len = wchar_data.size();
        if (S_OK != pImul->ConvertStringToUnicode(&encMode, code_page, &src[start_index], &src_size, &wchar_data[0], &dst_output_len)) {
            sts = AUO_CHAP_ERR_CONVERTION;
        }
        wchar_data.resize(dst_output_len);
    }
    return sts;
}

int chapter_file::get_unicode_data_from_file(wstring& wchar_data) {
    using std::ios;
    using std::ifstream;
    //ファイルを一気に読み込み
    vector<char> file_data;
    ifstream inputFile(filepath, ios::in | ios::binary);
    if (!inputFile.good()) {
        sts = AUO_CHAP_ERR_FILE_OPEN;
    } else {
        file_data.resize((size_t)inputFile.seekg(0, std::ios::end).tellg() + 1, '\0');
        inputFile.seekg(0, std::ios::beg).read(&file_data[0], static_cast<std::streamsize>(file_data.size()));

        if (0 == file_data.size()) {
            sts = AUO_CHAP_ERR_FILE_OPEN;
        //文字コード判定
        } else if (CODE_PAGE_UNSET == (code_page = check_code_page(file_data, code_page))) {
            sts = AUO_CHAP_ERR_CP_DETECT;
        } else {
            //文字コード変換
            sts = get_unicode_data(wchar_data, file_data);
        }
    }
    return sts;
}

int chapter_file::check_chap_type(const wstring& data) {
    if (0 == data.length())
        sts = CHAP_TYPE_UNKNOWN;
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
        && string::npos != data.find(L"<TextSample"))
        return CHAP_TYPE_APPLE;

    return CHAP_TYPE_UNKNOWN;
}

int chapter_file::check_chap_type_from_file() {
    //文字コード変換してファイル内容を取得
    if (AUO_CHAP_ERR_NONE != (sts = get_unicode_data_from_file(wchar_filedata))) {
        chapter_type = CHAP_TYPE_UNKNOWN;
    } else {
        chapter_type = check_chap_type(wchar_filedata);
    }
    return sts;
}

int chapter_file::read_chapter_nero() {
    if (0 == wchar_filedata.length() || nullptr == pImul) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    //行単位に分解
    const wchar_t delim = (string::npos != wchar_filedata.find(L"\n")) ? L'\n' : L'\r'; //適切な改行コードを見つける
    auto pw_line = split(wchar_filedata, delim);


    //読み取り
    static const wchar_t * const CHAP_KEY = L"CHAPTER";
    const wchar_t *pw_key[] = { CHAP_KEY, nullptr, nullptr }; // 時間行, 名前行, ダミー
    wchar_t *pw_data[2];
    const int total_lines = pw_line.size();
    for (int i = 0; i < total_lines && !sts; i++) {
        //末尾の改行空白を取り除く
        pw_line[i] = pw_line[i].substr(0, pw_line[i].find_last_not_of(L"\r\n ") + 1);
        if (string::npos == pw_line[i].find(pw_key[i&1], 0, wcslen(pw_key[i&1])))
            return AUO_CHAP_ERR_INVALID_FMT;
        pw_key[(i&1) + 1] = &pw_line[i][0];//CHAPTER KEY名を保存
        pw_data[i&1] = wcschr(&pw_line[i][0], L'=');
        *pw_data[i&1] = L'\0'; //CHAPTER KEY名を一つの文字列として扱えるように
        pw_data[i&1]++; //データは'='の次から
        if (i&1) {
            //読み取り
            std::unique_ptr<chapter> chap(new chapter());
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

int chapter_file::read_chapter_apple() {
    if (nullptr == filepath || nullptr == pImul) {
        sts = AUO_CHAP_ERR_NULL_PTR;
        return sts;
    }

    static const wchar_t * const ELEMENT_NAME = L"TextSample";
    static const wchar_t * const ATTRIBUTE_NAME = L"sampleTime";

    IXmlReader *pReader = nullptr;
    IStream *pStream = nullptr;

    CoInitialize(NULL);

    if (S_OK != CreateXmlReader(IID_PPV_ARGS(&pReader), NULL))
        sts = AUO_CHAP_ERR_INIT_XML_PARSER;
    else if (S_OK != SHCreateStreamOnFile(filepath, STGM_READ, &pStream))
        sts = AUO_CHAP_ERR_INIT_READ_STREAM;
    else if (S_OK != pReader->SetInput(pStream))
        sts = AUO_CHAP_ERR_FAIL_SET_STREAM;
    else {
        const wchar_t *pwLocalName = NULL, *pwValue = NULL;
        XmlNodeType nodeType;
        int time[4] = { 0 };
        bool flag_next_line_is_time = true; //次は時間を取得するべき

        while (S_OK == pReader->Read(&nodeType)) {
            switch (nodeType) {
                case XmlNodeType_Element:
                    if (S_OK != pReader->GetLocalName(&pwLocalName, NULL))
                        return AUO_CHAP_ERR_PARSE_XML;
                    if (wcscmp(ELEMENT_NAME, pwLocalName))
                        break;
                    if (S_OK != pReader->MoveToFirstAttribute())
                        break;
                    do {
                        const wchar_t *pwAttributeName = NULL;
                        const wchar_t *pwAttributeValue = NULL;
                        if (S_OK != pReader->GetLocalName(&pwAttributeName, NULL))
                            break;
                        if (_wcsicmp(ATTRIBUTE_NAME, pwAttributeName))
                            break;
                        if (S_OK != pReader->GetValue(&pwAttributeValue, NULL))
                            break;
                        //必要ならバッファ拡張(想定される最大限必要なバッファに設定)
                        if (   4 != swscanf_s(pwAttributeValue, L"%d:%d:%d:%d\r\n", &time[0], &time[1], &time[2], &time[3])
                            && 4 != swscanf_s(pwAttributeValue, L"%d:%d:%d.%d\r\n", &time[0], &time[1], &time[2], &time[3])
                            && 4 != swscanf_s(pwAttributeValue, L"%d:%d.%d.%d\r\n", &time[0], &time[1], &time[2], &time[3])
                            && 4 != swscanf_s(pwAttributeValue, L"%d.%d.%d.%d\r\n", &time[0], &time[1], &time[2], &time[3]))
                            return AUO_CHAP_ERR_PARSE_XML;
                        flag_next_line_is_time = FALSE;
                    } while (S_OK == pReader->MoveToNextAttribute());
                    break;
                case XmlNodeType_Text:
                    if (S_OK != pReader->GetValue(&pwValue, NULL))
                        break;
                    if (pwLocalName == NULL || wcscmp(pwLocalName, ELEMENT_NAME))
                        break;
                    if (flag_next_line_is_time)
                        break;
                    //変換
                    {
                        std::unique_ptr<chapter> chap(new chapter());
                        chap->h  = time[0];
                        chap->m  = time[1];
                        chap->s  = time[2];
                        chap->ms = time[3];
                        chap->name = pwValue;
                        chapters.push_back(std::move(chap));
                        flag_next_line_is_time = true;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    //リソース解放
    if (pReader)
        pReader->Release();
    if (pStream)
        pStream->Release();
    CoUninitialize();

    return sts;
}

int chapter_file::read_chapter() {
    if (0 == wchar_filedata.length()) {
        if (AUO_CHAP_ERR_NONE != (sts = check_chap_type_from_file())) {
            return sts;
        }
    }
    sts = (CHAP_TYPE_NERO == chapter_type) ? read_chapter_nero() : read_chapter_apple();
    return sts;
}

int chapter_file::read_file() {
    if (S_OK != CoCreateInstance(CLSID_CMultiLanguage, NULL, CLSCTX_INPROC_SERVER, IID_IMultiLanguage2, (void**)&pImul) || nullptr == pImul) {
        sts = AUO_CHAP_ERR_INIT_IMUL2;
    } else {
        sts = read_chapter();
    }
    return sts;
}

void chapter_file::add_dummy_chap_zero_pos() {
    if (chapters.size()) {
        if (0 != (chapters[0]->h | chapters[0]->m | chapters[0]->s | chapters[0]->ms)) {
            std::unique_ptr<chapter> chap(new chapter);
            chap->name = L"";
            chapters.insert(chapters.begin(), std::move(chap));
        }
    }
}

void chapter_file::delay_chapter(int delay_ms) {
    for (DWORD i = 0; i < chapters.size(); i++) {
        DWORD chap_time_ms = chapters[i]->get_ms();
        if (chap_time_ms) {
            chapters[i]->set_ms(chap_time_ms + delay_ms);
        }
    }
}
